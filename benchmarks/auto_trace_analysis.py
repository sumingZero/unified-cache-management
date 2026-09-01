"""UCM Trace hit-rate analysis CLI.

Collects trace logs, parses topology, dispatches to the appropriate
simulation engine (standard / mamba / fawa), and reports four-tier
hit-rate scenarios.

All core data structures (BlockPool, ByteLRUPool, GroupContext), parsing
logic (SimTopology, TraceRecord), and simulation engines reside in this
single file.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import random
import re
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Iterable

# ============================================================================
# Constants & regexes
# ============================================================================

GIB = 1024**3
PROMPT_TOKENS_TOTAL_METRICS = (
    "vllm:prompt_tokens_total",
    "prompt_tokens_total",
)
PROMPT_TOKENS_CACHE_HIT_METRICS = (
    'vllm:prompt_tokens_by_source_total{source="local_cache_hit"}',
    'prompt_tokens_by_source_total{source="local_cache_hit"}',
)

AVAILABLE_KV_RE = re.compile(
    r"\b(?:available|current)[_\s-]*(?:kv[_\s-]*)?cache[_\s-]*memory\b"
    r"[^0-9]*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[kmgt]?i?b|bytes?)?",
    re.IGNORECASE,
)
TP_SIZE_RE = re.compile(
    r"(?:['\"]?tensor[_-]parallel[_-]size['\"]?\s*[:=]\s*|"
    r"--tensor[-_]parallel[-_]size\s+)"
    r"(?P<value>\d+)",
    re.IGNORECASE,
)
DP_SIZE_RE = re.compile(
    r"(?:['\"]?data[_-]parallel[_-]size['\"]?\s*[:=]\s*|"
    r"--data[-_]parallel[-_]size\s+)"
    r"(?P<value>\d+)",
    re.IGNORECASE,
)
PROM_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)

_UCM_TRACE_META_RE = re.compile(r"UCMTraceMeta:\s*(?P<fields>.+)", re.IGNORECASE)
_UCM_TRACE_RE = re.compile(
    r"UCMTrace:\s*"
    r"timestamp:\s*(?P<timestamp>\d+(?:\.\d+)?),\s*"
    r"(?:request_id:\s*(?P<request_id>[^,]+),\s*)?"
    r"input_length:\s*(?P<input_length>\d+),\s*"
    r"output_length:\s*(?P<output_length>\d+),\s*"
    r"block_hashes:\s*(?P<block_hashes>\[.*?\])"
)
_TRACE_RE = re.compile(
    r"timestamp:\s*(?P<timestamp>\d+(?:\.\d+)?),\s*"
    r"(?:request_id:\s*(?P<request_id>[^,]+),\s*)?"
    r"input_length:\s*(?P<input_length>\d+),\s*"
    r"output_length:\s*(?P<output_length>\d+),\s*"
    r"ucm_block_ids:\s*(?P<ucm_block_ids>\[.*?\])"
)
_SYSTEM_TIME_RE = re.compile(r"^\[(?P<system_time>\d{4}-\d{2}-\d{2} [^\]]+)\]")


# ============================================================================
# Core data structures
# ============================================================================

class BlockPool:
    """Simulates vLLM's HBM BlockPool: alloc/free/touch/rescue lifecycle.

    All block_ids share one pool (vLLM global BlockPool). The free queue is
    ordered by release time: head = oldest released, tail = newest. ``alloc``
    reuses from the head (oldest); ``free`` with hash appends to tail (survives
    longer), without hash prepends to head (immediate reuse).

    Prefix cache entries are keyed by ``(hash, group_id)`` so different groups
    sharing the same content hash map to different physical blocks.

    block_id 0 is reserved as null_block (never allocated, never freed),
    matching vLLM's BlockPool.__init__.
    """

    def __init__(self, capacity_block_ids: int):
        self._capacity = max(0, capacity_block_ids)
        self._free_deque: OrderedDict[int, None] = OrderedDict()
        for i in range(1, self._capacity + 1):
            self._free_deque[i] = None
        self._in_use: dict[int, int] = {}
        self._hash_to_block: dict = {}
        self._block_hashes: dict[int, set] = {}

    def num_free(self) -> int:
        return len(self._free_deque)

    def num_in_use(self) -> int:
        return len(self._in_use)

    def alloc(self) -> int | None:
        if not self._free_deque:
            return None
        block_id, _ = self._free_deque.popitem(last=False)
        self._evict_hashes(block_id)
        self._in_use[block_id] = 1
        return block_id

    def touch(self, hash_val, group_id: int) -> bool:
        key = (hash_val, group_id)
        block_id = self._hash_to_block.get(key)
        if block_id is None:
            return False
        if block_id in self._in_use:
            self._in_use[block_id] += 1
            return True
        del self._free_deque[block_id]
        self._in_use[block_id] = 1
        return True

    def touch_get(self, hash_val, group_id: int) -> int | None:
        key = (hash_val, group_id)
        block_id = self._hash_to_block.get(key)
        if block_id is None:
            return None
        if block_id in self._in_use:
            self._in_use[block_id] += 1
            return block_id
        del self._free_deque[block_id]
        self._in_use[block_id] = 1
        return block_id

    def peek(self, hash_val, group_id: int) -> bool:
        return (hash_val, group_id) in self._hash_to_block

    def cache_block(self, block_id: int, hash_val, group_id: int) -> None:
        key = (hash_val, group_id)
        self._hash_to_block[key] = block_id
        self._block_hashes.setdefault(block_id, set()).add(key)

    def free(self, block_id: int, has_hash: bool) -> None:
        self._in_use[block_id] -= 1
        if self._in_use[block_id] > 0:
            return
        del self._in_use[block_id]
        if has_hash:
            self._free_deque[block_id] = None
        else:
            self._free_deque[block_id] = None
            self._free_deque.move_to_end(block_id, last=False)

    def free_reverse(self, blocks: list[tuple[int, bool]]) -> None:
        for block_id, has_hash in reversed(blocks):
            self.free(block_id, has_hash)

    def _evict_hashes(self, block_id: int) -> None:
        keys = self._block_hashes.pop(block_id, None)
        if keys is None:
            return
        for key in keys:
            cached = self._hash_to_block.get(key)
            if cached == block_id:
                del self._hash_to_block[key]


@dataclass(frozen=True)
class ByteCacheEntry:
    producer_index: int


class RequestGroups:
    """Union-find tracking request lifetime (first appearance → last hit)."""

    def __init__(self) -> None:
        self.parent: list[int] = []
        self.first_timestamp: list[float] = []
        self.last_hit_timestamp: list[float | None] = []

    def add(self, timestamp: float) -> int:
        index = len(self.parent)
        self.parent.append(index)
        self.first_timestamp.append(timestamp)
        self.last_hit_timestamp.append(None)
        return index

    def find(self, index: int) -> int:
        parent = self.parent[index]
        if parent != index:
            self.parent[index] = self.find(parent)
        return self.parent[index]

    def union_roots(self, roots: Iterable[int]) -> int:
        root_set = {self.find(r) for r in roots}
        if not root_set:
            raise ValueError("cannot union empty request group")
        root = min(root_set, key=lambda i: self.first_timestamp[i])
        for item in root_set:
            if item == root:
                continue
            self.parent[item] = root
            item_last = self.last_hit_timestamp[item]
            if item_last is not None:
                root_last = self.last_hit_timestamp[root]
                self.last_hit_timestamp[root] = (
                    item_last if root_last is None else max(root_last, item_last)
                )
        return root

    def record_hit(self, root: int, timestamp: float) -> None:
        root = self.find(root)
        last = self.last_hit_timestamp[root]
        self.last_hit_timestamp[root] = (
            timestamp if last is None else max(last, timestamp)
        )

    def lifetimes(self) -> list[float]:
        values: list[float] = []
        for i, last in enumerate(self.last_hit_timestamp):
            if self.find(i) != i or last is None:
                continue
            values.append(last - self.first_timestamp[i])
        return values


def _nearest_percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = ceil(len(s) * pct / 100) - 1
    return s[max(0, min(idx, len(s) - 1))]


def _lifetime_stats(values: list[float]) -> dict:
    return {
        "request_lifetime_sample_count": len(values),
        "average_request_lifetime_seconds": sum(values) / len(values) if values else 0.0,
        "p90_request_lifetime_seconds": _nearest_percentile(values, 90),
        "p95_request_lifetime_seconds": _nearest_percentile(values, 95),
    }


class ByteLRUPool:
    """Byte-level LRU pool for UCM DRAM/FS stores.

    Unlike BlockPool (block-count capacity), ByteLRUPool tracks bytes. Entries
    can have different sizes (FA ``fa_file_size`` vs WA ``wa_file_size``),
    so eviction is by byte budget, not entry count.
    """

    def __init__(self, capacity_bytes: int):
        self._capacity = max(0, capacity_bytes)
        self._items: OrderedDict = OrderedDict()
        self._used_bytes = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    def get(self, key) -> ByteCacheEntry | None:
        if self._capacity <= 0 or key not in self._items:
            return None
        self._items.move_to_end(key)
        return self._items[key][0]

    def peek(self, key) -> ByteCacheEntry | None:
        if self._capacity <= 0 or key not in self._items:
            return None
        return self._items[key][0]

    def put(self, key, size_bytes: int, entry: ByteCacheEntry) -> None:
        if self._capacity <= 0 or size_bytes > self._capacity:
            return
        if key in self._items:
            old_size = self._items[key][1]
            self._used_bytes -= old_size
            self._items[key] = (entry, size_bytes)
            self._items.move_to_end(key)
        else:
            self._items[key] = (entry, size_bytes)
        self._used_bytes += size_bytes
        self._evict(key)

    def _evict(self, keep_key) -> None:
        while self._used_bytes > self._capacity and len(self._items) > 1:
            evict_key, (_, evict_size) = self._items.popitem(last=False)
            self._used_bytes -= evict_size
            if evict_key == keep_key:
                self._items[evict_key] = (ByteCacheEntry(0), evict_size)
                self._used_bytes += evict_size


@dataclass
class GroupContext:
    """Per-group runtime state for HBM simulation.

    One instance per KV cache group, mirroring vLLM's
    ``SingleTypeKVCacheManager``. Tracks block allocation, hash derivation,
    and release rules specific to this group's manager type.
    """

    group_id: int
    logical_block_size: int
    vllm_hash_block_size: int
    manager_type: str
    sliding_window: int | None = None
    compress_ratio: int = 1
    alignment_tokens: int = 0
    block_ids: list[int | None] = field(default_factory=list)
    num_cached: int = 0

    @property
    def scale_factor(self) -> int:
        return self.logical_block_size // self.vllm_hash_block_size

    def derive_block_hash(self, block_idx: int, chain: list) -> object:
        return chain[(block_idx + 1) * self.scale_factor - 1]

    def reachable(self, block_idx: int) -> bool:
        if self.manager_type != "sliding_window" or self.sliding_window is None:
            return True
        need = ceil((self.sliding_window - 1) / self.logical_block_size)
        per_segment = self.alignment_tokens // self.logical_block_size
        if need >= per_segment:
            return True
        return block_idx % per_segment >= per_segment - need

    def reset(self) -> None:
        self.block_ids = []
        self.num_cached = 0


# ============================================================================
# Topology / Trace data structures & parsing
# ============================================================================

@dataclass
class GroupSpec:
    name: str
    logical_block_size: int
    manager_type: str
    sliding_window: int | None = None
    compress_ratio: int = 1


@dataclass
class SimTopology:
    model_type: str
    is_mla: bool
    vllm_hash_block_size: int
    hbm_block_data_size: int
    lcm_block_size: int = 0
    mamba_groups: int = 0
    ucm_hash_block_size: int = 0
    fa_file_size: int = 0
    wa_file_size: int = 0
    alignment_tokens: int = 0
    group_specs: list[GroupSpec] = field(default_factory=list)

    def build_group_contexts(self) -> list[GroupContext]:
        at = self.alignment_tokens or self.vllm_hash_block_size
        return [
            GroupContext(
                group_id=i,
                logical_block_size=spec.logical_block_size,
                vllm_hash_block_size=self.vllm_hash_block_size,
                manager_type=spec.manager_type,
                sliding_window=spec.sliding_window,
                compress_ratio=spec.compress_ratio,
                alignment_tokens=at,
            )
            for i, spec in enumerate(self.group_specs)
        ]


@dataclass
class TraceRecord:
    timestamp: float
    input_length: int
    output_length: int
    hash_ids: list[str]
    source: str
    request_id: str | None = None
    system_time: str | None = None


def _extract_int(line: str, name: str) -> int:
    m = re.search(rf"\b{name}=(\d+)", line)
    return int(m.group(1)) if m else 0


def _extract_bool(line: str, name: str) -> bool:
    m = re.search(rf"\b{name}=(true|false)", line, re.IGNORECASE)
    return bool(m) and m.group(1).lower() == "true"


def _extract_str(line: str, name: str) -> str:
    m = re.search(rf"\b{name}=(\w+)", line)
    return m.group(1) if m else ""


def _extract_groups(line: str) -> list[GroupSpec]:
    m = re.search(r"\bgroups=(\[.*\])", line)
    if not m:
        return []
    try:
        raw = ast.literal_eval(m.group(1))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    specs: list[GroupSpec] = []
    for tup in raw:
        if not isinstance(tup, (tuple, list)) or len(tup) < 3:
            continue
        name = str(tup[0])
        lbs = int(tup[1])
        mtype = str(tup[2])
        sw = int(tup[3]) if len(tup) > 3 and tup[3] is not None else None
        cr = int(tup[4]) if len(tup) > 4 and tup[4] is not None else 1
        specs.append(GroupSpec(name, lbs, mtype, sw, cr))
    return specs


def parse_trace_meta(line: str) -> SimTopology | None:
    if not _UCM_TRACE_META_RE.search(line):
        return None
    model_type = _extract_str(line, "type")
    if model_type not in ("standard", "mamba", "fawa"):
        return None
    topo = SimTopology(
        model_type=model_type,
        is_mla=_extract_bool(line, "is_mla"),
        vllm_hash_block_size=_extract_int(line, "vllm_hash_block_size"),
        hbm_block_data_size=_extract_int(line, "hbm_block_data_size"),
    )
    if model_type == "mamba":
        topo.lcm_block_size = _extract_int(line, "lcm_block_size")
        topo.mamba_groups = _extract_int(line, "mamba_groups")
        topo.alignment_tokens = topo.lcm_block_size or topo.vllm_hash_block_size
    elif model_type == "fawa":
        topo.ucm_hash_block_size = _extract_int(line, "ucm_hash_block_size")
        topo.fa_file_size = _extract_int(line, "fa_file_size")
        topo.wa_file_size = _extract_int(line, "wa_file_size")
        topo.alignment_tokens = _extract_int(line, "alignment_tokens") or topo.vllm_hash_block_size
        topo.group_specs = _extract_groups(line)
    else:
        topo.alignment_tokens = topo.vllm_hash_block_size
    return topo


def parse_trace_line(line: str, source: str) -> TraceRecord | None:
    m = _UCM_TRACE_RE.search(line)
    field_key = "block_hashes"
    if m is None:
        m = _TRACE_RE.search(line)
        field_key = "ucm_block_ids"
    if m is None:
        return None
    try:
        hash_ids = ast.literal_eval(m.group(field_key))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(hash_ids, list):
        return None
    sys_match = _SYSTEM_TIME_RE.search(line)
    req_id = m.group("request_id")
    return TraceRecord(
        timestamp=float(m.group("timestamp")),
        input_length=int(m.group("input_length")),
        output_length=int(m.group("output_length")),
        hash_ids=[str(h) for h in hash_ids],
        source=source,
        request_id=req_id.strip() if req_id else None,
        system_time=sys_match.group("system_time") if sys_match else None,
    )


# ============================================================================
# Cache topology & shared helpers
# ============================================================================

def _rate(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def block_token_weights(record: TraceRecord) -> list[int]:
    n = len(record.hash_ids)
    if n == 0:
        return []
    base = record.input_length // n
    rem = record.input_length % n
    return [base + 1 if i < rem else base for i in range(n)]


@dataclass
class Topology:
    """Cache-pool topology: how HBM/DRAM/FS pools are partitioned."""
    is_mla: bool
    unified: bool
    num_nodes: int
    dp_size: int
    tp_size: int

    @property
    def dp_per_node(self) -> int:
        return max(1, self.dp_size // self.num_nodes) if self.num_nodes > 0 else 1

    def num_dram_pools(self) -> int:
        if self.unified:
            return 1
        return self.num_nodes if self.is_mla else self.dp_size

    def pools_for(self, dp_rank, hbm_pools, dram_pools, fs_pool):
        hbm = hbm_pools[dp_rank]
        if self.unified:
            dram = dram_pools[0]
        elif self.is_mla:
            node = min(dp_rank // self.dp_per_node, self.num_nodes - 1)
            dram = dram_pools[node]
        else:
            dram = dram_pools[dp_rank]
        return hbm, dram, fs_pool


# ============================================================================
# Standard model simulation
# ============================================================================

def simulate_standard(
    records: Iterable[TraceRecord],
    topo: Topology,
    vllm_hash_block_size: int,
    hbm_block_data_size: int,
    gpu_capacity_blocks: int,
    dram_capacity_bytes: int,
    fs_capacity_bytes: int,
    random_seed: int | None = 0,
) -> dict:
    if gpu_capacity_blocks < 0 or dram_capacity_bytes < 0 or fs_capacity_bytes < 0:
        raise ValueError("cache capacities must be >= 0")

    rng = random.Random(random_seed)
    hbm_pools = [BlockPool(gpu_capacity_blocks) for _ in range(topo.dp_size)]
    dram_pools = [ByteLRUPool(dram_capacity_bytes) for _ in range(topo.num_dram_pools())]
    fs_pool = ByteLRUPool(fs_capacity_bytes)

    entry_size = hbm_block_data_size
    group_id = 0
    request_groups = RequestGroups()
    producer_map: dict = {}

    total_tokens = 0
    gpu_hit_tokens = 0
    dram_hit_tokens = 0
    fs_hit_tokens = 0
    miss_tokens = 0

    for record in records:
        total_tokens += record.input_length
        request_index = request_groups.add(record.timestamp)
        if not record.hash_ids:
            miss_tokens += record.input_length
            continue

        dp_rank = rng.randrange(topo.dp_size) if topo.dp_size > 1 else 0
        hbm, dram, fs = topo.pools_for(dp_rank, hbm_pools, dram_pools, fs_pool)

        block_track: list[tuple[int, bool]] = []
        prefix_available = True
        weights = block_token_weights(record)
        hit_roots: set[int] = set()

        for hash_id, weight in zip(record.hash_ids, weights):
            if not prefix_available:
                bid = hbm.alloc()
                if bid is None:
                    miss_tokens += weight
                    continue
                hbm.cache_block(bid, hash_id, group_id)
                producer_map[hash_id] = request_index
                block_track.append((bid, True))
                miss_tokens += weight
                continue

            bid = hbm.touch_get(hash_id, group_id)
            if bid is not None:
                gpu_hit_tokens += weight
                block_track.append((bid, True))
                hit_roots.add(request_groups.find(producer_map.get(hash_id, request_index)))
                continue

            dram_entry = dram.get(hash_id)
            if dram_entry is not None:
                bid = hbm.alloc()
                if bid is None:
                    prefix_available = False
                    miss_tokens += weight
                    continue
                hbm.cache_block(bid, hash_id, group_id)
                producer_map[hash_id] = request_index
                block_track.append((bid, True))
                dram_hit_tokens += weight
                hit_roots.add(request_groups.find(dram_entry.producer_index))
                continue

            fs_entry = fs.get(hash_id)
            if fs_entry is not None:
                bid = hbm.alloc()
                if bid is None:
                    prefix_available = False
                    miss_tokens += weight
                    continue
                hbm.cache_block(bid, hash_id, group_id)
                producer_map[hash_id] = request_index
                dram.put(hash_id, entry_size, fs_entry)
                block_track.append((bid, True))
                fs_hit_tokens += weight
                hit_roots.add(request_groups.find(fs_entry.producer_index))
                continue

            bid = hbm.alloc()
            if bid is None:
                prefix_available = False
                miss_tokens += weight
                continue
            hbm.cache_block(bid, hash_id, group_id)
            producer_map[hash_id] = request_index
            block_track.append((bid, True))
            miss_tokens += weight
            prefix_available = False

        if hit_roots:
            root = request_groups.union_roots([request_index, *hit_roots])
            request_groups.record_hit(root, record.timestamp)

        entry = ByteCacheEntry(request_index)
        for hash_id in record.hash_ids:
            dram.put(hash_id, entry_size, entry)
            fs.put(hash_id, entry_size, entry)

        hbm.free_reverse(block_track)

    total_hit = gpu_hit_tokens + dram_hit_tokens + fs_hit_tokens
    result = {
        "total_tokens": total_tokens,
        "gpu_hit_tokens": gpu_hit_tokens,
        "dram_hit_tokens": dram_hit_tokens,
        "fs_hit_tokens": fs_hit_tokens,
        "miss_tokens": miss_tokens,
        "total_hit_tokens": total_hit,
        "hit_rate": _rate(total_hit, total_tokens),
    }
    result.update(_lifetime_stats(request_groups.lifetimes()))
    return result


# ============================================================================
# Mamba model simulation
# ============================================================================

def _chunk_end_boundaries(
    start: int, num_tokens: int, lcm: int, chunk_size: int | None
) -> set[int]:
    if chunk_size is None or chunk_size <= 0:
        return {b for b in range(lcm, num_tokens + 1, lcm) if b > start}
    step = (chunk_size // lcm) * lcm
    if step <= 0:
        step = lcm
    dumped: set[int] = set()
    pos = start
    while pos < num_tokens:
        nxt = min(pos + step, num_tokens)
        if nxt < num_tokens:
            aligned = (nxt // lcm) * lcm
            nxt = aligned if aligned > pos else pos + lcm
        if nxt > start and nxt % lcm == 0:
            dumped.add(nxt)
        pos = nxt
    return dumped


def _rank_ids(rank0_hash: str, tp: int, derive: bool) -> list[str]:
    if not derive or tp <= 1:
        return [rank0_hash]
    return [rank0_hash if r == 0 else f"{rank0_hash}:{r}" for r in range(tp)]


def _derive_state_dicts(
    record: TraceRecord, lcm_block: int, num_mamba_groups: int
) -> list[dict[int, str]]:
    out: list[dict[int, str]] = []
    for g in range(num_mamba_groups):
        d: dict[int, str] = {}
        for i, prefix_hash in enumerate(record.hash_ids):
            boundary = (i + 1) * lcm_block
            d[boundary] = f"g{g}:B{boundary}:{prefix_hash}"
        out.append(d)
    return out


def simulate_mamba(
    records: Iterable[TraceRecord],
    topo: Topology,
    vllm_hash_block_size: int,
    hbm_block_data_size: int,
    lcm_block_size: int,
    mamba_groups: int,
    gpu_capacity_blocks: int,
    dram_capacity_bytes: int,
    fs_capacity_bytes: int,
    random_seed: int | None = 0,
    chunk_size: int | None = None,
) -> dict:
    if gpu_capacity_blocks < 0 or dram_capacity_bytes < 0 or fs_capacity_bytes < 0:
        raise ValueError("cache capacities must be >= 0")

    rng = random.Random(random_seed)
    hbm_pools = [BlockPool(gpu_capacity_blocks) for _ in range(topo.dp_size)]
    dram_pools = [ByteLRUPool(dram_capacity_bytes) for _ in range(topo.num_dram_pools())]
    fs_pool = ByteLRUPool(fs_capacity_bytes)

    entry_size = hbm_block_data_size
    group_id = 0
    tp = topo.tp_size
    derive_mamba = topo.is_mla
    request_groups = RequestGroups()
    producer_map: dict = {}

    total_tokens = 0
    gpu_hit_tokens = 0
    dram_hit_tokens = 0
    fs_hit_tokens = 0
    miss_tokens = 0

    for record in records:
        total_tokens += record.input_length
        request_index = request_groups.add(record.timestamp)
        if not record.hash_ids:
            miss_tokens += record.input_length
            continue

        dp_rank = rng.randrange(topo.dp_size) if topo.dp_size > 1 else 0
        hbm, dram, fs = topo.pools_for(dp_rank, hbm_pools, dram_pools, fs_pool)

        fa_track: list[tuple[int, bool]] = []
        prefix_available = True
        hit_roots: set[int] = set()

        prefix_hit_blocks = 0
        for hash_id in record.hash_ids:
            if not prefix_available:
                bid = hbm.alloc()
                if bid is None:
                    continue
                hbm.cache_block(bid, hash_id, group_id)
                producer_map[hash_id] = request_index
                fa_track.append((bid, True))
                continue

            bid = hbm.touch_get(hash_id, group_id)
            if bid is not None:
                fa_track.append((bid, True))
                prefix_hit_blocks += 1
                hit_roots.add(request_groups.find(producer_map.get(hash_id, request_index)))
                continue

            dram_entry = dram.get(hash_id)
            if dram_entry is not None:
                bid = hbm.alloc()
                if bid is None:
                    prefix_available = False
                    continue
                hbm.cache_block(bid, hash_id, group_id)
                producer_map[hash_id] = request_index
                fa_track.append((bid, True))
                prefix_hit_blocks += 1
                hit_roots.add(request_groups.find(dram_entry.producer_index))
                continue

            fs_entry = fs.get(hash_id)
            if fs_entry is not None:
                bid = hbm.alloc()
                if bid is None:
                    prefix_available = False
                    continue
                hbm.cache_block(bid, hash_id, group_id)
                producer_map[hash_id] = request_index
                dram.put(hash_id, entry_size, fs_entry)
                fa_track.append((bid, True))
                prefix_hit_blocks += 1
                hit_roots.add(request_groups.find(fs_entry.producer_index))
                continue

            bid = hbm.alloc()
            if bid is None:
                prefix_available = False
                continue
            hbm.cache_block(bid, hash_id, group_id)
            producer_map[hash_id] = request_index
            fa_track.append((bid, True))
            prefix_available = False

        lcm_block = lcm_block_size
        if lcm_block <= 0:
            miss_tokens += record.input_length
            if hit_roots:
                root = request_groups.union_roots([request_index, *hit_roots])
                request_groups.record_hit(root, record.timestamp)
            hbm.free_reverse(fa_track)
            continue

        max_boundary = prefix_hit_blocks * lcm_block
        state_dicts = _derive_state_dicts(record, lcm_block, mamba_groups)
        all_boundaries: set[int] = set()
        for d in state_dicts:
            all_boundaries.update(d.keys())
        candidates = sorted(b for b in all_boundaries if b <= max_boundary)

        gated_tokens = 0
        gated_tier = -1
        mamba_track: list[tuple[int, bool]] = []

        for boundary in reversed(candidates):
            tier_rank = 0
            ok = True
            winning: list = []
            for d in state_dicts:
                rank0 = d.get(boundary)
                if rank0 is None:
                    ok = False
                    break
                for cid in _rank_ids(rank0, tp, derive_mamba):
                    found_rank = 0
                    if hbm.peek(cid, group_id):
                        winning.append((cid, None, 0))
                        continue
                    dram_entry = dram.peek(cid)
                    if dram_entry is not None:
                        found_rank = 1
                        winning.append((cid, dram_entry, found_rank))
                        tier_rank = max(tier_rank, found_rank)
                        continue
                    fs_entry = fs.peek(cid)
                    if fs_entry is not None:
                        found_rank = 2
                        winning.append((cid, fs_entry, found_rank))
                        tier_rank = max(tier_rank, found_rank)
                        continue
                    ok = False
                    break
                if not ok:
                    break
            if not ok:
                continue

            gated_tokens = boundary
            gated_tier = tier_rank

            for cid, entry, found_rank in winning:
                if found_rank == 0:
                    hit_roots.add(request_groups.find(producer_map.get(cid, request_index)))
                    bid = hbm.touch_get(cid, group_id)
                    if bid is not None:
                        mamba_track.append((bid, True))
                elif found_rank >= 1:
                    hit_roots.add(request_groups.find(entry.producer_index))
                    bid = hbm.alloc()
                    if bid is not None:
                        hbm.cache_block(bid, cid, group_id)
                        producer_map[cid] = request_index
                        mamba_track.append((bid, True))
                    if found_rank >= 2 and entry is not None:
                        dram.put(cid, entry_size, entry)
            break

        if gated_tier == 0:
            gpu_hit_tokens += gated_tokens
        elif gated_tier == 1:
            dram_hit_tokens += gated_tokens
        elif gated_tier == 2:
            fs_hit_tokens += gated_tokens
        miss_tokens += record.input_length - gated_tokens

        if hit_roots:
            root = request_groups.union_roots([request_index, *hit_roots])
            request_groups.record_hit(root, record.timestamp)

        dump_boundaries = _chunk_end_boundaries(
            gated_tokens, record.input_length, lcm_block, chunk_size
        )
        entry = ByteCacheEntry(request_index)

        # FA DRAM/FS dump
        for hash_id in record.hash_ids:
            dram.put(hash_id, entry_size, entry)
            fs.put(hash_id, entry_size, entry)

        # Mamba DRAM/FS dump (separated from HBM block management)
        sorted_boundaries = sorted(dump_boundaries)
        for d in state_dicts:
            for boundary in sorted_boundaries:
                rank0 = d.get(boundary)
                if rank0 is None:
                    continue
                for cid in _rank_ids(rank0, tp, derive_mamba):
                    dram.put(cid, entry_size, entry)
                    fs.put(cid, entry_size, entry)

        # Mamba HBM block management: per-boundary (1 block/group/boundary)
        # - Non-last boundaries: chunk-end freed -> before FA (forward order)
        # - Last boundary: remaining -> after FA
        mamba_chunk_end = []
        mamba_remaining = []
        if sorted_boundaries:
            last_boundary = sorted_boundaries[-1]
            for d in state_dicts:
                for boundary in sorted_boundaries:
                    rank0 = d.get(boundary)
                    if rank0 is None:
                        continue
                    for cid in _rank_ids(rank0, tp, derive_mamba):
                        bid = hbm.alloc()
                        if bid is not None:
                            hbm.cache_block(bid, cid, group_id)
                            producer_map[cid] = request_index
                            if boundary == last_boundary:
                                mamba_remaining.append((bid, True))
                            else:
                                mamba_chunk_end.append((bid, True))

        # Free mamba chunk-end (forward) -> before FA
        for bid, has_hash in mamba_chunk_end:
            hbm.free(bid, has_hash)

        # Free FA -> after mamba chunk-end, before mamba remaining
        hbm.free_reverse(fa_track)

        # Free mamba track (Phase 2 hits) + remaining -> after FA
        for bid, has_hash in mamba_track:
            hbm.free(bid, has_hash)
        for bid, has_hash in mamba_remaining:
            hbm.free(bid, has_hash)

    total_hit = gpu_hit_tokens + dram_hit_tokens + fs_hit_tokens
    result = {
        "total_tokens": total_tokens,
        "gpu_hit_tokens": gpu_hit_tokens,
        "dram_hit_tokens": dram_hit_tokens,
        "fs_hit_tokens": fs_hit_tokens,
        "miss_tokens": miss_tokens,
        "total_hit_tokens": total_hit,
        "hit_rate": _rate(total_hit, total_tokens),
    }
    result.update(_lifetime_stats(request_groups.lifetimes()))
    return result


# ============================================================================
# FAWA (DS V4) simulation
# ============================================================================

def _fa_key(hash_val):
    return ("fa", hash_val)


def _wa_key(hash_val):
    return ("wa", hash_val)


def _pools_for_fawa(topo: Topology, dp_rank: int, hbm_pools, fa_dram_pools,
                    wa_dram_pools, fs_pool):
    hbm = hbm_pools[dp_rank]
    if topo.unified:
        fa_dram, wa_dram = fa_dram_pools[0], wa_dram_pools[0]
    elif topo.is_mla:
        node = min(dp_rank // topo.dp_per_node, topo.num_nodes - 1)
        fa_dram, wa_dram = fa_dram_pools[node], wa_dram_pools[node]
    else:
        fa_dram, wa_dram = fa_dram_pools[dp_rank], wa_dram_pools[dp_rank]
    return hbm, fa_dram, wa_dram, fs_pool


def simulate_fawa(
    records: Iterable[TraceRecord],
    topo: Topology,
    sim_topology: SimTopology,
    gpu_capacity_blocks: int,
    dram_capacity_bytes: int,
    fs_capacity_bytes: int,
    random_seed: int | None = 0,
    chunk_size: int | None = None,
    wa_dump_block_wise: bool = True,
) -> dict:
    if gpu_capacity_blocks < 0 or dram_capacity_bytes < 0 or fs_capacity_bytes < 0:
        raise ValueError("cache capacities must be >= 0")

    rng = random.Random(random_seed)
    hbm_pools = [BlockPool(gpu_capacity_blocks) for _ in range(topo.dp_size)]
    if topo.unified:
        shared_dram = [ByteLRUPool(dram_capacity_bytes)
                       for _ in range(topo.num_dram_pools())]
        fa_dram_pools = shared_dram
        wa_dram_pools = shared_dram
    else:
        half_dram = dram_capacity_bytes // 2
        fa_dram_pools = [ByteLRUPool(half_dram) for _ in range(topo.num_dram_pools())]
        wa_dram_pools = [ByteLRUPool(half_dram) for _ in range(topo.num_dram_pools())]
    fs_pool = ByteLRUPool(fs_capacity_bytes)

    group_contexts = sim_topology.build_group_contexts()
    fa_groups = [g for g in group_contexts if g.manager_type in ("compress", "full_attention")]
    wa_groups = [g for g in group_contexts if g.manager_type == "sliding_window"]
    ucm_hash_bs = sim_topology.ucm_hash_block_size
    vllm_hash_bs = sim_topology.vllm_hash_block_size
    ucm_scale = ucm_hash_bs // vllm_hash_bs
    fa_size = sim_topology.fa_file_size
    wa_size = sim_topology.wa_file_size
    wa_block_wise = wa_dump_block_wise
    request_groups = RequestGroups()
    producer_map: dict = {}

    total_tokens = 0
    gpu_hit_tokens = 0
    dram_hit_tokens = 0
    fs_hit_tokens = 0
    miss_tokens = 0

    for record in records:
        total_tokens += record.input_length
        request_index = request_groups.add(record.timestamp)
        chain = record.hash_ids
        if not chain:
            miss_tokens += record.input_length
            continue

        dp_rank = rng.randrange(topo.dp_size) if topo.dp_size > 1 else 0
        hbm, fa_dram, wa_dram, fs = _pools_for_fawa(
            topo, dp_rank, hbm_pools, fa_dram_pools, wa_dram_pools, fs_pool
        )

        num_ucm_blocks = len(chain) // ucm_scale
        hit_roots: set[int] = set()

        # --- Phase 1a: HBM FA prefix (per-group, min across FA groups) ---
        hbm_prefix_tokens = float("inf")
        for g in fa_groups:
            g.reset()
            g_num_full = len(chain) // g.scale_factor
            g_prefix = 0
            g_avail = True
            for block_idx in range(g_num_full):
                h = g.derive_block_hash(block_idx, chain)
                if g_avail:
                    bid = hbm.touch_get(h, g.group_id)
                    if bid is not None:
                        g.block_ids.append(bid)
                        g.num_cached += 1
                        g_prefix += 1
                        hit_roots.add(request_groups.find(producer_map.get(h, request_index)))
                        continue
                    bid = hbm.alloc()
                    if bid is not None:
                        hbm.cache_block(bid, h, g.group_id)
                        producer_map[h] = request_index
                        g.block_ids.append(bid)
                        g.num_cached += 1
                    g_avail = False
                else:
                    bid = hbm.alloc()
                    if bid is not None:
                        hbm.cache_block(bid, h, g.group_id)
                        producer_map[h] = request_index
                        g.block_ids.append(bid)
                        g.num_cached += 1
            if len(chain) % g.scale_factor != 0:
                bid = hbm.alloc()
                if bid is not None:
                    g.block_ids.append(bid)
            g_prefix_tokens = g_prefix * g.logical_block_size
            if g_prefix_tokens < hbm_prefix_tokens:
                hbm_prefix_tokens = g_prefix_tokens
        if hbm_prefix_tokens == float("inf"):
            hbm_prefix_tokens = 0

        # --- Phase 1b: UCM FA DRAM/FS forward (beyond HBM prefix) ---
        ucm_start = int(hbm_prefix_tokens) // ucm_hash_bs
        ucm_prefix = ucm_start
        fa_ext_tier = 0  # 0=HBM, 1=DRAM, 2=FS (worst tier in FA extension)
        for ucm_idx in range(ucm_start, num_ucm_blocks):
            ucm_hash = chain[(ucm_idx + 1) * ucm_scale - 1]
            dram_entry = fa_dram.get(_fa_key(ucm_hash))
            if dram_entry is not None:
                ucm_prefix = ucm_idx + 1
                fa_ext_tier = max(fa_ext_tier, 1)
                hit_roots.add(request_groups.find(dram_entry.producer_index))
                continue
            fs_entry = fs.get(_fa_key(ucm_hash))
            if fs_entry is not None:
                fa_dram.put(_fa_key(ucm_hash), fa_size, fs_entry)
                ucm_prefix = ucm_idx + 1
                fa_ext_tier = max(fa_ext_tier, 2)
                hit_roots.add(request_groups.find(fs_entry.producer_index))
                continue
            break
        ucm_prefix_tokens = ucm_prefix * ucm_hash_bs

        # --- Phase 2: WA reverse (HBM → DRAM → FS, peek) ---
        gated_tokens = 0
        gated_tier = -1
        for ucm_idx in range(min(ucm_prefix, num_ucm_blocks) - 1, -1, -1):
            ucm_hash = chain[(ucm_idx + 1) * ucm_scale - 1]
            all_hbm_hit = True
            wa_hashes_at_boundary = []
            for g in wa_groups:
                wa_blk_idx = (ucm_idx + 1) * (ucm_hash_bs // g.logical_block_size) - 1
                wa_hash = g.derive_block_hash(wa_blk_idx, chain)
                wa_hashes_at_boundary.append(wa_hash)
                if not hbm.peek(wa_hash, g.group_id):
                    all_hbm_hit = False
                    break
            if all_hbm_hit:
                gated_tokens = (ucm_idx + 1) * ucm_hash_bs
                gated_tier = 0
                for wh in wa_hashes_at_boundary:
                    hit_roots.add(request_groups.find(producer_map.get(wh, request_index)))
                break
            wa_entry = wa_dram.peek(_wa_key(ucm_hash))
            if wa_entry is not None:
                gated_tokens = (ucm_idx + 1) * ucm_hash_bs
                gated_tier = 1
                hit_roots.add(request_groups.find(wa_entry.producer_index))
                break
            fs_entry = fs.peek(_wa_key(ucm_hash))
            if fs_entry is not None:
                gated_tokens = (ucm_idx + 1) * ucm_hash_bs
                gated_tier = 2
                hit_roots.add(request_groups.find(fs_entry.producer_index))
                break

        if gated_tier >= 0:
            fa_tier = 0 if gated_tokens <= int(hbm_prefix_tokens) else fa_ext_tier
            overall_tier = max(fa_tier, gated_tier)
            if overall_tier == 0:
                gpu_hit_tokens += gated_tokens
            elif overall_tier == 1:
                dram_hit_tokens += gated_tokens
            else:
                fs_hit_tokens += gated_tokens
        miss_tokens += record.input_length - gated_tokens

        if hit_roots:
            root = request_groups.union_roots([request_index, *hit_roots])
            request_groups.record_hit(root, record.timestamp)

        # --- Phase 3: dump + free ---
        entry = ByteCacheEntry(request_index)

        # FA DRAM/FS dump
        for ucm_idx in range(num_ucm_blocks):
            ucm_hash = chain[(ucm_idx + 1) * ucm_scale - 1]
            fa_dram.put(_fa_key(ucm_hash), fa_size, entry)
            fs.put(_fa_key(ucm_hash), fa_size, entry)

        # WA DRAM/FS dump (separated from HBM block management)
        if wa_block_wise:
            wa_dump_indices = range(num_ucm_blocks)
        else:
            dump_b = _chunk_end_boundaries(
                gated_tokens, record.input_length, ucm_hash_bs, chunk_size
            )
            wa_dump_indices = [
                i for i in range(num_ucm_blocks)
                if (i + 1) * ucm_hash_bs in dump_b
            ]
        for ucm_idx in wa_dump_indices:
            ucm_hash = chain[(ucm_idx + 1) * ucm_scale - 1]
            wa_dram.put(_wa_key(ucm_hash), wa_size, entry)
            fs.put(_wa_key(ucm_hash), wa_size, entry)

        # WA HBM block management: per-chunk, per-group
        # - Alloc ceil(chunk/lbs) blocks per chunk, cache only reachable ones
        # - Carry over `need` running blocks to next chunk's head
        # - Free non-running (reversed) at chunk end -> before FA
        # - Last chunk's blocks freed after FA
        alignment = sim_topology.alignment_tokens or ucm_hash_bs
        if chunk_size and chunk_size > 0:
            chunk_step = max(alignment, (chunk_size // alignment) * alignment)
        else:
            chunk_step = None

        wa_remaining = []
        for wa_g in wa_groups:
            lbs = wa_g.logical_block_size
            need = ceil((wa_g.sliding_window - 1) / lbs) if wa_g.sliding_window else 1
            total_blocks = len(chain) // wa_g.scale_factor
            if chunk_step:
                blocks_per_chunk = max(1, chunk_step // lbs)
            else:
                blocks_per_chunk = total_blocks

            prev_running = []
            cs = 0
            while cs < total_blocks:
                ce = min(cs + blocks_per_chunk, total_blocks)
                chunk_blocks = list(prev_running)
                for blk_idx in range(cs, ce):
                    h = wa_g.derive_block_hash(blk_idx, chain)
                    bid = hbm.alloc()
                    if bid is not None:
                        if wa_g.reachable(blk_idx):
                            hbm.cache_block(bid, h, wa_g.group_id)
                            producer_map[h] = request_index
                            chunk_blocks.append((bid, True))
                        else:
                            chunk_blocks.append((bid, False))

                if ce < total_blocks:
                    to_free = chunk_blocks[:-need] if len(chunk_blocks) > need else []
                    hbm.free_reverse(to_free)
                    prev_running = chunk_blocks[-need:] if len(chunk_blocks) >= need else chunk_blocks
                else:
                    prev_running = chunk_blocks
                cs = ce
            wa_remaining.extend(prev_running)

        # FA free (free_reverse) -- after WA chunk-end, before WA remaining
        for g in fa_groups:
            blocks = [(b, i < g.num_cached) for i, b in enumerate(g.block_ids) if b is not None]
            hbm.free_reverse(blocks)

        # Last WA running blocks freed after FA (reversed)
        hbm.free_reverse(wa_remaining)

    total_hit = gpu_hit_tokens + dram_hit_tokens + fs_hit_tokens
    result = {
        "total_tokens": total_tokens,
        "gpu_hit_tokens": gpu_hit_tokens,
        "dram_hit_tokens": dram_hit_tokens,
        "fs_hit_tokens": fs_hit_tokens,
        "miss_tokens": miss_tokens,
        "total_hit_tokens": total_hit,
        "hit_rate": _rate(total_hit, total_tokens),
    }
    result.update(_lifetime_stats(request_groups.lifetimes()))
    return result


# ============================================================================
# Log collection & analysis
# ============================================================================

@dataclass
class LogFacts:
    log_files: list[str]
    records: list[TraceRecord]
    available_kv_cache_memory_bytes: list[int]
    tensor_parallel_sizes: list[int]
    data_parallel_sizes: list[int]
    sim_topology: SimTopology | None = None


def _iter_log_files(log_dir: Path) -> list[Path]:
    patterns = ("*.log", "*.log.*", "*.log.gz")
    files: dict[Path, None] = {}
    for pattern in patterns:
        for path in log_dir.rglob(pattern):
            if path.is_file():
                files[path] = None
    return sorted(files)


def _open_log_file(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def _parse_bytes(value: str, unit: str | None) -> int:
    number = float(value)
    if not unit:
        return int(number)
    multipliers = {
        "b": 1, "byte": 1, "bytes": 1,
        "kb": 1024, "kib": 1024,
        "mb": 1024**2, "mib": 1024**2,
        "gb": 1024**3, "gib": 1024**3,
        "tb": 1024**4, "tib": 1024**4,
    }
    return int(number * multipliers.get(unit.lower(), 1))


def collect_log_facts(log_dir: Path) -> LogFacts:
    if not log_dir.exists() or not log_dir.is_dir():
        raise ValueError(f"log directory does not exist: {log_dir}")

    log_files = _iter_log_files(log_dir)
    if not log_files:
        raise ValueError(f"no log files found in log directory: {log_dir}")

    records: list[TraceRecord] = []
    available_memory: list[int] = []
    tp_sizes: list[int] = []
    dp_sizes: list[int] = []
    sim_topology: SimTopology | None = None

    for path in log_files:
        with _open_log_file(path) as handle:
            for line in handle:
                record = parse_trace_line(line, str(path))
                if record is not None:
                    records.append(record)

                topo = parse_trace_meta(line)
                if topo is not None:
                    sim_topology = topo

                for match in AVAILABLE_KV_RE.finditer(line):
                    available_memory.append(
                        _parse_bytes(match.group("value"), match.group("unit"))
                    )
                for match in TP_SIZE_RE.finditer(line):
                    tp_sizes.append(int(match.group("value")))
                for match in DP_SIZE_RE.finditer(line):
                    dp_sizes.append(int(match.group("value")))

    if not records:
        raise ValueError("no trace records found in log files")
    if not available_memory:
        raise ValueError("available kv cache memory was not found in log files")
    if not tp_sizes:
        raise ValueError("tensor_parallel_size was not found in log files")
    if not dp_sizes:
        raise ValueError("data_parallel_size was not found in log files")

    records.sort(key=lambda item: item.timestamp)
    return LogFacts(
        log_files=[str(path) for path in log_files],
        records=records,
        available_kv_cache_memory_bytes=available_memory,
        tensor_parallel_sizes=tp_sizes,
        data_parallel_sizes=dp_sizes,
        sim_topology=sim_topology,
    )


def _resolve_single(values: list[int], name: str) -> int:
    unique = set(values)
    if len(unique) != 1:
        raise ValueError(
            f"conflicting {name} values: "
            + ", ".join(str(v) for v in sorted(unique))
        )
    val = next(iter(unique))
    if val <= 0:
        raise ValueError(f"{name} must be > 0")
    return val


def _resolve_gpu_cache_bytes(facts: LogFacts) -> int:
    return min(facts.available_kv_cache_memory_bytes)


def _parse_prometheus(metrics_text: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROM_SAMPLE_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        labels = match.group("labels")
        if labels:
            source = ""
            for pair in labels.split(","):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    if k.strip() == "source":
                        source = v.strip().strip('"')
            if source:
                key = f'{name}{{source="{source}"}}'
            else:
                key = name
        else:
            key = name
        samples[key] = samples.get(key, 0.0) + float(match.group("value"))
    return samples


def _fetch_service_hit_rate(service_url: str, timeout: float) -> dict:
    normalized = service_url.strip()
    if "://" not in normalized:
        normalized = "http://" + normalized
    metrics_url = normalized.rstrip("/")
    if not metrics_url.endswith("/metrics"):
        metrics_url += "/metrics"
    with urllib.request.urlopen(metrics_url, timeout=timeout) as response:
        text = response.read().decode("utf-8", errors="replace")
    samples = _parse_prometheus(text)

    total_tokens = next(
        (samples[n] for n in PROMPT_TOKENS_TOTAL_METRICS if n in samples), 0
    )
    cache_hit = next(
        (samples[n] for n in PROMPT_TOKENS_CACHE_HIT_METRICS if n in samples), 0
    )

    hit_rate = cache_hit / total_tokens if total_tokens > 0 else 0.0
    return {
        "service_url": service_url,
        "metrics_url": metrics_url,
        "prefix_cache_hits_total": cache_hit,
        "prefix_cache_queries_total": total_tokens,
        "actual_kv_cache_hit_rate": hit_rate,
    }


def _dram_per_pool_bytes(
    is_mla: bool, unified: bool, num_nodes: int, dp_size: int, tp_size: int,
    total_dram_bytes: int,
) -> int:
    if unified:
        return total_dram_bytes if is_mla else total_dram_bytes // tp_size
    if is_mla:
        return total_dram_bytes // num_nodes
    return total_dram_bytes // (dp_size * tp_size)


def _fs_capacity_bytes(is_mla: bool, tp_size: int, total_fs_bytes: int) -> int:
    return total_fs_bytes if is_mla else total_fs_bytes // tp_size


# ============================================================================
# CLI
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze theoretical UCM KV cache hit-rate from logs."
    )
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--dram-pool-size-gb", type=float, required=True)
    parser.add_argument("--fs-pool-size-gb", type=float, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--unified-memory-pool", action="store_true", default=False)
    parser.add_argument("--service-url")
    parser.add_argument("--metrics-timeout", type=float, default=5.0)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _build_sim_topology(facts: LogFacts) -> SimTopology:
    if facts.sim_topology is None:
        raise ValueError(
            "No UCMTraceMeta found in log. Use a Lite connector "
            "(use_lite=true) to produce the topology line."
        )
    return facts.sim_topology


def build_analysis(args: argparse.Namespace) -> dict:
    facts = collect_log_facts(args.log_dir)
    tp_size = _resolve_single(facts.tensor_parallel_sizes, "tensor_parallel_size")
    dp_size = _resolve_single(facts.data_parallel_sizes, "data_parallel_size")
    topo_info = _build_sim_topology(facts)

    is_mla = topo_info.is_mla
    hbm_block_data_size = topo_info.hbm_block_data_size
    if hbm_block_data_size <= 0:
        raise ValueError("hbm_block_data_size must be > 0 (from UCMTraceMeta topology line)")

    gpu_kv_bytes = _resolve_gpu_cache_bytes(facts)
    gpu_cap_blocks = gpu_kv_bytes // hbm_block_data_size

    dram_bytes = int(args.dram_pool_size_gb * GIB)
    fs_bytes = int(args.fs_pool_size_gb * GIB)
    unified = args.unified_memory_pool

    dram_per_pool = _dram_per_pool_bytes(
        is_mla, unified, args.num_nodes, dp_size, tp_size, dram_bytes
    )
    fs_cap = _fs_capacity_bytes(is_mla, tp_size, fs_bytes)

    topo = Topology(
        is_mla=is_mla, unified=unified,
        num_nodes=args.num_nodes, dp_size=dp_size, tp_size=tp_size,
    )

    model_type = topo_info.model_type
    sim_kwargs = dict(topo=topo, random_seed=args.random_seed)

    if model_type == "fawa":
        sim_fn = simulate_fawa
        sim_kwargs["sim_topology"] = topo_info
        sim_kwargs["chunk_size"] = args.max_num_batched_tokens
    elif model_type == "mamba":
        sim_fn = simulate_mamba
        sim_kwargs["vllm_hash_block_size"] = topo_info.vllm_hash_block_size or topo_info.lcm_block_size
        sim_kwargs["hbm_block_data_size"] = hbm_block_data_size
        sim_kwargs["lcm_block_size"] = topo_info.lcm_block_size
        sim_kwargs["mamba_groups"] = topo_info.mamba_groups
        sim_kwargs["chunk_size"] = args.max_num_batched_tokens
    else:
        sim_fn = simulate_standard
        sim_kwargs["vllm_hash_block_size"] = topo_info.vllm_hash_block_size or 128
        sim_kwargs["hbm_block_data_size"] = hbm_block_data_size

    unique_blocks = len({h for r in facts.records for h in r.hash_ids})
    if model_type == "fawa":
        large_bytes = unique_blocks * (topo_info.fa_file_size + topo_info.wa_file_size)
    else:
        large_bytes = unique_blocks * hbm_block_data_size

    # theoretical_max: capacity must hold all unique blocks without eviction.
    # For mamba/fawa, HBM and DRAM/FS also hold mamba/WA state entries beyond
    # FA hashes, so unique_blocks alone is insufficient. Use total hash count
    # across all records (an upper bound on distinct entries) scaled up.
    total_hash_count = sum(len(r.hash_ids) for r in facts.records)
    theoretical_blocks = max(unique_blocks, total_hash_count) * 100
    if model_type == "fawa":
        theoretical_bytes = theoretical_blocks * max(
            hbm_block_data_size,
            topo_info.fa_file_size + topo_info.wa_file_size,
        )
    else:
        theoretical_bytes = theoretical_blocks * hbm_block_data_size

    scenario_defs = [
        ("theoretical_max", theoretical_blocks, theoretical_bytes, theoretical_bytes),
        ("hbm", gpu_cap_blocks, 0, 0),
        ("hbm_dram", gpu_cap_blocks, dram_per_pool, 0),
        ("hbm_dram_fs", gpu_cap_blocks, dram_per_pool, fs_cap),
    ]

    def run_scenarios(**extra_kwargs):
        kwargs = {**sim_kwargs, **extra_kwargs}
        out = {}
        for name, gpu_cap, dram_cap, fs_cap_s in scenario_defs:
            out[name] = sim_fn(
                records=facts.records,
                gpu_capacity_blocks=gpu_cap,
                dram_capacity_bytes=dram_cap,
                fs_capacity_bytes=fs_cap_s,
                **kwargs,
            )
        return out

    if model_type == "fawa":
        scenarios = {
            "block_wise": run_scenarios(wa_dump_block_wise=True),
            "chunk_wise": run_scenarios(wa_dump_block_wise=False),
        }
    else:
        scenarios = run_scenarios()

    service_metrics = (
        _fetch_service_hit_rate(args.service_url, args.metrics_timeout)
        if args.service_url else None
    )

    def pct(r):
        return r["hit_rate"] * 100

    analysis = {
        "total_request_count": len(facts.records),
        "total_request_token_count": sum(r.input_length for r in facts.records),
    }

    if model_type == "fawa":
        for mode_name in ("block_wise", "chunk_wise"):
            ms = scenarios[mode_name]
            analysis[f"{mode_name}_theoretical_max_percent"] = pct(ms["theoretical_max"])
            analysis[f"{mode_name}_hbm_percent"] = pct(ms["hbm"])
            analysis[f"{mode_name}_hbm_dram_percent"] = pct(ms["hbm_dram"])
            analysis[f"{mode_name}_hbm_dram_fs_percent"] = pct(ms["hbm_dram_fs"])
    else:
        analysis["theoretical_max_kv_cache_hit_rate_percent"] = pct(scenarios["theoretical_max"])
        analysis["hbm_theoretical_hit_rate_percent"] = pct(scenarios["hbm"])
        analysis["hbm_dram_pool_theoretical_hit_rate_percent"] = pct(scenarios["hbm_dram"])
        analysis["hbm_dram_fs_pool_theoretical_hit_rate_percent"] = pct(scenarios["hbm_dram_fs"])

    if service_metrics:
        analysis["service_actual_kv_cache_hit_rate_percent"] = (
            service_metrics["actual_kv_cache_hit_rate"] * 100
        )

    if model_type == "fawa":
        tmax = scenarios["block_wise"]["theoretical_max"]
    else:
        tmax = scenarios["theoretical_max"]
    analysis["request_lifetime_sample_count"] = tmax["request_lifetime_sample_count"]
    analysis["average_request_lifetime_seconds"] = tmax["average_request_lifetime_seconds"]
    analysis["p90_request_lifetime_seconds"] = tmax["p90_request_lifetime_seconds"]
    analysis["p95_request_lifetime_seconds"] = tmax["p95_request_lifetime_seconds"]

    return {
        "inputs": {
            "log_dir": str(args.log_dir),
            "model_type": model_type,
            "is_mla": is_mla,
            "hbm_block_data_size": hbm_block_data_size,
            "dram_pool_size_gb": args.dram_pool_size_gb,
            "fs_pool_size_gb": args.fs_pool_size_gb,
            "chunk_size": args.max_num_batched_tokens,
            "unified_memory_pool": unified,
            "service_url": args.service_url,
        },
        "derived": {
            "log_files": facts.log_files,
            "tp_size": tp_size,
            "dp_size": dp_size,
            "num_nodes": args.num_nodes,
            "gpu_kv_cache_bytes": gpu_kv_bytes,
            "gpu_capacity_blocks": gpu_cap_blocks,
            "dram_per_pool_bytes": dram_per_pool,
            "fs_capacity_bytes": fs_cap,
        },
        "analysis": analysis,
        "simulation_details": scenarios,
    }


def print_summary(result: dict) -> None:
    a = result["analysis"]
    d = result["derived"]
    i = result["inputs"]
    print("Trace cache hit rate analysis")
    print(f"  Model type: {i['model_type']}")
    print(f"  Total request count: {a['total_request_count']}")
    print(f"  Total request token count: {a['total_request_token_count']}")
    print(f"  HBM available: {d['gpu_kv_cache_bytes'] / GIB:.2f} GiB")
    print(f"  TP={d['tp_size']}  DP={d['dp_size']}  Nodes={d['num_nodes']}")
    print(f"  DRAM pool: {i['dram_pool_size_gb']:.2f} GiB  FS pool: {i['fs_pool_size_gb']:.2f} GiB")
    if i["model_type"] == "fawa":
        for mode_name in ("block_wise", "chunk_wise"):
            print(f"  [{mode_name}]")
            print(f"    Theoretical max: {a[f'{mode_name}_theoretical_max_percent']:.6f}%")
            print(f"    HBM:              {a[f'{mode_name}_hbm_percent']:.6f}%")
            print(f"    HBM+DRAM:         {a[f'{mode_name}_hbm_dram_percent']:.6f}%")
            print(f"    HBM+DRAM+FS:      {a[f'{mode_name}_hbm_dram_fs_percent']:.6f}%")
    else:
        print(f"  Theoretical max: {a['theoretical_max_kv_cache_hit_rate_percent']:.6f}%")
        if "service_actual_kv_cache_hit_rate_percent" in a:
            print(f"  Service actual:    {a['service_actual_kv_cache_hit_rate_percent']:.6f}%")
        print(f"  HBM:               {a['hbm_theoretical_hit_rate_percent']:.6f}%")
        print(f"  HBM+DRAM:          {a['hbm_dram_pool_theoretical_hit_rate_percent']:.6f}%")
        print(f"  HBM+DRAM+FS:       {a['hbm_dram_fs_pool_theoretical_hit_rate_percent']:.6f}%")

    print(f"  Request lifetime sample count: {a['request_lifetime_sample_count']}")
    print(f"  Average request lifetime: {a['average_request_lifetime_seconds']:.6f} s")
    print(f"  P90 request lifetime: {a['p90_request_lifetime_seconds']:.6f} s")
    print(f"  P95 request lifetime: {a['p95_request_lifetime_seconds']:.6f} s")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = build_analysis(args)
        if args.trace_output:
            facts = collect_log_facts(args.log_dir)
            args.trace_output.parent.mkdir(parents=True, exist_ok=True)
            with args.trace_output.open("w", encoding="utf-8") as f:
                for r in facts.records:
                    f.write(json.dumps({
                        "timestamp": r.timestamp,
                        "input_length": r.input_length,
                        "output_length": r.output_length,
                        "hash_ids": r.hash_ids,
                    }, ensure_ascii=False) + "\n")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        print_summary(result)
    except Exception as exc:
        import sys
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
