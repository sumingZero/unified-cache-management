"""Tests for sim_engine FAWA (DS V4) model."""
from auto_trace_analysis import TraceRecord
from auto_trace_analysis import Topology, SimTopology, GroupSpec, simulate_fawa


def make_fawa_topology(
    vllm_hash_bs=4,
    ucm_hash_bs=16,
    hbm_data_size=1024,
    fa_file=512,
    wa_file=512,
    alignment=16,
    groups=None,
):
    if groups is None:
        groups = [
            GroupSpec("G0", 16, "compress", None, 4),
            GroupSpec("G1", 16, "sliding_window", 16, 1),
        ]
    return SimTopology(
        model_type="fawa",
        is_mla=True,
        vllm_hash_block_size=vllm_hash_bs,
        hbm_block_data_size=hbm_data_size,
        ucm_hash_block_size=ucm_hash_bs,
        fa_file_size=fa_file,
        wa_file_size=wa_file,
        alignment_tokens=alignment,
        group_specs=groups,
    )


def make_chain(n):
    return [f"h{i}" for i in range(n)]


def test_fa_prefix_hbm_rescue():
    """req0 cold, req1 same prefix: FA rescued from free_deque, WA also in HBM."""
    topo_info = make_fawa_topology()
    topo = Topology(is_mla=True, unified=False, num_nodes=1, dp_size=1, tp_size=1)
    chain = make_chain(16)
    records = [
        TraceRecord(0.0, 64, 0, chain, "t"),
        TraceRecord(1.0, 64, 0, chain, "t"),
    ]
    result = simulate_fawa(
        records, topo, topo_info,
        gpu_capacity_blocks=100, dram_capacity_bytes=0, fs_capacity_bytes=0,
        random_seed=0, chunk_size=16, wa_dump_block_wise=True,
    )
    assert result["gpu_hit_tokens"] == 64, f"gpu_hit={result['gpu_hit_tokens']}"
    assert result["miss_tokens"] == 64, f"miss={result['miss_tokens']}"
    print(f"  test_fa_prefix_hbm_rescue: gpu_hit={result['gpu_hit_tokens']}, miss={result['miss_tokens']}")


def test_fa_dram_promotion():
    """HBM too small for all FA blocks: partial HBM hit, rest from DRAM."""
    topo_info = make_fawa_topology()
    topo = Topology(is_mla=True, unified=False, num_nodes=1, dp_size=1, tp_size=1)
    chain = make_chain(16)
    records = [
        TraceRecord(0.0, 64, 0, chain, "t"),
        TraceRecord(1.0, 64, 0, chain, "t"),
    ]
    result = simulate_fawa(
        records, topo, topo_info,
        gpu_capacity_blocks=2, dram_capacity_bytes=100 * 1024, fs_capacity_bytes=0,
        random_seed=0, chunk_size=16, wa_dump_block_wise=True,
    )
    assert result["dram_hit_tokens"] == 64, f"dram_hit={result['dram_hit_tokens']}"
    assert result["miss_tokens"] == 64, f"miss={result['miss_tokens']}"
    print(f"  test_fa_dram_promotion: dram_hit={result['dram_hit_tokens']}, miss={result['miss_tokens']}")


def test_wa_dram_hit():
    """HBM has FA but WA only in DRAM (WA blocks evicted from HBM)."""
    topo_info = make_fawa_topology()
    topo = Topology(is_mla=True, unified=False, num_nodes=1, dp_size=1, tp_size=1)
    chain = make_chain(16)
    records = [
        TraceRecord(0.0, 64, 0, chain, "t"),
        TraceRecord(1.0, 64, 0, chain, "t"),
    ]
    result = simulate_fawa(
        records, topo, topo_info,
        gpu_capacity_blocks=4, dram_capacity_bytes=100 * 1024, fs_capacity_bytes=0,
        random_seed=0, chunk_size=16, wa_dump_block_wise=True,
    )
    assert result["dram_hit_tokens"] == 64, f"dram_hit={result['dram_hit_tokens']}"
    assert result["miss_tokens"] == 64, f"miss={result['miss_tokens']}"
    print(f"  test_wa_dram_hit: dram_hit={result['dram_hit_tokens']}, miss={result['miss_tokens']}")


def test_no_hit_different_prefix():
    """Different prefix: no FA or WA hit."""
    topo_info = make_fawa_topology()
    topo = Topology(is_mla=True, unified=False, num_nodes=1, dp_size=1, tp_size=1)
    chain0 = make_chain(16)
    chain1 = [f"x{i}" for i in range(16)]
    records = [
        TraceRecord(0.0, 64, 0, chain0, "t"),
        TraceRecord(1.0, 64, 0, chain1, "t"),
    ]
    result = simulate_fawa(
        records, topo, topo_info,
        gpu_capacity_blocks=100, dram_capacity_bytes=100 * 1024, fs_capacity_bytes=0,
        random_seed=0, chunk_size=16, wa_dump_block_wise=True,
    )
    assert result["gpu_hit_tokens"] == 0
    assert result["miss_tokens"] == 128, f"miss={result['miss_tokens']}"
    print(f"  test_no_hit_different_prefix: miss={result['miss_tokens']}")


def test_c128_mla_insufficient():
    """FA group with large lbs: request too short → 0 full blocks → hit=0."""
    groups = [
        GroupSpec("G0", 16, "compress", None, 4),
        GroupSpec("G1", 64, "compress", None, 16),
    ]
    topo_info = make_fawa_topology(vllm_hash_bs=4, ucm_hash_bs=16, alignment=16, groups=groups)
    topo = Topology(is_mla=True, unified=False, num_nodes=1, dp_size=1, tp_size=1)
    chain = make_chain(8)
    records = [
        TraceRecord(0.0, 32, 0, chain, "t"),
        TraceRecord(1.0, 32, 0, chain, "t"),
    ]
    result = simulate_fawa(
        records, topo, topo_info,
        gpu_capacity_blocks=100, dram_capacity_bytes=100 * 1024, fs_capacity_bytes=0,
        random_seed=0, chunk_size=16, wa_dump_block_wise=True,
    )
    assert result["gpu_hit_tokens"] == 0, f"gpu_hit={result['gpu_hit_tokens']}"
    print(f"  test_c128_mla_insufficient: gpu_hit={result['gpu_hit_tokens']}")


def test_reachable_filter():
    """WA group with need < per_segment: only reachable blocks get hash."""
    groups = [
        GroupSpec("G0", 16, "compress", None, 4),
        GroupSpec("G1", 4, "sliding_window", 4, 1),
    ]
    topo_info = make_fawa_topology(vllm_hash_bs=4, ucm_hash_bs=16, alignment=16, groups=groups)
    topo = Topology(is_mla=True, unified=False, num_nodes=1, dp_size=1, tp_size=1)
    chain = make_chain(16)
    records = [
        TraceRecord(0.0, 64, 0, chain, "t"),
        TraceRecord(1.0, 64, 0, chain, "t"),
    ]
    result = simulate_fawa(
        records, topo, topo_info,
        gpu_capacity_blocks=100, dram_capacity_bytes=0, fs_capacity_bytes=0,
        random_seed=0, chunk_size=16, wa_dump_block_wise=True,
    )
    assert result["gpu_hit_tokens"] == 64, f"gpu_hit={result['gpu_hit_tokens']}"
    assert result["miss_tokens"] == 64, f"miss={result['miss_tokens']}"
    print(f"  test_reachable_filter: gpu_hit={result['gpu_hit_tokens']}, miss={result['miss_tokens']}")


def test_block_wise_vs_chunk_wise():
    """Block-wise has more WA dump points than chunk-wise."""
    topo_info = make_fawa_topology()
    topo = Topology(is_mla=True, unified=False, num_nodes=1, dp_size=1, tp_size=1)
    chain = make_chain(16)
    records = [
        TraceRecord(0.0, 64, 0, chain, "t"),
        TraceRecord(1.0, 64, 0, chain, "t"),
    ]
    bw = simulate_fawa(
        records, topo, topo_info,
        gpu_capacity_blocks=100, dram_capacity_bytes=100 * 1024, fs_capacity_bytes=0,
        random_seed=0, chunk_size=16, wa_dump_block_wise=True,
    )
    cw = simulate_fawa(
        records, topo, topo_info,
        gpu_capacity_blocks=100, dram_capacity_bytes=100 * 1024, fs_capacity_bytes=0,
        random_seed=0, chunk_size=16, wa_dump_block_wise=False,
    )
    total_bw = bw["gpu_hit_tokens"] + bw["dram_hit_tokens"] + bw["fs_hit_tokens"]
    total_cw = cw["gpu_hit_tokens"] + cw["dram_hit_tokens"] + cw["fs_hit_tokens"]
    assert total_bw == 64, f"block_wise total_hit={total_bw}"
    assert total_cw == 64, f"chunk_wise total_hit={total_cw}"
    print(f"  test_block_wise_vs_chunk_wise: bw={total_bw}, cw={total_cw}")


def test_chunk_carry_over():
    """Multiple chunks: running blocks carried over, freed after FA."""
    topo_info = make_fawa_topology()
    topo = Topology(is_mla=True, unified=False, num_nodes=1, dp_size=1, tp_size=1)
    chain = make_chain(32)
    records = [
        TraceRecord(0.0, 128, 0, chain, "t"),
        TraceRecord(1.0, 128, 0, chain, "t"),
    ]
    result = simulate_fawa(
        records, topo, topo_info,
        gpu_capacity_blocks=100, dram_capacity_bytes=0, fs_capacity_bytes=0,
        random_seed=0, chunk_size=16, wa_dump_block_wise=True,
    )
    assert result["gpu_hit_tokens"] == 128, f"gpu_hit={result['gpu_hit_tokens']}"
    assert result["miss_tokens"] == 128, f"miss={result['miss_tokens']}"
    print(f"  test_chunk_carry_over: gpu_hit={result['gpu_hit_tokens']}, miss={result['miss_tokens']}")


def test_unified_dram():
    """Unified DRAM pool: FA+WA share one ByteLRUPool."""
    topo_info = make_fawa_topology()
    topo = Topology(is_mla=True, unified=True, num_nodes=1, dp_size=1, tp_size=1)
    chain = make_chain(16)
    records = [
        TraceRecord(0.0, 64, 0, chain, "t"),
        TraceRecord(1.0, 64, 0, chain, "t"),
    ]
    result = simulate_fawa(
        records, topo, topo_info,
        gpu_capacity_blocks=2, dram_capacity_bytes=100 * 1024, fs_capacity_bytes=0,
        random_seed=0, chunk_size=16, wa_dump_block_wise=True,
    )
    assert result["dram_hit_tokens"] == 64, f"dram_hit={result['dram_hit_tokens']}"
    assert result["miss_tokens"] == 64, f"miss={result['miss_tokens']}"
    print(f"  test_unified_dram: dram_hit={result['dram_hit_tokens']}, miss={result['miss_tokens']}")


if __name__ == "__main__":
    print("Running FAWA model tests...")
    test_fa_prefix_hbm_rescue()
    test_fa_dram_promotion()
    test_wa_dram_hit()
    test_no_hit_different_prefix()
    test_c128_mla_insufficient()
    test_reachable_filter()
    test_block_wise_vs_chunk_wise()
    test_chunk_carry_over()
    test_unified_dram()
    print("All FAWA tests passed!")
