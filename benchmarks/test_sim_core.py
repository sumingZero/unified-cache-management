"""Smoke tests for sim_core topology & trace parsing."""
from auto_trace_analysis import parse_trace_meta, parse_trace_line, SimTopology, GroupSpec

# --- standard topology ---
t1 = parse_trace_meta(
    "UCMTraceMeta: type=standard, is_mla=true, "
    "vllm_hash_block_size=128, hbm_block_data_size=1048576"
)
assert t1.model_type == "standard"
assert t1.is_mla is True
assert t1.vllm_hash_block_size == 128
assert t1.hbm_block_data_size == 1048576
assert t1.alignment_tokens == 128

# --- mamba topology ---
t2 = parse_trace_meta(
    "UCMTraceMeta: type=mamba, is_mla=false, "
    "vllm_hash_block_size=512, hbm_block_data_size=536576, "
    "lcm_block_size=512, mamba_groups=3"
)
assert t2.model_type == "mamba"
assert t2.is_mla is False
assert t2.lcm_block_size == 512
assert t2.mamba_groups == 3
assert t2.alignment_tokens == 512

# --- fawa topology ---
fawa_line = (
    "UCMTraceMeta: type=fawa, is_mla=true, "
    "vllm_hash_block_size=8, ucm_hash_block_size=512, "
    "hbm_block_data_size=4579072, fa_file_size=4558848, "
    "wa_file_size=9224192, "
    "alignment_tokens=128, "
    "groups=[('G0', 512, 'compress', None, 4), "
    "('G1', 16384, 'compress', None, 128), "
    "('G2', 128, 'sliding_window', 128, 1), "
    "('G3', 128, 'sliding_window', 128, 1), "
    "('G4', 8, 'sliding_window', 8, 4), "
    "('G5', 32, 'sliding_window', 32, 128)]"
)
t3 = parse_trace_meta(fawa_line)
assert t3.model_type == "fawa"
assert t3.ucm_hash_block_size == 512
assert t3.fa_file_size == 4558848
assert t3.wa_file_size == 9224192
assert t3.alignment_tokens == 128
assert len(t3.group_specs) == 6
assert t3.group_specs[0].name == "G0"
assert t3.group_specs[0].logical_block_size == 512
assert t3.group_specs[0].manager_type == "compress"
assert t3.group_specs[0].sliding_window is None
assert t3.group_specs[0].compress_ratio == 4
assert t3.group_specs[4].logical_block_size == 8
assert t3.group_specs[4].sliding_window == 8

# build_group_contexts
ctxs = t3.build_group_contexts()
assert len(ctxs) == 6
assert ctxs[0].logical_block_size == 512
assert ctxs[0].scale_factor == 64  # 512 // 8
assert ctxs[1].logical_block_size == 16384
assert ctxs[1].compress_ratio == 128
assert ctxs[2].manager_type == "sliding_window"
assert ctxs[2].sliding_window == 128
assert ctxs[4].logical_block_size == 8

# --- new trace format ---
r1 = parse_trace_line(
    "UCMTrace: timestamp: 123.45, request_id: req1, "
    "input_length: 1024, output_length: 512, "
    "block_hashes: ['h0', 'h1', 'h2']",
    "test.log",
)
assert r1 is not None
assert r1.timestamp == 123.45
assert r1.input_length == 1024
assert r1.output_length == 512
assert r1.hash_ids == ["h0", "h1", "h2"]
assert r1.request_id == "req1"

# --- old trace format (backward compat) ---
r2 = parse_trace_line(
    "timestamp: 67.89, input_length: 256, output_length: 128, "
    "ucm_block_ids: ['h0', 'h1']",
    "old.log",
)
assert r2 is not None
assert r2.timestamp == 67.89
assert r2.input_length == 256
assert r2.hash_ids == ["h0", "h1"]
assert r2.request_id is None

# --- non-trace line ---
assert parse_trace_line("some random log line", "x") is None

print("All topology/trace parsing tests passed!")
