"""Tests for sim_engine mamba model."""
from auto_trace_analysis import TraceRecord
from auto_trace_analysis import Topology, simulate_mamba


def make_h(i):
    return f"h{i}"


def test_mamba_state_hbm_rescue():
    """req0 dumps mamba state to HBM (alloc+cache+free).
    req1 finds it in HBM free queue → gpu hit."""
    records = [
        # req0: 4 blocks = 512 tokens, all miss (cold)
        TraceRecord(0.0, 512, 0, [make_h(0), make_h(1), make_h(2), make_h(3)], "t"),
        # req1: same prefix, FA rescued + mamba state found
        TraceRecord(1.0, 512, 0, [make_h(0), make_h(1), make_h(2), make_h(3)], "t"),
    ]
    topo = Topology(is_mla=False, unified=False, num_nodes=1, dp_size=1, tp_size=1)
    result = simulate_mamba(
        records, topo,
        vllm_hash_block_size=128, hbm_block_data_size=1024,
        lcm_block_size=128, mamba_groups=1,
        gpu_capacity_blocks=100, dram_capacity_bytes=0, fs_capacity_bytes=0,
        random_seed=0, chunk_size=256,
    )
    # req0: all miss (512 tokens)
    # req1: FA 4 blocks rescued (4*128=512 tokens gpu_hit from FA)
    #   mamba state at 512 found in HBM → gated_tokens=512, tier=0 (gpu)
    #   But gated_tokens replaces FA prefix_hit in the token counting!
    #   Actually: gated_tokens = 512 (mamba state boundary)
    #   gpu_hit_tokens += gated_tokens (if tier=0)
    #   miss_tokens += input_length - gated_tokens = 512 - 512 = 0
    # So req1: gpu_hit=512, miss=0
    # Total: gpu_hit=512, miss=512
    assert result["gpu_hit_tokens"] == 512, f"gpu_hit={result['gpu_hit_tokens']}"
    assert result["miss_tokens"] == 512, f"miss={result['miss_tokens']}"
    print(f"  test_mamba_state_hbm_rescue: gpu_hit={result['gpu_hit_tokens']}, miss={result['miss_tokens']}")


def test_mamba_state_dram():
    """HBM too small for mamba state dump (alloc fails).
    Mamba state only in DRAM → dram hit."""
    records = [
        TraceRecord(0.0, 512, 0, [make_h(0), make_h(1), make_h(2), make_h(3)], "t"),
        TraceRecord(1.0, 512, 0, [make_h(0), make_h(1), make_h(2), make_h(3)], "t"),
    ]
    topo = Topology(is_mla=False, unified=False, num_nodes=1, dp_size=1, tp_size=1)
    # HBM = 4 (exactly fits 4 FA blocks, no room for mamba alloc during dump)
    # Actually, after Phase 1 (FA all in-use = 4 blocks), Phase 3 mamba alloc
    # would fail (pool exhausted). So mamba state goes only to DRAM.
    result = simulate_mamba(
        records, topo,
        vllm_hash_block_size=128, hbm_block_data_size=1024,
        lcm_block_size=128, mamba_groups=1,
        gpu_capacity_blocks=4, dram_capacity_bytes=100 * 1024, fs_capacity_bytes=0,
        random_seed=0, chunk_size=256,
    )
    # req0: FA all miss (alloc 4, HBM full). Phase 3 mamba alloc fails.
    #   Mamba state only in DRAM. free_reverse FA (all 4 freed).
    # req1: FA 4 rescued (HBM). Mamba state at 512 in DRAM → dram hit.
    #   gated_tokens=512, tier=1 (dram)
    #   dram_hit=512, miss=0
    # Total: gpu_hit=0 (FA prefix is replaced by mamba gated), dram_hit=512, miss=512
    # Wait: the gated_tokens replaces the FA prefix hit. So FA's gpu_hit is not counted.
    # gpu_hit = 0 (no mamba state in HBM)
    # dram_hit = 512 (mamba state found in DRAM)
    # miss = 512 (req0) + 0 (req1) = 512
    print(f"  test_mamba_state_dram: gpu={result['gpu_hit_tokens']}, dram={result['dram_hit_tokens']}, miss={result['miss_tokens']}")
    assert result["dram_hit_tokens"] == 512, f"dram_hit={result['dram_hit_tokens']}"
    assert result["miss_tokens"] == 512, f"miss={result['miss_tokens']}"


def test_mamba_no_state():
    """Different prefix → no mamba state found → all miss."""
    records = [
        TraceRecord(0.0, 256, 0, [make_h(0), make_h(1)], "t"),
        TraceRecord(1.0, 256, 0, [make_h(2), make_h(3)], "t"),  # different prefix
    ]
    topo = Topology(is_mla=False, unified=False, num_nodes=1, dp_size=1, tp_size=1)
    result = simulate_mamba(
        records, topo,
        vllm_hash_block_size=128, hbm_block_data_size=1024,
        lcm_block_size=128, mamba_groups=1,
        gpu_capacity_blocks=100, dram_capacity_bytes=100*1024, fs_capacity_bytes=0,
        random_seed=0, chunk_size=256,
    )
    # req0: all miss, FA dumped, mamba state at 256 dumped
    # req1: FA different prefix → prefix_hit=0 → max_boundary=0 → no candidates
    #   gated_tokens=0, miss=256
    assert result["gpu_hit_tokens"] == 0
    assert result["miss_tokens"] == 512, f"miss={result['miss_tokens']}"
    print(f"  test_mamba_no_state: miss={result['miss_tokens']}")


def test_mamba_partial_prefix():
    """req1 shares 2-block prefix with req0.
    FA prefix hit = 2, mamba state at boundary 256 found."""
    records = [
        TraceRecord(0.0, 512, 0, [make_h(0), make_h(1), make_h(2), make_h(3)], "t"),
        # req1: shares h0,h1 prefix, h4,h5 new
        TraceRecord(1.0, 512, 0, [make_h(0), make_h(1), make_h(4), make_h(5)], "t"),
    ]
    topo = Topology(is_mla=False, unified=False, num_nodes=1, dp_size=1, tp_size=1)
    result = simulate_mamba(
        records, topo,
        vllm_hash_block_size=128, hbm_block_data_size=1024,
        lcm_block_size=128, mamba_groups=1,
        gpu_capacity_blocks=100, dram_capacity_bytes=100*1024, fs_capacity_bytes=0,
        random_seed=0, chunk_size=256,
    )
    # req0: all miss (512). FA dumped, mamba at 256 and 512 dumped.
    # req1: FA h0,h1 rescued (prefix_hit=2, max_boundary=256).
    #   mamba at 256: state key g0:B256:h1 → found in HBM (dumped by req0)
    #   gated_tokens=256, tier=0 (gpu)
    #   gpu_hit=256, miss=512-256=256
    # Total: gpu_hit=256, miss=512+256=768
    assert result["gpu_hit_tokens"] == 256, f"gpu_hit={result['gpu_hit_tokens']}"
    assert result["miss_tokens"] == 768, f"miss={result['miss_tokens']}"
    print(f"  test_mamba_partial_prefix: gpu={result['gpu_hit_tokens']}, miss={result['miss_tokens']}")


if __name__ == "__main__":
    print("Running mamba model tests...")
    test_mamba_state_hbm_rescue()
    test_mamba_state_dram()
    test_mamba_no_state()
    test_mamba_partial_prefix()
    print("All mamba tests passed!")
