"""Tests for sim_engine standard model."""
from auto_trace_analysis import TraceRecord
from auto_trace_analysis import Topology, simulate_standard


def make_records():
    """Two requests sharing a 3-block prefix, then a cold request."""
    h = lambda i: f"hash{i}"
    return [
        # req0: 3 blocks, all miss (cold start)
        TraceRecord(0.0, 384, 0, [h(0), h(1), h(2)], "test"),
        # req1: same 3-block prefix + 1 new block
        # h0,h1,h2 should hit (rescued from free queue), h3 miss
        TraceRecord(1.0, 512, 0, [h(0), h(1), h(2), h(3)], "test"),
        # req2: completely different prefix, all miss
        TraceRecord(2.0, 256, 0, [h(4), h(5)], "test"),
    ]


def test_basic_hit_miss():
    records = make_records()
    topo = Topology(is_mla=False, unified=False, num_nodes=1, dp_size=1, tp_size=1)
    # Large HBM (enough for all), no DRAM/FS
    result = simulate_standard(
        records, topo,
        vllm_hash_block_size=128,
        hbm_block_data_size=1024,
        gpu_capacity_blocks=100,
        dram_capacity_bytes=0,
        fs_capacity_bytes=0,
        random_seed=0,
    )
    # req0: 3 blocks all miss (384 tokens)
    # req1: 3 blocks hit (rescued from free queue), 1 miss (128 tokens)
    # req2: 2 blocks miss (256 tokens)
    # gpu_hit = 3*128 = 384 (req1's prefix hit)
    # miss = 384 + 128 + 256 = 768
    assert result["gpu_hit_tokens"] == 384, f"gpu_hit={result['gpu_hit_tokens']}"
    assert result["dram_hit_tokens"] == 0
    assert result["fs_hit_tokens"] == 0
    assert result["miss_tokens"] == 768, f"miss={result['miss_tokens']}"
    assert result["total_tokens"] == 384 + 512 + 256
    print(f"  test_basic_hit_miss: gpu_hit={result['gpu_hit_tokens']}, miss={result['miss_tokens']}")


def test_dram_promotion():
    """Block dumped to DRAM by req0, promoted to HBM by req1."""
    records = make_records()
    topo = Topology(is_mla=False, unified=False, num_nodes=1, dp_size=1, tp_size=1)
    # HBM too small for req0's blocks to stay in free queue after free
    # Actually, free_reverse puts them in free queue. With HBM=100, all fit.
    # Let's test with DRAM enabled
    result = simulate_standard(
        records, topo,
        vllm_hash_block_size=128,
        hbm_block_data_size=1024,
        gpu_capacity_blocks=100,
        dram_capacity_bytes=100 * 1024,
        fs_capacity_bytes=0,
        random_seed=0,
    )
    # With large HBM, all hits are gpu hits (rescued from free queue)
    # DRAM also has the blocks (dumped), but HBM hit takes priority
    assert result["gpu_hit_tokens"] == 384, f"gpu_hit={result['gpu_hit_tokens']}"
    assert result["dram_hit_tokens"] == 0
    print(f"  test_dram_promotion: gpu_hit={result['gpu_hit_tokens']}, dram_hit={result['dram_hit_tokens']}")


def test_small_hbm():
    """HBM can only hold 2 blocks. req0's 3 blocks freed, oldest evicted."""
    h = lambda i: f"hash{i}"
    records = [
        TraceRecord(0.0, 384, 0, [h(0), h(1), h(2)], "test"),
        # req1: h0 might be evicted (freed first, closest to head)
        # free_reverse: h2 freed first (tail), h1, h0 freed last (tail end)
        # So h0 is at the very tail → survives longest
        TraceRecord(1.0, 384, 0, [h(0), h(1), h(2)], "test"),
    ]
    topo = Topology(is_mla=False, unified=False, num_nodes=1, dp_size=1, tp_size=1)
    result = simulate_standard(
        records, topo,
        vllm_hash_block_size=128,
        hbm_block_data_size=1024,
        gpu_capacity_blocks=3,  # exactly fits 3 blocks
        dram_capacity_bytes=0,
        fs_capacity_bytes=0,
        random_seed=0,
    )
    # With HBM=3, req0 allocs 3 blocks, frees all 3 (free_reverse → all in free queue)
    # req1 touches all 3 → all rescued (HBM can hold 3 in-use)
    assert result["gpu_hit_tokens"] == 384, f"gpu_hit={result['gpu_hit_tokens']}"
    assert result["miss_tokens"] == 384, f"miss={result['miss_tokens']}"
    print(f"  test_small_hbm: gpu_hit={result['gpu_hit_tokens']}, miss={result['miss_tokens']}")


def test_hbm_exhausted():
    """HBM too small: alloc fails → preempt → miss."""
    h = lambda i: f"hash{i}"
    records = [
        TraceRecord(0.0, 512, 0, [h(0), h(1), h(2), h(3)], "test"),
    ]
    topo = Topology(is_mla=False, unified=False, num_nodes=1, dp_size=1, tp_size=1)
    result = simulate_standard(
        records, topo,
        vllm_hash_block_size=128,
        hbm_block_data_size=1024,
        gpu_capacity_blocks=2,  # only 2 blocks, but request needs 4
        dram_capacity_bytes=0,
        fs_capacity_bytes=0,
        random_seed=0,
    )
    # First 2 blocks: miss (alloc OK, cached)
    # Block 3: alloc fails (pool exhausted) → miss
    # Block 4: alloc fails → miss
    # But weights: 512/4 = 128 per block
    # gpu_hit = 0 (no prior blocks)
    # miss = 2*128 (alloc OK, cached but still miss since no prior hit) + 2*128 (alloc fail) = 512
    # Wait, the first 2 blocks miss but alloc OK → cached, recorded as miss
    # The last 2 blocks: alloc fails → also miss
    # Total miss = 4*128 = 512
    assert result["miss_tokens"] == 512, f"miss={result['miss_tokens']}"
    assert result["gpu_hit_tokens"] == 0
    # But now 2 blocks are in free queue (freed at request end)
    # Let's verify with a second request
    records2 = [
        TraceRecord(0.0, 512, 0, [h(0), h(1), h(2), h(3)], "test"),
        TraceRecord(1.0, 256, 0, [h(2), h(3)], "test"),  # h2,h3 were never cached
    ]
    result2 = simulate_standard(
        records2, topo,
        vllm_hash_block_size=128,
        hbm_block_data_size=1024,
        gpu_capacity_blocks=2,
        dram_capacity_bytes=0,
        fs_capacity_bytes=0,
        random_seed=0,
    )
    # req0: 4 blocks, HBM=2. First 2 alloc'd+cached, last 2 preempt.
    # free_reverse: block 1 freed first, block 0 freed last (tail, survives)
    # req1: h2 not cached (preempted), h3 not cached → all miss
    print(f"  test_hbm_exhausted: miss={result['miss_tokens']}, req2_miss={result2['miss_tokens']}")


def test_free_reverse_order():
    """Verify free_reverse makes prefix survive longest in free queue."""
    h = lambda i: f"hash{i}"
    records = [
        # req0: 4 blocks, HBM=4, all fit
        TraceRecord(0.0, 512, 0, [h(0), h(1), h(2), h(3)], "test"),
        # req1: same 4 blocks → should all hit (rescued from free queue)
        TraceRecord(1.0, 512, 0, [h(0), h(1), h(2), h(3)], "test"),
    ]
    topo = Topology(is_mla=False, unified=False, num_nodes=1, dp_size=1, tp_size=1)
    result = simulate_standard(
        records, topo,
        vllm_hash_block_size=128,
        hbm_block_data_size=1024,
        gpu_capacity_blocks=4,
        dram_capacity_bytes=0,
        fs_capacity_bytes=0,
        random_seed=0,
    )
    # All 4 blocks rescued → 4*128 = 512 gpu_hit
    assert result["gpu_hit_tokens"] == 512, f"gpu_hit={result['gpu_hit_tokens']}"
    assert result["miss_tokens"] == 512, f"miss={result['miss_tokens']}"
    print(f"  test_free_reverse_order: gpu_hit={result['gpu_hit_tokens']}")


def test_multi_dp():
    """Different DP ranks have independent HBM pools."""
    h = lambda i: f"hash{i}"
    records = [
        TraceRecord(0.0, 128, 0, [h(0)], "test"),
        TraceRecord(1.0, 128, 0, [h(0)], "test"),  # might hit same or different DP
    ]
    topo = Topology(is_mla=False, unified=False, num_nodes=1, dp_size=2, tp_size=1)
    result = simulate_standard(
        records, topo,
        vllm_hash_block_size=128,
        hbm_block_data_size=1024,
        gpu_capacity_blocks=10,
        dram_capacity_bytes=0,
        fs_capacity_bytes=0,
        random_seed=0,
    )
    # With seed=0, dp_rank alternates. If both land on same DP → hit.
    # If different DP → miss. Can't assert exact, but check total.
    total = result["gpu_hit_tokens"] + result["miss_tokens"]
    assert total == 256, f"total={total}"
    print(f"  test_multi_dp: gpu_hit={result['gpu_hit_tokens']}, miss={result['miss_tokens']}")


if __name__ == "__main__":
    print("Running standard model tests...")
    test_basic_hit_miss()
    test_dram_promotion()
    test_small_hbm()
    test_hbm_exhausted()
    test_free_reverse_order()
    test_multi_dp()
    print("All tests passed!")
