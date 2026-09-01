# UCM Trace Mode User Guide

This document describes how to use **UCM Trace Mode**, a lightweight diagnostic and evaluation mode that records per-request traces during inference **without** performing any actual KV cache dump/load operations. Trace Mode lets you collect real request traffic data and simulate the theoretical KV cache hit rate UCM could deliver before committing to a full UCM storage rollout.

It is recommended to first collect hit ratio statistics with Trace Mode and confirm with relevant project members whether to adopt UCM.

## 1. Overview

Trace Mode supports three model categories, each with a dedicated Lite connector variant:

| Model type   | Connector                | Examples                                      |
| :----------- | :----------------------- | :-------------------------------------------- |
| `standard` | `UCMLiteConnector`     | GQA / MLA models (e.g. Qwen2.5)               |
| `mamba`    | `UCMHLALiteConnector`  | Hybrid linear-attention (Qwen3-Next, Kimi-K3) |
| `fawa`     | `UCMFAWALiteConnector` | DeepSeek-V4 (FA + WA mixed attention)         |

Trace Mode is enabled by setting two options to `true` in the UCM configuration file:

| Option                   | Default   | Description                                                                                                                        |
| :----------------------- | :-------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| `enable_record_traces` | `false` | Logs per-request traces (timestamp, input_length, output_length, block_hashes). Each hash is a 16-byte hex string (32 characters). |
| `use_lite`             | `false` | Switches to the**UCM Lite Connector**, which works with a Fake Store that skips all actual KV dump/load operations.          |

When both are enabled, `UCMConnector` internally instantiates the appropriate Lite connector variant based on the model's KV cache layout. The Lite connector:

- Computes the same block hash IDs that a real UCM deployment would use.
- Logs a topology line at startup and a trace record for every request on its first lookup.
- Returns `0` external hit tokens (there is no real store to look up), so inference correctness is unaffected and no KV data is persisted.
- Implements all KV transfer hooks (`start_load_kv`, `save_kv_layer`, `wait_for_save`, etc.) as no-ops.

### Topology Line (startup, one per process)

At startup the Lite connector emits a single `UCMTraceMeta:` line that encodes the model's KV cache topology. The simulator parses this line automatically to determine model type, block sizes, and per-group configuration — **you don't need to specify any of these on the CLI**.

```text
UCMTraceMeta: type=<standard|mamba|fawa>, is_mla=<bool>, vllm_hash_block_size=<T>, hbm_block_data_size=<B>, ...
```

The exact fields vary by model type (mamba adds `lcm_block_size`/`mamba_groups`; fawa adds `ucm_hash_block_size`/`fa_file_size`/`wa_file_size`/`alignment_tokens`/`groups`). This line is for the simulator's internal use only.

### Trace Line (per request)

Each request produces one log line on first lookup:

```text
UCMTrace: timestamp: 1234567.890123, request_id: req-42, input_length: 8192, output_length: 128, block_hashes: ['a1b2...', 'c3d4...', ...]
```

| Field             | Description                                                                                                                                                                      |
| :---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `timestamp`     | `time.perf_counter()` value at lookup time, used to preserve request ordering during analysis.                                                                                 |
| `request_id`    | vLLM request identifier (present in Lite connector traces).                                                                                                                      |
| `input_length`  | Number of input tokens in the request (`request.num_tokens`).                                                                                                                  |
| `output_length` | Maximum output tokens for the request (`request.max_tokens`).                                                                                                                  |
| `block_hashes`  | List of hex-encoded block hash IDs (vLLM`request.block_hashes`). Each block corresponds to `vllm_hash_block_size` tokens; each hash is a 16-byte hex string (32 characters). |

## 2. Configuration

You can start from the sample file at `unified-cache-management/examples/ucm_config_example.yaml`

Trace Mode is enabled by setting two options to `true`:

```yaml
enable_record_traces: true
use_lite: true
```

### Log Configuration(Optional)

Trace line can be large because each hash is a 32-character hex string and a long request can contain many block hashes. Tune the following environment variables before launching the service:

| Environment Variable  | Default  | Description                                                                                                                     |
| :-------------------- | :------- | :------------------------------------------------------------------------------------------------------------------------------ |
| `UCM_LOG_PATH`      | `log`  | Directory for per-process log files (e.g.`ucm-<pid>.log`).                                                                    |
| `UCM_LOG_MAX_FILES` | `10`   | Maximum number of rotated log files kept per process.                                                                           |
| `UCM_LOG_MAX_SIZE`  | `5`    | Maximum size in**MiB** per log file before rotation. Increase this significantly when recording traces for long requests. |
| `UCM_LOG_LEVEL`     | `info` | Log level. Traces are emitted at`INFO` level, so keep this at `info` (or lower for extra debug output).                     |

Example for a trace-collection run:

```bash
export UCM_LOG_PATH=/workspace/ucm-trace-logs
export UCM_LOG_MAX_SIZE=256      # 256 MiB per file
export UCM_LOG_MAX_FILES=50     # keep up to 50 rotated files per process
export UCM_LOG_LEVEL=info
```

## 3. Launching the Inference Service

Trace Mode is deployed as an OpenAI-compatible vLLM server. Start it the same way as a normal UCM deployment — the only difference is the UCM config file contents.

Take the Qwen/Qwen2.5-14B-Instruct model as an example:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
  --max-model-len 32000 \
  --tensor-parallel-size 2 \
  --gpu_memory_utilization 0.87 \
  --block_size 128 \
  --trust-remote-code \
  --port 7800 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{
      "kv_connector": "UCMConnector",
      "kv_role": "kv_both",
      "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
      "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/workspace/unified-cache-management/examples/ucm_config_example.yaml"}
  }'
```

**⚠️ Replace the `UCM_CONFIG_FILE` path with the actual path to your trace-mode config file on your machine.**

You can now send production-equivalent traffic to the server. Every request will produce a trace record in the UCM log directory. No KV cache is dumped or loaded.

## 4. Trace Analysis

After collecting traces, run `benchmarks/auto_trace_analysis.py` to simulate the theoretical KV cache hit rate. The script parses the `UCMTraceMeta:` topology line **plus** the `available kv cache memory` (or `current kv cache memory`), `tensor_parallel_size`, and `data_parallel_size` values that vLLM/UCM emit at startup, then simulates a multi-tier cache (HBM → DRAM → FS) to estimate the hit rate UCM would achieve. The simulation engine (standard / mamba / fawa) is selected automatically based on the `type` field in the topology line.

```bash
python benchmarks/auto_trace_analysis.py \
  --log-dir <path to log folder> \
  --dram-pool-size-gb <dram_gb> \
  --fs-pool-size-gb <fs_gb> \
  --num-nodes <physical nodes> \
  [--max-num-batched-tokens <chunk_size>] \
  [--unified-memory-pool] \
  [--service-url <ip:port>] \
  [--output <path>] \
  [--trace-output <path>]
```

Required arguments:

| Argument                | Description                                                                                                                                                                                                                                                                                                                         |
| :---------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--log-dir`           | Directory containing the UCM log files (scanned recursively for`*.log`, `*.log.*`, `*.log.gz`). It must include vLLM's startup logs (captured to `<UCM_LOG_PATH>/vllm-<pid>.log` by default) so the available KV cache memory, tensor-parallel size, data-parallel size, and `UCMTraceMeta:` topology line can be parsed. |
| `--dram-pool-size-gb` | Simulated DRAM (host memory) pool size in GiB (total cluster budget).                                                                                                                                                                                                                                                               |
| `--fs-pool-size-gb`   | Simulated filesystem (SSD/NFS) pool size in GiB (total cluster budget).                                                                                                                                                                                                                                                             |
| `--num-nodes`         | Number of physical nodes (manual; not logged by vLLM).                                                                                                                                                                                                                                                                              |

Optional arguments:

| Argument                     | Description                                                                                                                                                                                          |
| :--------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--max-num-batched-tokens` | vLLM`--max-num-batched-tokens` (chunked-prefill budget, default 8192). For hybrid models, state is dumped at chunk-end boundaries; set to your service's actual value for accuracy.                |
| `--unified-memory-pool`    | If set, DRAM is a single global pool (FA+WA share one`ByteLRUPool` for FAWA). If unset, DRAM is partitioned per-node (MLA) or per-DP-rank (GQA); FAWA splits DRAM evenly between FA and WA stores. |
| `--service-url`            | vLLM`/metrics` endpoint (Prometheus). When set, the tool fetches the service's actual prefix-cache hit rate for comparison.                                                                        |
| `--metrics-timeout`        | Timeout for fetching Prometheus metrics (default 5.0 seconds).                                                                                                                                       |
| `--random-seed`            | Random seed for DP-rank routing (default 0).                                                                                                                                                         |
| `--trace-output`           | If set, exports parsed trace records as JSON Lines to the given path.                                                                                                                                |
| `--output`                 | If set, writes the full analysis result as JSON to the given path.                                                                                                                                   |

## 5. Report

`print_summary` prints the following to stdout (the same values are written to the `analysis` section of the `--output` JSON).

### Standard / Mamba models

```text
Trace cache hit rate analysis
  Model type: standard
  Total request count: 1200
  Total request token count: 19660800
  HBM available: 12.50 GiB
  TP=2  DP=2  Nodes=1
  DRAM pool: 64.00 GiB  FS pool: 1024.00 GiB
  Theoretical max KV cache hit rate: 78.340000%
  HBM theoretical hit rate: 21.560000%
  HBM + DRAM pool theoretical hit rate: 45.210000%
  HBM + DRAM pool + FS pool theoretical hit rate: 72.890000%
  Request lifetime sample count: 980
  Average request lifetime: 142.350000 s
  P90 request lifetime: 318.700000 s
  P95 request lifetime: 405.120000 s
```

### FAWA (DeepSeek-V4) models

FAWA models run each scenario twice — once with **block-wise** WA dump (every hash-block boundary) and once with **chunk-wise** WA dump (only chunk-end boundaries) — producing two groups of four scenarios:

```text
Trace cache hit rate analysis
  Model type: fawa
  Total request count: 1200
  Total request token count: 19660800
  HBM available: 12.50 GiB
  TP=2  DP=2  Nodes=1
  DRAM pool: 64.00 GiB  FS pool: 1024.00 GiB
  [block_wise]
    Theoretical max: 80.120000%
    HBM:              23.450000%
    HBM+DRAM:         48.670000%
    HBM+DRAM+FS:      75.340000%
  [chunk_wise]
    Theoretical max: 75.890000%
    HBM:              20.120000%
    HBM+DRAM:         43.560000%
    HBM+DRAM+FS:      70.230000%
  Request lifetime sample count: 980
  Average request lifetime: 142.350000 s
  P90 request lifetime: 318.700000 s
  P95 request lifetime: 405.120000 s
```

> Block-wise typically yields higher hit rates (any boundary can be matched) at the cost of more dump I/O. Chunk-wise has lower I/O but can only match at chunk boundaries. Choose the mode that best reflects your deployment's dump strategy.

**Workload & capacity echo** — confirms what was parsed from the logs and derived from your arguments:

| Metric                        | Meaning                                                                                                                                         |
| :---------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------- |
| `Model type`                | Model category (`standard` / `mamba` / `fawa`) parsed from the `UCMTraceMeta` topology line.                                            |
| `Total request count`       | Number of trace records parsed (= request count).                                                                                               |
| `Total request token count` | Sum of`input_length` across all requests.                                                                                                     |
| `HBM available`             | Single-card HBM KV budget in GiB. Parsed from vLLM's`Current/Available KV cache memory`; this is the per-DP-rank HBM, never multiplied by TP. |
| `TP`                        | Tensor-parallel size resolved from the logs.                                                                                                    |
| `DP`                        | Data-parallel size resolved from the logs.                                                                                                      |
| `DRAM pool` / `FS pool`   | The`--dram-pool-size-gb` / `--fs-pool-size-gb` you passed in.                                                                               |

**Hit-rate scenarios** — four simulations with different tier capacities; each rate is `hit_tokens / total_tokens`. For FAWA models, each scenario runs twice (block-wise and chunk-wise WA dump):

| Metric                                             | Tier capacity used                                         | What it tells you                                              |
| :------------------------------------------------- | :--------------------------------------------------------- | :------------------------------------------------------------- |
| `Theoretical max KV cache hit rate`              | every tier =`unique_block_count` (effectively unlimited) | Upper bound — the most UCM could ever reach for this traffic. |
| `HBM theoretical hit rate`                       | HBM only (DRAM=FS=0)                                       | What you get with no external pool — pure on-device KV.       |
| `HBM + DRAM pool theoretical hit rate`           | HBM + DRAM (FS=0)                                          | Marginal gain from adding a host-memory pool.                  |
| `HBM + DRAM pool + FS pool theoretical hit rate` | all three tiers                                            | Closest to a real UCM deployment's expected hit rate.          |

Read them as a monotonic ladder: `HBM` ≤ `HBM+DRAM` ≤ `HBM+DRAM+FS` ≤ `Theoretical max`. Key points:

- **`Theoretical max`** is the ceiling for this traffic. If it is already low, the workload has little prefix reuse and UCM will not deliver much uplift regardless of pool size — in that case adopting UCM may not be worthwhile.
- **`HBM`** is the no-external-pool baseline. Note the simulated value assumes a single in-flight request; under real **concurrency** multiple requests compete for the HBM KV budget and evict each other, so the actual on-device hit rate will be **lower** than this value.
- **`HBM + DRAM pool`** is what a DRAM pool of the specified size can reach. Compared against the live `Service actual KV cache hit rate` (only shown when `--service-url` is configured, fetched from vLLM's `/metrics`), the difference is the uplift the DRAM pool brings.
- **`HBM + DRAM pool + FS pool`** vs **`HBM + DRAM pool`**: the delta is the additional hit rate contributed by the filesystem (SSD/NFS) tier.
- If the three-tier value is already near the theoretical max, enlarging DRAM/FS yields little; if there is a big gap, bigger pools still help.

**Request lifetime** — how long a request's blocks stay reusable (time from first appearance to the last hit on any of its blocks):

| Metric                            | Meaning                                                      |
| :-------------------------------- | :----------------------------------------------------------- |
| `Request lifetime sample count` | Number of request groups that had at least one block reused. |
| `Average request lifetime`      | Mean reuse lifetime.                                         |
| `P90 request lifetime`          | 90% of reused blocks are hit again within this window.       |
| `P95 request lifetime`          | 95% of reused blocks are hit again within this window.       |

**`Average` / `P90` / `P95 request lifetime`** is the request's actual alive time — the span from when the request first appears (chat start) to the last time any of its blocks was reused. It measures how long a conversation's KV stays useful. Use it to size retention: if `P95 = 405 s`, blocks must stay in cache ~7 min to capture 95% of reuse — compare against your DRAM/FS capacity and eviction to judge whether the pool is large enough and retention is long enough.
