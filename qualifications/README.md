# Qualification results

## High-concurrency graph capacity — 2026-09-26

The [machine-readable comparison](qwen35-dp2-capacity-tuning-20260926.json)
records four Qwen3.5-35B-A3B TP4/DP2 runs on eight Ascend 910B2 devices.
Every run completed 128/128 synthetic requests without a reported request
error. At concurrency 64, raising per-engine `--max-num-seqs` from 16 to 32
and adding graph capture sizes 24 and 32 increased two-run mean output
throughput from 312.39 to 446.81 tokens/s (+43.0%). Mean TTFT fell from
11.67 to 2.26 s (−80.6%), while mean TPOT rose from 90.71 to 115.28 ms
(+27.1%). The 16-sequence configuration was rerun after both 32-sequence
runs and returned to its earlier throughput range.

The same [experimental MOD branch](https://github.com/vLLM-HUST/ascend-distributed-metadata/tree/perf/graph-downgrade-unpad)
was enabled in both configurations. This is a stack configuration gain for
the measured workload, not an ADM source-code speedup. The branch records
the separate graph-padding and MOD on/off checks. Generated text and raw
logs remain local; result hashes and aggregate metrics are in the JSON.

## Qwen3.5-35B-A3B model smoke test — 2026-09-25

The [machine-readable result](qwen35-35b-a3b-tp4-dp1-smoke-20260925.json)
records a 32-request subset of the benchmark repository's
`random-online-4chip` scenario. It used synthetic 1024-token inputs and
256-token outputs, seed 0, TP4/DP1, eager execution, and four Ascend 910B2
devices. All 32 requests completed with no request errors. The measured
duration was 353.80 s and output throughput was 23.15 tokens/s.

The model and source revisions, benchmark scenario hash, aggregate metrics,
and sanitized per-request lengths and latencies are in the JSON file. The
original detailed result remains local; its SHA256 is recorded in the JSON.
Generated text, host paths, and raw server logs are not published.

The plugin was enabled for this run, but DP1 takes the native metadata path.
This result establishes model execution only. A MOD speedup requires a
matched DP2 baseline, treatment, and recheck.

## Qwen3.5-35B-A3B TP4/DP2 baseline — 2026-09-25

The [baseline result](qwen35-35b-a3b-tp4-dp2-baseline-20260925.json) uses
the same 32-request subset, model, seed, token lengths, and concurrency on
eight Ascend 910B2 devices. The MOD was configured off. All 32 requests
completed without errors. Duration was 301.59 s and output throughput was
27.16 tokens/s. The original detailed result is retained locally and its
SHA256 is recorded in the public JSON.

This is the first DP2 baseline. A matched MOD run and independent recheck are
still required before reporting a performance difference.

## DP2 MOD comparison and sync-call microbenchmark — 2026-09-25

The [comparison result](qwen35-35b-a3b-dp2-mod-comparison-20260925.json)
records two baseline and two MOD full-model runs under the same 32-request
TP4/DP2 workload. Each completed 32/32 requests without errors. Baseline
output throughput was 27.16 and 26.76 tokens/s; MOD throughput was 26.82 and
26.86 tokens/s. These runs do not establish an end-to-end speedup.

Two independent local two-process CPU/Gloo measurements directly called the
pinned Ascend sync method and the MOD wrapper. Median per-call times were
0.415 vs 0.310 ms and 0.424 vs 0.299 ms, respectively (native vs MOD).
The MOD was 25.4% and 29.5% faster in this limited sync-call measurement.
Correctness was checked before timing. The result file includes all eight
alternating timing blocks per run and SHA256 hashes of the original results.
The exact [microbenchmark script](benchmark_dp_sync.py) is included here;
its SHA256 matches the hashes recorded for both runs.
This CPU/Gloo measurement does not measure NPU communication or Qwen serving.
