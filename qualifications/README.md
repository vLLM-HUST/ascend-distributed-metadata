# Qualification results

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
