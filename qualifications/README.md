# Qualification results

## DP2 scalar MOD

[Machine-readable record](qwen35-dp2-scalar-candidate-20260925.json):
Qwen3.5-35B-A3B, TP4/DP2, eight Ascend 910B2 devices, 32-request
`random-online-4chip` subset, eager mode. The candidate–published
MOD–candidate sequence completed 32/32 requests in every run. Candidate
output throughput was 26.979 and 27.052 tok/s; the intervening published MOD
run was 26.906 tok/s. The two deltas were +0.27% and +0.54%, with a +0.41%
mean. The difference is small relative to run variation; repeatable serving
speedup has not been established.

Two independent local CPU/Gloo comparisons of the same scalar candidate
against the previous MOD measured 0.253 vs 0.185 ms and 0.310 vs 0.210 ms
per metadata call (previous vs candidate). Correctness passed. The candidate
was 26.8% and 32.4% faster in this limited call-level scope. The exact
[candidate script](../benchmarks/dp2_scalar_candidate_cpu_gloo.py) and
original-result hashes are in the JSON record. These figures exclude NPU
communication and model execution.

## Earlier packed MOD call result

[CPU/Gloo call-level record](dp2-packed-sync-cpu-gloo-20260925.json):
two independent CPU/Gloo comparisons against the native method measured
0.415 vs 0.310 ms and 0.424 vs 0.299 ms (native vs original packed MOD),
or 25.4% and 29.5% less time. The [script](../benchmarks/dp_sync_cpu_gloo.py)
is byte-identical after its move to `benchmarks/`. The full serving comparison,
which did not favor that original MOD, remains on the
[`feat/epoch-bound-replica` experiment branch](https://github.com/vLLM-HUST/ascend-distributed-metadata/blob/feat/epoch-bound-replica/qualifications/qwen35-35b-a3b-dp2-mod-comparison-20260925.json).

## Model execution and baseline

The [TP4/DP1 smoke](qwen35-35b-a3b-tp4-dp1-smoke-20260925.json) completed
32/32 requests at 23.15 output tok/s. DP1 uses the native metadata path.
The [TP4/DP2 baseline](qwen35-35b-a3b-tp4-dp2-baseline-20260925.json)
completed 32/32 requests at 27.16 output tok/s with the original MOD off.
Neither single run is a speedup comparison. Both records contain source and
workload revisions, sanitized metrics, and hashes of the retained raw data.

Negative results and configuration-only gains are indexed in
[experimental work](../docs/experiments.md).
