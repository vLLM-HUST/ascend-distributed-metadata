# Qualification results

## Combined DP2 plugin and load-metadata replica

The [requested-replay smoke](qwen35-bf16-requested-replay-20260929.json)
used the paired plugin and core branches. The coordinator published sequence
7 in response to the injected recovery ID, and the API's durable receipt
consumed that same sequence. This verifies the control-message path under a
BF16 TP2/DP2 service. It does not measure a performance gain.

The [durable-receipt smoke](qwen35-bf16-durable-receipt-20260929.json)
repeated the BF16 TP2/DP2 combined service with an existing receipt directory.
After the injected local API replica loss, routing consumed the recovered
snapshot and the API wrote one content-addressed JSON file with `0600`
permissions. The file content matched the log receipt. Automatic loss
detection and worker restart were outside this check.

The [BF16 Qwen3.5 TP2/DP2 recheck](qwen35-bf16-combined-replica-r2-20260929.json)
used the published packed synchronization path alongside the optional
coordinator/API replica on the reviewed core-hook branch. The service became
healthy, answered a completion request, and emitted a recovery receipt after
an injected local API replica loss and subsequent routing read. The
[first smoke](qwen35-bf16-combined-replica-20260929.json) ran before an
additional consumer validation fix. Both runs passed. This is recovery
correctness evidence for a diagnostic loss, not a throughput comparison or a
worker-failure test.

## DP4 word MOD

The [machine-readable record](qwen3moe-dp4-int64-word-20260928.json) covers
Qwen3-30B-A3B-W8A8 with TP1/DP4, four Ascend 910B2 devices, a frozen
64-request trace, and `FULL_DECODE_ONLY` graphs. Six service sessions ran in
published–candidate–published–candidate–candidate–published order, with two
rounds each. Every round completed 64/64 requests. The paired session-mean
throughput differences were +0.38%, +0.71%, and +1.32%. Overall means were
463.376 tok/s published and 467.099 tok/s candidate. The first round of each
of the first two candidate sessions was slower than its preceding published
session, so these results do not imply a per-round gain.

Two independent CPU/Gloo direct-method comparisons passed correctness checks
and measured 0.559 vs 0.356 ms and 0.526 vs 0.366 ms per call (published vs
candidate). These exclude NPU communication and model execution. The record
includes raw result hashes, code and workload pins, and activation evidence
for all four DP ranks.

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
