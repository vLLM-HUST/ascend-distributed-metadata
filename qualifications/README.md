# Qualification results

## Experimental DP2 token-vector construction — 2026-09-26

The [machine-readable record](qwen35-dp2-frombuffer-token-vector-20260926.json)
covers a code change that builds a fresh, writable CPU token vector through
`torch.frombuffer` and a Python `array('i')`. Three alternating two-process
CPU/Gloo comparisons against the preceding plugin branch measured median
metadata-call reductions of 9.4%, 10.0%, and 11.3%. All 17 host tests passed,
including caller mutation and repeated-step checks.

A Qwen3.5 TP2/DP2 baseline–candidate–baseline–candidate service sequence
completed 64/64 requests in each of eight benchmark rounds. Warm second-round
throughput was 233.3/229.7 tokens/s for the baseline and 227.9/231.2 for
the candidate. This did not establish a service gain, so this branch remains
experimental. A separate two-NPU diagnostic found that moving the scalar
round trip from CPU/Gloo to NPU/HCCL was slower (0.124 versus 0.322 ms);
the communication backend was left unchanged.

## DP2 word and CPU group reuse — 2026-09-26

The [machine-readable result](qwen35-dp2-word-group-cache-20260926.json)
records three repeated, alternating two-process CPU/Gloo measurements against
the preceding experimental ADM source. Reusing the DP2 scalar collective
buffer and its CPU process group shortened the median metadata sync call by
2.3%–4.2% across the three direct comparisons. All 17 host tests passed,
including repeated-step and fallback cases. This result is limited to the
local metadata method and CPU/Gloo transport.

Four Qwen3.5-35B-A3B TP2/DP2 service sessions on NPU 0–3 completed 64/64
requests in each of two benchmark rounds. The sequence was baseline,
candidate, baseline, candidate. Warm second-round output throughput was
232.0 and 231.3 tokens/s for the baseline versus 234.3 and 232.5 for the
candidate. The 0.7% mean difference is too small to establish a reliable
serving speedup. The first baseline session's initial benchmark round was
much slower, so first-round measurements are reported separately. NPU 7 was
occupied; these TP2/DP2 runs are separate from the prior TP4/DP2 results.

## Graph padding and high-concurrency capacity — 2026-09-26

The [machine-readable comparison](qwen35-dp2-graph-padding-capacity-20260926.json)
records Qwen3.5-35B-A3B TP4/DP2 runs on eight Ascend 910B2 devices. All
recorded serving runs completed every request without a reported request
error. With `FULL_DECODE_ONLY`, this experimental MOD branch nearly removed
DP padding on steps whose synchronized graph mode was `NONE`. The A–B–A
MOD-enabled/disabled throughput runs did **not** establish a repeatable
end-to-end ADM speedup.

For a separate 128-request synthetic workload at concurrency 64, increasing
`--max-num-seqs` from 16 to 32 and adding graph capture sizes 24 and 32
raised mean output throughput from 312.39 to 446.81 tokens/s (+43.0%) and
reduced mean TTFT from 11.67 to 2.26 s (−80.6%) across two runs per
configuration. Mean TPOT rose from 90.71 to 115.28 ms (+27.1%). Both
configurations had the same MOD enabled. This is a stack configuration gain
for the measured workload, not evidence that ADM source code accelerated
serving. Generated text and raw logs remain local; result hashes and
sanitized aggregate metrics are recorded in the JSON file.

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

## DP2 scalar-sync candidate — 2026-09-25

The [candidate record](qwen35-dp2-scalar-candidate-20260925.json) covers an
experimental DP2 one-`int32` all-reduce. Two independent local CPU/Gloo
comparisons against the published MOD passed correctness and measured median
per-call times of 0.253 vs 0.185 ms and 0.310 vs 0.210 ms (published vs
candidate). The candidate was 26.8% and 32.4% faster in this sync-call scope.
The exact [candidate microbenchmark script](benchmark_dp2_scalar_candidate.py)
and original-result SHA256 hashes are included.

A Qwen3.5 TP4/DP2 candidate-published-candidate serving sequence completed
32/32 requests in every run. Candidate output throughput was 26.979 and
27.052 tokens/s versus 26.906 tokens/s for the intervening published MOD.
The 0.27% and 0.54% differences are too small to establish a service-level
gain against prior serving variation.
