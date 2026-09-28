# BF16 MoE routing output-budget experiment

Status: isolated experimental candidate; no NPU result for this code yet.

The frozen 64-request Qwen3.5-35B-A3B BF16 trace has an 11.72% output-token
budget imbalance if requests are alternated between two DP ranks. The earlier
workload-aware route (`feat/adm-workload-aware-routing@7b1be47`) did not
establish a serving gain: its first candidate session measured 334.792 and
332.208 output tokens/s, and its second measured 334.600 and 327.188. Two
unchanged baseline sessions measured 331.790/332.989 and 331.916/334.574.
All eight rounds completed 64/64 requests. Results are local under
`/workspace/kmx/results/adm/qwen35-tp2-dp2-routing-skew-*`.

This branch changes only the request work estimate from
`min(16, input_tokens/512 + output_budget/128)` to
`min(32, input_tokens/512 + output_budget/32)`. The native waiting/running
count score, internal-DP-only scope, opt-in switch, and exact core-method
fingerprint remain unchanged. The larger output weight is meant to avoid
placing several long decode requests on one rank during a burst.

Gate: host tests first, then one matched BF16 TP2/DP2 baseline-candidate
screen under the same frozen trace and graph settings. If either candidate
round is worse than the nearby baseline beyond ordinary run variation, keep
this branch experimental. Record the complete per-round output, latency,
error counts, and model/runtime/source hashes before any promotion decision.
