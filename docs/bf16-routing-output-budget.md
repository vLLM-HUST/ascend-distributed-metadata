# BF16 MoE routing output-budget experiment

Status: isolated experimental candidate with a repeated positive result on a
constructed output-skew stress trace. It is not integrated into `main`.

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

On the original, unreordered trace this branch measured 331.726/335.400 and
335.928/339.969 output tokens/s. The intervening baseline measured
337.797/338.805. This does not establish a gain on the original trace.

For a separate mechanism stress test, the same 64 real prompts and output
budgets were reordered so the 32 longest outputs occupy even arrival slots and
the 32 shortest occupy odd slots. Arrival times were kept. This produces a
33.1% output-budget imbalance under strict alternating assignment. The two
baseline–candidate session pairs both favored the candidate by 2.09% and
2.01% in session-mean output throughput, with 64/64 requests successful in
every round. Mean output throughput across four rounds was 337.389 vs
344.310 tokens/s (baseline vs candidate, +2.05%). The candidate also had a
lower average p99 request latency. Exact pins, per-round numbers, and raw
result hashes are in
[`qualifications/qwen35-bf16-routing-output-skew-20260928.json`](../qualifications/qwen35-bf16-routing-output-skew-20260928.json).

This supports the narrow claim that decode-budget-weighted routing can help
under a deliberate output-skew burst. The previous natural-trace result and
the construction of this stress trace must accompany any presentation of the
positive number. No main-branch or general-workload gain is claimed.
