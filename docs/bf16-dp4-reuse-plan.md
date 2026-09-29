# BF16 MoE DP4 metadata word reuse experiment

Status: experimental, not promoted. Three CPU/Gloo comparisons passed
correctness and shortened the median DP4 metadata call by 3.1%, 6.0%, and
4.0%. A W8A8 Qwen3-30B-A3B TP1/DP4 service screen completed 64/64 requests
in each round but the candidate's two-round mean was 0.91% lower than the
published MOD. See the [sanitized result record](../qualifications/dp4-reuse-word-exploration-20260928.json).
BF16 TP2/DP4 was not run because eight devices were not free during this
screen. The candidate stays on this branch and does not enter `main`.

## Frozen comparison

- Published MOD: `main@16362b2`, DP4 packed `int64` all-reduce.
- Candidate: this branch; each runner owns one reusable CPU `int64` word. The
  rank-local slot is reset before every synchronous collective. If Python's
  signed 64-bit array is unavailable, use a reusable Torch CPU tensor.
- Model: local Qwen3.5-35B-A3B ModelScope BF16 MoE weights. Verify the exact
  model files, runtime revisions, and eight free 910B2 devices at admission.
- Proposed topology: TP2/DP4. Prior TP2/DP2 serving establishes per-rank
  model feasibility, but TP2/DP4 itself is not yet admitted.
- First workload: a fixed rank-skewed 64-request text trace, then one uniform
  trace if the first comparison is positive. Keep prompts, request order,
  concurrency, graph mode, model length, and launch settings identical.

## Gates

1. Host correctness: reconstructed rank token vector, minimum graph mode,
   padding, fallback on every rank, repeated calls, and no state leakage.
2. Four-rank CPU/Gloo direct-method comparison against published MOD using
   `benchmarks/dp4_word_cpu_gloo.py`. This is a call-level filter, not a
   service-level claim. Keep the raw result and source hashes.
3. Admit BF16 TP2/DP4 with published MOD before running the candidate. Confirm
   that the plugin's DP4 path is active and all 64 requests complete.
4. Alternate published MOD and candidate service sessions, at least three
   matched pairs with two rounds per session. Compare output tokens/s as the
   primary metric; report TTFT, TPOT, failures, and per-round variation.
5. If positive, compare against native ADM-disabled serving under the same
   frozen carrier. Only publish a throughput claim supported by the full
   matched record. A call-level gain alone stays experimental.

The CPU/Gloo comparison and all NPU work are owner-executed under
`/workspace/kmx/adm/command-contract.md`.
