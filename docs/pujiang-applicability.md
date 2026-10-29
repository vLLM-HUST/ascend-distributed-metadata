# Pujiang five-dataset applicability and next-run contract

Reviewed inputs: ADM main `3bc9e876ee862b20accbe98149c0ad6ff458e16d`,
[ADM #5](https://github.com/vLLM-HUST/ascend-distributed-metadata/issues/5),
[#7](https://github.com/vLLM-HUST/ascend-distributed-metadata/issues/7),
[benchmark #254](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254),
[#403](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/403),
[dataset program #253](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/253),
benchmark snapshot `c17289b58c0e3d93033d8426d8d8b55ce17fc253`, subsequent
main `a181a5e94414bca9b51bb89e5ce2da6bdcef38ce`, and Extension
Manager main `a78c9681ee428589dc7666c0c159cb24c3f0c58a`. Issue #403's current
**plan-review-first / zero-new-measurement** section overrides its historical
SWE/Terminal queue and old AMD64-backend text. No agent, container, NPU server,
judge or dataset score was run as part of this ADM audit.

## Two different comparisons

1. **Organization B0/B1:** preserve the named organization baseline and actual
   B1 identities; B0 is vLLM 0.18 / vLLM-Ascend 0.18. The current
   [V5.5 delivery contract](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/c17289b58c0e3d93033d8426d8d8b55ce17fc253/docs/ACCEPTANCE_V5_5.md)
   specifies Qwen3.5-35B-A3B BF16, two 910B2, TP2. Under the normal TP×DP
   mapping this is DP1: the reviewed ADM runner does not call the DP sync hook.
   Existing historical B0/B1 runs may use other explicitly recorded settings;
   none becomes an ADM result without runtime evidence. The old V5.4 requirement
   remains preserved as historical; it is not silently rewritten.
2. **ADM OFF/ON ablation:** separately freeze identical model, precision,
   source, software stack, DP topology, workload and scheduler settings. Change
   only ADM activation. TP2/DP2 needs at least four exclusively allocated
   devices; historical TP4/DP2 used eight. Both require separate host, memory,
   communication and correctness qualification. TP1/DP4 needs four devices
   but Qwen3.5 per-replica fit/support is UNKNOWN. These are not replacement
   B0 definitions or automatically supported deployment recipes.

The ECPA manifest currently admits host versions `>=0.11.2.dev0,<0.12` and
the checked runtime qualification pins core `e1248fa...` and Ascend
`367b8e62...`. This excludes a 0.18 host. Independently, the locally available
Ascend-HUST `v0.18.0` tag resolves to `e18643f8a4d5bd9990727654318ad069ea0b56e2`;
its sync method has a `with_prefill` signature and normalized fingerprint
`5db4db388be800094cebc6c28e36bf70bf9f1ddc420b9537f1301a113a265fee`, not ADM's
`3e3cac40908b68c65c45c820efca81073c0a11bef5eab13b2f8cf9ce36e407f5`.
That tag is source evidence, not proof of the actual organization OCI image's
contents. A reviewed adapter and exact actual-image/source verification are
needed; widening the version range or bypassing the fingerprint is not a fix.

## Dataset matrix

`not-exercised` describes the specified two-card DP1 execution, not dataset
support. `blocked` describes an ADM experiment whose prerequisites are missing;
it does **not** mean the underlying dataset has no working benchmark adapter.
No dataset currently has a qualified ADM ON/OFF task-quality result.

The latest benchmark [client directory](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/a181a5e94414bca9b51bb89e5ce2da6bdcef38ce/docs/dataset-entrypoints/primary-clients.json)
also pins reusable MMLU-Pro (`b6215bf...`), HLE (`534c22c...`) and
FrontierScience (`ff7b431...`) clients with setup and command examples. Its
MMLU local extraction policy is not the original leaderboard scorer; HLE and
FrontierScience use substitute judges. Reported client installation/help checks
do not establish that data preparation, model requests, or scoring were run.

| Dataset | V5.5 TP2/DP1 ADM path | ADM DP2/DP4 ablation | Reusable organization evidence / runner | Quality and concrete blocker / owner |
| --- | --- | --- | --- | --- |
| MMLU-Pro | **not-exercised** | **blocked** | Fixed 200-task designated B0 151/200 and corrected B1 156/200 are organization runs, not ADM. [Original-row runner](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/c17289b58c0e3d93033d8426d8d8b55ce17fc253/scripts/run_mmlu_pro_original_samples.py) pins `lm-eval==0.4.11` and source checksums, but its first-three-row smoke is not the 200-task/full-set scorer. | Accuracy on frozen task IDs and scorer, retain truncation/failures; full-set manifest/scorer and matched ADM TP/DP support/activation evidence required. **Owner:** ADM maintainer + benchmark evaluation owner. |
| HLE-Verified | **not-exercised** | **blocked** | [0.18 native Gold/Revision](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/62c29b0abb4984c6cf1e8071734c38d8ac84657f/reports/hle-native018-completed-20261026/README.md): 157/668 and316/1143, with unresolved23/31; `hust_eval_hle_verified` solver/judge/summary contract and private originals must be requalified for use. | Frozen verified subset, multimodal inputs, pinned judge/scorer and unresolved denominator; dataset rights/judge/actual 0.18-source compatibility and DP runtime are blockers. **Owner:** benchmark evaluation/judge owner + ADM maintainer. |
| SWE-bench-Pro | **not-exercised** | **blocked** | [#403](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/403) reports 30 historic SWE sessions and a proposed, not admitted, fixed-trace replay. Existing sandbox/agent artifacts are not ADM results. | Original fresh-sandbox task resolved rate is quality; replay TPS/TTFT/TPOT is supporting telemetry and cannot create a task score. **New SWE measurement/agents/controls are stopped pending proposal review and authorization.** Freeze trace reconstruction, source/image/scorer and four-card compatibility first. **Owner:** benchmark profiling/CPU owner, storage/model owner, ADM maintainer. |
| FrontierScience | **not-exercised** | **blocked** | [Frozen input reader](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/c17289b58c0e3d93033d8426d8d8b55ce17fc253/src/vllm_hust_benchmark/frontierscience_inputs.py) and [0.18 paired provenance](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/098184a4610dab052ac7fc2f15fc337eb436ad75/reports/frontierscience-native018-paired-20261025/provenance.json) cover100 Olympiad/60 Research. B0 solver is 0.18; the frozen judge is a development-stack substitute, not official leaderboard. | Track-specific accuracy/rubric with full denominators and unresolved grades, pinned judge and inputs; same-stack ADM pair and task-quality checks missing. **Owner:** benchmark evaluation/judge owner + ADM maintainer. |
| Terminal-Bench 2.1 | **not-exercised** | **blocked** | Fixed [Harbor contract](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/c17289b58c0e3d93033d8426d8d8b55ce17fc253/reports/pyramidkv-terminal-harbor-contract-20261009/README.md), revision `7131e4375048a0e408a8fb404b5f499d726b695b`, 89 tasks, Harbor0.24.0/Terminus-2; partial historical attempts/controls do not establish ADM scoring. | Original task resolution/pass@1 is quality. Per #403 use existing ARM64 adaptation path; image identities/controls, fixed replay inputs and plan approval are unresolved. Do not request a new AMD64 backend, run Harbor, or count control/replayed requests as task scores. **Owner:** benchmark profiling/CPU owner, storage/model owner, @Apei-520; ADM maintainer for runtime. |

For all five, obtain actual worker path evidence; a running vLLM endpoint,
enabled plugin, Manager plan, or task accuracy score cannot establish DP sync
execution. Long prompt length alone does not prove the step-local >8191 fallback.
Collect TTFT, TPOT, ITL, throughput/goodput, failures/timeouts and latency
distributions as secondary metrics, preserving the prescribed task-quality
metric, denominator, unresolved cases and scorer identities. Profiler
attribution of any removable critical-path cost is currently UNKNOWN.

## Reproducible pre-run handoff — no dispatch

The benchmark repository owns dataset preparation, runner, scoring and canonical
artifact publication; ADM supplies the plugin and worker diagnostic API. Do not
create five parallel ADM-specific harnesses. Before review/execution, freeze:

- Dataset revision, selected task IDs and order, licenses, source and manifest
  SHA-256, tokenizer/chat-template/weight identities, prompts or private trace
  hashes, generation parameters, seeds, limits, scaffold/tools/image and scorer.
- Exact actual core/Ascend source, image RepoDigests and fingerprints, CANN,
  torch/NPU versions, model BF16, device lease, TP/DP/PP, runtime graph mode,
  attention/KV/APC settings and equal OFF/ON command/environment. No inferred
  0.18 compatibility; plugin absent/disabled is OFF and only ADM is ON.
- Actual per-worker before/after diagnostic snapshots and process identities in
  a **separate** observer-perturbation qualification; count successful word and
  group fallback calls in the frozen window. Require every expected rank, matching
  launch and process identity, and a nonzero relevant denominator. No observation
  means UNKNOWN. Manager lifecycle evidence and runtime path counts are distinct.
- A predeclared task-quality metric, SLO, primary throughput/latency metric,
  baseline tuning, warmup, arrival policy, negative regime, failures/timeouts,
  fresh-server cross-order repeats and profiler cost attribution. Preserve all
  outputs/hashes privately; publish sanitized immutable manifests in benchmark
  first. Do not edit historical receipts or infer missing values as zero.

### Resource/budget envelope for later review

No hardware is allocated by this document. For TP2/DP2, a standard four-card
layout is a **minimum**, not a support or memory guarantee. One ABBA pair means
four fresh-service windows per workload, so an exclusive four-card reservation
for `T` hours/window consumes approximately `16 × T` card-hours per workload;
T, startup, scorer/judge resources and cleanup budgets are **UNKNOWN** until
measured/planned. Historical TP4/DP2 uses eight cards, not four. Evaluation
minimum workload counts, if the same selected contracts are approved, are
MMLU-Pro 200 × four arms = 800 task attempts, HLE Gold+Revision 1811 × four =
7244 solver attempts, and FrontierScience160 × four = 640 solver attempts;
judge calls, retries, cost and time are additional and remain UNKNOWN. These
are ablation planning counts, not authorization or completed task coverage.

For SWE/Terminal, #403's only current proposal is four historical session
fragments: per arm16 valid requests plus1 rejection diagnostic, at most34 HTTP
requests across B0/B1 and four service windows, output cap139264 tokens, proposed
global stop6h. Reconstructed message candidates are not original wire/token
bytes; admissible executions remain zero until review and frozen contracts.
That B0/B1 proposal is not automatically an ADM OFF/ON ablation. A hypothetical
four-card exclusive six-hour ADM reservation would be 24 card-hours, excluding
other evaluator resources; it is an accounting example, not an approved budget.
Owner: release/evaluation engineering for official B0 artifacts, ADM maintainer
for compatibility and diagnostic capture, benchmark profiling for CPU/agents
and scoring, storage for exclusive model/device windows and independent review.
