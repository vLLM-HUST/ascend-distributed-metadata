# Experimental branches and unpromoted evidence

The `main` branch carries the DP2 scalar implementation and its bounded
positive measurements. Experimental branches retain alternative source,
negative comparisons, and open qualification questions. Their result files
remain in Git history.

| Branch | Scope | Current evidence |
| --- | --- | --- |
| [`feat/epoch-bound-replica`](https://github.com/vLLM-HUST/ascend-distributed-metadata/tree/feat/epoch-bound-replica) | Per-rank frontier validation and local API replica recovery; requires a separate vLLM-HUST hook branch | Gated local-copy recovery passed, 28 host tests passed, matched Qwen3 MoE serving was flat (451.44 vs 451.08 output tok/s mean). Worker crash and topology recovery remain untested. |
| [`perf/native-padding-array-word`](https://github.com/vLLM-HUST/ascend-distributed-metadata/tree/perf/native-padding-array-word) | Cached DP2 word/group and native padding semantics | CPU/Gloo metadata calls improved 37.5%–43.6% against the old public main, but matched serving did not improve repeatably. |
| [`perf/graph-downgrade-unpad`](https://github.com/vLLM-HUST/ascend-distributed-metadata/tree/perf/graph-downgrade-unpad) | Graph-mode padding experiment | Padding fell, but an A–B–A serving check did not establish a MOD gain. A separate +43.0% throughput result came from capacity and graph-capture configuration with the same MOD in both arms, so it is not attributed to ADM code. |
| [`perf/frombuffer-token-vector`](https://github.com/vLLM-HUST/ascend-distributed-metadata/tree/perf/frombuffer-token-vector) | Array-backed token vector | Local call improvement; matched service result was inconclusive. |

The early [published-MOD versus native serving comparison](https://github.com/vLLM-HUST/ascend-distributed-metadata/blob/feat/epoch-bound-replica/qualifications/qwen35-35b-a3b-dp2-mod-comparison-20260925.json)
and the [configuration-only comparison](https://github.com/vLLM-HUST/ascend-distributed-metadata/blob/perf/graph-downgrade-unpad/qualifications/qwen35-dp2-graph-padding-capacity-20260926.json)
remain available in those branches. A future promotion should identify the
exact candidate source, baseline, workload, recheck, and tradeoffs.

This boundary follows the same principle visible in
[BidKV](https://github.com/vLLM-HUST/vllm-hust-bidkv): the package exposes
a bounded runtime interface and default-off activation, while experiments
and legacy adapters are identified separately. BidKV's production path uses
a typed host hook; ADM's current version-gated general plugin still wraps
the reviewed Ascend method. The
[LatchMoE repository](https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE)
likewise separates its runtime mechanism from policy experiments and records
exact host pins and claim limits. ADM keeps one small plugin module in
`main`; splitting it into protocol/adapter layers would add no value until
the recovery experiment is promoted.
