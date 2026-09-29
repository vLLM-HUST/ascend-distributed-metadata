# Experiment branches

Development beyond the `main` plugin continues on separate branches:

| Branch | Work in progress |
| --- | --- |
| [`feat/epoch-bound-replica`](https://github.com/vLLM-HUST/ascend-distributed-metadata/tree/feat/epoch-bound-replica) | Per-rank metadata snapshots and local API replica recovery. Runtime use also needs the [vLLM-HUST hook branch](https://github.com/vLLM-HUST/vllm-hust/tree/feature/ascend-distributed-metadata). |
| `feat/adm-replica-integration-20260929` | Integrates that opt-in replica with the published DP2/DP4 word plugin; BF16 TP2/DP2 combined smoke passed. |
| [`perf/native-padding-array-word`](https://github.com/vLLM-HUST/ascend-distributed-metadata/tree/perf/native-padding-array-word) | Cached DP2 metadata word and native padding behavior. |
| [`perf/graph-downgrade-unpad`](https://github.com/vLLM-HUST/ascend-distributed-metadata/tree/perf/graph-downgrade-unpad) | Graph-mode padding and capacity experiments. |
| [`perf/frombuffer-token-vector`](https://github.com/vLLM-HUST/ascend-distributed-metadata/tree/perf/frombuffer-token-vector) | Array-backed token-vector construction. |
| [`perf/routing-output-budget-bf16-20260928`](https://github.com/vLLM-HUST/ascend-distributed-metadata/tree/perf/routing-output-budget-bf16-20260928) | BF16 DP2 output-budget-weighted routing. Two matched pairs improved output throughput about 2% on a constructed alternating-skew trace; natural-order benefit was not established. [Qualification record](https://github.com/vLLM-HUST/ascend-distributed-metadata/blob/perf/routing-output-budget-bf16-20260928/qualifications/qwen35-bf16-routing-output-skew-20260928.json). |
| [`perf/dp4-reuse-word-bf16-20260928`](https://github.com/vLLM-HUST/ascend-distributed-metadata/tree/perf/dp4-reuse-word-bf16-20260928) | DP4 buffer-reuse exploration. CPU/Gloo call time improved, but the W8A8 serving screen regressed; BF16 DP4 serving was not run. [Exploration record](https://github.com/vLLM-HUST/ascend-distributed-metadata/blob/perf/dp4-reuse-word-bf16-20260928/qualifications/dp4-reuse-word-exploration-20260928.json). |

Each branch keeps its own code, qualification records, and limitations. The
[earlier serving comparison](https://github.com/vLLM-HUST/ascend-distributed-metadata/blob/feat/epoch-bound-replica/qualifications/qwen35-35b-a3b-dp2-mod-comparison-20260925.json)
and [configuration comparison](https://github.com/vLLM-HUST/ascend-distributed-metadata/blob/perf/graph-downgrade-unpad/qualifications/qwen35-dp2-graph-padding-capacity-20260926.json)
remain available there.
