# Experiment branches

Development beyond the `main` plugin continues on separate branches:

| Branch | Work in progress |
| --- | --- |
| [`feat/epoch-bound-replica`](https://github.com/vLLM-HUST/ascend-distributed-metadata/tree/feat/epoch-bound-replica) | Per-rank metadata snapshots and local API replica recovery. Runtime use also needs the [vLLM-HUST hook branch](https://github.com/vLLM-HUST/vllm-hust/tree/feature/ascend-distributed-metadata). |
| [`perf/native-padding-array-word`](https://github.com/vLLM-HUST/ascend-distributed-metadata/tree/perf/native-padding-array-word) | Cached DP2 metadata word and native padding behavior. |
| [`perf/graph-downgrade-unpad`](https://github.com/vLLM-HUST/ascend-distributed-metadata/tree/perf/graph-downgrade-unpad) | Graph-mode padding and capacity experiments. |
| [`perf/frombuffer-token-vector`](https://github.com/vLLM-HUST/ascend-distributed-metadata/tree/perf/frombuffer-token-vector) | Array-backed token-vector construction. |

Each branch keeps its own code, qualification records, and limitations. The
[earlier serving comparison](https://github.com/vLLM-HUST/ascend-distributed-metadata/blob/feat/epoch-bound-replica/qualifications/qwen35-35b-a3b-dp2-mod-comparison-20260925.json)
and [configuration comparison](https://github.com/vLLM-HUST/ascend-distributed-metadata/blob/perf/graph-downgrade-unpad/qualifications/qwen35-dp2-graph-padding-capacity-20260926.json)
remain available there.
