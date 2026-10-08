# Ascend Distributed Metadata

An opt-in vLLM Ascend plugin for synchronizing data-parallel scheduling
metadata. DP2 uses one `int32` and DP4 uses one `int64` all-reduce. Other DP
groups use one packed `int32` per rank with all-gather. The plugin returns the
same token vector, maximum token count, and minimum graph mode as the reviewed
native method.

## Requirements

- vLLM-HUST core `e1248fa2655fdb8d72f80a5d6c28fa9c75b660c0`
- vLLM-Ascend-HUST `367b8e62da799870a7476ce34f5f7658589a8aad`
- Python 3.11 or later in an environment with the matching Ascend runtime

The plugin checks the target Ascend method's AST fingerprint at startup and
refuses to wrap a different implementation. It does not install vLLM,
Torch-NPU, or CANN as package dependencies.

## Installation and activation

Install this repository in the qualified runtime environment. The package
registers `adm_packed_sync` in `vllm.general_plugins`; installing it does not
enable the optimization. Set `ADM_PACKED_SYNC_ENABLE=1` and include
`adm_packed_sync` in `VLLM_PLUGINS` for the worker processes. The default is
disabled. The [dev-hub profile](.vllm-hust/optimization.json) provides the
named `adm` launch configuration.

The wheel also publishes the experimental ECPA Bundle
`org.vllm-hust.ascend-distributed-metadata`. Extension Manager can discover,
inspect, check, plan, and render the same activation without importing this
module. Enabling the Bundle adds the plugin name and the opt-in environment
variable to a host-owned launch; it does not start an external service or
prove that the patch became effective. Compatibility remains restricted to
the reviewed vLLM/vLLM-Ascend commits and runtime qualification profile, and
the plugin still verifies the target method's AST fingerprint in every worker.

DP1 and rank-local skip paths use the native method. DP2/DP4 token counts above
8191 or unsupported graph modes use a group-visible fallback to the native
collective. The native padding decision is preserved.

## Results

The [Qwen3-30B-A3B-W8A8 DP4 comparison](qualifications/qwen3moe-dp4-int64-word-20260928.json)
used the same 64-request trace for six TP1/DP4 service sessions, with two
rounds per session. All 12 rounds completed 64/64 requests. The three matched
session means favored the DP4 word path by 0.38%, 0.71%, and 1.32%; overall
mean output throughput was 467.099 vs 463.376 tok/s (+0.80%). Two local
CPU/Gloo method-call runs measured 30.5% and 36.2% less time. The service
difference is small and individual rounds varied; the call-level result has
a narrower scope than full-model serving.

The [Qwen3.5-35B-A3B TP4/DP2 comparison](qualifications/README.md) completed
32/32 requests in each of three runs. The candidate measured 26.979 and
27.052 output tok/s; the intervening previous MOD run measured 26.906
output tok/s. Two local CPU/Gloo metadata-call comparisons measured 26.8%
and 32.4% lower median time than that previous MOD. These measurements have
different scopes; the small serving difference may reflect run variation.
The JSON records include the source revisions, result hashes, and workload.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/ascend_distributed_metadata/` | Runtime plugin |
| `tests/` | Host correctness and fallback tests |
| `benchmarks/` | CPU/Gloo measurement scripts |
| `qualifications/` | Qwen3.5 and metadata-call results |
| `.vllm-hust/` | Dev-hub launch profile |
| `docs/experiments.md` | Work in progress on experiment branches |

The [experiment branches](docs/experiments.md) contain metadata-replica
recovery, alternative synchronization implementations, and their test records.
They are separate from this package's `main` runtime path.

## Canonical MOD metadata

Repository identity, directly responsible maintainers, advisor status, default-off
activation, rollback, scope, and evidence qualification are recorded in
[`MOD_METADATA.json`](MOD_METADATA.json). `advisor_status: unknown` is not the
same as confirmed `none`. Performance statements remain limited to the workloads
and evidence labels recorded there; they are not general online claims.
