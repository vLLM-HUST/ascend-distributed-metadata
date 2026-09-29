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

The optional load-metadata replica also needs the opt-in core hooks on
`vLLM-HUST/vllm-hust` `dbe6cc64d691f1a768d5409e0d4b5a8f55f5f4f7`,
which descends from the core revision above. The hooks are absent from the
base core revision. The combined package passed a BF16 TP2/DP2 service and
local-replica-loss smoke on this hook carrier; see the
[qualification record](qualifications/qwen35-bf16-combined-replica-20260929.json).

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

DP1 and rank-local skip paths use the native method. DP2/DP4 token counts above
8191 or unsupported graph modes use a group-visible fallback to the native
collective. The native padding decision is preserved.

## Experimental load-metadata replica

The separate `adm_epoch_replica` entry point validates coordinator snapshots
before the internal DP load balancer adopts them. It tracks each rank's
`current_wave`/`step_counter` frontier, rejects stale publications and requires
a complete observed snapshot after local replica invalidation. A recovery
receipt is created only after a routing read uses the restored snapshot. Enable
it with `ADM_EPOCH_REPLICA_ENABLE=1` and include `adm_epoch_replica` in
`VLLM_PLUGINS` on the reviewed core-hook carrier. The default is disabled.
The packed-sync switch can remain independently enabled or disabled.
The entry point is registered in both the general-plugin group (API and
worker processes) and the dedicated coordinator group; its process-local
hook factory is activated only when the switch is `1`.

This experimental generation identifies a coordinator publication session;
it is not a worker process/topology epoch or a worker-crash recovery protocol.
The existing [design and limitations](docs/epoch-replica.md) describe its
current boundary. The combined smoke does not establish an automatic loss
detector, worker-crash recovery, or a performance gain.

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
| `docs/epoch-replica.md` | Experimental replica protocol and limits |
| `docs/engineering-plan.md` | Upgrade gates and open capabilities |

The [experiment branches](docs/experiments.md) contain metadata-replica
recovery, alternative synchronization implementations, and their test records.
They are separate from this package's `main` runtime path.
