# Ascend Distributed Metadata MOD

An independent, opt-in `vllm.general_plugins` package for the reviewed
`NPUModelRunner._sync_metadata_across_dp` implementation. Its packed path
gathers one `int32` per DP rank and reconstructs the native token vector,
maximum token count, and minimum runtime graph mode. The native collective
uses a `2 x DP` `int32` tensor; the packed collective uses a `DP` `int32`
tensor. This is a payload-size reduction, not a measured latency gain.

## Scope and compatibility

- Reviewed Ascend source: `vllm-project/vllm-ascend` commit
  `367b8e62da799870a7476ce34f5f7658589a8aad`, target AST fingerprint
  `3e3cac40908b68c65c45c820efca81073c0a11bef5eab13b2f8cf9ce36e407f5`.
- Candidate vLLM-HUST core source: `vLLM-HUST/vllm-hust` commit
  `e1248fa2655fdb8d72f80a5d6c28fa9c75b660c0`, a packaging-only
  delta over the reviewed `vllm-project/vllm` commit
  `bc150f50299199599673614f80d12a196f377655`.
- The plugin refuses a different target function body. Host tests check
  return values and collective shape under simulated rank inputs; native
  NPU/DP2 qualification remains to be run.
- This MOD optimizes per-call token and graph-mode synchronization. It does
  not implement generation/epoch-bound metadata recovery.

## Install and enable

Install this directory into the reviewed vLLM/Ascend environment. Select only
this general plugin and set `ADM_PACKED_SYNC_ENABLE=1` in **every** worker
process before starting vLLM. The default is disabled.

The repository also provides `.vllm-hust/optimization.json` for the
vLLM-HUST dev-hub's named `adm` optimization profile. The profile selects
Ascend's installed platform/general plugins plus this plugin and configures
TP4/DP2. Its Qwen3.5-35B-A3B qualification is pending; the manifest does not
certify a performance gain.

```bash
VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model,ascend_model_loader,ascend_service_profiling,adm_packed_sync \
ADM_PACKED_SYNC_ENABLE=1 vllm serve /path/to/model
```

Run the identical baseline with `ADM_PACKED_SYNC_ENABLE=0`. The concrete
serving command, model path, NPU allocation, source revisions and result
directory must be frozen for each comparison.

The rank-local skip and DP1 paths remain native. For active DP collectives,
all ranks always call the same packed all-gather, irrespective of local
padding flags. An out-of-range token count or graph mode is encoded as a
group-visible sentinel; all ranks then run the original collective.

## Verification status

The [Qwen3.5-35B-A3B qualification results](qualifications/README.md) include
a TP4/DP1 model smoke test and a TP4/DP2 baseline, each with 32/32 completed
requests on Ascend 910B2. DP1 uses the native metadata path. A positive
performance claim requires a matched MOD treatment and independent recheck
with completed requests and no rank errors.
