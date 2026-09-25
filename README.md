# Ascend Distributed Metadata MOD

An independent, opt-in `vllm.general_plugins` package for the reviewed
`NPUModelRunner._sync_metadata_across_dp` implementation. For DP2, this
candidate packs both ranks into one `int32` and uses an all-reduce. Larger DP
groups use one `int32` per rank with all-gather. Both paths reconstruct the
native token vector, maximum token count, and minimum runtime graph mode.
The native collective uses a `2 x DP` `int32` tensor. This candidate is
experimental; its service-level speedup is unconfirmed.

## Scope and compatibility

- Reviewed Ascend source: `vllm-project/vllm-ascend` commit
  `367b8e62da799870a7476ce34f5f7658589a8aad`, target AST fingerprint
  `3e3cac40908b68c65c45c820efca81073c0a11bef5eab13b2f8cf9ce36e407f5`.
- Candidate vLLM-HUST core source: `vLLM-HUST/vllm-hust` commit
  `e1248fa2655fdb8d72f80a5d6c28fa9c75b660c0`, a packaging-only
  delta over the reviewed `vllm-project/vllm` commit
  `bc150f50299199599673614f80d12a196f377655`.
- The plugin refuses a different target function body. Host tests check
  return values, fallback, and collective shape under simulated rank inputs.
  Qwen3.5 TP4/DP2 serving results are recorded in `qualifications/`.
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

The rank-local skip and DP1 paths remain native. For active DP2 collectives,
all ranks call the same one-`int32` all-reduce, irrespective of local padding
flags. DP2 token counts above 8191 or unsupported graph modes use a
group-visible sentinel; all ranks then run the original collective. Larger
DP groups retain the packed all-gather path and its original range checks.

## Verification status

The [Qwen3.5-35B-A3B qualification results](qualifications/README.md) include
a TP4/DP1 model smoke test and four TP4/DP2 baseline/MOD serving runs, all
with 32/32 completed requests on Ascend 910B2. The matched serving runs did
not establish an end-to-end speedup. Two independent local CPU/Gloo DP2
sync-call measurements found the MOD 25.4% and 29.5% faster than the native
method. This is a narrower result than NPU communication or model throughput.

The [scalar DP2 candidate](qualifications/qwen35-dp2-scalar-candidate-20260925.json)
was 26.8% and 32.4% faster than the published MOD in two separate local
CPU/Gloo sync-call measurements. A single matched Qwen3.5 TP4/DP2 serving
comparison completed 32/32 requests in each mode: candidate 26.979 and
published MOD 26.906 output tokens/s. That 0.27% difference does not establish
a service-level speedup; a candidate serving recheck remains pending.
