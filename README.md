# Ascend Distributed Metadata (ADM)

ADM is an opt-in `vllm.general_plugins` package for distributed-parallel
metadata synchronization in the pinned vLLM-HUST / Ascend-HUST stack. The
promoted implementation packs both DP2 ranks into one `int32` all-reduce;
larger DP groups use one packed `int32` per rank with all-gather. The output
retains the native token vector, maximum token count, and minimum graph mode.

## Measured result

On Qwen3.5-35B-A3B TP4/DP2, the promoted scalar candidate completed all
32 requests in each candidate–published MOD–candidate run. Output throughput
was **26.979 and 27.052 tok/s**, versus **26.906 tok/s** for the intervening
published MOD run: **+0.27% and +0.54%**, mean **+0.41%**. Both observations
meet the project's 0.05% positive-result threshold communicated on
2026-09-28. The difference is small relative to prior run variation, so
repeatable end-to-end acceleration is not established. Two separate local
CPU/Gloo sync-call comparisons measured 26.8% and 32.4% less time than the
previous MOD; those measurements do not include model execution or NPU
communication. See [qualification records](qualifications/README.md).

## Compatibility and activation

- Reviewed Ascend source: `367b8e62da799870a7476ce34f5f7658589a8aad`.
  The plugin checks the target method's AST fingerprint
  `3e3cac40908b68c65c45c820efca81073c0a11bef5eab13b2f8cf9ce36e407f5`
  and refuses a different implementation.
- Reviewed vLLM-HUST core source: `e1248fa2655fdb8d72f80a5d6c28fa9c75b660c0`.
- Entry point: `vllm.general_plugins/adm_packed_sync`. The default is off;
  `ADM_PACKED_SYNC_ENABLE=1` enables the hook in every worker process.
- DP1 and rank-local skip paths remain native. DP2 token counts above 8191
  or unsupported graph modes use a group-visible fallback to the native
  collective. The native padding decision is preserved.
- `.vllm-hust/optimization.json` is the dev-hub `adm` launch profile. Its
  model qualification status remains pending because a numerical positive
  observation is narrower than a stable deployment claim.

Use the profile with a qualified stack, or install this package in that stack
and select `adm_packed_sync` in `VLLM_PLUGINS`. Keep the disabled variant as
the control for any new performance comparison.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/ascend_distributed_metadata/` | Version-gated runtime plugin |
| `tests/` | Host correctness and fallback checks |
| `benchmarks/` | Reusable local CPU/Gloo measurement scripts |
| `qualifications/` | Promoted results, hashes, and scope limits |
| `docs/experiments.md` | Experimental branches and unpromoted outcomes |
| `.vllm-hust/` | Dev-hub optimization profile |

The generation-bound DP load-metadata replica and other exploratory changes
remain on their own branches. This `main` package does not implement worker
failure recovery. See [experimental work](docs/experiments.md).
