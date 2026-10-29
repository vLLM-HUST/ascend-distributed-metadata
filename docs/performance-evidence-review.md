# ADM performance evidence and path accounting

Scope: [issue #7](https://github.com/vLLM-HUST/ascend-distributed-metadata/issues/7),
reviewed against ADM main `3bc9e876ee862b20accbe98149c0ad6ff458e16d` and the
[archived qualification](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-ascend-distributed-metadata/snapshot/qualifications/qwen3moe-dp4-int64-word-20260928.json).
No experiment was run for this review. A source-level mechanism is not an exposed
critical-path measurement; its removable end-to-end fraction is **UNKNOWN**.

## Version chain and measurements

| Evidence | Actual comparison and scope | Result and limitation |
| --- | --- | --- |
| Original DP2 packed MOD | Native → `6d7f0c6`, CPU/Gloo, two ranks, token counts 8/12, mode NONE | Method calls 25.4%/29.5% faster in two runs; no NPU/model inference. [record](../qualifications/dp2-packed-sync-cpu-gloo-20260925.json) |
| Original DP2 serving | Native OFF ↔ original packed MOD ON `6d7f0c6`, Qwen3.5, TP4/DP2, eager, 32 random 1024/256 requests, seed0, max concurrency8, A/B/B/A | 128/128 completed, Native mean 26.9608 vs MOD 26.8444 tok/s, -0.432%; variation overlaps. This is **not** the later scalar implementation. [historical branch](https://github.com/vLLM-HUST/ascend-distributed-metadata/blob/feat/epoch-bound-replica/qualifications/qwen35-35b-a3b-dp2-mod-comparison-20260925.json) |
| DP2 scalar | Previous packed MOD → scalar `26b8ac3`; same Qwen3.5 TP4/DP2 eager scenario, candidate/previous/candidate | 32/32 each; 26.979/26.906/27.052 tok/s, candidate mean +0.409% vs previous MOD. CPU/Gloo calls -26.8%/-32.4%; serving benefit unconfirmed. [record](../qualifications/qwen35-dp2-scalar-candidate-20260925.json) |
| DP4 int64 word | Previous MOD `5d1a2e47304622ba1a3109ffff38cec1b11ed185` → word candidate `e346ea5f489f5151f539f69db100929ce9d9250b`, Qwen3-30B-A3B **W8A8**, TP1/DP4, FULL_DECODE_ONLY | Six fresh sessions, two rounds/session, all 768/768 completed; 463.376 → 467.099 output tok/s, +0.803% vs previous MOD. Three paired session-mean deltas +0.384%, +0.709%, +1.316%; first candidate round lost in two pairs. Sample SD of three session means: previous 1.281, candidate 3.258 tok/s. No general benefit or relative-to-Native gain established. [record](../qualifications/qwen3moe-dp4-int64-word-20260928.json) |
| DP4 CPU/Gloo | Previous MOD → word candidate, four local CPU ranks, direct sync call | -36.2% in NONE and -30.5% in FULL per-call duration. No model or NPU execution; not an end-to-end serving result. |

The DP4 published/candidate source SHA-256 values `4bb42bbd2eaceb6dfab95f4d5a9bb9f4195e6d0bb3cac34cead2e06b430eddfd` /
`daa17b4d246653e6f98a99c3141fd84df556c60af4ca38fea5588085be33a7c0` match
the Git contents of the two recorded revisions. The archived qualification blob
`d7181da09bf32d6bbafbf24b16af2e1491420139` matches this repository's qualification
file. The qualification lists the 12 raw-result SHA-256 values and three sets of
observed DP ranks, but the original per-round payloads and worker logs were not
located in the accessible checkout/archive search. Their bytes and event timestamps
therefore were **not** independently rehashed or inspected; listed hashes are not
a substitute for recovered raw evidence.

## What the runtime actually does

The patched method is guarded by Ascend method AST fingerprint
`3e3cac40908b68c65c45c820efca81073c0a11bef5eab13b2f8cf9ce36e407f5`.
Previously, Python 3.12 added empty `type_params` AST fields and the identical
reviewed method hashed differently from Python 3.11, despite the declared
Python >=3.11 requirement. The new normalizer removes **only empty** such
fields; nonempty generic parameters still fail closed. The exact pinned method
now hashes to the recorded fingerprint on both Python 3.11 and 3.12. This
does not establish any new host revision compatibility. Python 3.13+ has not
been tested here and other AST changes remain fail-closed.
On the reviewed host (`vllm-ascend-hust@367b8e62`), the model runner normally
calls it only when DP > 1. If directly called at DP1 or the host's rank-local
skip predicate applies, it delegates to Native. The skip predicate includes
model/communication and KV-consumer cases; its actual value is workload/runtime
dependent. DP2 uses an int32 and DP4 an int64 one-word CPU-group all-reduce.
All ranks see a sentinel and re-enter Native if any rank's scheduled token count
exceeds 8191 or its **runtime** graph mode is not concrete NONE/PIECEWISE/FULL.
8191 is a per-rank synchronization-step bound, not a maximum prompt length.
For other DP sizes, a packed int32 per rank is all-gathered; that path is not
qualified by the DP2/DP4 serving claims.

The old `ADM_DP4_REPORT_ACTIVE=1` event was emitted **before** the collective,
sentinel check and reconstruction. The qualification reports ranks 0–3 as seen
in each candidate session, but this establishes at most branch entry at an
unverified time; it cannot establish successful word-path use, measurement-window
coverage, or the rate of native fallback. Both fast-path and fallback proportions
for those historical windows are **UNKNOWN**, not zero. Client request completion
also cannot provide a per-step denominator.

Current code moves this optional one-shot marker after successful reconstruction.
This changes future evidence semantics; old logs retain their original meaning.
For new *diagnostic* runs only, set `ADM_PACKED_SYNC_COUNT_PATHS=1` before worker
spawn. An owning-process observer can call
`snapshot_path_counts(NPUModelRunner)` before and after a frozen measurement
window. Disabled/uninstalled counting returns `None`, not a zero snapshot.
Counters are process- and rank-local; fork resets them. Each outcome records a
completed method call: `word_dp2`, `word_dp4`, `packed_gather`, `native_dp1`,
`native_skip`, or `native_sentinel_fallback` (Native re-entry after a sentinel).
An exception before return is not counted. The short opt-in lock/counter overhead
is **NOT_MEASURED** and requires a separate observer-on/off perturbation check;
it is disabled by default and cannot substantiate a speedup.

An observer must capture snapshots inside the same worker, bind the host, PID,
`/proc` start ticks, launch identity, ranks, worker coverage and actual window
timestamps, then retain/hash the original records. Offline:

```sh
python -m ascend_distributed_metadata.evidence before.json after.json
```

This rejects unknown/different PID-start identities, missing/decreasing/duplicate
counts and invalid rank/outcome combinations. Word success and sentinel fallback
fractions use **completed DP2/DP4 word-or-sentinel calls** on each rank; native
skip and failed calls are not in that denominator. Zero denominator produces
`null`. Counters are not request rates and rank totals cannot be summed as unique
collectives. No snapshot → UNKNOWN. Deployment integration and a perturbation
measurement are still required before a performance-qualified NPU window.
Neither a marker nor a counter snapshot is an ECPA attestation or automatically
proves Manager `runtime_effective`.

An experimental diagnostic adapter is now available: set
`ADM_PACKED_SYNC_COUNT_PATHS=1` and `ADM_PACKED_SYNC_CAPTURE_DIR` to a fresh,
private, existing directory unique to the launch. Create `begin`, trigger a
sync call on every worker and verify all expected `begin-*.json` receipts;
after the workload create `end` and trigger another sync call on every worker.
Pair `end-*.json` by PID/start ticks and use
`python -m ascend_distributed_metadata.evidence --capture-envelopes BEGIN END`.
Independently require complete TP/DP coverage. Workers observe markers at
different call boundaries: begin-triggering calls are included, end-triggering
calls are excluded, and background calls inside the boundaries are included.
Missing or stale markers and workers are UNKNOWN. File polling and writes are
observer overhead; use these diagnostics only for path evidence, not performance.
