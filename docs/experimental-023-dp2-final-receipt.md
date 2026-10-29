# 0.23 four-card DP2 diagnostic: HOLD

The user confirmed exclusive logical devices 0–3. Actual model Qwen3.5-35B-A3B
BF16, Ascend 910B2 ×4, TP2/DP2, eager, no MTP, same four synthetic prompts,
32 output tokens, seed17. All four fresh service windows completed 4/4 requests,
and corresponding generated-text SHA-256 values were identical across arms.
All four ON workers reported the successful one-word path during initialization.
No measurement-window entry/fallback rate was captured.

| Order | Arm | Mean of four concurrent request elapsed times | Raw receipt SHA-256 |
| --- | --- | ---: | --- |
| 1 | OFF | 12.861 s | `842ec594936158aae37dd7c557a1271e5b40f98a83b7d643e1e336fbd4060c66` |
| 2 | ON | 9.436 s | `adf8020df316af6e5901d467f14d031d30c2bf09698be9ab95a274930c5dd63f` |
| 3 | ON | 9.307 s | `a46c6f75199f96f76141d19ca1940bb2da8eae1f8b2a0e870006f802b26f4180` |
| 4 | OFF | 9.237 s | `3ff2774e1683fd90a76f438f169c987e20d42d2b83dcea44f5498cb444608bee` |

The initial apparent reduction did not recur in reverse order: in the reverse
pair the ON mean was slightly higher than OFF. This small, cold, concurrent
four-request diagnostic cannot establish performance, TTFT, TPOT, ITL, goodput,
dataset task quality, graph or MTP correctness. Do not pool its four request
latencies as independent repeated service experiments. This is a same-stack
OFF/ON diagnostic and does not replace the organization B0/B1 evaluation.

One attempted ON invocation stopped before server startup because the prior
port was not immediately reusable; subsequent arms used separate local ports.
After requests, simultaneous process-group shutdown generated API engine-dead
logs. The Python resource tracker reported cleanup of eight shared-memory and
four semaphore objects on the final OFF arm. No vLLM/NPU processes remained
visible and each HBM reading returned to the pre-run approximately 3.4 GiB.
Retain these cleanup limitations; process-group logs and raw receipts remain
local under `/root/kmx/adm-dp2-probe-20261028/`.

Decision: **HOLD**. Exact-source experimental direct-plugin switch remains
default OFF, and the ECPA host range remains unchanged. This is not a supported
0.23 release or a measured speedup. Next required work is capture of actual
worker counters over a frozen window, observer overhead qualification, graph
and MTP correctness, dataset/input/scorer authorization, representative warmed
workloads, and tuned fresh-server repeats with failure and quality accounting.
