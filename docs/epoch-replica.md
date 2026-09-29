# Epoch-bound DP load-metadata replica: engineering design

Status: the prior `feat/epoch-bound-replica` branch implemented and host-tested
this state machine with isolated vLLM-HUST hooks. Its TP1/DP2 NPU serving and
gated local replica-loss recovery check passed; two matched serving rounds did
not establish a throughput gain. The combined package on this branch passed
a BF16 TP2/DP2 service and injected local-loss
[smoke](../qualifications/qwen35-bf16-combined-replica-20260929.json). This
does not qualify automatic loss detection, worker-process recovery, or a
serving-speed claim.

## Real object and path

| Stage | Reviewed vLLM-HUST `e1248fa` symbol | Role |
| --- | --- | --- |
| Producer | `DPEngineCoreProc._maybe_publish_request_counts` | Emits `SchedulerStats` containing waiting/running, `current_wave`, and `step_counter` when counts change. |
| Aggregator | `DPCoordinatorProc.process_input_socket` | Receives each rank's stats and publishes a full count list. |
| Replica | `DPAsyncMPClient._ensure_stats_update_task` | Receives coordinator publications and updates `lb_engines`. |
| Consumer | `DPLBAsyncMPClient.get_core_engine_for_request` | Uses `lb_engines` to route new requests. |

The existing packed token/graph-mode collective is separate. Its output is
not a persistent replica and has no source-backed epoch or recovery event.

## Protocol boundary

The coordinator assigns a publication generation at startup. For each rank it
keeps the producer's `(current_wave, step_counter)`, an incrementing accepted
version, and the last waiting/running counts. Comparisons are **per rank**:
rank 1 can lag rank 0 without being treated as stale. A duplicate identity
with different counts is an error; a lower identity cannot replace newer
counts. Every publication contains a complete rank set, global wave/running
state, and a strictly increasing publication sequence.
The `counts_update` bit preserves the native distinction between periodic
count publications and wave-only notifications. A wave-only notification
advances the validated replica but leaves the API's speculative waiting
counts intact. After explicit invalidation, even a wave-only publication
installs its complete snapshot so recovery uses authoritative counts.
The coordinator's initial zero counts are marked unobserved; they can seed
ordinary startup routing but cannot satisfy a recovery receipt. Recovery
requires a real stats update from every rank.

The frontend validates the full snapshot before replacing the count data used
for routing. Lower publication sequence, rank frontier, or global wave is
rejected. A generation change requires explicit invalidation; prior
generations remain retired. After local replica invalidation, the next valid
complete snapshot can restore readiness. A recovery receipt is created only
when a subsequent routing read uses the restored state. The receipt binds the
recovery ID, generations, publication sequence, rank frontiers, and snapshot
hash. The gated diagnostic logs the local invalidation and receipt; a durable
receipt sink is not yet implemented.

This generation identifies a **coordinator publication session**; it is not
a worker process/topology epoch. `current_wave` identifies the existing DP
request wave, not an arbitrary method invocation. The design does not claim
worker crash recovery, elasticity, or the full research acceptance of #31.

## Runtime hooks in the isolated core branch

1. Load only the DP metadata plugin entry point in the coordinator process.
2. Before native coordinator count adoption, ask the publisher to validate
   each rank's `SchedulerStats` update. Rejected stale stats do not overwrite
   the authoritative counts.
   Each rank must publish its initial `(0, 0)` counts on its first loop, even
   when unchanged, so an idle participant can contribute a real recovery
   observation rather than a coordinator default.
3. Encode each coordinator publication through the publisher, preserving the
   native message when the hook is disabled.
4. In the API stats task, validate the encoded snapshot before assigning
   `lb_engines`; a rejected snapshot leaves the previous accepted state.
5. Immediately before the internal DP routing decision, mark the validated
   snapshot as consumed. While invalidated, routing fails closed until a
   complete snapshot arrives.

The hooks must be opt-in and preserve the original wire format when disabled.
An isolated core branch carries only these hooks; the protocol and tests
remain in this ADM package. During local replica invalidation, the asynchronous
request entry waits up to 12 seconds while the stats task receives a new
snapshot. A missing rank is a transient rejected snapshot; a timeout is an
explicit request failure.

For a bounded recovery qualification only,
`ADM_REPLICA_INJECT_LOSS_AT_PUBLICATION_SEQ=N` invalidates the API process's
local replica after it has accepted an initial snapshot and received a full
two-rank publication with sequence at least `N`. It drops that one publication,
then waits for a later complete snapshot. The gated diagnostic prints the
injection sequence and the recovery receipt after a real routing read. This
simulates local-copy loss; it is not a worker crash or a transport failure.
