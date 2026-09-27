# Epoch-bound DP load-metadata replica: engineering design

Status: plugin state machine implemented and host-tested; runtime integration
requires two narrow hooks in the reviewed vLLM-HUST core. No native recovery
receipt or throughput claim exists yet.

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

The frontend validates the full snapshot before replacing the count data used
for routing. Lower publication sequence, rank frontier, or global wave is
rejected. A generation change requires explicit invalidation; prior
generations remain retired. After local replica invalidation, the next valid
complete snapshot can restore readiness. A recovery receipt is created only
when a subsequent routing read uses the restored state. The receipt binds the
recovery ID, generations, publication sequence, rank frontiers, and snapshot
hash. Raw events and receipt publication still require runtime integration.

This generation identifies a **coordinator publication session**; it is not
a worker process/topology epoch. `current_wave` identifies the existing DP
request wave, not an arbitrary method invocation. The design does not claim
worker crash recovery, elasticity, or the full research acceptance of #31.

## Runtime hooks needed

1. Load only the DP metadata plugin entry point in the coordinator process.
2. Before native coordinator count adoption, ask the publisher to validate
   each rank's `SchedulerStats` update. Rejected stale stats do not overwrite
   the authoritative counts.
3. Encode each coordinator publication through the publisher, preserving the
   native message when the hook is disabled.
4. In the API stats task, validate the encoded snapshot before assigning
   `lb_engines`; a rejected snapshot leaves the previous accepted state.
5. Immediately before the internal DP routing decision, mark the validated
   snapshot as consumed. While invalidated, routing fails closed until a
   complete snapshot arrives.

The hooks must be opt-in and preserve the original wire format when disabled.
An isolated core branch will carry only these hooks; the protocol and tests
remain in this ADM package.
