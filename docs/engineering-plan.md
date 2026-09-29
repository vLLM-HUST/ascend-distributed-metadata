# ADM plugin engineering plan

The published packed collective remains the stable serving feature. The
replica path is an independent, opt-in feature on the reviewed core-hook
carrier. Evidence for one path does not qualify the other.

## Stage 1: integrate and qualify

- Package both entry points so the API consumer and coordinator publisher
  load in their own processes. Keep both switches off by default.
- Verify package discovery, hook registration, rank-skew snapshot behavior,
  stale-data rejection, and receipt-after-routing behavior on the host.
- Run a pinned BF16 Qwen3.5 DP2 service smoke with the two features enabled,
  then a controlled local-replica-loss exercise. Record revisions, package
  identity, model, devices, logs, and the actual recovery receipt.

## Stage 2: make recovery operational

- Define a real invalidation trigger and a bounded replay request through
  the coordinator transport. Today invalidation is only an explicit API call
  or gated diagnostic fault; a subscriber gap is not yet detected.
- Bind each rank frontier to a worker lifetime or topology epoch. The current
  `current_wave` and `step_counter` fields do not establish that lifetime.
- The optional durable receipt sink passed a fresh BF16 service check with
  injected loss, matching on-disk content, and `0600` permissions. Acceptance
  and fallback counters plus a service metrics export remain open.

## Stage 3: measure value

- Compare identical BF16 traces with paired baseline/treatment sessions and
  a recheck, including skewed DP routing and throughput/latency distributions.
- Profile metadata publication, message size, routing adoption, and recovery
  time separately. Promote only effects that survive repeated matched runs.

The Stage 1 host gate can close before the NPU gate. Stages 2 and 3 require
new evidence before a claim about production recovery or performance.
