# Worker-window evidence diagnostic — preregistration

Purpose: validate the opt-in per-worker capture pipeline on the already
qualified four-card eager diagnostic, not measure performance or dataset score.
Use exact pinned 0.23 source/model, BF16 Qwen3.5-35B-A3B, TP2/DP2, seed17,
four synthetic prompts, 32 tokens, no MTP; same command as the previous probe
plus `--arm on --capture-window`. Each of three 4-request batches serves one
purpose: trigger begin receipts, exercise the target batch, trigger end receipts.
The begin-triggering batch and background work after each worker's begin call
are included in its count window; end-triggering calls are excluded. Worker
boundaries are not simultaneous. Compare no timing from this run.

Require four begin/end pairs matched by PID and process start ticks, same host
and unique launch directory, expected DP ranks 0/0/1/1, nondecreasing counters
and a nonzero completed-word-or-sentinel denominator per worker. A missing,
inconsistent or zero-denominator receipt is UNKNOWN, not a zero fallback rate.
Preserve counts separately for native skip and native-device fallback; do not
sum worker totals as unique collectives. Raw per-worker receipts stay private.
This run does not establish uninstrumented overhead, host release support, task
quality, graph/MTP correctness, or official organization B0/B1.
