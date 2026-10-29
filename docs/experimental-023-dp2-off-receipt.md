# 0.23 TP2/DP2 Native OFF diagnostic

On the user-confirmed exclusive logical devices 0–3 (910B2), the preregistered
[synthetic eager probe](experimental-023-dp2-probe.md) reached service readiness
with TP2/DP2 and completed 4/4 requests, each producing 32 output tokens. The
model was Qwen3.5-35B-A3B BF16, seed17, no speculative decoding. The per-request
durations were approximately 12.61–13.10 seconds; this first cold diagnostic is
not a throughput, TTFT, ITL, dataset-quality, or official B0/B1 measurement.

Runtime: core `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`, Ascend
`5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`; model metadata revision
`59d61f3ce65a6d9863b86d2e96597125219dc754`, full weight hashes not checked.
Raw local receipt SHA-256 `842ec594936158aae37dd7c557a1271e5b40f98a83b7d643e1e336fbd4060c66`.
After all requests completed, the process-group shutdown generated engine-dead
messages while API workers exited; the server and device processes were gone
and all four HBM readings returned to approximately 3.4 GiB. No ON result or
speedup is established yet. Next: same-stack ON diagnostic.
