# 0.23 TP2/DP2 first ON diagnostic and reverse-order plan

The first OFF/ON pair used the [frozen probe](experimental-023-dp2-probe.md).
Both arms completed 4/4 requests, each with 32 output tokens; all four generated
text SHA-256 values matched between arms. All four ON workers reported a
completed int32 word path: DP0/TP0, DP0/TP1, DP1/TP0 and DP1/TP1. These markers
occurred during initialization and do not measure path entry or fallback rate
during the request window. ON receipt SHA-256:
`adf8020df316af6e5901d467f14d031d30c2bf09698be9ab95a274930c5dd63f`.

The four request mean elapsed times were 12.861 s OFF and 9.436 s ON. These
are concurrent cold diagnostic requests, without warmup, token timings,
representative workload or repeated order controls; they do not establish a
speedup. The ON server used port 18762 after the original port 18761 was not
immediately reusable following cleanup; model/configuration was unchanged.
All four devices returned to the pre-run HBM level, and no process was visible.

Next bounded diagnostic, preregistered here before execution: fresh ON then
fresh OFF, same prompts/configuration, new local ports, same 900/240 second
limits. Preserve all results, including mismatches and failures. This reverse
order can expose order effects but still cannot establish an end-to-end gain.
The earlier post-request process-group shutdown logged API `EngineDeadError`;
it is a cleanup log, not an inference failure. HOLD for dataset and performance
claims and for ECPA host compatibility.
