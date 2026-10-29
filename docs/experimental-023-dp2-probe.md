# Experimental 0.23 TP2/DP2 qualification probe

Decision: **HOLD pending actual diagnostics**. This does not change the ECPA
qualification or admit 0.23 generally. Exact Ascend sync and graph-helper AST
fingerprints are required, and `ADM_EXPERIMENTAL_023_ENABLE=1` is a second
explicit switch in addition to the default-off plugin. The host's
`dp_allreduce_on_npu` branch remains Native. Shared runtime and model are
read-only. The metadata-only model audit does not verify weight bytes.

The operator confirmed an exclusive lease on logical 0–3. Current devices are
910B2; the read-only source gate passed core `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`
and Ascend `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`. BF16 Qwen3.5,
TP2/DP2, eager, no MTP, four synthetic prompts, four concurrent requests,
32 output tokens, seed 17. Exact command is in `benchmarks/experimental_023_dp2_smoke.py`.
Each fresh server is bounded by 900 seconds to ready and 240 seconds per request.
Run OFF then ON only if OFF works, with identical flags apart from enablement;
compare completion counts, output hashes and server-side worker markers. Stop
on any error or mismatch. If both pass, this remains a diagnostic: no workload
quality, TTFT, ITL, activation rate, graph-mode or reliable speedup claim.
The one-shot DP2 marker proves a completed word path in that worker, not its
measurement-window rate. Native OFF/ON here is a same-stack ablation, not
the organization B0/B1 comparison. No SWE/Terminal work is authorized.
