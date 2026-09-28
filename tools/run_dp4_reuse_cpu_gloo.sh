#!/usr/bin/env bash
# Owner-run call-level filter. Does not launch the model or use an NPU.

PYTHON=/workspace/kmx/envs/adm-runtimequal-d7f1c842/bin/python
PUBLISHED=/workspace/kmx/adm-mod-delivery-20260924
CANDIDATE=/workspace/kmx/worktrees/adm-dp4-reuse-bf16-20260928
CORE=/workspace/kmx/worktrees/adm-hostqual-d7f1c842/third_party/vllm-hust
ASCEND=/workspace/kmx/worktrees/adm-hostqual-d7f1c842/third_party/vllm-ascend-hust
RUN_ROOT=$(/usr/bin/mktemp -d /workspace/kmx/results/adm/qwen35-dp4-reuse-cpu-gloo-20260928-XXXXXX)

printf 'run_root=%s\npython=%s\npublished=%s\ncandidate=%s\n' \
    "$RUN_ROOT" "$PYTHON" "$PUBLISHED" "$CANDIDATE" \
    | /usr/bin/tee "$RUN_ROOT/admission.txt"
for ITEM in \
    "$PUBLISHED:16362b2d6c229ec1c900b87cdb2c974039db95a6" \
    "$CORE:e1248fa2655fdb8d72f80a5d6c28fa9c75b660c0" \
    "$ASCEND:367b8e62da799870a7476ce34f5f7658589a8aad"; do
    REPO=${ITEM%:*}
    EXPECTED=${ITEM##*:}
    ACTUAL=$(/usr/bin/git -C "$REPO" rev-parse HEAD)
    DIRTY=$(/usr/bin/git -C "$REPO" status --porcelain)
    printf 'repo=%s revision=%s dirty=%s\n' "$REPO" "$ACTUAL" "${DIRTY:-clean}" \
        | /usr/bin/tee -a "$RUN_ROOT/admission.txt"
    if [ "$ACTUAL" != "$EXPECTED" ] || [ -n "$DIRTY" ]; then
        printf 'admission_exit=2\nrun_root=%s\n' "$RUN_ROOT"
        exit 2
    fi
done
ACTUAL=$(/usr/bin/git -C "$CANDIDATE" rev-parse HEAD)
DIRTY=$(/usr/bin/git -C "$CANDIDATE" status --porcelain)
BRANCH=$(/usr/bin/git -C "$CANDIDATE" branch --show-current)
printf 'repo=%s revision=%s branch=%s dirty=%s\n' \
    "$CANDIDATE" "$ACTUAL" "$BRANCH" "${DIRTY:-clean}" \
    | /usr/bin/tee -a "$RUN_ROOT/admission.txt"
if [ "$BRANCH" != perf/dp4-reuse-word-bf16-20260928 ] || [ -n "$DIRTY" ]; then
    printf 'candidate_admission_exit=2\nrun_root=%s\n' "$RUN_ROOT"
    exit 2
fi
SOURCE=$CANDIDATE/src/ascend_distributed_metadata/packed_sync.py
ACTUAL_SHA=$(/usr/bin/sha256sum "$SOURCE")
printf 'candidate_source_sha256=%s\n' "$ACTUAL_SHA" | /usr/bin/tee -a "$RUN_ROOT/admission.txt"
if [ "${ACTUAL_SHA%% *}" != 1ae1e4387bbdbd78f5da4fc9637abfb301e508656ed6db48d15cacebbb5b7cbd ] \
    || [ ! -x "$PYTHON" ]; then
    printf 'source_admission_exit=2\nrun_root=%s\n' "$RUN_ROOT"
    exit 2
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh || {
    printf 'cann_source_exit=2\nrun_root=%s\n' "$RUN_ROOT"; exit 2;
}
source /usr/local/Ascend/nnal/atb/set_env.sh || {
    printf 'atb_source_exit=2\nrun_root=%s\n' "$RUN_ROOT"; exit 2;
}
export VLLM_VERSION=0.20.2
export VLLM_ASCEND_TORCH_PREFLIGHT=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_PLUGINS=ascend

"$PYTHON" - <<'PY' >"$RUN_ROOT/import-origins.log" 2>&1
import importlib.util
import sys
print("python_origin=", sys.executable)
for name in ("torch", "torch_npu", "vllm", "vllm_ascend"):
    spec = importlib.util.find_spec(name)
    print(f"{name}_origin=", None if spec is None else spec.origin)
    assert spec is not None
PY
IMPORT_EXIT=$?
/usr/bin/cat "$RUN_ROOT/import-origins.log"
if [ "$IMPORT_EXIT" -ne 0 ]; then
    printf 'import_exit=%s\nrun_root=%s\n' "$IMPORT_EXIT" "$RUN_ROOT"
    exit "$IMPORT_EXIT"
fi

"$PYTHON" "$CANDIDATE/benchmarks/dp4_word_cpu_gloo.py" \
    "$RUN_ROOT" \
    "$PUBLISHED/src/ascend_distributed_metadata/packed_sync.py" \
    "$SOURCE" FULL \
    >"$RUN_ROOT/bench.log" 2>&1
BENCH_EXIT=$?
/usr/bin/tail -n 35 "$RUN_ROOT/bench.log"
printf 'bench_exit=%s\nrun_root=%s\n' "$BENCH_EXIT" "$RUN_ROOT"
exit "$BENCH_EXIT"
