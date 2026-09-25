"""Owner-run, CPU/Gloo timing of the pinned Ascend DP metadata methods.

This measures one synchronization call with two local ranks. It cannot
establish a Qwen serving speedup or an NPU network collective speedup.
"""

import datetime
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace


SEQUENCE = ("native", "mod", "mod", "native", "native", "mod", "mod", "native")
ITERATIONS = 400
WARMUP = 30


def worker(rank: int, run_root: str) -> None:
    import torch
    import torch.distributed as dist
    from ascend_distributed_metadata import packed_sync
    from vllm_ascend.worker import model_runner_v1 as runner_module

    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{run_root}/gloo-init",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=120),
    )
    native = runner_module.NPUModelRunner._sync_metadata_across_dp
    assert packed_sync._fingerprint(native) == packed_sync.TARGET_FINGERPRINT
    runner_module.get_dp_group = lambda: SimpleNamespace(cpu_group=dist.group.WORLD)
    runner_module.should_skip_allreduce_across_dp_group = lambda *_: False
    mod = packed_sync._wrap(native, runner_module)
    runner = SimpleNamespace(dp_size=2, dp_rank=rank, vllm_config=object())
    tokens = (8, 12)[rank]
    mode = runner_module.CUDAGraphMode.NONE

    def invoke(method):
        maximum, vector, synced_mode = method(
            runner, tokens, cudagraph_mode=mode, allow_dp_padding=False
        )
        return maximum, vector.tolist(), synced_mode

    assert invoke(native) == invoke(mod) == (12, [8, 12], mode)
    for method in (native, mod):
        for _ in range(WARMUP):
            invoke(method)

    blocks = []
    for name in SEQUENCE:
        method = native if name == "native" else mod
        dist.barrier()
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            method(runner, tokens, cudagraph_mode=mode, allow_dp_padding=False)
        elapsed_ms = (time.perf_counter() - start) * 1000
        worst_ms = torch.tensor([elapsed_ms], dtype=torch.float64)
        dist.all_reduce(worst_ms, op=dist.ReduceOp.MAX)
        blocks.append({"method": name, "worst_rank_ms": worst_ms.item()})

    if rank == 0:
        native_ms = statistics.median(
            block["worst_rank_ms"] for block in blocks if block["method"] == "native"
        ) / ITERATIONS
        mod_ms = statistics.median(
            block["worst_rank_ms"] for block in blocks if block["method"] == "mod"
        ) / ITERATIONS
        result = {
            "scope": "local two-process CPU/Gloo DP metadata call only",
            "source_fingerprint": packed_sync.TARGET_FINGERPRINT,
            "ranks": 2,
            "iterations_per_block": ITERATIONS,
            "warmup_per_method": WARMUP,
            "sequence": list(SEQUENCE),
            "blocks": blocks,
            "median_native_ms_per_call": native_ms,
            "median_mod_ms_per_call": mod_ms,
            "mod_change_percent": (mod_ms / native_ms - 1) * 100,
            "correctness": "both methods returned (12, [8, 12], CUDAGraphMode.NONE)",
        }
        path = Path(run_root) / "result.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print("correctness=passed")
        print(f"median_native_ms_per_call={native_ms:.6f}")
        print(f"median_mod_ms_per_call={mod_ms:.6f}")
        print(f"mod_change_percent={result['mod_change_percent']:.3f}")
        print(f"result_path={path}")
    dist.destroy_process_group()


if __name__ == "__main__":
    import torch.multiprocessing as mp

    root = Path(sys.argv[1]).resolve()
    assert root.is_dir() and not (root / "result.json").exists()
    assert os.environ.get("ADM_PACKED_SYNC_ENABLE") == "0"
    mp.spawn(worker, args=(str(root),), nprocs=2, join=True)
