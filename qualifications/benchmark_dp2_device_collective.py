"""Isolated DP2 scalar round-trip timing for Gloo CPU and HCCL NPU."""

import datetime
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch_npu  # noqa: F401: register the NPU backend


ORDER = ("cpu", "npu", "npu", "cpu") * 2
ITERATIONS = 400


def worker(rank: int, run_root: str):
    import torch.distributed as dist

    torch.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl",
        init_method=f"file://{run_root}/hccl-init",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=120),
    )
    cpu_group = dist.new_group(ranks=[0, 1], backend="gloo")
    torch.set_num_threads(1)
    slot = ((8, 12)[rank] << 2) << (15 * rank)
    expected = (8 << 2) + ((12 << 2) << 15)
    cpu_word = torch.empty(1, dtype=torch.int32, device="cpu")
    npu_word = torch.empty(1, dtype=torch.int32, device=f"npu:{rank}")

    def run_cpu():
        cpu_word.fill_(slot)
        dist.all_reduce(cpu_word, group=cpu_group)
        return int(cpu_word.item())

    def run_npu():
        npu_word.fill_(slot)
        dist.all_reduce(npu_word, group=dist.group.WORLD)
        return int(npu_word.item())

    methods = {"cpu": run_cpu, "npu": run_npu}
    for name, method in methods.items():
        assert method() == expected, f"{name}: wrong packed word"
        for _ in range(30):
            method()
    blocks = []
    for name in ORDER:
        dist.barrier(group=cpu_group)
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            methods[name]()
        elapsed = torch.tensor([(time.perf_counter() - start) * 1000], dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=cpu_group)
        blocks.append({"name": name, "worst_rank_ms": elapsed.item()})

    if rank == 0:
        medians = {name: statistics.median(x["worst_rank_ms"] for x in blocks
                                            if x["name"] == name) / ITERATIONS
                   for name in methods}
        record = {"scope": "two local NPU workers; one int32 fill, collective, and host read",
                  "devices": [0, 1], "iterations_per_block": ITERATIONS,
                  "order": ORDER, "blocks": blocks,
                  "median_ms_per_round_trip": medians,
                  "npu_change_percent": (medians["npu"] / medians["cpu"] - 1) * 100,
                  "correctness": "passed"}
        path = Path(run_root) / "result.json"
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"cpu_ms={medians['cpu']:.6f}")
        print(f"npu_ms={medians['npu']:.6f}")
        print(f"npu_change_percent={record['npu_change_percent']:.3f}")
        print(f"result_path={path}")
    dist.destroy_process_group()


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    assert root.is_dir() and not (root / "result.json").exists()
    mp.spawn(worker, args=(str(root),), nprocs=2, join=True)
