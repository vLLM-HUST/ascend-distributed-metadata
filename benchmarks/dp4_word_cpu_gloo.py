"""Compare pinned ADM DP4 all-gather with a four-slot int64 all-reduce."""

import datetime
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SEQUENCE = ("published", "candidate", "candidate", "published") * 2
ITERATIONS = 400
WARMUP = 30
COUNTS = (8, 12, 20, 4)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(rank, run_root, published_path, candidate_path, mode_name):
    import torch
    import torch.distributed as dist
    from vllm_ascend.worker import model_runner_v1 as runner_module

    published = load_module("adm_dp4_published", published_path)
    candidate = load_module("adm_dp4_candidate", candidate_path)
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{run_root}/gloo-init",
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=120),
    )
    native = runner_module.NPUModelRunner._sync_metadata_across_dp
    assert published._fingerprint(native) == candidate._fingerprint(native)
    assert published.TARGET_FINGERPRINT == candidate.TARGET_FINGERPRINT
    runner_module.get_dp_group = lambda: SimpleNamespace(cpu_group=dist.group.WORLD)
    runner_module.should_skip_allreduce_across_dp_group = lambda *_: False
    methods = {
        "published": published._wrap(native, runner_module),
        "candidate": candidate._wrap(native, runner_module),
    }
    runner = SimpleNamespace(dp_size=4, dp_rank=rank, vllm_config=object())
    mode = runner_module.CUDAGraphMode[mode_name]
    expected = (max(COUNTS), list(COUNTS), mode)

    def invoke(method):
        maximum, vector, synced_mode = method(
            runner, COUNTS[rank], cudagraph_mode=mode, allow_dp_padding=False
        )
        return maximum, vector.tolist(), synced_mode

    assert invoke(native) == expected
    for method in methods.values():
        assert invoke(method) == expected
        for _ in range(WARMUP):
            invoke(method)

    blocks = []
    for name in SEQUENCE:
        dist.barrier()
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            methods[name](runner, COUNTS[rank], cudagraph_mode=mode)
        elapsed_ms = (time.perf_counter() - start) * 1000
        worst_ms = torch.tensor([elapsed_ms], dtype=torch.float64)
        dist.all_reduce(worst_ms, op=dist.ReduceOp.MAX)
        blocks.append({"method": name, "worst_rank_ms": worst_ms.item()})

    if rank == 0:
        medians = {
            name: statistics.median(
                block["worst_rank_ms"] for block in blocks
                if block["method"] == name
            ) / ITERATIONS
            for name in methods
        }
        result = {
            "scope": "four local CPU/Gloo ranks, one DP4 metadata method call",
            "mode": mode_name,
            "rank_tokens": list(COUNTS),
            "published_sha256": hashlib.sha256(Path(published_path).read_bytes()).hexdigest(),
            "candidate_sha256": hashlib.sha256(Path(candidate_path).read_bytes()).hexdigest(),
            "source_fingerprint": published.TARGET_FINGERPRINT,
            "iterations_per_block": ITERATIONS,
            "sequence": list(SEQUENCE),
            "blocks": blocks,
            "median_ms_per_call": medians,
            "candidate_change_percent": (
                medians["candidate"] / medians["published"] - 1
            ) * 100,
            "correctness": "native, published and candidate returned matching four-rank counts and graph mode",
        }
        path = Path(run_root) / "result.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print("correctness=passed", flush=True)
        print(f"published_ms_per_call={medians['published']:.6f}", flush=True)
        print(f"candidate_ms_per_call={medians['candidate']:.6f}", flush=True)
        print(f"candidate_change_percent={result['candidate_change_percent']:.3f}", flush=True)
        print(f"result_path={path}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    import torch.multiprocessing as mp

    root = Path(sys.argv[1]).resolve()
    assert root.is_dir() and not (root / "result.json").exists()
    mp.spawn(
        worker,
        args=(str(root), sys.argv[2], sys.argv[3], sys.argv[4]),
        nprocs=4,
        join=True,
    )
