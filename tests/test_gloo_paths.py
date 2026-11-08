"""Real CPU collectives for word path and group-wide fallback; no vLLM/NPU."""

from datetime import timedelta
from enum import Enum
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ascend_distributed_metadata import packed_sync


class Mode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


def native(runner, num_tokens, is_draft_model=False, cudagraph_mode=Mode.NONE,
           allow_dp_padding=False):
    runner.native_calls += 1
    metadata = torch.zeros((2, runner.dp_size), dtype=torch.int32)
    metadata[0, runner.dp_rank] = num_tokens
    metadata[1, runner.dp_rank] = cudagraph_mode.value
    dist.all_reduce(metadata)
    tokens = metadata[0].tolist()
    maximum = max(tokens)
    vector = [maximum] * runner.dp_size if allow_dp_padding or is_draft_model else tokens
    return maximum, torch.tensor(vector, dtype=torch.int32), Mode(min(metadata[1].tolist()))


def worker(rank, world_size, root):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/gloo-init", rank=rank,
        world_size=world_size, timeout=timedelta(seconds=30),
    )
    try:
        module = SimpleNamespace(
            torch=torch, dist=dist,
            get_dp_group=lambda: SimpleNamespace(cpu_group=dist.group.WORLD),
            should_skip_allreduce_across_dp_group=lambda *_: False,
            CUDAGraphMode=Mode,
        )
        runner = SimpleNamespace(dp_size=world_size, dp_rank=rank,
                                 vllm_config=object(), native_calls=0)
        wrapped = packed_sync._wrap(native, module)
        first = wrapped(runner, 11 + rank, cudagraph_mode=Mode.FULL)
        assert (first[0], first[1].tolist(), first[2]) == (
            10 + world_size, list(range(11, 11 + world_size)), Mode.FULL,
        )
        # Only one rank is out of range. Every rank must enter the same
        # native fallback collective, in the same order.
        count = 8192 if rank == world_size - 1 else 20 + rank
        second = wrapped(runner, count, cudagraph_mode=Mode.PIECEWISE,
                         allow_dp_padding=True)
        assert (second[0], second[1].tolist(), second[2]) == (
            8192, [8192] * world_size, Mode.PIECEWISE,
        )
        assert runner.native_calls == 1
        snapshot = wrapped._adm_path_counts_snapshot()
        Path(root, f"rank-{rank}.json").write_text(json.dumps(snapshot))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_real_gloo_group_wide_fallback(tmp_path, monkeypatch, world_size):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo not available")
    monkeypatch.setenv(packed_sync.COUNT_ENV, "1")
    # CPU-only child processes must not auto-load an installed NPU backend.
    monkeypatch.setenv("PYTORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    mp.spawn(worker, args=(world_size, str(tmp_path)), nprocs=world_size, join=True)
    for rank in range(world_size):
        snapshot = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert {x["outcome"]: x["calls"] for x in snapshot["counts"]} == {
            f"word_dp{world_size}": 1, "native_sentinel_fallback": 1,
        }
        assert all(x["dp_rank"] == rank for x in snapshot["counts"])
