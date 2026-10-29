"""Differential checks against the exact 0.23 source, entirely on CPU."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ascend_distributed_metadata import packed_sync


def fixture():
    path = Path(__file__).parent / "fixtures/reviewed_sync_023.py"
    spec = importlib.util.spec_from_file_location("reviewed_023", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Collective:
    def __init__(self, counts, modes):
        self.counts, self.modes, self.calls = counts, modes, []

    def all_reduce(self, tensor, *, group):
        self.calls.append((tensor.numel(), group))
        if tensor.ndim == 2:
            tensor.copy_(torch.tensor([self.counts, self.modes], dtype=torch.int32))
        else:
            word = sum(((n << 2) | mode) << (15 * rank)
                       for rank, (n, mode) in enumerate(zip(self.counts, self.modes)))
            tensor.fill_(word)


@pytest.mark.parametrize("padding,draft", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("rank", [0, 1])
def test_exact_native_matches_candidate(padding, draft, rank):
    source = fixture()
    dist = Collective([15, 47], [2, 1])
    module = SimpleNamespace(torch=torch, dist=dist, CUDAGraphMode=source.CUDAGraphMode,
        get_dp_group=lambda: SimpleNamespace(cpu_group="cpu", device_group="device"),
        should_skip_allreduce_across_dp_group=lambda *_: False,
        _post_process_cudagraph_mode=source._post_process_cudagraph_mode)
    source.__dict__.update(vars(module))
    runner = SimpleNamespace(dp_size=2, dp_rank=rank, vllm_config=object(),
                             ascend_config=SimpleNamespace(dp_allreduce_on_npu=False))
    args = (runner, [15, 47][rank], draft, source.CUDAGraphMode.FULL, padding)
    native = source._sync_metadata_across_dp(*args)
    candidate = packed_sync._wrap(source._sync_metadata_across_dp, module, device_guard=True)(*args)
    assert native[0] == candidate[0]
    assert torch.equal(native[1], candidate[1])
    assert native[2] == candidate[2]
    assert dist.calls == [(4, "cpu"), (1, "cpu")]


def test_npu_collective_setting_delegates_without_cpu_collective(monkeypatch):
    source = fixture()
    dist = Collective([1, 2], [0, 0])
    module = SimpleNamespace(torch=torch, dist=dist, CUDAGraphMode=source.CUDAGraphMode,
        get_dp_group=lambda: SimpleNamespace(cpu_group="cpu"),
        should_skip_allreduce_across_dp_group=lambda *_: False)
    calls = []
    def native(self, num_tokens, is_draft_model=False,
               cudagraph_mode=source.CUDAGraphMode.NONE, allow_dp_padding=False):
        calls.append(num_tokens)
        return num_tokens, None, cudagraph_mode
    runner = SimpleNamespace(dp_size=2, dp_rank=0, vllm_config=object(),
                             ascend_config=SimpleNamespace(dp_allreduce_on_npu=True))
    monkeypatch.setenv(packed_sync.COUNT_ENV, "1")
    wrapped = packed_sync._wrap(native, module, device_guard=True)
    result = wrapped(runner, 7)
    assert calls == [7] and dist.calls == [] and result[0] == 7
    assert wrapped._adm_path_counts_snapshot()["counts"][0]["outcome"] == "native_device_fallback"


def test_023_is_explicitly_opt_in_and_checks_both_source_functions(monkeypatch):
    source = fixture()
    assert packed_sync._fingerprint(source._sync_metadata_across_dp) == packed_sync.EXPERIMENTAL_023_FINGERPRINT
    assert packed_sync._fingerprint(source._post_process_cudagraph_mode) == packed_sync.EXPERIMENTAL_023_HELPER_FINGERPRINT
    runner_type = type("Runner", (), {"_sync_metadata_across_dp": source._sync_metadata_across_dp})
    with pytest.raises(packed_sync.PackedSyncViolation):
        packed_sync.install(runner_type, source)
    monkeypatch.setenv(packed_sync.EXPERIMENTAL_023_ENV, "1")
    source._post_process_cudagraph_mode = lambda _: 0
    with pytest.raises(packed_sync.PackedSyncViolation):
        packed_sync.install(runner_type, source)


def test_023_positive_install_and_completed_dp2_marker(monkeypatch, capsys):
    source = fixture()
    dist = Collective([3, 4], [0, 0])
    source.torch = torch
    source.dist = dist
    source.get_dp_group = lambda: SimpleNamespace(cpu_group="cpu")
    source.should_skip_allreduce_across_dp_group = lambda *_: False
    runner_type = type("Runner", (), {"_sync_metadata_across_dp": source._sync_metadata_across_dp})
    monkeypatch.setenv(packed_sync.EXPERIMENTAL_023_ENV, "1")
    monkeypatch.setenv("ADM_DP2_REPORT_ACTIVE", "1")
    assert packed_sync.install(runner_type, source)
    runner = runner_type()
    runner.dp_size, runner.dp_rank = 2, 0
    runner.vllm_config = object()
    runner.ascend_config = SimpleNamespace(dp_allreduce_on_npu=False)
    runner._sync_metadata_across_dp(3)
    runner._sync_metadata_across_dp(3)
    assert capsys.readouterr().out.count("adm_dp2_word_active=rank:0,dtype:int32") == 1
