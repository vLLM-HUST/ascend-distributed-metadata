from enum import Enum
from types import SimpleNamespace

import pytest
import torch

from ascend_distributed_metadata import packed_sync


class Mode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


class Runner:
    dp_size = 4
    vllm_config = object()

    def __init__(self, rank=0):
        self.dp_rank = rank
        self.native_calls = 0

    def _sync_metadata_across_dp(
        self,
        num_tokens,
        is_draft_model=False,
        cudagraph_mode=Mode.NONE,
        allow_dp_padding=False,
    ):
        self.native_calls += 1
        return num_tokens, None, cudagraph_mode


class FakeDist:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def all_gather_into_tensor(self, output, local, *, group):
        self.calls.append((local.clone(), output.numel(), group))
        output.copy_(torch.tensor(self.values, dtype=torch.int32))

    def all_reduce(self, tensor, *, group):
        self.calls.append((tensor.clone(), tensor.numel(), group))
        word = sum(value << (packed_sync.DP2_SLOT_BITS * rank)
                   for rank, value in enumerate(self.values))
        tensor.fill_(word)


def module_for(values, *, skip=False):
    dist = FakeDist(values)
    module = SimpleNamespace(
        torch=torch,
        dist=dist,
        get_dp_group=lambda: SimpleNamespace(cpu_group="reviewed-cpu-group"),
        should_skip_allreduce_across_dp_group=lambda _config, _draft: skip,
        CUDAGraphMode=Mode,
    )
    return module, dist


def test_packed_collective_reconstructs_sparse_vector_and_minimum_mode():
    counts = [32, 48, 16, 40]
    modes = [Mode.FULL, Mode.PIECEWISE, Mode.FULL, Mode.NONE]
    values = [packed_sync._encode(n, m) for n, m in zip(counts, modes)]
    module, dist = module_for(values)
    runner = Runner(rank=1)
    wrapped = packed_sync._wrap(Runner._sync_metadata_across_dp, module)

    maximum, vector, mode = wrapped(runner, 48, cudagraph_mode=Mode.PIECEWISE)

    assert maximum == 48
    assert vector.tolist() == counts
    assert vector.dtype == torch.int32 and vector.device.type == "cpu"
    assert mode is Mode.NONE
    assert runner.native_calls == 0
    assert len(dist.calls) == 1
    local, output_length, group = dist.calls[0]
    assert local.tolist() == [values[1]]
    assert output_length == 4
    assert group == "reviewed-cpu-group"


def test_dp2_scalar_collective_preserves_sparse_and_padded_results():
    counts = [17, 8191]
    modes = [Mode.FULL, Mode.NONE]
    values = [packed_sync._encode(n, m) for n, m in zip(counts, modes)]
    module, dist = module_for(values)
    wrapped = packed_sync._wrap(Runner._sync_metadata_across_dp, module)
    runners = [Runner(rank=0), Runner(rank=1)]
    for runner in runners:
        runner.dp_size = 2

    sparse = wrapped(runners[0], counts[0], cudagraph_mode=modes[0])
    padded = wrapped(runners[1], counts[1], cudagraph_mode=modes[1],
                     allow_dp_padding=True)

    assert (sparse[0], sparse[1].tolist(), sparse[2]) == (8191, counts, Mode.NONE)
    assert (padded[0], padded[1].tolist(), padded[2]) == (8191, [8191] * 2, Mode.NONE)
    assert [call[1] for call in dist.calls] == [1, 1]
    assert [call[0].tolist() for call in dist.calls] == [[values[0]],
                                                         [values[1] << 15]]
    assert [runner.native_calls for runner in runners] == [0, 0]


@pytest.mark.parametrize("bad_rank", [0, 1])
def test_dp2_out_of_range_rank_falls_back_on_both_ranks(bad_rank):
    counts = [7, 8]
    counts[bad_rank] = packed_sync.DP2_MAX_TOKENS + 1
    values = [packed_sync._encode(n, Mode.FULL) for n in counts]
    values[bad_rank] = packed_sync.DP2_SENTINEL
    module, dist = module_for(values)
    wrapped = packed_sync._wrap(Runner._sync_metadata_across_dp, module)
    runners = [Runner(rank=0), Runner(rank=1)]
    for runner in runners:
        runner.dp_size = 2

    for rank, runner in enumerate(runners):
        wrapped(runner, counts[rank], cudagraph_mode=Mode.FULL)

    assert len(dist.calls) == 2
    assert [call[1] for call in dist.calls] == [1, 1]
    assert [runner.native_calls for runner in runners] == [1, 1]


@pytest.mark.parametrize("draft,padding", [(False, True), (True, False)])
def test_padded_and_draft_paths_return_uniform_vector(draft, padding):
    counts = [32, 48, 16, 40]
    values = [packed_sync._encode(n, Mode.FULL) for n in counts]
    module, dist = module_for(values)
    runner = Runner()
    wrapped = packed_sync._wrap(Runner._sync_metadata_across_dp, module)

    maximum, vector, mode = wrapped(
        runner, 32, is_draft_model=draft,
        cudagraph_mode=Mode.FULL, allow_dp_padding=padding,
    )

    assert (maximum, vector.tolist(), mode) == (48, [48] * 4, Mode.FULL)
    assert len(dist.calls) == 1


def test_rank_local_padding_choice_does_not_change_collective_shape():
    values = [packed_sync._encode(7, Mode.NONE),
              packed_sync._encode(12, Mode.FULL),
              packed_sync._encode(3, Mode.PIECEWISE),
              packed_sync._encode(9, Mode.FULL)]
    module, dist = module_for(values)
    wrapped = packed_sync._wrap(Runner._sync_metadata_across_dp, module)

    sparse = wrapped(Runner(rank=0), 7, cudagraph_mode=Mode.NONE,
                     allow_dp_padding=False)[1]
    padded = wrapped(Runner(rank=1), 12, cudagraph_mode=Mode.FULL,
                     allow_dp_padding=True)[1]

    assert sparse.tolist() == [7, 12, 3, 9]
    assert padded.tolist() == [12, 12, 12, 12]
    assert [call[1] for call in dist.calls] == [4, 4]


def test_group_visible_sentinel_uses_native_collective_on_every_rank():
    values = [packed_sync._encode(7, Mode.FULL), -1,
              packed_sync._encode(3, Mode.NONE),
              packed_sync._encode(9, Mode.PIECEWISE)]
    module, dist = module_for(values)
    wrapped = packed_sync._wrap(Runner._sync_metadata_across_dp, module)
    runners = [Runner(rank=i) for i in range(4)]

    for runner in runners:
        wrapped(runner, 7, cudagraph_mode=Mode.FULL)

    assert len(dist.calls) == 4
    assert [runner.native_calls for runner in runners] == [1] * 4


@pytest.mark.parametrize("tokens,mode", [(-1, Mode.NONE),
                                           (packed_sync.MAX_PACKED_TOKENS + 1, Mode.FULL),
                                           (1, object())])
def test_out_of_range_inputs_mark_fallback(tokens, mode):
    assert packed_sync._encode(tokens, mode) == packed_sync.FALLBACK_SENTINEL


def test_dp1_and_native_skip_do_not_enter_packed_collective():
    module, dist = module_for([], skip=True)
    wrapped = packed_sync._wrap(Runner._sync_metadata_across_dp, module)
    runner = Runner()
    assert wrapped(runner, 4, cudagraph_mode=Mode.NONE) == (4, None, Mode.NONE)
    assert runner.native_calls == 1
    assert not dist.calls

    module, dist = module_for([])
    wrapped = packed_sync._wrap(Runner._sync_metadata_across_dp, module)
    runner.dp_size = 1
    assert wrapped(runner, 4, cudagraph_mode=Mode.NONE) == (4, None, Mode.NONE)
    assert not dist.calls


def test_omitted_graph_mode_keeps_native_default():
    values = [packed_sync._encode(4, Mode.NONE)] * 4
    module, _ = module_for(values)
    wrapped = packed_sync._wrap(Runner._sync_metadata_across_dp, module)

    maximum, vector, mode = wrapped(Runner(), 4)

    assert maximum == 4
    assert vector.tolist() == [4] * 4
    assert mode is Mode.NONE


def test_install_is_idempotent_and_rejects_other_source(monkeypatch):
    class Target(Runner):
        pass

    module, _ = module_for([])
    with pytest.raises(packed_sync.PackedSyncViolation, match="reviewed pin"):
        packed_sync.install(Target, module)
    monkeypatch.setattr(packed_sync, "_fingerprint", lambda _fn: packed_sync.TARGET_FINGERPRINT)
    assert packed_sync.install(Target, module)
    assert not packed_sync.install(Target, module)


def test_disabled_entrypoint_does_not_import_ascend(monkeypatch):
    monkeypatch.delenv(packed_sync.ENABLE_ENV, raising=False)
    monkeypatch.setattr(packed_sync.importlib, "import_module",
                        lambda name: pytest.fail(f"unexpected import: {name}"))
    packed_sync.register()
    monkeypatch.setenv(packed_sync.ENABLE_ENV, "invalid")
    with pytest.raises(packed_sync.PackedSyncViolation, match="must be 0 or 1"):
        packed_sync.register()
