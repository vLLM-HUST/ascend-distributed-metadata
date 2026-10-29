import copy
import json
from types import SimpleNamespace

import pytest

from ascend_distributed_metadata.evidence import captured_window
from ascend_distributed_metadata.packed_sync import _PathCounters
from ascend_distributed_metadata.window_capture import WindowCapture


def test_worker_capture_tracks_call_window_and_never_overwrites(tmp_path):
    counters = _PathCounters()
    runner = SimpleNamespace(dp_size=2, dp_rank=1)
    capture = WindowCapture(tmp_path, counters)
    counters.record(runner, "word_dp2")  # warmup outside window
    (tmp_path / "begin").touch()
    capture.poll(runner)
    counters.record(runner, "word_dp2")
    counters.record(runner, "native_sentinel_fallback")
    (tmp_path / "end").touch()
    capture.poll(runner)
    counters.record(runner, "word_dp2")  # outside the end boundary
    capture.poll(runner)
    files = list(tmp_path.glob("begin-*.json")), list(tmp_path.glob("end-*.json"))
    assert tuple(map(len, files)) == (1, 1)
    before, after = [json.loads(paths[0].read_text()) for paths in files]
    result = captured_window(before, after)
    group = result["delta"]["groups"][0]
    assert group["word_attempts_completed"] == 2
    assert group["word_success_fraction"] == group["sentinel_fallback_fraction"] == 0.5
    assert result["dp_rank"] == 1


def test_stale_end_or_missing_directory_does_not_invent_snapshot(tmp_path):
    counters = _PathCounters()
    runner = SimpleNamespace(dp_size=2, dp_rank=0)
    cap = WindowCapture(tmp_path, counters)
    (tmp_path / "begin").touch()
    (tmp_path / "end").touch()
    cap.poll(runner)
    assert not list(tmp_path.glob("*.json"))
    WindowCapture(tmp_path / "missing", counters).poll(runner)


@pytest.mark.parametrize("mutation", ["phase", "time", "rank", "launch", "host", "pid"])
def test_capture_envelopes_reject_invalid_pair(tmp_path, mutation):
    counters = _PathCounters()
    runner = SimpleNamespace(dp_size=2, dp_rank=0)
    cap = WindowCapture(tmp_path, counters)
    (tmp_path / "begin").touch()
    cap.poll(runner)
    counters.record(runner, "word_dp2")
    (tmp_path / "end").touch()
    cap.poll(runner)
    before = json.loads(next(tmp_path.glob("begin-*.json")).read_text())
    after = json.loads(next(tmp_path.glob("end-*.json")).read_text())
    after = copy.deepcopy(after)
    if mutation == "phase": after["phase"] = "begin"
    elif mutation == "time": after["monotonic_ns"] = before["monotonic_ns"]
    elif mutation == "rank": after["dp_rank"] = 1
    elif mutation == "launch": after["launch_id"] += "-other"
    elif mutation == "host": after["host"] += "-other"
    else: after["snapshot"]["pid"] += 1
    with pytest.raises(ValueError):
        captured_window(before, after)


def test_wrong_counter_rank_is_not_accepted(tmp_path):
    counters = _PathCounters()
    runner = SimpleNamespace(dp_size=2, dp_rank=0)
    cap = WindowCapture(tmp_path, counters)
    (tmp_path / "begin").touch()
    cap.poll(runner)
    counters.record(SimpleNamespace(dp_size=2, dp_rank=1), "word_dp2")
    (tmp_path / "end").touch()
    cap.poll(runner)
    with pytest.raises(ValueError, match="capture rank"):
        captured_window(json.loads(next(tmp_path.glob("begin-*.json")).read_text()),
                        json.loads(next(tmp_path.glob("end-*.json")).read_text()))


def test_wrapper_captures_between_worker_calls(tmp_path, monkeypatch):
    from enum import Enum
    import torch
    from ascend_distributed_metadata import packed_sync

    class Mode(Enum):
        NONE = 0
    class Dist:
        @staticmethod
        def all_reduce(tensor, *, group):
            tensor.fill_((2 << 2) | ((3 << 2) << 15))
    def native(runner, num_tokens, is_draft_model=False, cudagraph_mode=Mode.NONE,
               allow_dp_padding=False):
        raise AssertionError("unexpected fallback")
    module = SimpleNamespace(torch=torch, dist=Dist(), CUDAGraphMode=Mode,
        get_dp_group=lambda: SimpleNamespace(cpu_group=None),
        should_skip_allreduce_across_dp_group=lambda *_: False)
    monkeypatch.setenv(packed_sync.COUNT_ENV, "1")
    monkeypatch.setenv(packed_sync.CAPTURE_DIR_ENV, str(tmp_path))
    wrapped = packed_sync._wrap(native, module)
    runner = SimpleNamespace(dp_size=2, dp_rank=0, vllm_config=object())
    (tmp_path / "begin").touch()
    wrapped(runner, 2)
    (tmp_path / "end").touch()
    wrapped(runner, 2)
    result = captured_window(json.loads(next(tmp_path.glob("begin-*.json")).read_text()),
                             json.loads(next(tmp_path.glob("end-*.json")).read_text()))
    assert result["delta"]["groups"][0]["completed_calls"] == {"word_dp2": 1}
