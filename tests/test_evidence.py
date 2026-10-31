import copy
import json
import subprocess
import sys

import pytest

from ascend_distributed_metadata.evidence import window_delta
from ascend_distributed_metadata.packed_sync import _PathCounters


def snapshot(pid=123, start=456, rows=()):
    return {"schema": "adm-path-counts/1", "pid": pid, "start_ticks": start,
            "counts": [dict(dp_size=2, dp_rank=0, outcome=name, calls=count)
                       for name, count in rows]}


def test_window_rates_use_completed_collective_attempts_not_skip_or_requests():
    before = snapshot(rows=[("word_dp2", 10), ("native_sentinel_fallback", 3),
                            ("native_skip", 6)])
    after = snapshot(rows=[("word_dp2", 17), ("native_sentinel_fallback", 4),
                           ("native_skip", 8)])
    result = window_delta(before, after)
    group = result["groups"][0]
    assert group["completed_calls"]["native_skip"] == 2
    assert group["word_attempts_completed"] == 8
    assert group["word_success_fraction"] == 7 / 8
    assert group["sentinel_fallback_fraction"] == 1 / 8
    assert result["ecpa_runtime_effective"] == "NOT_ASSERTED"


def test_zero_denominator_is_unknown_and_other_dp_is_not_a_word_path():
    one = snapshot(rows=[("native_skip", 1)])
    group = window_delta(one, one)["groups"][0]
    assert group["word_attempts_completed"] == 0
    assert group["word_success_fraction"] is None
    assert group["sentinel_fallback_fraction"] is None
    before = snapshot()
    after = snapshot()
    after["counts"] = [dict(dp_size=3, dp_rank=1, outcome="packed_gather", calls=4)]
    group = window_delta(before, after)["groups"][0]
    assert group["word_attempts_completed"] is None
    assert group["word_success_fraction"] is None


@pytest.mark.parametrize("mutation", ["pid", "start", "missing", "decrease", "duplicate", "wrong_rank", "wrong_size"])
def test_invalid_or_different_worker_snapshots_fail_closed(mutation):
    before = snapshot(rows=[("word_dp2", 10)])
    after = copy.deepcopy(before)
    if mutation == "pid":
        after["pid"] += 1
    elif mutation == "start":
        after["start_ticks"] = None
    elif mutation == "missing":
        after["counts"] = []
    elif mutation == "decrease":
        after["counts"][0]["calls"] = 9
    elif mutation == "duplicate":
        after["counts"].append(copy.deepcopy(after["counts"][0]))
    elif mutation == "wrong_rank":
        after["counts"][0]["dp_rank"] = 2
    else:
        after["counts"][0]["dp_size"] = 4
    with pytest.raises(ValueError):
        window_delta(before, after)


def test_cli_refuses_duplicate_json_keys(tmp_path):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(snapshot()))
    after.write_text('{"schema":"adm-path-counts/1","pid":123,"pid":124,"start_ticks":456,"counts":[]}')
    result = subprocess.run([sys.executable, "-m", "ascend_distributed_metadata.evidence",
                             str(before), str(after)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "duplicate JSON key" in result.stderr


def test_fork_identity_resets_inherited_counts(monkeypatch):
    from types import SimpleNamespace
    from ascend_distributed_metadata import packed_sync

    counters = _PathCounters()
    runner = SimpleNamespace(dp_size=2, dp_rank=0)
    counters.record(runner, "word_dp2")
    pid = counters.snapshot()["pid"]
    monkeypatch.setattr(packed_sync.os, "getpid", lambda: pid + 1)
    child = counters.snapshot()
    assert child["pid"] == pid + 1
    assert child["counts"] == []
