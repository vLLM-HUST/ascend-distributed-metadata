"""Offline deltas for worker-owned snapshots; never starts vLLM or an NPU."""

import argparse
import json
from pathlib import Path
from typing import Any

OUTCOMES = frozenset({"native_dp1", "native_skip", "native_sentinel_fallback",
                      "native_device_fallback", "word_dp2", "word_dp4", "packed_gather"})


def _snapshot(data: Any) -> tuple[tuple[int, int], dict[tuple[int, int, str], int]]:
    if not isinstance(data, dict) or set(data) != {"schema", "pid", "start_ticks", "counts"}:
        raise ValueError("invalid snapshot fields")
    if data["schema"] != "adm-path-counts/1":
        raise ValueError("unsupported snapshot schema")
    pid, start = data["pid"], data["start_ticks"]
    if type(pid) is not int or pid <= 0 or type(start) is not int or start <= 0:
        raise ValueError("unknown worker process identity")
    if not isinstance(data["counts"], list):
        raise ValueError("counts must be a list")
    counts = {}
    for row in data["counts"]:
        if not isinstance(row, dict) or set(row) != {"dp_size", "dp_rank", "outcome", "calls"}:
            raise ValueError("invalid count entry")
        size, rank, outcome, count = (row[k] for k in ("dp_size", "dp_rank", "outcome", "calls"))
        if (type(size) is not int or size < 1 or type(rank) is not int
                or not 0 <= rank < size or type(count) is not int or count < 0
                or not isinstance(outcome, str) or outcome not in OUTCOMES):
            raise ValueError("invalid path count")
        if ((outcome == "native_dp1" and size != 1)
                or (outcome in ("native_skip", "native_sentinel_fallback", "native_device_fallback") and size == 1)
                or (outcome == "word_dp2" and size != 2)
                or (outcome == "word_dp4" and size != 4)
                or (outcome == "packed_gather" and size in (1, 2, 4))):
            raise ValueError("outcome incompatible with DP size")
        key = (size, rank, outcome)
        if key in counts:
            raise ValueError("duplicate path count")
        counts[key] = count
    return (pid, start), counts


def window_delta(before: Any, after: Any) -> dict[str, Any]:
    """Difference complete snapshots from one worker and one measured window.

    Rates describe completed method calls, not requests, time or ECPA status.
    DP1, native skip, and exceptions are not silently folded into word-path
    denominators. Missing telemetry is an error, not a zero observation.
    """
    old_id, old = _snapshot(before)
    new_id, new = _snapshot(after)
    if old_id != new_id:
        raise ValueError("worker identity changed between snapshots")
    if any(key not in new or new[key] < value for key, value in old.items()):
        raise ValueError("counter disappeared or decreased")
    groups: dict[tuple[int, int], dict[str, int]] = {}
    for (size, rank, outcome), value in new.items():
        groups.setdefault((size, rank), {})[outcome] = value - old.get((size, rank, outcome), 0)
    results = []
    for (size, rank), counts in sorted(groups.items()):
        word = counts.get("word_dp2" if size == 2 else "word_dp4", 0) if size in (2, 4) else None
        fallback = counts.get("native_sentinel_fallback", 0) if size in (2, 4) else None
        denominator = word + fallback if word is not None and fallback is not None else 0
        results.append({"dp_size": size, "dp_rank": rank, "completed_calls": counts,
                        "word_attempts_completed": denominator if size in (2, 4) else None,
                        "word_success_fraction": word / denominator if denominator else None,
                        "sentinel_fallback_fraction": fallback / denominator if denominator else None})
    return {"schema": "adm-path-window/1", "pid": old_id[0], "start_ticks": old_id[1],
            "groups": results,
            "scope": "completed worker method calls; fractions exclude native skip, device fallback and failed calls",
            "ecpa_runtime_effective": "NOT_ASSERTED"}


def captured_window(before: Any, after: Any) -> dict[str, Any]:
    """Validate per-worker boundary metadata, then derive completed-call deltas."""
    fields = {"schema", "phase", "launch_id", "host", "monotonic_ns", "dp_size", "dp_rank", "snapshot"}
    if not isinstance(before, dict) or not isinstance(after, dict) or set(before) != fields or set(after) != fields:
        raise ValueError("invalid capture envelope")
    if before["schema"] != "adm-worker-capture/1" or after["schema"] != before["schema"]:
        raise ValueError("invalid capture schema")
    if before["phase"] != "begin" or after["phase"] != "end":
        raise ValueError("invalid capture order")
    for key in ("launch_id", "host"):
        if not isinstance(before[key], str) or not before[key] or before[key] != after[key]:
            raise ValueError(f"mismatched {key}")
    for key in ("dp_size", "dp_rank", "monotonic_ns"):
        if type(before[key]) is not int or type(after[key]) is not int:
            raise ValueError(f"invalid {key}")
    if (before["dp_size"] <= 0 or not 0 <= before["dp_rank"] < before["dp_size"]
            or (before["dp_size"], before["dp_rank"]) != (after["dp_size"], after["dp_rank"])
            or before["monotonic_ns"] >= after["monotonic_ns"]):
        raise ValueError("invalid capture rank or time")
    delta = window_delta(before["snapshot"], after["snapshot"])
    if any((group["dp_size"], group["dp_rank"]) != (before["dp_size"], before["dp_rank"])
           for group in delta["groups"]):
        raise ValueError("counts do not belong to capture rank")
    return {"launch_id": before["launch_id"], "host": before["host"],
            "dp_size": before["dp_size"], "dp_rank": before["dp_rank"],
            "begin_monotonic_ns": before["monotonic_ns"], "end_monotonic_ns": after["monotonic_ns"],
            "delta": delta, "window_scope": "per-worker call boundaries, not synchronized across ranks"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--capture-envelopes", action="store_true",
                        help="validate worker capture envelopes before computing the delta")
    args = parser.parse_args()
    def unique_pairs(pairs):
        data = {}
        for key, value in pairs:
            if key in data:
                raise ValueError(f"duplicate JSON key: {key}")
            data[key] = value
        return data

    evaluate = captured_window if args.capture_envelopes else window_delta
    print(json.dumps(evaluate(
        json.loads(args.before.read_text(), object_pairs_hook=unique_pairs),
        json.loads(args.after.read_text(), object_pairs_hook=unique_pairs),
    ), indent=2))


if __name__ == "__main__":
    main()
