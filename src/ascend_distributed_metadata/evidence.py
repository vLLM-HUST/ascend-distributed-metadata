"""Offline deltas for worker-owned snapshots; never starts vLLM or an NPU."""

import argparse
import json
from pathlib import Path
from typing import Any

OUTCOMES = frozenset({"native_dp1", "native_skip", "native_sentinel_fallback",
                      "word_dp2", "word_dp4", "packed_gather"})


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
                or (outcome in ("native_skip", "native_sentinel_fallback") and size == 1)
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
        total = sum(counts.values())
        native = sum(value for outcome, value in counts.items() if outcome.startswith("native_"))
        results.append({"dp_size": size, "dp_rank": rank, "completed_calls": counts,
                        "all_completed_calls": total,
                        "word_fraction_all_completed": word / total if word is not None and total else None,
                        "native_fraction_all_completed": native / total if total else None,
                        "word_attempts_completed": denominator if size in (2, 4) else None,
                        "word_success_fraction": word / denominator if denominator else None,
                        "sentinel_fallback_fraction": fallback / denominator if denominator else None})
    return {"schema": "adm-path-window/1", "pid": old_id[0], "start_ticks": old_id[1],
            "groups": results,
            "scope": "completed worker method calls; fractions exclude native skip and failed calls",
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
            or before["monotonic_ns"] <= 0 or before["monotonic_ns"] >= after["monotonic_ns"]):
        raise ValueError("invalid capture rank or time")
    delta = window_delta(before["snapshot"], after["snapshot"])
    if any((group["dp_size"], group["dp_rank"]) != (before["dp_size"], before["dp_rank"])
           for group in delta["groups"]):
        raise ValueError("counts do not belong to capture rank")
    return {"launch_id": before["launch_id"], "host": before["host"],
            "dp_size": before["dp_size"], "dp_rank": before["dp_rank"],
            "begin_monotonic_ns": before["monotonic_ns"], "end_monotonic_ns": after["monotonic_ns"],
            "delta": delta, "window_scope": "per-worker call boundaries, not synchronized across ranks"}


def captured_workers(pairs: list[tuple[Any, Any]], dp_size: int, workers_per_rank: int) -> dict[str, Any]:
    """Require complete, distinct worker coverage and a common overlapping window."""
    if type(dp_size) is not int or dp_size < 1 or type(workers_per_rank) is not int or workers_per_rank < 1:
        raise ValueError("invalid expected topology")
    if len(pairs) != dp_size * workers_per_rank:
        raise ValueError("incomplete worker coverage")
    windows = [captured_window(before, after) for before, after in pairs]
    if len({(w["host"], w["launch_id"]) for w in windows}) != 1:
        raise ValueError("mixed host or launch")
    identities = [(w["host"], w["delta"]["pid"], w["delta"]["start_ticks"]) for w in windows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate worker identity")
    if any(w["dp_size"] != dp_size for w in windows):
        raise ValueError("unexpected DP size")
    if sorted(w["dp_rank"] for w in windows) != [r for r in range(dp_size) for _ in range(workers_per_rank)]:
        raise ValueError("incomplete DP rank coverage")
    if max(w["begin_monotonic_ns"] for w in windows) >= min(w["end_monotonic_ns"] for w in windows):
        raise ValueError("worker windows do not overlap")
    observable = all(w["delta"]["groups"] and sum(
        sum(g["completed_calls"].values()) for g in w["delta"]["groups"]) > 0 for w in windows)
    return {"schema": "adm-worker-set/1", "status": "observed" if observable else "unknown",
            "workers": windows,
            "scope": "per-worker call boundaries overlap but are not synchronized; TP identity must be checked externally"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path, nargs="?")
    parser.add_argument("after", type=Path, nargs="?")
    parser.add_argument("--capture-envelopes", action="store_true",
                        help="validate worker capture envelopes before computing the delta")
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--dp-size", type=int)
    parser.add_argument("--workers-per-rank", type=int)
    args = parser.parse_args()
    def unique_pairs(pairs):
        data = {}
        for key, value in pairs:
            if key in data:
                raise ValueError(f"duplicate JSON key: {key}")
            data[key] = value
        return data

    def read(path):
        return json.loads(path.read_text(), object_pairs_hook=unique_pairs)

    if args.capture_dir:
        if args.before or args.after or args.capture_envelopes or args.dp_size is None or args.workers_per_rank is None:
            parser.error("capture-dir needs topology and no positional paths")
        begins = sorted(args.capture_dir.glob("begin-*.json"))
        ends = sorted(args.capture_dir.glob("end-*.json"))
        expected_end_names = {p.name.replace("begin-", "end-", 1) for p in begins}
        if expected_end_names != {p.name for p in ends}:
            raise ValueError("unpaired capture files")
        pairs = []
        for path in begins:
            before = read(path)
            after = read(args.capture_dir / path.name.replace("begin-", "end-", 1))
            captured_window(before, after)
            identity = before["snapshot"]
            if path.name != f"begin-{identity['pid']}-{identity['start_ticks']}.json":
                raise ValueError("capture filename does not match worker identity")
            pairs.append((before, after))
        print(json.dumps(captured_workers(pairs, args.dp_size, args.workers_per_rank), indent=2))
        return
    if not args.before or not args.after or args.dp_size is not None or args.workers_per_rank is not None:
        parser.error("supply both snapshot paths, or capture-dir and topology")
    evaluate = captured_window if args.capture_envelopes else window_delta
    print(json.dumps(evaluate(
        read(args.before), read(args.after),
    ), indent=2))


if __name__ == "__main__":
    main()
