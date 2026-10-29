"""Opt-in diagnostic snapshots at worker call boundaries, never a timing tool."""

import json
import os
from pathlib import Path
import socket
import time


class WindowCapture:
    """Observe a single begin/end pair in a fresh launch-specific directory.

    Callers create `begin` then, only after every worker captured begin, `end`.
    Each worker observes a marker at its next method call; the end-triggering
    call is outside its snapshot. Boundaries are per-worker, not synchronized.
    """

    def __init__(self, directory, counters):
        self.root = Path(directory)
        self.counters = counters
        self.begin = False
        self.end = False
        self.pid = os.getpid()

    def poll(self, runner):
        if os.getpid() != self.pid:
            self.pid, self.begin, self.end = os.getpid(), False, False
        if self.end:
            return
        try:
            begin = (self.root / "begin").is_file()
            end = (self.root / "end").is_file()
            if not self.begin:
                # Never invent a window if end was already requested.
                if not begin or end:
                    return
                self._write("begin", runner)
                self.begin = True
            elif end:
                self._write("end", runner)
                self.end = True
        except (OSError, AttributeError, TypeError, ValueError):
            # Diagnostics must never break inference. Missing evidence is unknown.
            return

    def _write(self, phase, runner):
        snapshot = self.counters.snapshot()
        if snapshot["start_ticks"] is None:
            raise ValueError("unknown process identity")
        data = {"schema": "adm-worker-capture/1", "phase": phase,
                "launch_id": self.root.name, "host": socket.gethostname(),
                "monotonic_ns": time.monotonic_ns(), "dp_size": runner.dp_size,
                "dp_rank": runner.dp_rank, "snapshot": snapshot}
        name = f"{phase}-{snapshot['pid']}-{snapshot['start_ticks']}.json"
        target = self.root / name
        tmp = self.root / f".{name}.{time.monotonic_ns()}.tmp"
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                json.dump(data, out, sort_keys=True)
                out.write("\n")
            os.link(tmp, target)  # atomic, never overwrite an existing receipt
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
