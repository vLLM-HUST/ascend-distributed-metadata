"""Epoch-bound snapshots for the internal DP load-balancing metadata path.

The coordinator owns publication order. A consumer accepts complete snapshots
only, compares each rank against its own frontier, and records recovery only
after the accepted state is used by routing.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from uuid import uuid4

SCHEMA = 1


class ReplicaViolation(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _number(value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ReplicaViolation("invalid_number")
    return value


@dataclass(frozen=True)
class RankSnapshot:
    rank: int
    epoch: int
    step: int
    version: int
    waiting: int
    running: int

    def to_wire(self) -> list[int]:
        return [self.rank, self.epoch, self.step, self.version,
                self.waiting, self.running]

    @classmethod
    def from_wire(cls, value: object) -> "RankSnapshot":
        if not isinstance(value, (list, tuple)) or len(value) != 6:
            raise ReplicaViolation("invalid_rank_snapshot")
        rank, epoch, step, version, waiting, running = value
        rank, epoch, step = _number(rank), _number(epoch, -1), _number(step, -1)
        version, waiting, running = (_number(version), _number(waiting),
                                     _number(running))
        if (epoch, step) == (-1, -1):
            if (version, waiting, running) != (0, 0, 0):
                raise ReplicaViolation("invalid_initial_rank")
        elif epoch < 0 or step < 0 or version == 0:
            raise ReplicaViolation("invalid_rank_frontier")
        return cls(rank, epoch, step, version, waiting, running)


def _digest(wire: dict) -> str:
    return sha256(json.dumps(wire, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class CoordinatorPublisher:
    """Build full DP snapshots from real per-rank SchedulerStats updates."""

    def __init__(self, rank_count: int, generation: str | None = None):
        if _number(rank_count, 1) < 2:
            raise ReplicaViolation("invalid_rank_count")
        self.rank_count = rank_count
        self.generation = uuid4().hex if generation is None else generation
        if not isinstance(self.generation, str) or not self.generation:
            raise ReplicaViolation("invalid_generation")
        self.ranks = [RankSnapshot(i, -1, -1, 0, 0, 0)
                      for i in range(rank_count)]
        self.publication_seq = -1

    def observe(self, rank: int, epoch: int, step: int,
                waiting: int, running: int) -> bool:
        rank = _number(rank)
        if rank >= self.rank_count:
            raise ReplicaViolation("unexpected_rank")
        next_frontier = (_number(epoch), _number(step))
        waiting, running = _number(waiting), _number(running)
        previous = self.ranks[rank]
        old_frontier = (previous.epoch, previous.step)
        if next_frontier < old_frontier:
            return False
        if next_frontier == old_frontier:
            if (waiting, running) != (previous.waiting, previous.running):
                raise ReplicaViolation("same_identity_different_payload")
            return False
        self.ranks[rank] = RankSnapshot(rank, epoch, step, previous.version + 1,
                                        waiting, running)
        return True

    def publish(self, current_wave: int, engines_running: bool) -> dict:
        _number(current_wave)
        if type(engines_running) is not bool:
            raise ReplicaViolation("invalid_running_state")
        self.publication_seq += 1
        return {
            "adm_schema": SCHEMA,
            "generation": self.generation,
            "publication_seq": self.publication_seq,
            "current_wave": current_wave,
            "engines_running": engines_running,
            "ranks": [rank.to_wire() for rank in self.ranks],
        }


class FrontendReplica:
    """Validate full snapshots before their counts reach the router."""

    def __init__(self, rank_count: int):
        self.rank_count = _number(rank_count, 1)
        self.generation: str | None = None
        self.publication_seq = -1
        self.ranks: tuple[RankSnapshot, ...] = ()
        self.current_wave = 0
        self.engines_running = False
        self._digest: str | None = None
        self._invalidated: tuple[str, str, str | None] | None = None
        self._pending_receipt: dict | None = None
        self.recovery_receipts: list[dict] = []
        self.retired_generations: set[str] = set()
        self.rejected = {"stale_publication": 0, "stale_rank": 0, "stale_wave": 0,
                         "retired_generation": 0, "unobserved_rank": 0}

    def invalidate(self, recovery_id: str, reason: str) -> None:
        if not recovery_id or not reason:
            raise ReplicaViolation("invalid_recovery_request")
        if self._invalidated is not None or self._pending_receipt is not None:
            raise ReplicaViolation("recovery_already_pending")
        self._invalidated = (recovery_id, reason, self.generation)

    @property
    def ready(self) -> bool:
        return bool(self.ranks) and self._invalidated is None

    def apply(self, wire: object) -> bool:
        if not isinstance(wire, dict) or wire.get("adm_schema") != SCHEMA:
            raise ReplicaViolation("invalid_schema")
        generation = wire.get("generation")
        if not isinstance(generation, str) or not generation:
            raise ReplicaViolation("invalid_generation")
        publication_seq = _number(wire.get("publication_seq"))
        current_wave = _number(wire.get("current_wave"))
        engines_running = wire.get("engines_running")
        if type(engines_running) is not bool:
            raise ReplicaViolation("invalid_running_state")
        raw_ranks = wire.get("ranks")
        if not isinstance(raw_ranks, (list, tuple)) or len(raw_ranks) != self.rank_count:
            raise ReplicaViolation("incomplete_rank_set")
        ranks = tuple(RankSnapshot.from_wire(value) for value in raw_ranks)
        if tuple(rank.rank for rank in ranks) != tuple(range(self.rank_count)):
            raise ReplicaViolation("invalid_rank_set")
        # Coordinator defaults are useful at startup, but they are not a
        # recovery response from an observed rank.
        if self._invalidated is not None and any(rank.version == 0 for rank in ranks):
            self.rejected["unobserved_rank"] += 1
            return False
        digest = _digest(wire)

        if generation in self.retired_generations:
            self.rejected["retired_generation"] += 1
            return False
        changed_generation = self.generation is not None and generation != self.generation
        if changed_generation and self._invalidated is None:
            raise ReplicaViolation("unauthorized_generation_change")
        if not changed_generation and generation == self.generation:
            if publication_seq < self.publication_seq:
                self.rejected["stale_publication"] += 1
                return False
            if publication_seq == self.publication_seq:
                if digest != self._digest:
                    raise ReplicaViolation("same_identity_different_payload")
                return False
            if current_wave < self.current_wave:
                self.rejected["stale_wave"] += 1
                return False
            for old, new in zip(self.ranks, ranks):
                if (new.epoch, new.step, new.version) < (
                    old.epoch, old.step, old.version
                ) or new.version < old.version:
                    self.rejected["stale_rank"] += 1
                    return False
                if new.version == old.version and new != old:
                    raise ReplicaViolation("same_identity_different_payload")
        if changed_generation:
            assert self.generation is not None
            self.retired_generations.add(self.generation)

        self.generation = generation
        self.publication_seq = publication_seq
        self.ranks = ranks
        self.current_wave = current_wave
        self.engines_running = engines_running
        self._digest = digest
        if self._invalidated is not None:
            recovery_id, reason, before = self._invalidated
            self._pending_receipt = {
                "recovery_id": recovery_id,
                "reason": reason,
                "generation_before": before,
                "generation_after": generation,
                "publication_seq": publication_seq,
                "snapshot_sha256": digest,
                "rank_frontiers": [[r.rank, r.epoch, r.step, r.version]
                                   for r in ranks],
            }
            self._invalidated = None
        return True

    def snapshot_counts(self) -> list[list[int]]:
        if not self.ready:
            raise ReplicaViolation("replica_not_ready")
        return [[rank.waiting, rank.running] for rank in self.ranks]

    def counts_for_route(self) -> list[list[int]]:
        counts = self.snapshot_counts()
        if self._pending_receipt is not None:
            self.recovery_receipts.append(self._pending_receipt)
            self._pending_receipt = None
        return counts
