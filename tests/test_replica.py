from copy import deepcopy

import pytest
import msgspec.msgpack

from ascend_distributed_metadata.replica import (
    CoordinatorPublisher,
    FrontendReplica,
    ReplicaViolation,
)


def test_rank_skew_uses_independent_frontiers_and_routes_from_full_snapshot():
    publisher = CoordinatorPublisher(2, "coordinator-A")
    assert publisher.observe(0, 0, 20, 1, 3)
    assert publisher.observe(1, 0, 2, 5, 0)
    replica = FrontendReplica(2)
    assert replica.apply(publisher.publish(0, True))
    assert replica.counts_for_route() == [[1, 3], [5, 0]]

    # Rank 1 is behind rank 0, but its own next update is still valid.
    assert publisher.observe(1, 0, 3, 4, 1)
    assert replica.apply(publisher.publish(0, True))
    assert replica.counts_for_route() == [[1, 3], [4, 1]]


def test_stale_duplicate_and_conflicting_publications():
    publisher = CoordinatorPublisher(2, "coordinator-A")
    publisher.observe(0, 0, 1, 1, 0)
    replica = FrontendReplica(2)
    first = publisher.publish(0, True)
    assert replica.apply(first)
    assert not replica.apply(first)

    changed = deepcopy(first)
    changed["ranks"][0][4] = 7
    with pytest.raises(ReplicaViolation, match="same_identity_different_payload"):
        replica.apply(changed)

    publisher.observe(0, 0, 2, 2, 0)
    assert replica.apply(publisher.publish(0, True))
    assert not replica.apply(first)
    assert replica.rejected["stale_publication"] == 1
    assert replica.counts_for_route() == [[2, 0], [0, 0]]


def test_stale_rank_update_never_overwrites_newer_counts():
    publisher = CoordinatorPublisher(2, "coordinator-A")
    publisher.observe(0, 0, 3, 3, 0)
    publisher.observe(1, 0, 1, 0, 1)
    replica = FrontendReplica(2)
    assert replica.apply(publisher.publish(0, True))
    assert not publisher.observe(0, 0, 2, 99, 0)
    assert replica.apply(publisher.publish(0, True))
    assert replica.counts_for_route() == [[3, 0], [0, 1]]

    forged = publisher.publish(0, True)
    forged["ranks"][1][2] = 0
    assert not replica.apply(forged)
    assert replica.rejected["stale_rank"] == 1
    assert replica.counts_for_route() == [[3, 0], [0, 1]]


def test_recovery_requires_complete_snapshot_and_actual_consumer_use():
    publisher = CoordinatorPublisher(2, "coordinator-A")
    publisher.observe(0, 0, 1, 1, 0)
    replica = FrontendReplica(2)
    assert replica.apply(publisher.publish(0, True))
    replica.invalidate("recovery-1", "local_replica_lost")
    with pytest.raises(ReplicaViolation, match="replica_not_ready"):
        replica.counts_for_route()
    incomplete = publisher.publish(0, True)
    incomplete["ranks"].pop()
    with pytest.raises(ReplicaViolation, match="incomplete_rank_set"):
        replica.apply(incomplete)
    assert not replica.recovery_receipts

    publisher.observe(1, 0, 2, 3, 1)
    assert replica.apply(publisher.publish(0, True))
    assert not replica.recovery_receipts
    assert replica.counts_for_route() == [[1, 0], [3, 1]]
    assert len(replica.recovery_receipts) == 1
    assert replica.recovery_receipts[0]["recovery_id"] == "recovery-1"
    replica.counts_for_route()
    assert len(replica.recovery_receipts) == 1


def test_generation_change_needs_invalidation_and_retires_old_generation():
    old = CoordinatorPublisher(2, "coordinator-A")
    new = CoordinatorPublisher(2, "coordinator-B")
    replica = FrontendReplica(2)
    old_snapshot = old.publish(0, True)
    assert replica.apply(old_snapshot)
    with pytest.raises(ReplicaViolation, match="unauthorized_generation_change"):
        replica.apply(new.publish(0, True))
    replica.invalidate("recovery-2", "coordinator_generation_change")
    assert replica.apply(new.publish(0, True))
    assert not replica.apply(old_snapshot)
    assert replica.rejected["retired_generation"] == 1
    assert replica.counts_for_route() == [[0, 0], [0, 0]]


def test_producer_rejects_same_epoch_step_conflicting_payload():
    publisher = CoordinatorPublisher(2, "coordinator-A")
    assert publisher.observe(0, 0, 1, 1, 0)
    assert not publisher.observe(0, 0, 1, 1, 0)
    with pytest.raises(ReplicaViolation, match="same_identity_different_payload"):
        publisher.observe(0, 0, 1, 2, 0)


def test_wire_round_trip_and_invalid_generation():
    publisher = CoordinatorPublisher(2, "coordinator-A")
    publisher.observe(0, 1, 7, 2, 4)
    wire = msgspec.msgpack.decode(msgspec.msgpack.encode(publisher.publish(1, True)))
    replica = FrontendReplica(2)
    assert replica.apply(wire)
    assert replica.snapshot_counts() == [[2, 4], [0, 0]]
    with pytest.raises(ReplicaViolation, match="invalid_generation"):
        CoordinatorPublisher(2, "")


def test_global_wave_regression_and_unconsumed_receipt_are_rejected():
    publisher = CoordinatorPublisher(2, "coordinator-A")
    replica = FrontendReplica(2)
    assert replica.apply(publisher.publish(1, True))
    stale_wave = publisher.publish(0, True)
    assert not replica.apply(stale_wave)
    assert replica.rejected["stale_wave"] == 1
    replica.invalidate("recovery-3", "local_replica_lost")
    assert replica.apply(publisher.publish(1, True))
    with pytest.raises(ReplicaViolation, match="recovery_already_pending"):
        replica.invalidate("recovery-4", "second_loss")
    replica.counts_for_route()
    replica.invalidate("recovery-4", "second_loss")
