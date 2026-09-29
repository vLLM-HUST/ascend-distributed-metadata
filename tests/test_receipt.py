import json
import os

import pytest

from ascend_distributed_metadata.receipt import FileReceiptSink
from ascend_distributed_metadata.replica import (
    MAX_RECENT_RECEIPTS,
    CoordinatorPublisher,
    FrontendReplica,
)


def _observed_publisher():
    publisher = CoordinatorPublisher(2, "coordinator-A")
    publisher.observe(0, 0, 1, 1, 0)
    publisher.observe(1, 0, 1, 0, 1)
    return publisher


def test_receipt_is_durable_only_after_routing_read(tmp_path):
    publisher = _observed_publisher()
    replica = FrontendReplica(2, FileReceiptSink(str(tmp_path)))
    assert replica.apply(publisher.publish(0, True))
    replica.invalidate("loss-1", "local_replica_lost")
    assert replica.apply(publisher.publish(0, True))
    assert list(tmp_path.iterdir()) == []

    assert replica.counts_for_route() == [[1, 0], [0, 1]]
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert json.loads(files[0].read_text()) == replica.recovery_receipts[0]
    assert os.stat(files[0]).st_mode & 0o777 == 0o600
    assert replica.recovery_count == 1
    replica.counts_for_route()
    assert len(list(tmp_path.iterdir())) == 1


def test_sink_failure_keeps_receipt_pending_and_blocks_routing_read():
    publisher = _observed_publisher()

    def failed_sink(_receipt):
        raise OSError("disk unavailable")

    replica = FrontendReplica(2, failed_sink)
    assert replica.apply(publisher.publish(0, True))
    replica.invalidate("loss-1", "local_replica_lost")
    assert replica.apply(publisher.publish(0, True))
    with pytest.raises(OSError, match="disk unavailable"):
        replica.counts_for_route()
    assert replica.recovery_count == 0
    assert not replica.recovery_receipts


def test_in_memory_receipts_are_bounded():
    publisher = _observed_publisher()
    replica = FrontendReplica(2)
    assert replica.apply(publisher.publish(0, True))
    for index in range(MAX_RECENT_RECEIPTS + 1):
        replica.invalidate(f"loss-{index}", "local_replica_lost")
        assert replica.apply(publisher.publish(0, True))
        replica.counts_for_route()
    assert replica.recovery_count == MAX_RECENT_RECEIPTS + 1
    assert len(replica.recovery_receipts) == MAX_RECENT_RECEIPTS
    assert replica.recovery_receipts[0]["recovery_id"] == "loss-1"


def test_receipt_directory_must_exist(tmp_path):
    with pytest.raises(ValueError, match="receipt_directory_missing"):
        FileReceiptSink(str(tmp_path / "absent"))


def test_receipt_file_retry_is_idempotent_and_conflict_fails(tmp_path):
    sink = FileReceiptSink(str(tmp_path))
    receipt = {"recovery_id": "loss-1", "publication_seq": 2}
    sink(receipt)
    sink(receipt)
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    files[0].write_text("partial")
    with pytest.raises(ValueError, match="receipt_file_conflict"):
        sink(receipt)
