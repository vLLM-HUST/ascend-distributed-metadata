"""Opt-in registration of the ADM DP statistics publisher and consumer."""

import json
import os

from .replica import CoordinatorPublisher, FrontendReplica, ReplicaViolation
from .receipt import FileReceiptSink

ENABLE_ENV = "ADM_EPOCH_REPLICA_ENABLE"
INJECT_AT_ENV = "ADM_REPLICA_INJECT_LOSS_AT_PUBLICATION_SEQ"
RECEIPT_DIR_ENV = "ADM_RECOVERY_RECEIPT_DIR"


class _DiagnosticFrontendReplica(FrontendReplica):
    """Gated local-copy fault injection for a native recovery receipt."""

    def __init__(self, rank_count: int, inject_at: int,
                 receipt_sink: FileReceiptSink | None = None):
        super().__init__(rank_count, receipt_sink)
        self.inject_at = inject_at
        self.injected = False

    def apply(self, wire: object) -> bool:
        if (
            not self.injected
            and self.generation is not None
            and isinstance(wire, dict)
            and type(wire.get("publication_seq")) is int
            and wire["publication_seq"] >= self.inject_at
            and isinstance(wire.get("ranks"), list)
            and len(wire["ranks"]) == self.rank_count
            and all(isinstance(rank, list) and len(rank) == 6 and
                    type(rank[3]) is int and rank[3] > 0 for rank in wire["ranks"])
        ):
            self.invalidate(f"injected-{self.inject_at}", "injected_local_replica_loss")
            self.injected = True
            print(f"adm_replica_fault_injected_publication={wire['publication_seq']}",
                  flush=True)
            return False
        return super().apply(wire)

    def counts_for_route(self) -> list[list[int]]:
        before = self.recovery_count
        counts = super().counts_for_route()
        if self.recovery_count > before:
            print("adm_recovery_receipt=" + json.dumps(self.recovery_receipts[-1],
                                                       sort_keys=True), flush=True)
        return counts


def register() -> None:
    enabled = os.environ.get(ENABLE_ENV, "0")
    if enabled == "0":
        return
    if enabled != "1":
        raise ReplicaViolation("invalid_enable_flag")
    # The reviewed core exposes only the narrow, process-local factory hook.
    # Enabled mode fails at startup when that hook is unavailable.
    from vllm.v1.engine.dp_metadata_hooks import register as register_factories

    receipt_dir = os.environ.get(RECEIPT_DIR_ENV)
    receipt_sink = FileReceiptSink(receipt_dir) if receipt_dir is not None else None
    inject_at_raw = os.environ.get(INJECT_AT_ENV)
    if inject_at_raw is None:
        def consumer_factory(rank_count: int) -> FrontendReplica:
            return FrontendReplica(rank_count, receipt_sink)
    else:
        try:
            inject_at = int(inject_at_raw)
        except ValueError as error:
            raise ReplicaViolation("invalid_injection_sequence") from error
        if inject_at < 0:
            raise ReplicaViolation("invalid_injection_sequence")

        def consumer_factory(rank_count: int) -> FrontendReplica:
            return _DiagnosticFrontendReplica(rank_count, inject_at, receipt_sink)

    register_factories(CoordinatorPublisher, consumer_factory)
