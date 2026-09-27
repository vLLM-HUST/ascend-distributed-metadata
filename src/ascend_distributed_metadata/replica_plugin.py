"""Opt-in registration of the ADM DP statistics publisher and consumer."""

import os

from .replica import CoordinatorPublisher, FrontendReplica, ReplicaViolation

ENABLE_ENV = "ADM_EPOCH_REPLICA_ENABLE"


def register() -> None:
    enabled = os.environ.get(ENABLE_ENV, "0")
    if enabled == "0":
        return
    if enabled != "1":
        raise ReplicaViolation("invalid_enable_flag")
    # The reviewed core exposes only the narrow, process-local factory hook.
    # Enabled mode fails at startup when that hook is unavailable.
    from vllm.v1.engine.dp_metadata_hooks import register as register_factories

    register_factories(CoordinatorPublisher, FrontendReplica)
