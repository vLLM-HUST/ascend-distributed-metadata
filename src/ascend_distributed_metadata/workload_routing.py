"""Opt-in request-work metadata for the reviewed internal DP load balancer.

The coordinator publishes per-rank request counts. During a burst, counts do
not describe how much prefill and decode work this API process just assigned.
This wrapper keeps that bounded, local estimate until the request finishes.
"""

import os
from collections.abc import Callable
from typing import Any

from .packed_sync import _fingerprint

ENABLE_ENV = "ADM_WORKLOAD_ROUTING_ENABLE"
TARGET_FINGERPRINT = "fc983a9be252a00514f0d2a58342d9c67d1708efca8cf6b90b0433e2228c57f1"
ORIGINAL_ATTR = "_adm_workload_routing_original"


class RoutingViolation(RuntimeError):
    """The enabled routing plugin cannot safely wrap this vLLM source."""


def _work_units(request: Any) -> float | None:
    tokens = request.prompt_token_ids
    params = request.sampling_params
    if tokens is None or params is None:
        return None
    max_tokens = params.max_tokens
    if type(max_tokens) is not int or max_tokens < 1:
        return None
    # The cap prevents a single declared maximum from pinning a rank forever.
    return min(16.0, len(tokens) / 512.0 + max_tokens / 128.0)


def _wrap(original: Callable[..., Any]) -> Callable[..., Any]:
    def metadata_aware_route(self: Any, request: Any) -> Any:
        # Preserve explicit rank selection and pooling's special routing.
        if request.data_parallel_rank is not None or request.pooling_params is not None:
            return original(self, request)
        work = _work_units(request)
        counts = self.lb_engines
        engines = self.core_engines
        if work is None or len(counts) != len(engines) or not engines:
            return original(self, request)

        pending = getattr(self, "_adm_pending_work", None)
        if pending is None:
            pending = {}
            self._adm_pending_work = pending
        # Native output handling removes completed IDs from reqs_in_flight.
        # Prune our estimates at the next dispatch; no output-path hook needed.
        active = self.reqs_in_flight
        for req_id in tuple(pending):
            if req_id not in active:
                del pending[req_id]

        extra = [0.0] * len(engines)
        for rank, assigned_work in pending.values():
            if rank < len(engines):
                extra[rank] += max(0.0, assigned_work - 2.0)

        chosen = 0
        best_score = float("inf")
        for offset in range(len(engines)):
            rank = (self.eng_start_index + offset) % len(engines)
            waiting, running = counts[rank]
            score = waiting * 4 + running + extra[rank]
            if score < best_score:
                best_score = score
                chosen = rank

        counts[chosen][0] += self.client_count
        identity = engines[chosen]
        active[request.request_id] = identity
        pending[request.request_id] = (chosen, work)
        return identity

    metadata_aware_route.__name__ = original.__name__
    metadata_aware_route.__doc__ = original.__doc__
    return metadata_aware_route


def install(client_type: type[Any]) -> bool:
    """Wrap only the exact reviewed vLLM-HUST internal DP routing method."""
    if hasattr(client_type, ORIGINAL_ATTR):
        return False
    original = client_type.get_core_engine_for_request
    if _fingerprint(original) != TARGET_FINGERPRINT:
        raise RoutingViolation("vLLM-HUST DP routing source does not match reviewed pin")
    setattr(client_type, ORIGINAL_ATTR, original)
    client_type.get_core_engine_for_request = _wrap(original)
    return True


def register() -> None:
    enabled = os.environ.get(ENABLE_ENV, "0")
    if enabled == "0":
        return
    if enabled != "1":
        raise RoutingViolation(f"{ENABLE_ENV} must be 0 or 1")
    from vllm.v1.engine.core_client import DPLBAsyncMPClient

    install(DPLBAsyncMPClient)
