"""Opt-in, version-pinned replacement for Ascend's DP metadata collective.

The native path all-reduces a 2 x DP int32 tensor. DP2 and DP4 use one
packed-word all-reduce with disjoint rank slots; other groups all-gather one
int32 per rank. Both reconstruct the same token vector and minimum graph mode.
Every participating rank uses the same collective shape, even when local
padding decisions differ.
"""

import ast
import hashlib
import importlib
import inspect
import os
import textwrap
import threading
from collections.abc import Callable
from typing import Any

ENABLE_ENV = "ADM_PACKED_SYNC_ENABLE"
TARGET_FINGERPRINT = "3e3cac40908b68c65c45c820efca81073c0a11bef5eab13b2f8cf9ce36e407f5"
EXPERIMENTAL_023_FINGERPRINT = "14121d45961602089f2f8525c44a7ad7bf73922495eb243003480a927b7c0948"
EXPERIMENTAL_023_HELPER_FINGERPRINT = "8e689cb1d725ea9c8de81f0e8f7c24037b2cb8033597d414e21c8464a7d2a19b"
EXPERIMENTAL_023_ENV = "ADM_EXPERIMENTAL_023_ENABLE"
ORIGINAL_ATTR = "_adm_packed_sync_original"
# Two low bits hold the concrete runtime CUDAGraphMode (0, 1, or 2).
# The remaining signed int32 bits hold a nonnegative token count.
MAX_PACKED_TOKENS = (2**31 - 1) >> 2
FALLBACK_SENTINEL = -1
# A 15-bit slot holds 13 token bits and two graph-mode bits. Four slots fit
# below the sign bit of int64; mode 3 is a group-visible fallback sentinel.
WORD_SLOT_BITS = 15
WORD_SLOT_MASK = (1 << WORD_SLOT_BITS) - 1
WORD_MAX_TOKENS = WORD_SLOT_MASK >> 2
WORD_MAX_VALUE = (WORD_MAX_TOKENS << 2) | 2
WORD_SENTINEL = 3
COUNT_ENV = "ADM_PACKED_SYNC_COUNT_PATHS"


def _process_start_ticks() -> int | None:
    """Linux process identity for pairing snapshots; unknown fails closed."""
    try:
        with open("/proc/self/stat", encoding="ascii") as stream:
            stat = stream.read()
        tail = stat[stat.rindex(")") + 2 :].split()
        ticks = int(tail[19])  # /proc stat field 22; tail begins at field 3.
        return ticks if ticks > 0 else None
    except (OSError, ValueError, IndexError):
        return None


class _PathCounters:
    """Opt-in completed-call counts, local to one process and DP rank.

    A forked child starts a fresh set of counters. Counts are only diagnostic:
    the caller must take both snapshots in the owning worker and preserve the
    process/launch identity separately. They are not ECPA attestations.
    """

    def __init__(self) -> None:
        self._pid = os.getpid()
        self._start_ticks = _process_start_ticks()
        self._lock = threading.Lock()
        self._counts: dict[tuple[int, int | None, str], int] = {}

    def _after_fork_if_needed(self) -> None:
        pid = os.getpid()
        if pid != self._pid:
            # Never acquire a lock inherited from another process.
            self._pid = pid
            self._start_ticks = _process_start_ticks()
            self._lock = threading.Lock()
            self._counts = {}

    def record(self, runner: Any, outcome: str) -> None:
        self._after_fork_if_needed()
        key = (runner.dp_size, getattr(runner, "dp_rank", None), outcome)
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        self._after_fork_if_needed()
        with self._lock:
            counts = [
                {"dp_size": dp, "dp_rank": rank, "outcome": outcome, "calls": count}
                for (dp, rank, outcome), count in self._counts.items()
            ]
        counts.sort(key=lambda item: (item["dp_size"], str(item["dp_rank"]), item["outcome"]))
        return {"schema": "adm-path-counts/1", "pid": self._pid,
                "start_ticks": self._start_ticks, "counts": counts}


def snapshot_path_counts(runner_type: type[Any]) -> dict[str, Any] | None:
    """Read local counters from the installed wrapper, or None if disabled.

    Only a process-owned observer can call this in the worker being measured.
    Take before/after snapshots in the same process over the desired window;
    startup and warmup calls must not be relabeled as measurement calls.
    """
    method = runner_type._sync_metadata_across_dp
    snapshot = getattr(method, "_adm_path_counts_snapshot", None)
    return snapshot() if snapshot is not None else None


class PackedSyncViolation(RuntimeError):
    """The enabled plugin cannot safely wrap the installed Ascend source."""


def _fingerprint(function: Callable[..., Any]) -> str | None:
    try:
        source = inspect.getsource(function)
        tree = ast.parse(textwrap.dedent(source))
        # Python 3.12 added an empty type_params field to function/class ASTs.
        # The reviewed fingerprint was produced by Python 3.11; omit only
        # that new *empty* field so identical source hashes identically. Never
        # erase real generic parameters, which are outside the reviewed source.
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if getattr(node, "type_params", ()):
                    return None
                if "type_params" in node._fields:
                    node._fields = tuple(field for field in node._fields if field != "type_params")
        normalized = ast.dump(tree, include_attributes=False)
        return hashlib.sha256(normalized.encode()).hexdigest()
    except (OSError, TypeError, SyntaxError):
        return None


def _encode(num_tokens: Any, mode: Any) -> int:
    """Return a packed int32 value or a group-visible fallback marker."""
    try:
        tokens = int(num_tokens)
        mode_value = mode.value
    except (AttributeError, TypeError, ValueError, OverflowError):
        return FALLBACK_SENTINEL
    if not 0 <= tokens <= MAX_PACKED_TOKENS:
        return FALLBACK_SENTINEL
    if type(mode_value) is not int or mode_value not in (0, 1, 2):
        return FALLBACK_SENTINEL
    return (tokens << 2) | mode_value


def _wrap(original: Callable[..., Any], runner_module: Any, *, device_guard: bool = False) -> Callable[..., Any]:
    torch = runner_module.torch
    dist = runner_module.dist
    get_dp_group = runner_module.get_dp_group
    should_skip = runner_module.should_skip_allreduce_across_dp_group
    graph_mode_type = runner_module.CUDAGraphMode
    default_mode = inspect.signature(original).parameters["cudagraph_mode"].default
    report_dp4 = os.environ.get("ADM_DP4_REPORT_ACTIVE") == "1"
    report_dp2 = os.environ.get("ADM_DP2_REPORT_ACTIVE") == "1"
    counters = _PathCounters() if os.environ.get(COUNT_ENV) == "1" else None

    def packed_sync(
        self: Any,
        num_tokens: int,
        is_draft_model: bool = False,
        cudagraph_mode: Any = default_mode,
        allow_dp_padding: bool = False,
    ) -> tuple[int, Any, Any]:
        nonlocal report_dp4, report_dp2
        if self.dp_size == 1:
            result = original(
                self, num_tokens, is_draft_model, cudagraph_mode, allow_dp_padding
            )
            if counters is not None:
                counters.record(self, "native_dp1")
            return result
        if should_skip(self.vllm_config, is_draft_model):
            result = original(
                self, num_tokens, is_draft_model, cudagraph_mode, allow_dp_padding
            )
            if counters is not None:
                counters.record(self, "native_skip")
            return result

        # The reviewed 0.23 host can deliberately route this collective to an
        # NPU to avoid CPU collective corruption. Never replace that path with
        # the CPU packed algorithm; preserve its original device group.
        if device_guard and self.ascend_config.dp_allreduce_on_npu:
            return original(self, num_tokens, is_draft_model, cudagraph_mode, allow_dp_padding)

        encoded = _encode(num_tokens, cudagraph_mode)
        if self.dp_size in (2, 4):
            slot = encoded if 0 <= encoded <= WORD_MAX_VALUE else WORD_SENTINEL
            packed = torch.tensor(
                [slot << (WORD_SLOT_BITS * self.dp_rank)],
                device="cpu", dtype=torch.int32 if self.dp_size == 2 else torch.int64,
            )
            dist.all_reduce(packed, group=get_dp_group().cpu_group)
            word = int(packed.item())
            values = [
                (word >> (WORD_SLOT_BITS * rank)) & WORD_SLOT_MASK
                for rank in range(self.dp_size)
            ]
        else:
            local = torch.tensor([encoded], device="cpu", dtype=torch.int32)
            gathered = torch.empty(self.dp_size, device="cpu", dtype=torch.int32)
            dist.all_gather_into_tensor(
                gathered, local, group=get_dp_group().cpu_group
            )
            values = gathered.tolist()
        if any(value < 0 or (value & 3) == WORD_SENTINEL for value in values):
            # All ranks see the sentinel and enter the same native collective.
            result = original(
                self, num_tokens, is_draft_model, cudagraph_mode, allow_dp_padding
            )
            if counters is not None:
                counters.record(self, "native_sentinel_fallback")
            return result

        tokens = [value >> 2 for value in values]
        max_tokens = max(tokens)
        synced_mode = graph_mode_type(min(value & 3 for value in values))
        if allow_dp_padding or is_draft_model:
            token_vector = torch.full(
                (self.dp_size,), max_tokens, device="cpu", dtype=torch.int32
            )
        else:
            token_vector = torch.tensor(tokens, device="cpu", dtype=torch.int32)
        if counters is not None:
            outcome = "word_dp2" if self.dp_size == 2 else (
                "word_dp4" if self.dp_size == 4 else "packed_gather"
            )
            counters.record(self, outcome)
        if self.dp_size == 4 and report_dp4:
            # Historical versions printed before the collective and could
            # falsely mark a later fallback (or failed collective) as active.
            report_dp4 = False
            try:
                print(f"adm_dp4_word_active=rank:{self.dp_rank},dtype:int64", flush=True)
            except (OSError, ValueError):
                # Optional evidence must not change serving correctness.
                pass
        if self.dp_size == 2 and report_dp2:
            report_dp2 = False
            try:
                print(f"adm_dp2_word_active=rank:{self.dp_rank},dtype:int32", flush=True)
            except (OSError, ValueError):
                pass
        return max_tokens, token_vector, synced_mode

    packed_sync.__name__ = original.__name__
    packed_sync.__doc__ = original.__doc__
    if counters is not None:
        packed_sync._adm_path_counts_snapshot = counters.snapshot  # type: ignore[attr-defined]
    return packed_sync


def install(runner_type: type[Any], runner_module: Any) -> bool:
    """Install only on the exact reviewed source implementation."""
    if hasattr(runner_type, ORIGINAL_ATTR):
        return False
    if hasattr(runner_type, "_adm_local_observer_original_sync_metadata_across_dp"):
        raise PackedSyncViolation("ADM local observer is already installed")
    if hasattr(runner_type, "_adm_runtime_original_sync_metadata_across_dp"):
        raise PackedSyncViolation("ADM runtime observer is already installed")
    original = runner_type._sync_metadata_across_dp
    fingerprint = _fingerprint(original)
    experimental_023 = (
        os.environ.get(EXPERIMENTAL_023_ENV) == "1"
        and fingerprint == EXPERIMENTAL_023_FINGERPRINT
        and _fingerprint(getattr(runner_module, "_post_process_cudagraph_mode", None))
        == EXPERIMENTAL_023_HELPER_FINGERPRINT
    )
    if fingerprint != TARGET_FINGERPRINT and not experimental_023:
        raise PackedSyncViolation("Ascend DP sync source does not match reviewed pin")
    setattr(runner_type, ORIGINAL_ATTR, original)
    runner_type._sync_metadata_across_dp = _wrap(original, runner_module, device_guard=experimental_023)
    return True


def register() -> None:
    """vLLM general-plugin entry point; disabled mode imports no Ascend code."""
    enabled = os.environ.get(ENABLE_ENV, "0")
    if enabled == "0":
        return
    if enabled != "1":
        raise PackedSyncViolation(f"{ENABLE_ENV} must be 0 or 1")
    # Match the reviewed Ascend worker import order to avoid an import cycle.
    importlib.import_module("vllm_ascend.ops.triton.triton_utils")
    module = importlib.import_module("vllm_ascend.worker.model_runner_v1")
    install(module.NPUModelRunner, module)
