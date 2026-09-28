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
from array import array
from collections.abc import Callable
from typing import Any

ENABLE_ENV = "ADM_PACKED_SYNC_ENABLE"
TARGET_FINGERPRINT = "3e3cac40908b68c65c45c820efca81073c0a11bef5eab13b2f8cf9ce36e407f5"
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
INT64_ARRAY_COMPAT = array("q").itemsize == 8


class PackedSyncViolation(RuntimeError):
    """The enabled plugin cannot safely wrap the installed Ascend source."""


def _fingerprint(function: Callable[..., Any]) -> str | None:
    try:
        source = inspect.getsource(function)
        normalized = ast.dump(
            ast.parse(textwrap.dedent(source)), include_attributes=False
        )
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


def _wrap(original: Callable[..., Any], runner_module: Any) -> Callable[..., Any]:
    torch = runner_module.torch
    dist = runner_module.dist
    get_dp_group = runner_module.get_dp_group
    should_skip = runner_module.should_skip_allreduce_across_dp_group
    graph_mode_type = runner_module.CUDAGraphMode
    default_mode = inspect.signature(original).parameters["cudagraph_mode"].default
    report_dp4 = os.environ.get("ADM_DP4_REPORT_ACTIVE") == "1"

    def packed_sync(
        self: Any,
        num_tokens: int,
        is_draft_model: bool = False,
        cudagraph_mode: Any = default_mode,
        allow_dp_padding: bool = False,
    ) -> tuple[int, Any, Any]:
        nonlocal report_dp4
        if self.dp_size == 1 or should_skip(self.vllm_config, is_draft_model):
            return original(
                self, num_tokens, is_draft_model, cudagraph_mode, allow_dp_padding
            )

        encoded = _encode(num_tokens, cudagraph_mode)
        if self.dp_size in (2, 4):
            if self.dp_size == 4 and report_dp4:
                print(f"adm_dp4_word_active=rank:{self.dp_rank},dtype:int64", flush=True)
                report_dp4 = False
            slot = encoded if 0 <= encoded <= WORD_MAX_VALUE else WORD_SENTINEL
            local_word = slot << (WORD_SLOT_BITS * self.dp_rank)
            if self.dp_size == 4:
                packed = getattr(self, "_adm_dp4_word", None)
                if packed is None:
                    if INT64_ARRAY_COMPAT:
                        word_buffer = array("q", [0])
                        packed = torch.frombuffer(word_buffer, dtype=torch.int64)
                        self._adm_dp4_word_buffer = word_buffer
                    else:
                        packed = torch.empty(1, device="cpu", dtype=torch.int64)
                    self._adm_dp4_word = packed
                # The CPU collective completes before this rank's next step.
                if INT64_ARRAY_COMPAT:
                    self._adm_dp4_word_buffer[0] = local_word
                else:
                    packed.fill_(local_word)
            else:
                packed = torch.tensor([local_word], device="cpu", dtype=torch.int32)
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
            return original(
                self, num_tokens, is_draft_model, cudagraph_mode, allow_dp_padding
            )

        tokens = [value >> 2 for value in values]
        max_tokens = max(tokens)
        synced_mode = graph_mode_type(min(value & 3 for value in values))
        if allow_dp_padding or is_draft_model:
            token_vector = torch.full(
                (self.dp_size,), max_tokens, device="cpu", dtype=torch.int32
            )
        else:
            token_vector = torch.tensor(tokens, device="cpu", dtype=torch.int32)
        return max_tokens, token_vector, synced_mode

    packed_sync.__name__ = original.__name__
    packed_sync.__doc__ = original.__doc__
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
    if _fingerprint(original) != TARGET_FINGERPRINT:
        raise PackedSyncViolation("Ascend DP sync source does not match reviewed pin")
    setattr(runner_type, ORIGINAL_ATTR, original)
    runner_type._sync_metadata_across_dp = _wrap(original, runner_module)
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
