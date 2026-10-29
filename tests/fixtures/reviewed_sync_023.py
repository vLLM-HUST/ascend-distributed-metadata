# Source: vllm-ascend commit 5cb98caaadeff42b5b62b996e34bb2aaa29d20fd
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from enum import Enum
class CUDAGraphMode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2

def _sync_metadata_across_dp(
        self,
        num_tokens: int,
        is_draft_model: bool = False,
        cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        allow_dp_padding: bool = False,
    ) -> tuple[int, torch.Tensor | None, CUDAGraphMode]:
        # TODO: In vLLM, the only thing that needs to be synced is num_tokens, but in
        # our case, we still need to sync the other two flags as well. So we need to
        # include them in the all_reduce operation, and more over, we CANNOT skip it
        # even if we are running in eager mode, which harms performance.
        # FIXME: Restore the `or self.vllm_config.model_config.enforce_eager` here
        # immediately once the other two flags are no longer needed.
        if self.dp_size == 1:
            return num_tokens, None, cudagraph_mode

        if should_skip_allreduce_across_dp_group(self.vllm_config, is_draft_model):
            num_tokens_after_padding = torch.tensor([num_tokens] * self.dp_size, device="cpu", dtype=torch.int32)
            return num_tokens, num_tokens_after_padding, cudagraph_mode

        # On certain devices, CPU-side all_reduce may return dirty data.
        # When dp_allreduce_on_npu is True, route DP metadata
        # synchronization through the NPU device group to avoid data corruption.
        device_str, group = (
            ("npu", get_dp_group().device_group)
            if self.ascend_config.dp_allreduce_on_npu
            else ("cpu", get_dp_group().cpu_group)
        )
        packed_tensor = torch.zeros(2, self.dp_size, device=device_str, dtype=torch.int32)
        packed_tensor[0][self.dp_rank] = num_tokens
        packed_tensor[1][self.dp_rank] = cudagraph_mode.value
        dist.all_reduce(packed_tensor, group=group)
        if device_str == "npu":
            packed_tensor = packed_tensor.cpu()

        # Unpack the results
        num_tokens_across_dp = packed_tensor[0, :]
        max_tokens_across_dp = int(num_tokens_across_dp.max().item())
        synced_cudagraph_mode = CUDAGraphMode(_post_process_cudagraph_mode(packed_tensor))

        # Create a tensor for num_tokens_after_padding
        if allow_dp_padding or is_draft_model:
            num_tokens_after_padding = torch.tensor(
                [max_tokens_across_dp] * self.dp_size, device="cpu", dtype=torch.int32
            )
        else:
            num_tokens_after_padding = num_tokens_across_dp.cpu()

        return max_tokens_across_dp, num_tokens_after_padding, synced_cudagraph_mode

def _post_process_cudagraph_mode(tensor: torch.Tensor) -> int:
    """
    Synchronize cudagraph_mode across DP ranks by taking the minimum.
    If any rank has NONE (0), all ranks use NONE.
    This ensures all ranks send consistent values (all padded or all unpadded).
    """
    return int(tensor[1, :].min().item())
