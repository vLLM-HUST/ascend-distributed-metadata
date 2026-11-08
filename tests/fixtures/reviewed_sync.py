# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The vLLM team.
# Licensed under the Apache License, Version 2.0.
# Source: vllm-ascend-hust@367b8e62da799870a7476ce34f5f7658589a8aad,
# vllm_ascend/worker/model_runner_v1.py, _sync_metadata_across_dp.
# This fixture is not executed; it exists to test the source fingerprint.
from __future__ import annotations

class CUDAGraphMode:
    NONE = 0

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

    packed_tensor = torch.zeros(2, self.dp_size, device="cpu", dtype=torch.int32)
    packed_tensor[0][self.dp_rank] = num_tokens
    packed_tensor[1][self.dp_rank] = cudagraph_mode.value
    dist.all_reduce(packed_tensor, group=get_dp_group().cpu_group)

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
