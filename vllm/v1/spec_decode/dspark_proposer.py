# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

import torch
from typing_extensions import override

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.dspark import (
    DSparkDiagnostics,
    make_dspark_warmup_draft_token_ids,
    score_prefix_lengths,
)
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

logger = init_logger(__name__)


class DSparkProposer(SpecDecodeBaseProposer):
    """DSpark proposer for DeepSeek V4 Flash DSpark.

    DSpark's draft model owns a small internal sliding-window cache over
    target-layer features. It does not allocate draft KV blocks through vLLM's
    normal speculative-decoding KV cache path.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dspark"
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
        hf_config = self.draft_model_config.hf_config
        self.target_hidden_size = hf_config.hidden_size * len(
            hf_config.dspark_target_layer_ids
        )
        self.noise_token_id = int(hf_config.dspark_noise_token_id)
        self._prefilled = False
        self._runner = runner
        self.diagnostics = DSparkDiagnostics(
            max_spec_tokens=self.num_speculative_tokens
        )

    @override
    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        del kv_cache_config, kernel_block_sizes
        self.block_size = 1

    @override
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        del num_tokens, use_cudagraphs, is_graph_capturing, slot_mappings

    @override
    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        del cudagraph_mode

    def _batch_size(self, next_token_ids: torch.Tensor) -> int:
        return int(next_token_ids.shape[0])

    def _view_by_request(
        self,
        values: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if values.shape[0] % batch_size != 0:
            raise ValueError(
                "DSpark currently requires uniform flattened per-request inputs; "
                f"got {values.shape[0]} rows for batch_size={batch_size}."
            )
        seq_len = values.shape[0] // batch_size
        return values.view(batch_size, seq_len, values.shape[-1])

    def _positions_by_request(
        self,
        positions: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if positions.ndim != 1:
            positions = positions.reshape(-1)
        if positions.shape[0] % batch_size != 0:
            raise ValueError(
                "DSpark currently requires uniform flattened positions; "
                f"got {positions.shape[0]} rows for batch_size={batch_size}."
            )
        return positions.view(batch_size, positions.shape[0] // batch_size)

    def _warmup_drafts(self, batch_size: int) -> torch.Tensor:
        return make_dspark_warmup_draft_token_ids(
            batch_size=batch_size,
            num_speculative_tokens=self.num_speculative_tokens,
            noise_token_id=self.noise_token_id,
            device=self.device,
        )

    def _observe_confidence(self, confidence: torch.Tensor) -> None:
        confidence_rows = confidence.detach().float().cpu().tolist()
        schedule = score_prefix_lengths(
            confidence_rows,
            [min(self.num_speculative_tokens, len(row)) for row in confidence_rows],
            steps_per_second=lambda _batch_tokens: 1.0,
        )
        self.diagnostics.observe(confidence_rows, schedule)

    @override
    @torch.inference_mode()
    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        del (
            target_token_ids,
            token_indices_to_sample,
            common_attn_metadata,
            sampling_metadata,
            mm_embed_inputs,
            num_rejected_tokens_gpu,
            slot_mappings,
        )
        batch_size = self._batch_size(next_token_ids)
        hidden_by_req = self._view_by_request(target_hidden_states, batch_size)
        positions_by_req = self._positions_by_request(target_positions, batch_size)

        self.model.prefill_main(hidden_by_req, positions_by_req)
        if not self._prefilled:
            self._prefilled = True
            return self._warmup_drafts(batch_size)

        last_hidden = hidden_by_req[:, -1].contiguous()
        last_positions = positions_by_req[:, -1].contiguous()
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=batch_size * self.num_speculative_tokens,
        ):
            draft_token_ids = self.model.draft(
                next_token_ids.to(torch.long),
                last_hidden,
                last_positions,
            )
        confidence = self.model.take_last_confidence()
        if confidence is not None:
            self._observe_confidence(confidence)
        return draft_token_ids[:, : self.num_speculative_tokens].to(torch.int32)

    def get_diagnostics_snapshot(self) -> Any:
        return self.diagnostics.snapshot()
