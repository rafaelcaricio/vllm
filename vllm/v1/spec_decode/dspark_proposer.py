# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from typing import Any

import torch
from typing_extensions import override

from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.dspark import (
    DSparkDiagnostics,
    confidence_threshold_prefix_length,
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
        self._draft_graph_runner: CUDAGraphWrapper | None = None
        self._draft_graph_batch_size = 0
        self._draft_input_ids_buffer = torch.zeros(
            self.max_batch_size,
            dtype=torch.long,
            device=device,
        )
        self._draft_hidden_buffer = torch.zeros(
            self.max_batch_size,
            self.target_hidden_size,
            dtype=self.dtype,
            device=device,
        )
        self._draft_positions_buffer = torch.zeros(
            self.max_batch_size,
            dtype=torch.long,
            device=device,
        )
        self.diagnostics = DSparkDiagnostics(
            max_spec_tokens=self.num_speculative_tokens
        )
        self.confidence_threshold = self._read_confidence_threshold()
        self._last_draft_lengths: list[int] | None = None
        self._last_draft_probs: torch.Tensor | None = None
        self._export_draft_probs = self._read_export_draft_probs()
        self._collect_confidence_diagnostics = (
            self._read_collect_confidence_diagnostics()
        )
        if self.confidence_threshold > 0.0:
            logger.info(
                "DSpark confidence-scheduled verification enabled with "
                "threshold %.4f.",
                self.confidence_threshold,
            )
        if self._export_draft_probs:
            logger.info(
                "DSpark draft probability export enabled for quality profiling. "
                "This adds a draft-logit softmax on greedy requests."
            )
        if self._collect_confidence_diagnostics:
            logger.info(
                "DSpark confidence diagnostics enabled. This copies confidence "
                "scores to CPU on every draft step."
            )

    @staticmethod
    def _read_confidence_threshold() -> float:
        raw = os.getenv("VLLM_DSPARK_CONFIDENCE_THRESHOLD", "0.0")
        try:
            threshold = float(raw)
        except ValueError as exc:
            raise ValueError(
                "VLLM_DSPARK_CONFIDENCE_THRESHOLD must be a float in [0, 1], "
                f"got {raw!r}"
            ) from exc
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(
                "VLLM_DSPARK_CONFIDENCE_THRESHOLD must be in [0, 1], "
                f"got {threshold}"
            )
        return threshold

    @staticmethod
    def _read_export_draft_probs() -> bool:
        raw = os.getenv("VLLM_DSPARK_EXPORT_DRAFT_PROBS", "0")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _read_collect_confidence_diagnostics() -> bool:
        raw = os.getenv("VLLM_DSPARK_COLLECT_CONFIDENCE_DIAGNOSTICS", "0")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

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
        del is_graph_capturing, slot_mappings
        batch_size = max(1, min(int(num_tokens), self.max_batch_size))
        (
            cudagraph_runtime_mode,
            padded_batch_size,
            num_tokens_across_dp,
            batch_descriptor,
        ) = self._determine_graph_batch(batch_size, use_cudagraphs=use_cudagraphs)
        self._prepare_draft_buffers(
            input_ids=torch.zeros(batch_size, dtype=torch.long, device=self.device),
            hidden_states=torch.zeros(
                batch_size,
                self.target_hidden_size,
                dtype=self.dtype,
                device=self.device,
            ),
            positions=torch.arange(batch_size, dtype=torch.long, device=self.device),
            padded_batch_size=padded_batch_size,
        )
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=padded_batch_size * self.num_speculative_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
        ):
            self._run_draft_for_current_context()

    @override
    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        if (
            not self.speculative_config.enforce_eager
            and cudagraph_mode.mixed_mode()
            in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        ):
            dspark_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            dspark_cudagraph_mode = CUDAGraphMode.NONE
        self.cudagraph_dispatcher.initialize_cudagraph_keys(dspark_cudagraph_mode)
        if dspark_cudagraph_mode != CUDAGraphMode.NONE and self.device.type == "cuda":
            self._draft_graph_runner = CUDAGraphWrapper(
                self._run_draft_from_buffers,
                self.vllm_config,
                runtime_mode=CUDAGraphMode.PIECEWISE,
            )

    def _determine_graph_batch(
        self,
        batch_size: int,
        *,
        use_cudagraphs: bool = True,
    ) -> tuple[CUDAGraphMode, int, torch.Tensor | None, BatchDescriptor]:
        cudagraph_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
            batch_size,
            valid_modes=({CUDAGraphMode.NONE} if not use_cudagraphs else None),
        )
        padded_batch_size = batch_descriptor.num_tokens
        num_tokens_across_dp = None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            from vllm.v1.worker.dp_utils import coordinate_batch_across_dp

            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=batch_size,
                    parallel_config=self.vllm_config.parallel_config,
                    allow_microbatching=False,
                    num_tokens_padded=padded_batch_size,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )
            assert not should_ubatch, "DBO ubatching not implemented for DSpark"
            if num_tokens_across_dp is not None:
                dp_rank = self.dp_rank
                padded_batch_size = int(num_tokens_across_dp[dp_rank].item())
                cudagraph_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
                    padded_batch_size,
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
                )
                assert batch_descriptor.num_tokens == padded_batch_size
                num_tokens_across_dp[dp_rank] = padded_batch_size
        return (
            cudagraph_mode,
            padded_batch_size,
            num_tokens_across_dp,
            batch_descriptor,
        )

    def _prepare_draft_buffers(
        self,
        *,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        padded_batch_size: int,
    ) -> None:
        batch_size = input_ids.shape[0]
        self._draft_graph_batch_size = padded_batch_size
        self._draft_input_ids_buffer[:batch_size].copy_(input_ids.to(torch.long))
        self._draft_hidden_buffer[:batch_size].copy_(hidden_states.to(self.dtype))
        self._draft_positions_buffer[:batch_size].copy_(positions.to(torch.long))
        if padded_batch_size > batch_size:
            pad_slice = slice(batch_size, padded_batch_size)
            self._draft_input_ids_buffer[pad_slice].fill_(self.noise_token_id)
            self._draft_hidden_buffer[pad_slice].zero_()
            self._draft_positions_buffer[pad_slice].zero_()

    def _run_draft_from_buffers(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = self._draft_graph_batch_size
        return self.model.draft_with_confidence(
            self._draft_input_ids_buffer[:batch_size],
            self._draft_hidden_buffer[:batch_size],
            self._draft_positions_buffer[:batch_size],
        )

    def _run_draft_for_current_context(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._draft_graph_runner is not None:
            return self._draft_graph_runner()
        return self._run_draft_from_buffers()

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

    def _trim_rejected_target_context(
        self,
        target_hidden_states: torch.Tensor,
        target_positions: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata | None,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Drop rejected verification suffixes before updating DSpark context.

        Padded speculative decoding keeps rejected tokens in the target forward
        as padding and expects proposers to ignore them. DSpark stores target
        hidden states in an internal context cache, so rejected suffix states
        must be removed before `prefill_main()`.
        """
        if num_rejected_tokens_gpu is None or common_attn_metadata is None:
            return target_hidden_states, target_positions

        rejected = num_rejected_tokens_gpu.detach().cpu().tolist()
        if not any(int(value) for value in rejected):
            return target_hidden_states, target_positions

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        if query_start_loc_cpu is None:
            query_start_loc_cpu = common_attn_metadata.query_start_loc.detach().cpu()
        query_starts = query_start_loc_cpu.tolist()
        if len(query_starts) != len(rejected) + 1:
            raise ValueError(
                "DSpark rejected-context trimming requires query_start_loc "
                "to have batch_size + 1 entries; got "
                f"{len(query_starts)} starts for {len(rejected)} requests."
            )

        hidden_chunks: list[torch.Tensor] = []
        position_chunks: list[torch.Tensor] = []
        effective_lengths: list[int] = []
        flat_positions = target_positions.reshape(-1)
        for req_index, num_rejected in enumerate(rejected):
            start = int(query_starts[req_index])
            end = int(query_starts[req_index + 1]) - int(num_rejected)
            if end <= start:
                raise ValueError(
                    "DSpark rejected-context trimming removed every token for "
                    f"request {req_index}: start={start}, end={end}."
                )
            hidden_chunks.append(target_hidden_states[start:end])
            position_chunks.append(flat_positions[start:end])
            effective_lengths.append(end - start)

        if len(set(effective_lengths)) != 1:
            raise ValueError(
                "DSpark currently requires uniform effective per-request "
                "target context lengths after rejection trimming; got "
                f"{effective_lengths}."
            )

        return torch.cat(hidden_chunks, dim=0), torch.cat(position_chunks, dim=0)

    def _warmup_drafts(self, batch_size: int) -> torch.Tensor:
        self._last_draft_lengths = [self.num_speculative_tokens] * batch_size
        return make_dspark_warmup_draft_token_ids(
            batch_size=batch_size,
            num_speculative_tokens=self.num_speculative_tokens,
            noise_token_id=self.noise_token_id,
            device=self.device,
        )

    def _draft_lengths_from_confidence(
        self,
        confidence_rows: list[list[float]],
    ) -> list[int]:
        if self.confidence_threshold <= 0.0:
            return [
                min(self.num_speculative_tokens, len(row))
                for row in confidence_rows
            ]
        return [
            confidence_threshold_prefix_length(
                row[: self.num_speculative_tokens],
                self.confidence_threshold,
            )
            for row in confidence_rows
        ]

    def _observe_confidence(self, confidence: torch.Tensor) -> list[int]:
        confidence_rows = confidence.detach().float().cpu().tolist()
        lengths = self._draft_lengths_from_confidence(confidence_rows)
        schedule = score_prefix_lengths(
            confidence_rows,
            lengths,
            steps_per_second=lambda _batch_tokens: 1.0,
        )
        self.diagnostics.observe(confidence_rows, schedule)
        return lengths

    def _should_observe_confidence(self) -> bool:
        return (
            self.confidence_threshold > 0.0
            or self._collect_confidence_diagnostics
        )

    def take_last_draft_lengths(self) -> list[int] | None:
        lengths = self._last_draft_lengths
        self._last_draft_lengths = None
        return lengths

    def take_last_draft_probs(self) -> torch.Tensor | None:
        draft_probs = self._last_draft_probs
        self._last_draft_probs = None
        return draft_probs

    def _maybe_store_draft_probs(
        self,
        draft_logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        batch_size: int,
    ) -> None:
        self._last_draft_probs = None
        if not getattr(self, "_export_draft_probs", False):
            return
        if not sampling_metadata.all_greedy:
            logger.warning_once(
                "VLLM_DSPARK_EXPORT_DRAFT_PROBS is currently limited to "
                "greedy DSpark requests because DSpark non-greedy drafting "
                "must sample the Markov-corrected block left-to-right."
            )
            return
        self._last_draft_probs = draft_logits[
            :batch_size, : self.num_speculative_tokens
        ].softmax(dim=-1, dtype=torch.float32)

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
            mm_embed_inputs,
            slot_mappings,
        )
        self._last_draft_probs = None
        batch_size = self._batch_size(next_token_ids)
        target_hidden_states, target_positions = self._trim_rejected_target_context(
            target_hidden_states,
            target_positions,
            common_attn_metadata,
            num_rejected_tokens_gpu,
        )
        hidden_by_req = self._view_by_request(target_hidden_states, batch_size)
        positions_by_req = self._positions_by_request(target_positions, batch_size)

        self.model.prefill_main(hidden_by_req, positions_by_req)
        if not self._prefilled:
            self._prefilled = True
            return self._warmup_drafts(batch_size)

        last_hidden = hidden_by_req[:, -1].contiguous()
        last_positions = positions_by_req[:, -1].contiguous()
        (
            cudagraph_runtime_mode,
            padded_batch_size,
            num_tokens_across_dp,
            batch_descriptor,
        ) = self._determine_graph_batch(batch_size)
        self._prepare_draft_buffers(
            input_ids=next_token_ids,
            hidden_states=last_hidden,
            positions=last_positions,
            padded_batch_size=padded_batch_size,
        )
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=padded_batch_size * self.num_speculative_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
        ):
            draft_token_ids, draft_logits, confidence = (
                self._run_draft_for_current_context()
            )
        self._maybe_store_draft_probs(draft_logits, sampling_metadata, batch_size)
        if confidence is not None and self._should_observe_confidence():
            self._last_draft_lengths = self._observe_confidence(
                confidence[:batch_size]
            )
        else:
            self._last_draft_lengths = [self.num_speculative_tokens] * batch_size
        return draft_token_ids[:batch_size, : self.num_speculative_tokens].to(
            torch.int32
        )

    def get_diagnostics_snapshot(self) -> Any:
        return self.diagnostics.snapshot()
