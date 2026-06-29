# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import torch
from typing_extensions import override

from vllm.compilation.cuda_graph import CUDAGraphOptions, CUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.dspark import (
    DSparkDiagnostics,
    confidence_threshold_prefix_length,
    full_prefix_dominates_sps_curve,
    hardware_aware_prefix_schedule,
    make_dspark_warmup_draft_token_ids,
    score_prefix_lengths,
)
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


@dataclass(frozen=True)
class _DSparkPrefillBatch:
    hidden: torch.Tensor
    positions: torch.Tensor
    num_rejected_tokens: torch.Tensor | None = None
    request_indices: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None
    valid_lengths: torch.Tensor | None = None

    @property
    def is_ragged(self) -> bool:
        return self.query_start_loc is not None


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
        self._draft_graph_pool: Any | None = None
        self._draft_graph_batch_size = 0
        self._draft_active_batch_size = 0
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
        self._draft_identity_slot_index_buffer = torch.arange(
            self.max_batch_size,
            dtype=torch.long,
            device=device,
        )
        self._draft_slot_index_buffer = self._draft_identity_slot_index_buffer.clone()
        self._req_id_to_slot: dict[str, int] = {}
        self._free_slots: list[int] = list(range(self.max_batch_size))
        self.diagnostics = DSparkDiagnostics(
            max_spec_tokens=self.num_speculative_tokens
        )
        self.confidence_threshold = self._read_confidence_threshold()
        self.confidence_scheduler = self._read_confidence_scheduler(
            self.confidence_threshold
        )
        self._forced_draft_length = self._read_forced_draft_length(
            self.num_speculative_tokens
        )
        self._sts_temperatures = self._read_sts_temperatures(
            self.num_speculative_tokens
        )
        self._sts_temperature_tensor = self._make_sts_temperature_tensor(
            self._sts_temperatures,
            device,
        )
        self._sps_curve = self._read_sps_curve()
        self._hardware_scheduler_early_stop = (
            self._read_hardware_scheduler_early_stop()
        )
        self._last_draft_lengths: list[int] | None = None
        self._last_draft_probs: torch.Tensor | None = None
        self._export_draft_probs = self._read_export_draft_probs()
        self._collect_confidence_diagnostics = (
            self._read_collect_confidence_diagnostics()
        )
        self._confidence_diagnostics_log_every = (
            self._read_confidence_diagnostics_log_every()
        )
        self._confidence_diagnostics_log_next = (
            self._confidence_diagnostics_log_every
        )
        self._collect_sts_calibration_diagnostics = (
            self._read_sts_calibration_diagnostics()
        )
        self._collect_position0_diagnostics = (
            self._read_position0_diagnostics()
        )
        self._logged_scheduler_confidence_bypass = False
        self._gpu_rejected_context_mask = self._read_gpu_rejected_context_mask()
        self._stage_timing = self._read_stage_timing()
        self._stage_timing_log_every = self._read_stage_timing_log_every()
        self._stage_timing_count = 0
        self._stage_timing_totals_ms: dict[str, float] = {}
        self._last_confidence: torch.Tensor | None = None
        self._last_raw_confidence: torch.Tensor | None = None
        if self.confidence_threshold > 0.0:
            logger.info(
                "DSpark confidence-scheduled verification enabled with "
                "threshold %.4f.",
                self.confidence_threshold,
            )
        if self.confidence_scheduler == "hardware":
            logger.info(
                "DSpark hardware-aware confidence scheduler enabled with "
                "early_stop=%s and SPS curve=%s.",
                self._hardware_scheduler_early_stop,
                self._sps_curve or "constant",
            )
        if self._forced_draft_length is not None:
            logger.info(
                "DSpark forced draft verification length enabled for "
                "profiling: %d.",
                self._forced_draft_length,
            )
        if self._sts_temperatures:
            logger.info(
                "DSpark STS confidence calibration enabled with temperatures %s.",
                self._sts_temperatures,
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
        if (
            self._confidence_diagnostics_log_every > 0
            and self._should_observe_confidence()
        ):
            logger.info(
                "DSpark confidence diagnostics will be logged every %d "
                "observed draft steps.",
                self._confidence_diagnostics_log_every,
            )
        if self._collect_position0_diagnostics:
            logger.info(
                "DSpark position-0 diagnostics enabled. The confidence head "
                "runs on every draft step; the runner logs first-token "
                "target-argmax agreement."
            )
        if self._collect_sts_calibration_diagnostics:
            logger.info(
                "DSpark STS calibration diagnostics enabled. Raw confidence "
                "rows are copied to CPU and paired with acceptance labels for "
                "offline temperature fitting."
            )
        if self._gpu_rejected_context_mask:
            logger.info(
                "DSpark GPU rejected-context mask enabled. Rejected target "
                "suffix rows are masked during draft main-KV cache update "
                "without synchronizing rejection counts to CPU."
            )
        if self._stage_timing:
            logger.info(
                "DSpark stage timing enabled. This synchronizes CUDA work and "
                "is intended for diagnostics, not speed-gate benchmarks."
            )
        if not self._needs_draft_logits() and not self._needs_confidence():
            logger.info(
                "DSpark fast draft-output mode enabled: confidence head and "
                "returned draft logits are skipped on the hot path."
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
    def _read_confidence_scheduler(confidence_threshold: float) -> str:
        raw = os.getenv("VLLM_DSPARK_CONFIDENCE_SCHEDULER", "auto")
        scheduler = raw.strip().lower()
        if scheduler in {"", "auto"}:
            return "threshold" if confidence_threshold > 0.0 else "off"
        if scheduler not in {"off", "threshold", "hardware"}:
            raise ValueError(
                "VLLM_DSPARK_CONFIDENCE_SCHEDULER must be one of "
                "'off', 'threshold', 'hardware', or 'auto', "
                f"got {raw!r}"
            )
        return scheduler

    @staticmethod
    def _read_forced_draft_length(max_draft_length: int) -> int | None:
        raw = os.getenv("VLLM_DSPARK_FORCE_DRAFT_LENGTH", "").strip()
        if raw == "":
            return None
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                "VLLM_DSPARK_FORCE_DRAFT_LENGTH must be an integer in "
                f"[0, {max_draft_length}] or empty, got {raw!r}"
            ) from exc
        if value < 0 or value > max_draft_length:
            raise ValueError(
                "VLLM_DSPARK_FORCE_DRAFT_LENGTH must be in "
                f"[0, {max_draft_length}], got {value}"
            )
        return value

    @staticmethod
    def _read_sps_curve() -> tuple[tuple[int, float], ...]:
        raw = os.getenv("VLLM_DSPARK_SPS_CURVE", "").strip()
        if not raw:
            return ()

        entries: dict[int, float] = {}
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                batch_tokens_raw, rate_raw = item.split(":", 1)
                batch_tokens = int(batch_tokens_raw)
                rate = float(rate_raw)
            except ValueError as exc:
                raise ValueError(
                    "VLLM_DSPARK_SPS_CURVE must be a comma-separated table "
                    "of '<batch_tokens>:<steps_per_second>' entries, "
                    f"got {raw!r}"
                ) from exc
            if batch_tokens <= 0:
                raise ValueError(
                    "VLLM_DSPARK_SPS_CURVE batch-token keys must be positive, "
                    f"got {batch_tokens}"
                )
            if rate < 0.0:
                raise ValueError(
                    "VLLM_DSPARK_SPS_CURVE rates must be non-negative, "
                    f"got {rate}"
                )
            entries[batch_tokens] = rate
        return tuple(sorted(entries.items()))

    @staticmethod
    def _read_sts_temperatures(
        max_draft_length: int,
    ) -> tuple[float, ...]:
        raw = os.getenv("VLLM_DSPARK_STS_TEMPERATURES", "").strip()
        if not raw:
            return ()

        temperatures: list[float] = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                temperature = float(item)
            except ValueError as exc:
                raise ValueError(
                    "VLLM_DSPARK_STS_TEMPERATURES must be a comma-separated "
                    "list of positive floats, "
                    f"got {raw!r}"
                ) from exc
            if temperature <= 0.0:
                raise ValueError(
                    "VLLM_DSPARK_STS_TEMPERATURES values must be positive, "
                    f"got {temperature}"
                )
            temperatures.append(temperature)

        if not temperatures:
            return ()
        if len(temperatures) not in {1, int(max_draft_length)}:
            raise ValueError(
                "VLLM_DSPARK_STS_TEMPERATURES must contain either one "
                f"temperature or {max_draft_length} per-position "
                f"temperatures, got {len(temperatures)}"
            )
        return tuple(temperatures)

    @staticmethod
    def _make_sts_temperature_tensor(
        temperatures: tuple[float, ...],
        device: torch.device,
    ) -> torch.Tensor | None:
        if len(temperatures) <= 1:
            return None
        return torch.tensor(
            temperatures,
            dtype=torch.float32,
            device=device,
        ).view(1, -1)

    @staticmethod
    def _read_hardware_scheduler_early_stop() -> bool:
        raw = os.getenv("VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP", "1")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _steps_per_second(self, batch_tokens: int) -> float:
        curve = getattr(self, "_sps_curve", ())
        if not curve:
            return 1.0

        batch_tokens = int(batch_tokens)
        selected_rate = curve[0][1]
        for profiled_tokens, rate in curve:
            if batch_tokens < profiled_tokens:
                break
            selected_rate = rate
        return selected_rate

    def _effective_confidence_scheduler(self) -> str:
        scheduler = getattr(self, "confidence_scheduler", None)
        if scheduler is not None:
            return scheduler
        threshold = getattr(self, "confidence_threshold", 0.0)
        return "threshold" if threshold > 0.0 else "off"

    def _confidence_request_count(self) -> int | None:
        active_batch_size = int(getattr(self, "_draft_active_batch_size", 0) or 0)
        if active_batch_size > 0:
            return active_batch_size
        graph_batch_size = int(getattr(self, "_draft_graph_batch_size", 0) or 0)
        if graph_batch_size > 0:
            return graph_batch_size
        return None

    def _hardware_scheduler_full_prefix_dominates(self) -> bool:
        request_count = self._confidence_request_count()
        if request_count is None:
            return False
        return full_prefix_dominates_sps_curve(
            request_count=request_count,
            max_spec_tokens=self.num_speculative_tokens,
            steps_per_second=self._steps_per_second,
        )

    def _scheduler_requires_confidence(self) -> bool:
        if getattr(self, "_forced_draft_length", None) is not None:
            return False

        scheduler = self._effective_confidence_scheduler()
        if scheduler == "off":
            return False
        if scheduler == "threshold":
            return self.confidence_threshold > 0.0
        if scheduler == "hardware":
            if self._hardware_scheduler_full_prefix_dominates():
                if not getattr(self, "_logged_scheduler_confidence_bypass", False):
                    logger.info(
                        "DSpark hardware scheduler bypassing confidence head: "
                        "full prefix dominates the profiled SPS curve for "
                        "request_count=%d.",
                        self._confidence_request_count(),
                    )
                    self._logged_scheduler_confidence_bypass = True
                return False
            return True
        return True

    @staticmethod
    def _read_export_draft_probs() -> bool:
        raw = os.getenv("VLLM_DSPARK_EXPORT_DRAFT_PROBS", "0")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _read_collect_confidence_diagnostics() -> bool:
        raw = os.getenv("VLLM_DSPARK_COLLECT_CONFIDENCE_DIAGNOSTICS", "0")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _read_confidence_diagnostics_log_every() -> int:
        raw = os.getenv("VLLM_DSPARK_CONFIDENCE_DIAGNOSTICS_LOG_EVERY", "0")
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                "VLLM_DSPARK_CONFIDENCE_DIAGNOSTICS_LOG_EVERY must be a "
                f"non-negative integer, got {raw!r}"
            ) from exc
        if value < 0:
            raise ValueError(
                "VLLM_DSPARK_CONFIDENCE_DIAGNOSTICS_LOG_EVERY must be "
                f"non-negative, got {value}"
            )
        return value

    @staticmethod
    def _read_position0_diagnostics() -> bool:
        raw = os.getenv("VLLM_DSPARK_POSITION0_DIAGNOSTICS", "0")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _read_sts_calibration_diagnostics() -> bool:
        raw = os.getenv("VLLM_DSPARK_STS_CALIBRATION_DIAGNOSTICS", "0")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _read_gpu_rejected_context_mask() -> bool:
        raw = os.getenv("VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK", "0")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _read_stage_timing() -> bool:
        raw = os.getenv("VLLM_DSPARK_STAGE_TIMING", "0")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _read_stage_timing_log_every() -> int:
        raw = os.getenv("VLLM_DSPARK_STAGE_TIMING_LOG_EVERY", "20")
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                "VLLM_DSPARK_STAGE_TIMING_LOG_EVERY must be an integer, "
                f"got {raw!r}"
            ) from exc
        return max(1, value)

    def _record_stage_timing(self, name: str, elapsed_ms: float) -> None:
        if not hasattr(self, "_stage_timing_totals_ms"):
            self._stage_timing_totals_ms = {}
        self._stage_timing_totals_ms[name] = (
            self._stage_timing_totals_ms.get(name, 0.0) + float(elapsed_ms)
        )

    def _timed_stage(self, name: str, fn):
        def run_stage():
            with record_function_or_nullcontext(f"dspark_proposer: {name}"):
                return fn()

        if not getattr(self, "_stage_timing", False):
            return run_stage()
        if self.device.type != "cuda":
            started = time.perf_counter()
            result = run_stage()
            self._record_stage_timing(name, (time.perf_counter() - started) * 1000.0)
            return result

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = run_stage()
        end.record()
        end.synchronize()
        self._record_stage_timing(name, start.elapsed_time(end))
        return result

    def _maybe_log_stage_timing(self) -> None:
        if not getattr(self, "_stage_timing", False):
            return
        self._stage_timing_count += 1
        if self._stage_timing_count % self._stage_timing_log_every != 0:
            return

        names = (
            "context_prepare",
            "prefill_project",
            "prefill_main",
            "graph_prepare",
            "draft",
            "postprocess",
            "total",
        )
        parts = []
        for name in names:
            total_ms = self._stage_timing_totals_ms.get(name, 0.0)
            parts.append(f"{name}={total_ms / self._stage_timing_count:.3f}ms")
        logger.info(
            "DSpark stage timing avg over %d proposals: %s",
            self._stage_timing_count,
            ", ".join(parts),
        )

    @staticmethod
    def _format_diagnostics_tuple(values: tuple[float, ...]) -> str:
        return "[" + ", ".join(f"{value:.3f}" for value in values) + "]"

    def _maybe_log_confidence_diagnostics(self) -> None:
        log_every = getattr(self, "_confidence_diagnostics_log_every", 0)
        if log_every <= 0:
            return

        snapshot = self.diagnostics.snapshot()
        log_next = getattr(self, "_confidence_diagnostics_log_next", log_every)
        if snapshot.num_steps < log_next:
            return

        logger.info(
            "DSpark confidence diagnostics: steps=%d, requests=%d, "
            "avg_scheduled_length=%.3f, prune_rate=%.3f, "
            "expected_acceptance_length=%.3f, "
            "expected_tokens_per_second=%.3f, scheduled_hist=%s, "
            "avg_confidence=%s, avg_raw_confidence=%s, calibration_delta=%s, "
            "survival=%s, scheduled_fraction=%s",
            snapshot.num_steps,
            snapshot.num_requests,
            snapshot.avg_scheduled_length,
            snapshot.draft_token_prune_rate,
            snapshot.expected_acceptance_length,
            snapshot.avg_expected_tokens_per_second,
            snapshot.scheduled_length_histogram,
            self._format_diagnostics_tuple(snapshot.avg_confidence_per_pos),
            self._format_diagnostics_tuple(snapshot.avg_raw_confidence_per_pos),
            self._format_diagnostics_tuple(
                snapshot.avg_confidence_calibration_delta_per_pos
            ),
            self._format_diagnostics_tuple(snapshot.avg_survival_per_pos),
            self._format_diagnostics_tuple(snapshot.scheduled_fraction_per_pos),
        )
        self._confidence_diagnostics_log_next = snapshot.num_steps + log_every

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
            cudagraph_options = CUDAGraphOptions()
            if getattr(self._runner, "_dspark_draft_stream_enabled", False):
                self._draft_graph_pool = current_platform.graph_pool_handle()
                cudagraph_options.graph_pool = self._draft_graph_pool
                logger.info(
                    "DSpark draft CUDA graphs use an isolated graph pool for "
                    "draft-stream replay."
                )
            self._draft_graph_runner = CUDAGraphWrapper(
                self._run_draft_from_buffers,
                self.vllm_config,
                runtime_mode=CUDAGraphMode.PIECEWISE,
                cudagraph_options=cudagraph_options,
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
        slot_index: torch.Tensor | None = None,
    ) -> None:
        batch_size = input_ids.shape[0]
        self._draft_active_batch_size = batch_size
        self._draft_graph_batch_size = padded_batch_size
        self._draft_input_ids_buffer[:batch_size].copy_(input_ids.to(torch.long))
        self._draft_hidden_buffer[:batch_size].copy_(hidden_states.to(self.dtype))
        self._draft_positions_buffer[:batch_size].copy_(positions.to(torch.long))
        if slot_index is None:
            self._draft_slot_index_buffer[:batch_size].copy_(
                self._draft_identity_slot_index_buffer[:batch_size]
            )
        else:
            slot_index = slot_index.to(
                device=self.device,
                dtype=torch.long,
                non_blocking=True,
            ).view(-1)
            if slot_index.shape[0] != batch_size:
                raise ValueError(
                    "DSpark slot_index must have one row per draft request; "
                    f"got {slot_index.shape[0]} indices for batch_size={batch_size}."
                )
            self._draft_slot_index_buffer[:batch_size].copy_(
                slot_index
            )
        if padded_batch_size > batch_size:
            pad_slice = slice(batch_size, padded_batch_size)
            self._draft_input_ids_buffer[pad_slice].fill_(self.noise_token_id)
            self._draft_hidden_buffer[pad_slice].zero_()
            self._draft_positions_buffer[pad_slice].zero_()
            self._draft_slot_index_buffer[pad_slice].zero_()

    def _run_draft_from_buffers(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = self._draft_graph_batch_size
        return self.model.draft_with_confidence(
            self._draft_input_ids_buffer[:batch_size],
            self._draft_hidden_buffer[:batch_size],
            self._draft_positions_buffer[:batch_size],
            return_logits=self._needs_draft_logits(),
            return_confidence=self._needs_confidence(),
            store_main_kv=False,
            slot_index=self._draft_slot_index_buffer[:batch_size],
        )

    def _run_draft_for_current_context(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._draft_graph_runner is not None:
            return self._draft_graph_runner()
        return self._run_draft_from_buffers()

    def _batch_size(self, next_token_ids: torch.Tensor) -> int:
        return int(next_token_ids.shape[0])

    def _row_to_slot(self, req_ids: list[str]) -> list[int]:
        live_req_ids = set(req_ids)
        for stale_req_id in [
            req_id for req_id in self._req_id_to_slot if req_id not in live_req_ids
        ]:
            self._free_slots.append(self._req_id_to_slot.pop(stale_req_id))
        self._free_slots.sort()

        slots: list[int] = []
        for req_id in req_ids:
            slot = self._req_id_to_slot.get(req_id)
            if slot is None:
                if not self._free_slots:
                    raise ValueError(
                        "DSpark request-stable slot map exhausted "
                        f"for batch_size={len(req_ids)} and "
                        f"max_batch_size={self.max_batch_size}."
                    )
                slot = self._free_slots.pop(0)
                self._req_id_to_slot[req_id] = slot
            slots.append(slot)
        return slots

    def _prefill_request_indices_for_slots(
        self,
        request_indices: torch.Tensor | None,
        slot_index: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if slot_index is None:
            return request_indices
        if request_indices is None:
            return slot_index
        group_indices = request_indices.to(
            device=slot_index.device,
            dtype=torch.long,
            non_blocking=True,
        ).view(-1)
        return slot_index.index_select(0, group_indices)

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
                "DSpark cannot infer ragged request boundaries after rejected "
                "context trimming without attention metadata; got effective "
                f"lengths {effective_lengths}."
            )

        return torch.cat(hidden_chunks, dim=0), torch.cat(position_chunks, dim=0)

    def _query_starts_cpu(
        self,
        common_attn_metadata: CommonAttentionMetadata | None,
        batch_size: int,
        num_target_rows: int,
    ) -> list[int] | None:
        if common_attn_metadata is None:
            return None
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        if query_start_loc_cpu is None:
            query_start_loc_cpu = common_attn_metadata.query_start_loc.detach().cpu()
        query_starts = [int(value) for value in query_start_loc_cpu.tolist()]
        if len(query_starts) != batch_size + 1:
            raise ValueError(
                "DSpark target context batching requires query_start_loc to "
                "have batch_size + 1 entries; got "
                f"{len(query_starts)} starts for {batch_size} requests."
            )
        if query_starts[0] != 0 or query_starts[-1] != num_target_rows:
            raise ValueError(
                "DSpark target context batching received inconsistent "
                f"query_start_loc={query_starts} for {num_target_rows} rows."
            )
        return query_starts

    def _rejected_counts_cpu(
        self,
        num_rejected_tokens_gpu: torch.Tensor | None,
        batch_size: int,
    ) -> list[int]:
        if num_rejected_tokens_gpu is None:
            return [0] * batch_size
        rejected = [
            int(value)
            for value in num_rejected_tokens_gpu.detach().cpu().reshape(-1).tolist()
        ]
        if len(rejected) != batch_size:
            raise ValueError(
                "DSpark rejected-token metadata must have one value per "
                f"request; got {len(rejected)} values for batch_size={batch_size}."
            )
        return rejected

    def _prepare_target_context_batches(
        self,
        target_hidden_states: torch.Tensor,
        target_positions: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata | None,
        num_rejected_tokens_gpu: torch.Tensor | None,
        batch_size: int,
    ) -> tuple[
        list[_DSparkPrefillBatch],
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build DSpark main-cache update batches while preserving request rows."""
        use_gpu_rejected_mask = (
            getattr(self, "_gpu_rejected_context_mask", False)
            and num_rejected_tokens_gpu is not None
        )
        rejected_for_gpu_mask = (
            num_rejected_tokens_gpu if use_gpu_rejected_mask else None
        )
        query_start_loc = None
        query_start_loc_cpu = None
        if common_attn_metadata is not None:
            query_start_loc = common_attn_metadata.query_start_loc
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            if query_start_loc.numel() != batch_size + 1:
                raise ValueError(
                    "DSpark target context batching requires query_start_loc "
                    "to have batch_size + 1 entries; got "
                    f"{query_start_loc.numel()} starts for {batch_size} "
                    "requests."
                )
            if query_start_loc_cpu.numel() != batch_size + 1:
                raise ValueError(
                    "DSpark target context batching requires CPU "
                    "query_start_loc to have batch_size + 1 entries; got "
                    f"{query_start_loc_cpu.numel()} starts for {batch_size} "
                    "requests."
                )
            if int(query_start_loc_cpu[0]) != 0 or int(
                query_start_loc_cpu[-1]
            ) != int(target_hidden_states.shape[0]):
                raise ValueError(
                    "DSpark target context batching received inconsistent "
                    f"query_start_loc boundaries for "
                    f"{int(target_hidden_states.shape[0])} rows."
                )

        if query_start_loc is None:
            if not use_gpu_rejected_mask:
                target_hidden_states, target_positions = (
                    self._trim_rejected_target_context(
                        target_hidden_states,
                        target_positions,
                        common_attn_metadata,
                        num_rejected_tokens_gpu,
                    )
                )
            hidden_by_req = self._view_by_request(target_hidden_states, batch_size)
            positions_by_req = self._positions_by_request(
                target_positions,
                batch_size,
            )

            if rejected_for_gpu_mask is not None:
                rejected = rejected_for_gpu_mask.to(
                    device=hidden_by_req.device,
                    dtype=torch.long,
                    non_blocking=True,
                ).view(batch_size)
                last_indices = (hidden_by_req.shape[1] - rejected - 1).clamp(min=0)
                last_hidden = hidden_by_req.gather(
                    1,
                    last_indices.view(batch_size, 1, 1).expand(
                        -1,
                        -1,
                        hidden_by_req.shape[-1],
                    ),
                ).squeeze(1).contiguous()
                last_positions = positions_by_req.gather(
                    1,
                    last_indices.view(batch_size, 1),
                ).squeeze(1).contiguous()
            else:
                last_hidden = hidden_by_req[:, -1].contiguous()
                last_positions = positions_by_req[:, -1].contiguous()
            return (
                [
                    _DSparkPrefillBatch(
                        hidden_by_req,
                        positions_by_req,
                        num_rejected_tokens=rejected_for_gpu_mask,
                    )
                ],
                last_hidden,
                last_positions,
            )

        assert query_start_loc is not None
        query_start_loc = query_start_loc.to(
            device=target_hidden_states.device,
            dtype=torch.long,
            non_blocking=True,
        ).view(-1)
        assert query_start_loc_cpu is not None
        raw_lengths_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        raw_lengths = query_start_loc[1:] - query_start_loc[:-1]
        if num_rejected_tokens_gpu is None:
            rejected_lengths = torch.zeros_like(raw_lengths)
        else:
            rejected_lengths = num_rejected_tokens_gpu.to(
                device=target_hidden_states.device,
                dtype=torch.long,
                non_blocking=True,
            ).view(batch_size)
        valid_lengths = raw_lengths - rejected_lengths
        if num_rejected_tokens_gpu is None:
            invalid = raw_lengths_cpu <= 0
            if bool(invalid.any().item()):
                raw_list = [int(value) for value in raw_lengths_cpu.tolist()]
                rejected_list = [0] * batch_size
                raise ValueError(
                    "DSpark target context batching requires at least one valid "
                    "target token per request; got raw_lengths="
                    f"{raw_list}, rejected={rejected_list}."
                )
        elif not rejected_lengths.is_cuda:
            invalid = (
                (raw_lengths <= 0)
                | (rejected_lengths < 0)
                | (rejected_lengths >= raw_lengths)
                | (valid_lengths <= 0)
            )
            if bool(invalid.any().item()):
                raw_list = raw_lengths.tolist()
                rejected_list = rejected_lengths.tolist()
                raise ValueError(
                    "DSpark target context batching requires at least one valid "
                    "target token per request; got raw_lengths="
                    f"{raw_list}, rejected={rejected_list}."
                )
        elif getattr(self, "_stage_timing", False):
            invalid = (
                (raw_lengths <= 0)
                | (rejected_lengths < 0)
                | (rejected_lengths >= raw_lengths)
                | (valid_lengths <= 0)
            )
            if bool(invalid.any().item()):
                raw_list = raw_lengths.detach().cpu().tolist()
                rejected_list = rejected_lengths.detach().cpu().tolist()
                raise ValueError(
                    "DSpark target context batching requires at least one valid "
                    "target token per request; got raw_lengths="
                    f"{raw_list}, rejected={rejected_list}."
                )
        elif bool((raw_lengths_cpu <= 0).any().item()):
            raw_list = [int(value) for value in raw_lengths_cpu.tolist()]
            rejected_list = ["gpu"] * batch_size
            raise ValueError(
                "DSpark target context batching requires at least one valid "
                "target token per request; got raw_lengths="
                f"{raw_list}, rejected={rejected_list}."
            )

        flat_positions = target_positions.reshape(-1)
        last_indices = query_start_loc[:-1] + valid_lengths - 1
        last_hidden = target_hidden_states.index_select(
            0,
            last_indices.to(dtype=torch.long),
        ).contiguous()
        last_positions = flat_positions.index_select(
            0,
            last_indices.to(dtype=torch.long),
        ).contiguous()

        is_ragged = bool((raw_lengths_cpu != raw_lengths_cpu[0]).any().item())
        if not is_ragged:
            raw_len = int(raw_lengths_cpu[0])
            hidden_by_req = target_hidden_states.reshape(
                batch_size,
                raw_len,
                target_hidden_states.shape[-1],
            )
            positions_by_req = flat_positions.reshape(batch_size, raw_len)
            rejected_for_prefill = (
                rejected_lengths.to(dtype=torch.int32)
                if num_rejected_tokens_gpu is not None
                else None
            )
            return (
                [
                    _DSparkPrefillBatch(
                        hidden_by_req.contiguous(),
                        positions_by_req.contiguous(),
                        num_rejected_tokens=rejected_for_prefill,
                    )
                ],
                last_hidden,
                last_positions,
            )

        return (
            [
                _DSparkPrefillBatch(
                    target_hidden_states,
                    flat_positions,
                    query_start_loc=query_start_loc,
                    valid_lengths=valid_lengths,
                )
            ],
            last_hidden,
            last_positions,
        )

    def _warmup_drafts(self, batch_size: int) -> torch.Tensor:
        self._last_draft_lengths = None
        return make_dspark_warmup_draft_token_ids(
            batch_size=batch_size,
            num_speculative_tokens=self.num_speculative_tokens,
            noise_token_id=self.noise_token_id,
            device=self.device,
        )

    def _set_last_draft_lengths(self, lengths: list[int]) -> None:
        if all(int(length) == self.num_speculative_tokens for length in lengths):
            self._last_draft_lengths = None
            return
        self._last_draft_lengths = [int(length) for length in lengths]

    def _calibrate_confidence(self, confidence: torch.Tensor) -> torch.Tensor:
        temperatures = getattr(self, "_sts_temperatures", ())
        if not temperatures or confidence.numel() == 0:
            return confidence

        confidence_f = confidence.float().clamp(1.0e-6, 1.0 - 1.0e-6)
        logits = torch.logit(confidence_f)
        if len(temperatures) == 1:
            return torch.sigmoid(logits / float(temperatures[0]))

        temp_tensor = getattr(self, "_sts_temperature_tensor", None)
        if (
            temp_tensor is None
            or temp_tensor.device != confidence.device
            or temp_tensor.shape[1] < confidence.shape[1]
        ):
            temp_tensor = self._make_sts_temperature_tensor(
                temperatures,
                confidence.device,
            )
            self._sts_temperature_tensor = temp_tensor
        assert temp_tensor is not None
        return torch.sigmoid(logits / temp_tensor[:, : confidence.shape[1]])

    def _draft_lengths_from_confidence(
        self,
        confidence_rows: list[list[float]],
    ) -> list[int]:
        return list(self._schedule_from_confidence(confidence_rows).lengths)

    def _schedule_from_confidence(
        self,
        confidence_rows: list[list[float]],
    ):
        confidence_rows = [
            row[: self.num_speculative_tokens] for row in confidence_rows
        ]
        forced_length = getattr(self, "_forced_draft_length", None)
        if forced_length is not None:
            lengths = [
                min(int(forced_length), self.num_speculative_tokens, len(row))
                for row in confidence_rows
            ]
            return score_prefix_lengths(
                confidence_rows,
                lengths,
                steps_per_second=self._steps_per_second,
            )

        scheduler = self._effective_confidence_scheduler()

        if scheduler == "hardware":
            return hardware_aware_prefix_schedule(
                confidence_rows,
                steps_per_second=self._steps_per_second,
                early_stop=getattr(self, "_hardware_scheduler_early_stop", True),
            )

        if scheduler == "off" or self.confidence_threshold <= 0.0:
            lengths = [
                min(self.num_speculative_tokens, len(row))
                for row in confidence_rows
            ]
        else:
            lengths = [
                confidence_threshold_prefix_length(
                    row,
                    self.confidence_threshold,
                )
                for row in confidence_rows
            ]
        return score_prefix_lengths(
            confidence_rows,
            lengths,
            steps_per_second=self._steps_per_second,
        )

    def _observe_confidence(
        self,
        confidence: torch.Tensor,
        raw_confidence: torch.Tensor | None = None,
    ) -> list[int]:
        confidence_rows = confidence.detach().float().cpu().tolist()
        raw_confidence_rows = (
            None
            if raw_confidence is None
            else raw_confidence.detach().float().cpu().tolist()
        )
        schedule = self._schedule_from_confidence(confidence_rows)
        self.diagnostics.observe(
            confidence_rows,
            schedule,
            raw_confidence_rows=raw_confidence_rows,
        )
        self._maybe_log_confidence_diagnostics()
        return list(schedule.lengths)

    def _should_observe_confidence(self) -> bool:
        return (
            self._scheduler_requires_confidence()
            or getattr(self, "_collect_confidence_diagnostics", False)
        )

    def _needs_confidence(self) -> bool:
        return (
            self._should_observe_confidence()
            or getattr(self, "_collect_position0_diagnostics", False)
            or getattr(self, "_collect_sts_calibration_diagnostics", False)
        )

    def _needs_draft_logits(self) -> bool:
        return self._export_draft_probs

    def take_last_draft_lengths(self) -> list[int] | None:
        lengths = self._last_draft_lengths
        self._last_draft_lengths = None
        return lengths

    def take_last_draft_probs(self) -> torch.Tensor | None:
        draft_probs = self._last_draft_probs
        self._last_draft_probs = None
        return draft_probs

    def take_last_confidence(self) -> torch.Tensor | None:
        confidence = self._last_confidence
        self._last_confidence = None
        return confidence

    def take_last_raw_confidence(self) -> torch.Tensor | None:
        confidence = self._last_raw_confidence
        self._last_raw_confidence = None
        return confidence

    def _maybe_store_draft_probs(
        self,
        draft_logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        batch_size: int,
    ) -> None:
        self._last_draft_probs = None
        if not getattr(self, "_export_draft_probs", False):
            return
        if draft_logits.numel() == 0:
            logger.warning_once(
                "DSpark draft probability export requested but draft logits "
                "were not returned by the draft model."
            )
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
        req_ids: list[str] | None = None,
    ) -> torch.Tensor:
        del (
            target_token_ids,
            token_indices_to_sample,
            mm_embed_inputs,
            slot_mappings,
        )
        self._last_draft_probs = None
        self._last_confidence = None
        self._last_raw_confidence = None
        total_started = time.perf_counter()
        batch_size = self._batch_size(next_token_ids)
        slot_index: torch.Tensor | None = None
        if req_ids is not None:
            if len(req_ids) != batch_size:
                raise ValueError(
                    "DSpark req_ids must have one entry per draft request; "
                    f"got {len(req_ids)} ids for batch_size={batch_size}."
                )
            slot_list = self._row_to_slot(req_ids)
            if slot_list != list(range(batch_size)):
                slot_index = torch.tensor(
                    slot_list,
                    device=self.device,
                    dtype=torch.long,
                )

        def prepare_context():
            return self._prepare_target_context_batches(
                target_hidden_states,
                target_positions,
                common_attn_metadata,
                num_rejected_tokens_gpu,
                batch_size,
            )

        (
            prefill_batches,
            last_hidden,
            last_positions,
        ) = self._timed_stage("context_prepare", prepare_context)

        def project_prefill_batches() -> list[
            tuple[_DSparkPrefillBatch, torch.Tensor | None]
        ]:
            project_main_context = getattr(self.model, "project_main_context", None)
            prefill_main_projected = getattr(self.model, "prefill_main_projected", None)
            prefill_main_ragged_projected = getattr(
                self.model,
                "prefill_main_ragged_projected",
                None,
            )
            if (
                project_main_context is None
                or prefill_main_projected is None
                or prefill_main_ragged_projected is None
            ):
                return [(batch, None) for batch in prefill_batches]

            return [
                (batch, project_main_context(batch.hidden))
                for batch in prefill_batches
            ]

        projected_prefill_batches = self._timed_stage(
            "prefill_project",
            project_prefill_batches,
        )

        def prefill_main():
            for batch, projected_hidden in projected_prefill_batches:
                prefill_request_indices = self._prefill_request_indices_for_slots(
                    batch.request_indices,
                    slot_index,
                )
                if batch.is_ragged:
                    assert batch.query_start_loc is not None
                    assert batch.valid_lengths is not None
                    if projected_hidden is None:
                        self.model.prefill_main_ragged(
                            batch.hidden,
                            batch.positions,
                            batch.query_start_loc,
                            batch.valid_lengths,
                            request_indices=prefill_request_indices,
                        )
                    else:
                        self.model.prefill_main_ragged_projected(
                            projected_hidden,
                            batch.positions,
                            batch.query_start_loc,
                            batch.valid_lengths,
                            request_indices=prefill_request_indices,
                        )
                elif prefill_request_indices is None:
                    if projected_hidden is None:
                        self.model.prefill_main(
                            batch.hidden,
                            batch.positions,
                            num_rejected_tokens=batch.num_rejected_tokens,
                        )
                    else:
                        self.model.prefill_main_projected(
                            projected_hidden,
                            batch.positions,
                            num_rejected_tokens=batch.num_rejected_tokens,
                        )
                else:
                    if projected_hidden is None:
                        self.model.prefill_main(
                            batch.hidden,
                            batch.positions,
                            num_rejected_tokens=batch.num_rejected_tokens,
                            request_indices=prefill_request_indices,
                        )
                    else:
                        self.model.prefill_main_projected(
                            projected_hidden,
                            batch.positions,
                            num_rejected_tokens=batch.num_rejected_tokens,
                            request_indices=prefill_request_indices,
                        )

        self._timed_stage(
            "prefill_main",
            prefill_main,
        )
        if not self._prefilled:
            self._prefilled = True
            return self._warmup_drafts(batch_size)

        def prepare_graph():
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
                slot_index=slot_index,
            )
            return (
                cudagraph_runtime_mode,
                padded_batch_size,
                num_tokens_across_dp,
                batch_descriptor,
            )

        (
            cudagraph_runtime_mode,
            padded_batch_size,
            num_tokens_across_dp,
            batch_descriptor,
        ) = self._timed_stage("graph_prepare", prepare_graph)
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=padded_batch_size * self.num_speculative_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
        ):
            draft_token_ids, draft_logits, confidence = (
                self._timed_stage("draft", self._run_draft_for_current_context)
            )

        def postprocess():
            self._maybe_store_draft_probs(draft_logits, sampling_metadata, batch_size)
            confidence_for_batch = None
            raw_confidence_for_batch = None
            if confidence is not None and confidence.numel() > 0:
                raw_confidence_for_batch = confidence[
                    :batch_size, : self.num_speculative_tokens
                ]
                confidence_for_batch = self._calibrate_confidence(
                    raw_confidence_for_batch
                )
                if getattr(self, "_collect_sts_calibration_diagnostics", False):
                    self._last_raw_confidence = (
                        raw_confidence_for_batch.detach().clone()
                    )
                if getattr(self, "_collect_position0_diagnostics", False):
                    self._last_confidence = confidence_for_batch.detach().clone()
            forced_length = getattr(self, "_forced_draft_length", None)
            if forced_length is not None:
                self._set_last_draft_lengths([int(forced_length)] * batch_size)
            elif (
                confidence_for_batch is not None
                and self._should_observe_confidence()
            ):
                self._set_last_draft_lengths(
                    self._observe_confidence(
                        confidence_for_batch,
                        raw_confidence=raw_confidence_for_batch,
                    )
                )
            else:
                self._last_draft_lengths = None
            return draft_token_ids[:batch_size, : self.num_speculative_tokens].to(
                torch.int32
            )

        result = self._timed_stage("postprocess", postprocess)
        self._record_stage_timing(
            "total",
            (time.perf_counter() - total_started) * 1000.0,
        )
        self._maybe_log_stage_timing()
        return result

    def get_diagnostics_snapshot(self) -> Any:
        return self.diagnostics.snapshot()
