# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import prometheus_client

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger
from vllm.v1.metrics.utils import create_metric_per_engine

logger = init_logger(__name__)

_DSPARK_STS_CALIBRATION_BINS = 10


def _new_calibration_matrix(num_spec_tokens: int, value: int = 0) -> list[list[int]]:
    return [[value] * _DSPARK_STS_CALIBRATION_BINS for _ in range(num_spec_tokens)]


def _new_calibration_float_matrix(num_spec_tokens: int) -> list[list[float]]:
    return [[0.0] * _DSPARK_STS_CALIBRATION_BINS for _ in range(num_spec_tokens)]


def _add_int_matrix(dst: list[list[int]], src: list[list[int]]) -> None:
    for row_idx, row in enumerate(src):
        for col_idx, value in enumerate(row):
            dst[row_idx][col_idx] += int(value)


def _add_float_matrix(dst: list[list[float]], src: list[list[float]]) -> None:
    for row_idx, row in enumerate(src):
        for col_idx, value in enumerate(row):
            dst[row_idx][col_idx] += float(value)


@dataclass
class SpecDecodingStats:
    """Per-step iteration decoding stats from scheduler.

    Each scheduler step, statistics on spec decoding performance are
    aggregated across requests by the scheduler and returned to the
    frontend in EngineCoreOutputs->SchedulerStats.
    """

    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)
    num_drafts_by_draft_length: list[int] = field(default_factory=list)
    dspark_confidence_bin_counts: list[list[int]] | None = None
    dspark_confidence_bin_accepted: list[list[int]] | None = None
    dspark_confidence_bin_sums: list[list[float]] | None = None

    @classmethod
    def new(cls, num_spec_tokens: int) -> "SpecDecodingStats":
        return cls(
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens_per_pos=[0] * num_spec_tokens,
            num_drafts_by_draft_length=[0] * (num_spec_tokens + 1),
        )

    def observe_draft(
        self,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        dspark_confidence: Sequence[float] | None = None,
    ):
        assert 0 <= num_draft_tokens <= self.num_spec_tokens
        assert num_accepted_tokens <= self.num_spec_tokens
        self.num_drafts += 1
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        self.num_drafts_by_draft_length[num_draft_tokens] += 1
        for i in range(num_accepted_tokens):
            self.num_accepted_tokens_per_pos[i] += 1
        self.observe_dspark_confidence(
            dspark_confidence,
            num_draft_tokens,
            num_accepted_tokens,
        )

    def observe_dspark_confidence(
        self,
        confidence: Sequence[float] | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> None:
        if confidence is None:
            return
        if self.dspark_confidence_bin_counts is None:
            self.dspark_confidence_bin_counts = _new_calibration_matrix(
                self.num_spec_tokens
            )
            self.dspark_confidence_bin_accepted = _new_calibration_matrix(
                self.num_spec_tokens
            )
            self.dspark_confidence_bin_sums = _new_calibration_float_matrix(
                self.num_spec_tokens
            )

        assert self.dspark_confidence_bin_accepted is not None
        assert self.dspark_confidence_bin_sums is not None
        num_positions = min(num_draft_tokens, len(confidence), self.num_spec_tokens)
        for position in range(num_positions):
            value = float(confidence[position])
            if not math.isfinite(value):
                continue
            clamped = min(max(value, 0.0), 1.0)
            bin_index = min(
                int(clamped * _DSPARK_STS_CALIBRATION_BINS),
                _DSPARK_STS_CALIBRATION_BINS - 1,
            )
            self.dspark_confidence_bin_counts[position][bin_index] += 1
            self.dspark_confidence_bin_sums[position][bin_index] += clamped
            if position < num_accepted_tokens:
                self.dspark_confidence_bin_accepted[position][bin_index] += 1


class SpecDecodingLogging:
    """Aggregate and log spec decoding metrics.

    LoggingStatLogger aggregates per-iteration metrics over a set
    time interval using observe() and then logs them using log()
    before resetting to zero.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.num_drafts: list[int] = []
        self.num_draft_tokens: list[int] = []
        self.num_accepted_tokens: list[int] = []
        self.accepted_tokens_per_pos_lists: list[list[int]] = []
        self.drafts_by_draft_length_lists: list[list[int]] = []
        self.dspark_confidence_bin_counts: list[list[int]] | None = None
        self.dspark_confidence_bin_accepted: list[list[int]] | None = None
        self.dspark_confidence_bin_sums: list[list[float]] | None = None
        self.last_log_time = time.monotonic()

    def observe(self, spec_decoding_stats: SpecDecodingStats):
        self.num_drafts.append(spec_decoding_stats.num_drafts)
        self.num_draft_tokens.append(spec_decoding_stats.num_draft_tokens)
        self.num_accepted_tokens.append(spec_decoding_stats.num_accepted_tokens)
        self.accepted_tokens_per_pos_lists.append(
            spec_decoding_stats.num_accepted_tokens_per_pos
        )
        self.drafts_by_draft_length_lists.append(
            spec_decoding_stats.num_drafts_by_draft_length
        )
        counts = spec_decoding_stats.dspark_confidence_bin_counts
        accepted = spec_decoding_stats.dspark_confidence_bin_accepted
        sums = spec_decoding_stats.dspark_confidence_bin_sums
        if counts is not None and any(sum(row) for row in counts):
            if self.dspark_confidence_bin_counts is None:
                self.dspark_confidence_bin_counts = _new_calibration_matrix(
                    spec_decoding_stats.num_spec_tokens
                )
                self.dspark_confidence_bin_accepted = _new_calibration_matrix(
                    spec_decoding_stats.num_spec_tokens
                )
                self.dspark_confidence_bin_sums = _new_calibration_float_matrix(
                    spec_decoding_stats.num_spec_tokens
                )
            assert self.dspark_confidence_bin_accepted is not None
            assert self.dspark_confidence_bin_sums is not None
            assert accepted is not None
            assert sums is not None
            _add_int_matrix(self.dspark_confidence_bin_counts, counts)
            _add_int_matrix(self.dspark_confidence_bin_accepted, accepted)
            _add_float_matrix(self.dspark_confidence_bin_sums, sums)

    def log(self, log_fn=logger.info):
        if not self.num_drafts:
            return
        num_drafts = np.sum(self.num_drafts)
        num_draft_tokens = np.sum(self.num_draft_tokens)
        num_accepted_tokens = np.sum(self.num_accepted_tokens)
        draft_throughput = 0
        accepted_throughput = 0

        elapsed_time = time.monotonic() - self.last_log_time
        if elapsed_time > 0:
            draft_throughput = num_draft_tokens / elapsed_time
            accepted_throughput = num_accepted_tokens / elapsed_time

        draft_acceptance_rate = (
            num_accepted_tokens / num_draft_tokens * 100
            if num_draft_tokens > 0
            else float("nan")
        )

        # Conventionally, mean acceptance length includes the bonus token
        mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)

        pos_matrix = np.array(self.accepted_tokens_per_pos_lists)
        acceptance_rates = np.sum(pos_matrix, axis=0) / num_drafts
        rates_str = ", ".join(f"{p:.3f}" for p in acceptance_rates)
        length_matrix = np.array(self.drafts_by_draft_length_lists)
        draft_length_histogram = np.sum(length_matrix, axis=0)
        draft_length_histogram_str = ", ".join(
            f"{length}:{count}"
            for length, count in enumerate(draft_length_histogram)
            if count
        )

        log_fn(
            "SpecDecoding metrics: "
            "Mean acceptance length: %.2f, "
            "Accepted throughput: %.2f tokens/s, "
            "Drafted throughput: %.2f tokens/s, "
            "Accepted: %d tokens, "
            "Drafted: %d tokens, "
            "Per-position acceptance rate: %s, "
            "Draft length histogram: %s, "
            "Avg Draft acceptance rate: %.1f%%",
            mean_acceptance_length,
            accepted_throughput,
            draft_throughput,
            num_accepted_tokens,
            num_draft_tokens,
            rates_str,
            draft_length_histogram_str,
            draft_acceptance_rate,
        )
        self._log_dspark_sts_calibration(log_fn)
        self.reset()

    def _log_dspark_sts_calibration(self, log_fn=logger.info):
        if self.dspark_confidence_bin_counts is None:
            return

        assert self.dspark_confidence_bin_accepted is not None
        assert self.dspark_confidence_bin_sums is not None
        payload = {
            "bins": _DSPARK_STS_CALIBRATION_BINS,
            "counts": self.dspark_confidence_bin_counts,
            "accepted": self.dspark_confidence_bin_accepted,
            "confidence_sums": [
                [round(value, 6) for value in row]
                for row in self.dspark_confidence_bin_sums
            ],
        }
        log_fn(
            "DSpark STS calibration bins: %s",
            json.dumps(payload, separators=(",", ":")),
        )


class SpecDecodingProm:
    """Record spec decoding metrics in Prometheus.

    The acceptance rate can be calculated using a PromQL query:

      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_draft_tokens_total[$interval])

    The mean acceptance length (conventionally including bonus tokens)
    can be calculated using:

      1 + (
      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_drafts[$interval]))

    A per-position acceptance rate vector can be computed using

      vllm:spec_decode_num_accepted_tokens_per_pos[$interval] /
      vllm:spec_decode_num_drafts[$interval]
    """

    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        speculative_config: SpeculativeConfig | None,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        self.spec_decoding_enabled = speculative_config is not None
        if not self.spec_decoding_enabled:
            return

        counter_drafts = self._counter_cls(
            name="vllm:spec_decode_num_drafts",
            documentation="Number of spec decoding drafts.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_drafts = create_metric_per_engine(
            counter_drafts, per_engine_labelvalues
        )

        counter_draft_tokens = self._counter_cls(
            name="vllm:spec_decode_num_draft_tokens",
            documentation="Number of draft tokens.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_draft_tokens = create_metric_per_engine(
            counter_draft_tokens, per_engine_labelvalues
        )

        counter_accepted_tokens = self._counter_cls(
            name="vllm:spec_decode_num_accepted_tokens",
            documentation="Number of accepted tokens.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_accepted_tokens = create_metric_per_engine(
            counter_accepted_tokens, per_engine_labelvalues
        )

        assert speculative_config is not None
        num_spec_tokens = (
            speculative_config.num_speculative_tokens
            if self.spec_decoding_enabled
            else 0
        )
        pos_labelnames = labelnames + ["position"]
        base_counter = self._counter_cls(
            name="vllm:spec_decode_num_accepted_tokens_per_pos",
            documentation="Accepted tokens per draft position.",
            labelnames=pos_labelnames,
        )
        self.counter_spec_decode_num_accepted_tokens_per_pos: dict[
            int, list[prometheus_client.Counter]
        ] = {
            idx: [base_counter.labels(*lv, str(pos)) for pos in range(num_spec_tokens)]
            for idx, lv in per_engine_labelvalues.items()
        }

        draft_length_labelnames = labelnames + ["draft_length"]
        draft_length_counter = self._counter_cls(
            name="vllm:spec_decode_num_drafts_by_draft_length",
            documentation="Number of spec decoding drafts by scheduled draft length.",
            labelnames=draft_length_labelnames,
        )
        self.counter_spec_decode_num_drafts_by_draft_length: dict[
            int, list[prometheus_client.Counter]
        ] = {
            idx: [
                draft_length_counter.labels(*lv, str(length))
                for length in range(num_spec_tokens + 1)
            ]
            for idx, lv in per_engine_labelvalues.items()
        }

    def observe(self, spec_decoding_stats: SpecDecodingStats, engine_idx: int = 0):
        if not self.spec_decoding_enabled:
            return
        self.counter_spec_decode_num_drafts[engine_idx].inc(
            spec_decoding_stats.num_drafts
        )
        self.counter_spec_decode_num_draft_tokens[engine_idx].inc(
            spec_decoding_stats.num_draft_tokens
        )
        self.counter_spec_decode_num_accepted_tokens[engine_idx].inc(
            spec_decoding_stats.num_accepted_tokens
        )
        for pos, counter in enumerate(
            self.counter_spec_decode_num_accepted_tokens_per_pos[engine_idx]
        ):
            counter.inc(spec_decoding_stats.num_accepted_tokens_per_pos[pos])
        for length, counter in enumerate(
            self.counter_spec_decode_num_drafts_by_draft_length[engine_idx]
        ):
            counter.inc(spec_decoding_stats.num_drafts_by_draft_length[length])
