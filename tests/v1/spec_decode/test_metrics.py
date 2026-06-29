# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.v1.spec_decode.metrics import SpecDecodingLogging, SpecDecodingStats


def test_spec_decoding_stats_tracks_draft_length_histogram() -> None:
    stats = SpecDecodingStats.new(num_spec_tokens=5)

    stats.observe_draft(num_draft_tokens=5, num_accepted_tokens=3)
    stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=2)
    stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=1)

    assert stats.num_drafts == 3
    assert stats.num_draft_tokens == 11
    assert stats.num_accepted_tokens == 6
    assert stats.num_drafts_by_draft_length == [0, 0, 0, 2, 0, 1]
    assert stats.num_accepted_tokens_per_pos == [3, 2, 1, 0, 0]
    assert stats.dspark_confidence_bin_counts is None
    assert stats.dspark_confidence_bin_accepted is None
    assert stats.dspark_confidence_bin_sums is None


def test_spec_decoding_stats_rejects_impossible_draft_length() -> None:
    stats = SpecDecodingStats.new(num_spec_tokens=5)

    with pytest.raises(AssertionError):
        stats.observe_draft(num_draft_tokens=6, num_accepted_tokens=0)


def test_spec_decoding_stats_bins_dspark_confidence_labels() -> None:
    stats = SpecDecodingStats.new(num_spec_tokens=3)

    stats.observe_draft(
        num_draft_tokens=3,
        num_accepted_tokens=2,
        dspark_confidence=(-0.1, 0.15, 1.2),
    )

    assert stats.dspark_confidence_bin_counts[0][0] == 1
    assert stats.dspark_confidence_bin_accepted[0][0] == 1
    assert stats.dspark_confidence_bin_sums[0][0] == 0.0
    assert stats.dspark_confidence_bin_counts[1][1] == 1
    assert stats.dspark_confidence_bin_accepted[1][1] == 1
    assert stats.dspark_confidence_bin_sums[1][1] == pytest.approx(0.15)
    assert stats.dspark_confidence_bin_counts[2][9] == 1
    assert stats.dspark_confidence_bin_accepted[2][9] == 0
    assert stats.dspark_confidence_bin_sums[2][9] == 1.0


def test_spec_decoding_logging_aggregates_dspark_calibration_bounded() -> None:
    logger = SpecDecodingLogging()
    first = SpecDecodingStats.new(num_spec_tokens=2)
    second = SpecDecodingStats.new(num_spec_tokens=2)

    first.observe_draft(
        num_draft_tokens=2,
        num_accepted_tokens=1,
        dspark_confidence=(0.24, 0.91),
    )
    second.observe_draft(
        num_draft_tokens=1,
        num_accepted_tokens=1,
        dspark_confidence=(0.26,),
    )
    logger.observe(first)
    logger.observe(second)

    messages: list[str] = []

    def capture(message: str, *args) -> None:
        messages.append(message % args)

    logger.log(capture)

    calibration = next(
        message
        for message in messages
        if message.startswith("DSpark STS calibration bins: ")
    )
    payload = json.loads(calibration.removeprefix("DSpark STS calibration bins: "))
    assert payload["bins"] == 10
    assert payload["counts"][0][2] == 2
    assert payload["accepted"][0][2] == 2
    assert payload["confidence_sums"][0][2] == pytest.approx(0.5)
    assert payload["counts"][1][9] == 1
    assert payload["accepted"][1][9] == 0
    assert logger.dspark_confidence_bin_counts is None
    assert logger.dspark_confidence_bin_accepted is None
    assert logger.dspark_confidence_bin_sums is None
