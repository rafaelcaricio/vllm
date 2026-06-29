# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.spec_decode.metrics import SpecDecodingStats


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


def test_spec_decoding_stats_rejects_impossible_draft_length() -> None:
    stats = SpecDecodingStats.new(num_spec_tokens=5)

    with pytest.raises(AssertionError):
        stats.observe_draft(num_draft_tokens=6, num_accepted_tokens=0)
