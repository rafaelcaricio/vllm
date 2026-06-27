# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product

import pytest
import torch
from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.v1.spec_decode.dspark import (
    DSparkDiagnostics,
    DSparkModelSpec,
    confidence_threshold_prefix_length,
    cumulative_survival,
    hardware_aware_prefix_schedule,
    infer_dspark_num_draft_layers,
    infer_dspark_weight_prefix,
    make_dspark_warmup_draft_token_ids,
    map_dspark_stacked_param_name,
    score_prefix_lengths,
    speculative_acceptance_confidence,
    unpack_mhc_pre_outputs,
)
from vllm.v1.spec_decode.dspark_proposer import DSparkProposer


def _dspark_kernels():
    return pytest.importorskip(
        "vllm.models.deepseek_v4.nvidia.dspark_kernels",
        reason="DeepSeek V4 CUDA/flash-attention extensions are unavailable",
        exc_type=ImportError,
    )


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    total = sum(float(value) for value in values)
    assert total > 0.0
    return tuple(float(value) / total for value in values)


def _probability_grid(
    *,
    vocab_size: int,
    total_mass: int = 6,
) -> list[tuple[float, ...]]:
    """Small exact grid of strictly positive categorical distributions."""

    out: list[tuple[float, ...]] = []

    def extend(prefix: tuple[int, ...], remaining: int) -> None:
        if len(prefix) == vocab_size - 1:
            if remaining > 0:
                out.append(_normalize((*prefix, remaining)))
            return

        min_tail_mass = vocab_size - len(prefix) - 1
        for value in range(1, remaining - min_tail_mass + 1):
            extend((*prefix, value), remaining - value)

    extend((), total_mass)
    return out


def _residual_distribution(
    target_probs: Sequence[float],
    draft_probs: Sequence[float],
) -> tuple[float, ...]:
    residual = tuple(
        max(float(target) - float(draft), 0.0)
        for target, draft in zip(target_probs, draft_probs, strict=True)
    )
    total = sum(residual)
    if total <= 1e-12:
        return _normalize(target_probs)
    return tuple(prob / total for prob in residual)


def _exact_one_step_speculative_distribution(
    *,
    target_probs: Sequence[float],
    draft_probs: Sequence[float],
    scheduler: Callable[[int], int],
) -> tuple[float, ...]:
    """Enumerate one speculative token exactly.

    `scheduler` receives the sampled draft token index and returns 0 or 1,
    which lets the tests compare non-anticipating and token-peeking schedulers.
    """

    target = _normalize(target_probs)
    draft = _normalize(draft_probs)
    residual = _residual_distribution(target, draft)
    out = [0.0] * len(target)

    for draft_token_id, draft_prob in enumerate(draft):
        verify_len = scheduler(draft_token_id)
        assert verify_len in (0, 1)

        if verify_len == 0:
            for token_id, target_prob in enumerate(target):
                out[token_id] += draft_prob * target_prob
            continue

        accept_prob = min(1.0, target[draft_token_id] / draft_prob)
        out[draft_token_id] += draft_prob * accept_prob
        for token_id, residual_prob in enumerate(residual):
            out[token_id] += draft_prob * (1.0 - accept_prob) * residual_prob

    return tuple(out)


def _brute_force_prefix_schedule(
    confidence_rows: Sequence[Sequence[float]],
    *,
    steps_per_second: Callable[[int], float],
):
    best = None
    for lengths in product(*(range(len(row) + 1) for row in confidence_rows)):
        result = score_prefix_lengths(
            confidence_rows,
            lengths,
            steps_per_second=steps_per_second,
        )
        if best is None or (
            result.expected_tokens_per_second > best.expected_tokens_per_second
        ):
            best = result
    assert best is not None
    return best


def test_dspark_model_spec_matches_deepseek_v4_flash_release_fields() -> None:
    hf_config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "hidden_size": 4096,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [40, 41, 42],
        "dspark_markov_rank": 256,
        "_weight_names": [
            "mtp.0.main_proj.weight",
            "mtp.1.attn.wq_a.weight",
            "mtp.2.confidence_head.proj.weight",
            "mtp.2.markov_head.markov_w1.weight",
            "mtp.2.markov_head.markov_w2.weight",
        ],
    }

    spec = DSparkModelSpec.from_hf_config(hf_config)

    assert spec.block_size == 5
    assert spec.noise_token_id == 128799
    assert spec.target_layer_ids == (40, 41, 42)
    assert spec.markov_rank == 256
    assert spec.markov_head_type == "vanilla"
    assert spec.confidence_head_with_markov is True
    assert spec.confidence_input_dim(hf_config["hidden_size"]) == 4352
    assert spec.weight_prefix == "mtp.2"
    assert spec.num_draft_layers == 3


def test_speculative_config_override_detects_deepseek_v4_dspark() -> None:
    hf_config = PretrainedConfig(architectures=["DeepseekV4ForCausalLM"])
    hf_config.model_type = "deepseek_v4"
    hf_config.num_hidden_layers = 43
    hf_config.num_nextn_predict_layers = 1
    hf_config.dspark_block_size = 5
    hf_config.dspark_noise_token_id = 128799
    hf_config.dspark_target_layer_ids = [40, 41, 42]
    hf_config.dspark_markov_rank = 256

    override = SpeculativeConfig.hf_config_override(hf_config)

    assert override.model_type == "deepseek_v4_dspark"
    assert override.architectures == ["DeepSeekV4DSparkModel"]
    assert override.n_predict == 5
    assert override.dspark_num_draft_layers == 3


def test_dspark_model_spec_rejects_invalid_target_layers() -> None:
    hf_config = {
        "num_hidden_layers": 4,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [1, 1],
        "dspark_markov_rank": 256,
    }

    with pytest.raises(ValueError, match="strictly increasing"):
        DSparkModelSpec.from_hf_config(hf_config)


def test_infer_dspark_weight_prefix_from_release_weight_names() -> None:
    assert (
        infer_dspark_weight_prefix(
            [
                "mtp.0.attn.wq_a.weight",
                "mtp.2.confidence_head.proj.weight",
                "mtp.2.markov_head.markov_w1.weight",
                "mtp.2.markov_head.markov_w2.weight",
            ]
        )
        == "mtp.2"
    )


def test_infer_dspark_num_draft_layers_from_release_weight_names() -> None:
    assert (
        infer_dspark_num_draft_layers(
            [
                "mtp.0.main_proj.weight",
                "mtp.0.attn.wq_a.weight",
                "mtp.1.attn.wq_a.weight",
                "mtp.2.confidence_head.proj.weight",
                "mtp.2.markov_head.markov_w1.weight",
            ]
        )
        == 3
    )


def test_infer_dspark_num_draft_layers_rejects_gap() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        infer_dspark_num_draft_layers(
            [
                "mtp.0.main_proj.weight",
                "mtp.2.confidence_head.proj.weight",
            ]
        )


def test_map_dspark_stacked_param_name_is_segment_aware() -> None:
    assert map_dspark_stacked_param_name("model.layers.43.attn.wq_a.weight") == (
        "model.layers.43.attn.fused_wqa_wkv.weight",
        0,
    )
    assert map_dspark_stacked_param_name("model.layers.43.attn.wkv.weight") == (
        "model.layers.43.attn.fused_wqa_wkv.weight",
        1,
    )
    assert (
        map_dspark_stacked_param_name("model.layers.45.markov_head.markov_w1.weight")
        is None
    )
    assert (
        map_dspark_stacked_param_name("model.layers.43.ffn.experts.0.w1.weight") is None
    )


def test_unpack_mhc_pre_outputs_returns_layer_input_first() -> None:
    post = torch.empty(2, 3, 1)
    comb = torch.empty(2, 3, 3)
    layer_input = torch.empty(2, 7)

    unpacked_layer_input, unpacked_post, unpacked_comb = unpack_mhc_pre_outputs(
        (post, comb, layer_input)
    )

    assert unpacked_layer_input is layer_input
    assert unpacked_post is post
    assert unpacked_comb is comb


def test_make_dspark_warmup_draft_token_ids_returns_valid_tensor() -> None:
    draft_token_ids = make_dspark_warmup_draft_token_ids(
        batch_size=2,
        num_speculative_tokens=5,
        noise_token_id=128799,
        device=torch.device("cpu"),
    )

    assert draft_token_ids.shape == (2, 5)
    assert draft_token_ids.dtype == torch.int32
    assert draft_token_ids.tolist() == [[128799] * 5, [128799] * 5]


def test_make_dspark_warmup_draft_token_ids_rejects_placeholder_token() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_dspark_warmup_draft_token_ids(
            batch_size=1,
            num_speculative_tokens=5,
            noise_token_id=-1,
            device=torch.device("cpu"),
        )


def _manual_dspark_sparse_attention(
    q: torch.Tensor,
    draft_kv: torch.Tensor,
    main_kv_cache: torch.Tensor,
    valid_main_lengths: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    batch_size, block_size, num_heads, head_dim = q.shape
    rows = []
    for batch_idx in range(batch_size):
        main_len = int(valid_main_lengths[batch_idx].item())
        kv = torch.cat(
            [
                main_kv_cache[batch_idx, :main_len],
                draft_kv[batch_idx],
            ],
            dim=0,
        )
        scores = torch.einsum("qhd,kd->qhk", q[batch_idx].float(), kv.float())
        scores.mul_(softmax_scale)
        normalizer = torch.maximum(
            scores.max(dim=-1, keepdim=True).values,
            attn_sink[:num_heads].view(1, num_heads, 1),
        )
        weights = torch.exp(scores - normalizer)
        denom = weights.sum(dim=-1, keepdim=True) + torch.exp(
            attn_sink[:num_heads].view(1, num_heads, 1) - normalizer
        )
        out = torch.einsum("qhk,kd->qhd", weights.to(kv.dtype), kv) / denom.to(
            kv.dtype
        )
        rows.append(out)
    return torch.stack(rows, dim=0).reshape(
        batch_size * block_size, num_heads, head_dim
    )


def test_dspark_sparse_attention_reference_matches_manual_window_semantics() -> None:
    dspark_sparse_attention_torch = _dspark_kernels().dspark_sparse_attention_torch
    batch_size = 2
    block_size = 3
    num_heads = 2
    head_dim = 4
    window_size = 5
    q = torch.arange(
        batch_size * block_size * num_heads * head_dim,
        dtype=torch.float32,
    ).view(batch_size, block_size, num_heads, head_dim)
    q = (q / 17.0).to(torch.float32)
    draft_kv = torch.arange(
        batch_size * block_size * head_dim,
        dtype=torch.float32,
    ).view(batch_size, block_size, head_dim)
    draft_kv = (draft_kv / 13.0).to(torch.float32)
    main_kv_cache = torch.arange(
        batch_size * window_size * head_dim,
        dtype=torch.float32,
    ).view(batch_size, window_size, head_dim)
    main_kv_cache = (main_kv_cache / 11.0).to(torch.float32)
    valid_main_lengths = torch.tensor([2, 5], dtype=torch.int64)
    attn_sink = torch.tensor([-0.25, 0.5], dtype=torch.float32)

    actual = dspark_sparse_attention_torch(
        q,
        draft_kv,
        main_kv_cache,
        valid_main_lengths,
        attn_sink,
        softmax_scale=0.125,
    )
    expected = _manual_dspark_sparse_attention(
        q,
        draft_kv,
        main_kv_cache,
        valid_main_lengths,
        attn_sink,
        softmax_scale=0.125,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dspark_sparse_attention_triton_matches_reference() -> None:
    kernels = _dspark_kernels()
    device = torch.device("cuda")
    batch_size = 2
    block_size = 5
    num_heads = 4
    head_dim = 64
    window_size = 9
    generator = torch.Generator(device=device).manual_seed(123)
    q = torch.randn(
        batch_size,
        block_size,
        num_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    draft_kv = torch.randn(
        batch_size,
        block_size,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    main_kv_cache = torch.randn(
        batch_size,
        window_size,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    valid_main_lengths = torch.tensor(
        [3, window_size], device=device, dtype=torch.int64
    )
    attn_sink = torch.randn(num_heads, device=device, dtype=torch.float32)
    scores = torch.empty(
        batch_size,
        block_size,
        num_heads,
        window_size + block_size,
        device=device,
        dtype=torch.float32,
    )

    actual = kernels.dspark_sparse_attention(
        q,
        draft_kv,
        main_kv_cache,
        valid_main_lengths,
        attn_sink,
        softmax_scale=head_dim**-0.5,
        scores_buffer=scores,
    )
    expected = kernels.dspark_sparse_attention_torch(
        q,
        draft_kv,
        main_kv_cache,
        valid_main_lengths,
        attn_sink,
        softmax_scale=head_dim**-0.5,
    )

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dspark_sparse_attention_triton_is_cuda_graph_safe() -> None:
    dspark_sparse_attention = _dspark_kernels().dspark_sparse_attention
    device = torch.device("cuda")
    batch_size = 1
    block_size = 5
    num_heads = 2
    head_dim = 64
    window_size = 8
    q = torch.randn(
        batch_size, block_size, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    draft_kv = torch.randn(
        batch_size, block_size, head_dim, device=device, dtype=torch.bfloat16
    )
    main_kv_cache = torch.randn(
        batch_size, window_size, head_dim, device=device, dtype=torch.bfloat16
    )
    valid_main_lengths = torch.tensor([window_size], device=device, dtype=torch.int64)
    attn_sink = torch.randn(num_heads, device=device, dtype=torch.float32)
    scores = torch.empty(
        batch_size,
        block_size,
        num_heads,
        window_size + block_size,
        device=device,
        dtype=torch.float32,
    )

    for _ in range(3):
        dspark_sparse_attention(
            q,
            draft_kv,
            main_kv_cache,
            valid_main_lengths,
            attn_sink,
            softmax_scale=head_dim**-0.5,
            scores_buffer=scores,
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = dspark_sparse_attention(
            q,
            draft_kv,
            main_kv_cache,
            valid_main_lengths,
            attn_sink,
            softmax_scale=head_dim**-0.5,
            scores_buffer=scores,
        )
    eager = dspark_sparse_attention(
        q,
        draft_kv,
        main_kv_cache,
        valid_main_lengths,
        attn_sink,
        softmax_scale=head_dim**-0.5,
        scores_buffer=scores,
    )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured, eager, rtol=3e-2, atol=3e-2)


def test_dspark_proposer_wraps_draft_in_forward_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.spec_decode.dspark_proposer as dspark_proposer_module

    in_forward_context = False
    context_num_tokens: list[int] = []

    class FakeForwardContext:
        def __init__(self, num_tokens: int) -> None:
            self.num_tokens = num_tokens

        def __enter__(self) -> None:
            nonlocal in_forward_context
            in_forward_context = True
            context_num_tokens.append(self.num_tokens)

        def __exit__(self, *_args: object) -> None:
            nonlocal in_forward_context
            in_forward_context = False

    def fake_set_forward_context(
        _attn_metadata: object,
        _vllm_config: object,
        *,
        num_tokens: int,
        **_kwargs: object,
    ) -> FakeForwardContext:
        return FakeForwardContext(num_tokens)

    class FakeDSparkModel:
        def prefill_main(
            self,
            _hidden_by_req: torch.Tensor,
            _positions_by_req: torch.Tensor,
        ) -> None:
            assert not in_forward_context

        def draft(
            self,
            input_ids: torch.Tensor,
            _last_hidden: torch.Tensor,
            _last_positions: torch.Tensor,
        ) -> torch.Tensor:
            assert in_forward_context
            return torch.full((input_ids.shape[0], 5), 7, dtype=torch.long)

        def take_last_confidence(self) -> None:
            return None

    monkeypatch.setattr(
        dspark_proposer_module,
        "set_forward_context",
        fake_set_forward_context,
    )
    monkeypatch.setattr(
        DSparkProposer,
        "_determine_graph_batch",
        lambda _self, batch_size: (None, batch_size, None, None),
    )
    monkeypatch.setattr(
        DSparkProposer,
        "_prepare_draft_buffers",
        lambda _self, **_kwargs: None,
    )
    monkeypatch.setattr(
        DSparkProposer,
        "_run_draft_for_current_context",
        lambda _self: (
            torch.full((2, 5), 7, dtype=torch.long),
            None,
        ),
    )

    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.vllm_config = object()
    proposer.device = torch.device("cpu")
    proposer.num_speculative_tokens = 5
    proposer._prefilled = True
    proposer.model = FakeDSparkModel()

    draft_ids = DSparkProposer.propose(
        proposer,
        target_token_ids=torch.empty(6, dtype=torch.long),
        target_positions=torch.arange(6),
        target_hidden_states=torch.zeros(6, 4),
        next_token_ids=torch.tensor([11, 12], dtype=torch.int32),
        token_indices_to_sample=None,
        common_attn_metadata=None,  # type: ignore[arg-type]
        sampling_metadata=None,  # type: ignore[arg-type]
    )

    assert context_num_tokens == [10]
    assert draft_ids.tolist() == [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7]]
    assert draft_ids.dtype == torch.int32


def test_dspark_proposer_confidence_threshold_sets_prefix_lengths() -> None:
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.num_speculative_tokens = 5
    proposer.confidence_threshold = 0.73
    proposer.diagnostics = DSparkDiagnostics(max_spec_tokens=5)

    lengths = DSparkProposer._observe_confidence(
        proposer,
        torch.tensor(
            [
                [0.90, 0.80, 0.50, 0.90, 0.90],
                [0.95, 0.95, 0.95, 0.95, 0.95],
            ],
            dtype=torch.float32,
        ),
    )

    assert lengths == [1, 5]
    snapshot = proposer.diagnostics.snapshot()
    assert snapshot.num_requests == 2
    assert snapshot.num_scheduled_draft_tokens == 6
    assert snapshot.scheduled_length_histogram == (0, 1, 0, 0, 0, 1)


def test_dspark_proposer_threshold_zero_keeps_full_block() -> None:
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.num_speculative_tokens = 5
    proposer.confidence_threshold = 0.0

    lengths = DSparkProposer._draft_lengths_from_confidence(
        proposer,
        [[0.10, 0.10], [0.10, 0.10, 0.10, 0.10, 0.10]],
    )

    assert lengths == [2, 5]


def test_gpu_model_runner_trims_dspark_draft_rows_by_confidence_lengths() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    class FakeEvent:

        def synchronize(self) -> None:
            return None

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner._draft_token_ids = torch.empty(2, 5)
    runner._draft_token_req_ids = ["a", "b"]
    runner.draft_token_ids_event = FakeEvent()
    runner.draft_token_ids_cpu = torch.tensor(
        [
            [11, 12, 13, 14, 15],
            [21, 22, 23, 24, 25],
        ],
        dtype=torch.int64,
    )
    runner._draft_token_lengths_cpu = [2, 0]

    draft_token_ids, req_ids = GPUModelRunner._get_draft_token_ids_cpu(runner)

    assert req_ids == ["a", "b"]
    assert draft_token_ids == [[11, 12], []]


def test_infer_dspark_weight_prefix_rejects_multiple_prefixes() -> None:
    with pytest.raises(ValueError, match="multiple prefixes"):
        infer_dspark_weight_prefix(
            [
                "mtp.1.markov_head.markov_w1.weight",
                "mtp.2.confidence_head.proj.weight",
            ]
        )


def test_dspark_diagnostics_tracks_scheduler_signals() -> None:
    rows = [
        [0.9, 0.5],
        [0.8, 0.4, 0.2],
    ]
    result = score_prefix_lengths(
        rows,
        [2, 1],
        steps_per_second=lambda _batch_tokens: 10.0,
    )

    diagnostics = DSparkDiagnostics(max_spec_tokens=3)
    diagnostics.observe(rows, result)
    snapshot = diagnostics.snapshot()

    assert snapshot.num_steps == 1
    assert snapshot.num_requests == 2
    assert snapshot.num_possible_draft_tokens == 5
    assert snapshot.num_scheduled_draft_tokens == 3
    assert snapshot.scheduled_length_histogram == (0, 1, 1, 0)
    assert snapshot.avg_scheduled_length == pytest.approx(1.5)
    assert snapshot.draft_token_prune_rate == pytest.approx(0.4)
    assert snapshot.expected_acceptance_length == pytest.approx(
        result.expected_accepted_tokens / 2
    )
    assert snapshot.avg_expected_tokens_per_second == pytest.approx(
        result.expected_tokens_per_second
    )
    assert snapshot.avg_confidence_per_pos == pytest.approx((0.85, 0.45, 0.2))
    assert snapshot.avg_survival_per_pos == pytest.approx((0.85, 0.385, 0.064))
    assert snapshot.scheduled_fraction_per_pos == pytest.approx((1.0, 0.5, 0.0))


def test_dspark_diagnostics_accumulates_multiple_steps() -> None:
    diagnostics = DSparkDiagnostics(max_spec_tokens=2)
    first = score_prefix_lengths(
        [[0.9, 0.8]],
        [2],
        steps_per_second=lambda _batch_tokens: 100.0,
    )
    second = score_prefix_lengths(
        [[0.4, 0.2], [0.7]],
        [0, 1],
        steps_per_second=lambda _batch_tokens: 50.0,
    )

    diagnostics.observe([[0.9, 0.8]], first)
    diagnostics.observe([[0.4, 0.2], [0.7]], second)
    snapshot = diagnostics.snapshot()

    assert snapshot.num_steps == 2
    assert snapshot.num_requests == 3
    assert snapshot.num_possible_draft_tokens == 5
    assert snapshot.num_scheduled_draft_tokens == 3
    assert snapshot.scheduled_length_histogram == (1, 1, 1)
    assert snapshot.avg_scheduled_length == pytest.approx(1.0)
    assert snapshot.draft_token_prune_rate == pytest.approx(0.4)
    assert snapshot.expected_acceptance_length == pytest.approx(
        (first.expected_accepted_tokens + second.expected_accepted_tokens) / 3
    )


def test_dspark_diagnostics_rejects_invalid_schedule_lengths() -> None:
    diagnostics = DSparkDiagnostics(max_spec_tokens=2)
    result = score_prefix_lengths(
        [[0.8]],
        [1],
        steps_per_second=lambda _batch_tokens: 1.0,
    )
    bad_result = result.__class__(
        lengths=(2,),
        expected_accepted_tokens=result.expected_accepted_tokens,
        batch_tokens=result.batch_tokens,
        expected_tokens_per_second=result.expected_tokens_per_second,
    )

    with pytest.raises(ValueError, match="scheduled length"):
        diagnostics.observe([[0.8]], bad_result)


def test_cumulative_survival_multiplies_conditional_confidences() -> None:
    assert cumulative_survival([0.9, 0.8, 0.5]) == pytest.approx((0.9, 0.72, 0.36))


def test_cumulative_survival_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match=r"confidences\[1\]"):
        cumulative_survival([0.5, 1.01])


def test_confidence_threshold_prefix_length_uses_prefix_survival() -> None:
    confidences = [0.9, 0.8, 0.5]

    assert confidence_threshold_prefix_length(confidences, 0.0) == 3
    assert confidence_threshold_prefix_length(confidences, 0.73) == 1
    assert confidence_threshold_prefix_length(confidences, 0.72) == 2
    assert confidence_threshold_prefix_length(confidences, 1.0) == 0


def test_speculative_acceptance_confidence_matches_total_variation() -> None:
    for target_probs in _probability_grid(vocab_size=3):
        for draft_probs in _probability_grid(vocab_size=3):
            confidence = speculative_acceptance_confidence(draft_probs, target_probs)
            total_variation_acceptance = 1.0 - 0.5 * sum(
                abs(target - draft)
                for target, draft in zip(target_probs, draft_probs, strict=True)
            )
            assert confidence == pytest.approx(total_variation_acceptance)


def test_confidence_threshold_scheduler_is_lossless_on_probability_grid() -> None:
    thresholds = (0.0, 0.25, 0.5, 0.75, 1.0)

    for target_probs in _probability_grid(vocab_size=3):
        for draft_probs in _probability_grid(vocab_size=3):
            confidence = speculative_acceptance_confidence(draft_probs, target_probs)

            for threshold in thresholds:
                verify_len = confidence_threshold_prefix_length(
                    [confidence],
                    threshold,
                )
                out = _exact_one_step_speculative_distribution(
                    target_probs=target_probs,
                    draft_probs=draft_probs,
                    scheduler=lambda _token_id, verify_len=verify_len: verify_len,
                )
                assert out == pytest.approx(target_probs, abs=1e-12)


def test_token_lookahead_scheduler_biases_output_distribution() -> None:
    target_probs = (0.8, 0.2)
    draft_probs = (0.2, 0.8)

    out = _exact_one_step_speculative_distribution(
        target_probs=target_probs,
        draft_probs=draft_probs,
        scheduler=lambda draft_token_id: 1 if draft_token_id == 0 else 0,
    )

    assert out != pytest.approx(target_probs, abs=1e-12)
    assert out[0] > target_probs[0]


def test_hardware_scheduler_prioritizes_high_survival_prefixes() -> None:
    result = hardware_aware_prefix_schedule(
        [
            [0.95, 0.90, 0.80],
            [0.60, 0.60, 0.60],
        ],
        steps_per_second=lambda _batch_tokens: 100.0,
    )

    assert result.lengths[0] >= result.lengths[1]
    assert result.lengths == (3, 3)
    assert result.batch_tokens == 8


def test_hardware_scheduler_prunes_when_capacity_drops() -> None:
    result = hardware_aware_prefix_schedule(
        [
            [0.95, 0.90, 0.80],
            [0.60, 0.60, 0.60],
        ],
        steps_per_second=lambda batch_tokens: {2: 100.0, 3: 100.0, 4: 55.0}.get(
            batch_tokens,
            30.0,
        ),
    )

    assert result.lengths == (1, 0)
    assert result.batch_tokens == 3


def test_unconstrained_scheduler_can_cross_jagged_capacity_cliffs() -> None:
    rows = [
        [0.99, 0.99, 0.99],
        [0.99, 0.99, 0.99],
    ]
    jagged = lambda batch_tokens: {2: 100.0, 3: 50.0, 4: 150.0, 5: 140.0}.get(
        batch_tokens,
        130.0,
    )

    early = hardware_aware_prefix_schedule(
        rows,
        steps_per_second=jagged,
        early_stop=True,
    )
    unconstrained = hardware_aware_prefix_schedule(
        rows,
        steps_per_second=jagged,
        early_stop=False,
    )

    assert early.lengths == (0, 0)
    assert unconstrained.expected_tokens_per_second > early.expected_tokens_per_second
    assert sum(unconstrained.lengths) >= 2


def test_unconstrained_scheduler_matches_bruteforce_for_smooth_profiles() -> None:
    confidence_values = (0.25, 0.50, 0.75, 0.95)

    for values in product(confidence_values, repeat=4):
        rows = [values[:2], values[2:]]
        row_count = len(rows)
        for decay in (1.0, 0.98, 0.93):
            steps_per_second = lambda batch_tokens, decay=decay, row_count=row_count: (
                100.0 * (decay ** max(batch_tokens - row_count, 0))
            )

            greedy = hardware_aware_prefix_schedule(
                rows,
                steps_per_second=steps_per_second,
                early_stop=False,
            )
            brute = _brute_force_prefix_schedule(
                rows,
                steps_per_second=steps_per_second,
            )

            assert greedy.expected_tokens_per_second == pytest.approx(
                brute.expected_tokens_per_second
            )
            assert greedy.lengths == brute.lengths
