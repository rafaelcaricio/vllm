# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.v1.spec_decode.dspark import (
    DSparkDiagnostics,
    DSparkModelSpec,
    DSparkPosition0Diagnostics,
    confidence_threshold_prefix_length,
    cumulative_survival,
    full_prefix_dominates_sps_curve,
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


def _dspark_model_module():
    return pytest.importorskip(
        "vllm.models.deepseek_v4.nvidia.dspark",
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


def test_full_prefix_dominates_sps_curve_for_single_stream_profile() -> None:
    curve = {
        1: 0.038045,
        2: 9.840034,
        3: 10.400448,
        4: 10.796546,
        5: 10.508421,
        6: 14.244464,
    }

    assert full_prefix_dominates_sps_curve(
        request_count=1,
        max_spec_tokens=5,
        steps_per_second=lambda batch_tokens: curve[batch_tokens],
    )


def test_full_prefix_dominates_sps_curve_rejects_concurrency_profile() -> None:
    def steps_per_second(batch_tokens: int) -> float:
        if batch_tokens < 12:
            return 7.235357
        if batch_tokens < 16:
            return 7.749286
        if batch_tokens < 48:
            return 6.334372
        return 5.396963

    assert not full_prefix_dominates_sps_curve(
        request_count=8,
        max_spec_tokens=5,
        steps_per_second=steps_per_second,
    )


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


def test_dspark_vocab_parallel_argmax_masks_padding(monkeypatch) -> None:
    dspark_model = _dspark_model_module()
    monkeypatch.setattr(
        dspark_model,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    class ShardIndices:
        org_vocab_start_index = 100
        num_org_vocab_padding = 2

    class FakeLMHead:
        shard_indices = ShardIndices()

    local_logits = torch.tensor(
        [
            [1.0, 8.0, 7.0, 99.0, 100.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )

    top_tokens = dspark_model._vocab_parallel_argmax(local_logits, FakeLMHead())

    assert top_tokens.tolist() == [101, 100]


def test_dspark_markov_argmax_torch_matches_materialized_logits() -> None:
    kernels = _dspark_kernels()
    base_logits = torch.tensor(
        [
            [1.0, 0.5, 0.0, 100.0],
            [0.1, 0.2, 0.3, 100.0],
        ],
        dtype=torch.float32,
    )
    markov_embed = torch.tensor(
        [
            [1.0, 2.0],
            [-1.0, 0.5],
        ],
        dtype=torch.float32,
    )
    markov_w2_weight = torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
        ],
        dtype=torch.float32,
    )

    max_vals, local_indices = kernels.dspark_markov_argmax_torch(
        base_logits,
        markov_embed,
        markov_w2_weight,
        num_pad=1,
    )

    materialized = base_logits + markov_embed @ markov_w2_weight.t()
    materialized[:, -1] = -float("inf")
    expected_vals, expected_indices = materialized.max(dim=-1)
    assert torch.allclose(max_vals, expected_vals)
    assert local_indices.tolist() == expected_indices.tolist()


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


def _manual_quant_dequant_nope(
    kv: torch.Tensor,
    *,
    rope_dim: int,
    group_size: int,
) -> torch.Tensor:
    head_dim = kv.shape[-1]
    nope_dim = head_dim - rope_dim
    out = kv.clone()
    groups = out[..., :nope_dim].reshape(-1, nope_dim // group_size, group_size)
    groups_fp32 = groups.float()
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    amax = groups_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-4)
    scale = torch.pow(
        torch.full((), 2.0, device=kv.device, dtype=torch.float32),
        torch.ceil(torch.log2(amax / fp8_max)),
    )
    quantized = torch.clamp(groups_fp32 / scale, -fp8_max, fp8_max).to(
        torch.float8_e4m3fn
    )
    out[..., :nope_dim].copy_(
        (quantized.float() * scale).reshape_as(out[..., :nope_dim])
    )
    return out


def test_dspark_quant_dequant_nope_torch_matches_reference_and_preserves_rope() -> None:
    kernels = _dspark_kernels()
    rope_dim = 4
    group_size = 8
    kv = torch.linspace(-3.0, 2.75, steps=2 * 3 * 20, dtype=torch.float32).view(
        2, 3, 20
    )
    kv = kv.to(torch.bfloat16)
    original_rope = kv[..., -rope_dim:].clone()
    expected = _manual_quant_dequant_nope(
        kv,
        rope_dim=rope_dim,
        group_size=group_size,
    )

    actual = kv.clone()
    returned = kernels.dspark_quant_dequant_nope_torch(
        actual,
        rope_dim=rope_dim,
        group_size=group_size,
    )

    assert returned is actual
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[..., -rope_dim:], original_rope, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dspark_quant_dequant_nope_triton_matches_reference() -> None:
    kernels = _dspark_kernels()
    device = torch.device("cuda")
    rope_dim = 8
    group_size = 8
    generator = torch.Generator(device=device).manual_seed(529)
    kv = torch.randn(
        4,
        5,
        40,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    expected = _manual_quant_dequant_nope(
        kv,
        rope_dim=rope_dim,
        group_size=group_size,
    )

    actual = kv.clone()
    returned = kernels.dspark_quant_dequant_nope(
        actual,
        rope_dim=rope_dim,
        group_size=group_size,
    )
    torch.cuda.synchronize()

    assert returned is actual
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dspark_quant_dequant_nope_triton_is_cuda_graph_safe() -> None:
    kernels = _dspark_kernels()
    device = torch.device("cuda")
    rope_dim = 8
    group_size = 8
    kv_base = torch.randn(2, 5, 40, device=device, dtype=torch.bfloat16)

    for _ in range(3):
        kernels.dspark_quant_dequant_nope(
            kv_base.clone(),
            rope_dim=rope_dim,
            group_size=group_size,
        )
    torch.cuda.synchronize()

    eager_input = kv_base.clone()
    expected = kernels.dspark_quant_dequant_nope(
        eager_input,
        rope_dim=rope_dim,
        group_size=group_size,
    ).clone()
    captured_input = kv_base.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kernels.dspark_quant_dequant_nope(
            captured_input,
            rope_dim=rope_dim,
            group_size=group_size,
        )
    graph.replay()
    torch.cuda.synchronize()

    assert captured is captured_input
    torch.testing.assert_close(captured_input, expected, rtol=0, atol=0)


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
            *,
            num_rejected_tokens: torch.Tensor | None = None,
        ) -> None:
            del num_rejected_tokens
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
            torch.zeros(2, 5, 13, dtype=torch.float32),
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


def test_dspark_proposer_exports_greedy_draft_probs_for_quality_probe() -> None:
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.num_speculative_tokens = 3
    proposer._export_draft_probs = True
    proposer._last_draft_probs = None

    class SamplingMetadataStub:
        all_greedy = True

    logits = torch.tensor(
        [
            [[3.0, 1.0, 0.0, -1.0], [0.0, 2.0, 1.0, -2.0], [1.0, 1.0, 1.0, 1.0]],
            [[-1.0, 0.0, 1.0, 2.0], [4.0, 0.0, 0.0, 0.0], [0.0, -1.0, -2.0, -3.0]],
        ],
        dtype=torch.float32,
    )

    DSparkProposer._maybe_store_draft_probs(
        proposer,
        logits,
        SamplingMetadataStub(),  # type: ignore[arg-type]
        batch_size=2,
    )

    draft_probs = DSparkProposer.take_last_draft_probs(proposer)
    assert draft_probs is not None
    assert draft_probs.shape == logits.shape
    torch.testing.assert_close(draft_probs, logits.softmax(dim=-1))
    assert DSparkProposer.take_last_draft_probs(proposer) is None


def test_dspark_proposer_skips_confidence_observation_when_threshold_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.spec_decode.dspark_proposer as dspark_proposer_module

    class FakeForwardContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeDSparkModel:
        def prefill_main(
            self,
            _hidden_by_req: torch.Tensor,
            _positions_by_req: torch.Tensor,
            *,
            num_rejected_tokens: torch.Tensor | None = None,
        ) -> None:
            del num_rejected_tokens
            return None

    monkeypatch.setattr(
        dspark_proposer_module,
        "set_forward_context",
        lambda *_args, **_kwargs: FakeForwardContext(),
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
    model_confidence = torch.full((1, 5), 0.95, dtype=torch.float32)

    monkeypatch.setattr(
        DSparkProposer,
        "_run_draft_for_current_context",
        lambda _self: (
            torch.full((1, 5), 7, dtype=torch.long),
            torch.zeros(1, 5, 13, dtype=torch.float32),
            model_confidence,
        ),
    )

    def fail_observe(*_args, **_kwargs):
        raise AssertionError("confidence observation should be skipped")

    monkeypatch.setattr(DSparkProposer, "_observe_confidence", fail_observe)

    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.vllm_config = object()
    proposer.device = torch.device("cpu")
    proposer.num_speculative_tokens = 5
    proposer.confidence_threshold = 0.0
    proposer._collect_confidence_diagnostics = False
    proposer._collect_position0_diagnostics = True
    proposer._export_draft_probs = False
    proposer._last_draft_probs = None
    proposer._last_confidence = None
    proposer._prefilled = True
    proposer.model = FakeDSparkModel()

    draft_ids = DSparkProposer.propose(
        proposer,
        target_token_ids=torch.empty(1, dtype=torch.long),
        target_positions=torch.arange(1),
        target_hidden_states=torch.zeros(1, 4),
        next_token_ids=torch.tensor([11], dtype=torch.int32),
        token_indices_to_sample=None,
        common_attn_metadata=None,  # type: ignore[arg-type]
        sampling_metadata=None,  # type: ignore[arg-type]
    )

    assert draft_ids.tolist() == [[7, 7, 7, 7, 7]]
    assert DSparkProposer.take_last_draft_lengths(proposer) is None
    assert proposer._last_confidence is not None
    torch.testing.assert_close(proposer._last_confidence, model_confidence)
    assert proposer._last_confidence.data_ptr() != model_confidence.data_ptr()


@pytest.mark.parametrize(
    (
        "confidence_threshold",
        "collect_confidence_diagnostics",
        "collect_position0_diagnostics",
        "export_draft_probs",
        "expected_return_logits",
        "expected_return_confidence",
    ),
    [
        (0.0, False, False, False, False, False),
        (0.5, False, False, False, False, True),
        (0.0, True, False, False, False, True),
        (0.0, False, True, False, False, True),
        (0.0, False, False, True, True, False),
    ],
)
def test_dspark_proposer_requests_only_needed_draft_outputs(
    confidence_threshold: float,
    collect_confidence_diagnostics: bool,
    collect_position0_diagnostics: bool,
    export_draft_probs: bool,
    expected_return_logits: bool,
    expected_return_confidence: bool,
) -> None:
    observed_flags: list[tuple[bool, bool, bool]] = []

    class FakeModel:
        def draft_with_confidence(
            self,
            input_ids: torch.Tensor,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
            *,
            return_logits: bool,
            return_confidence: bool,
            store_main_kv: bool,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            observed_flags.append(
                (return_logits, return_confidence, store_main_kv)
            )
            batch_size = input_ids.shape[0]
            logits = (
                torch.zeros(batch_size, 5, 13)
                if return_logits
                else torch.empty(0, 0, 0)
            )
            confidence = (
                torch.ones(batch_size, 5)
                if return_confidence
                else torch.empty(batch_size, 0)
            )
            return torch.full((batch_size, 5), 7), logits, confidence

    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.model = FakeModel()
    proposer._draft_graph_batch_size = 2
    proposer._draft_input_ids_buffer = torch.tensor([11, 12], dtype=torch.long)
    proposer._draft_hidden_buffer = torch.zeros(2, 4)
    proposer._draft_positions_buffer = torch.arange(2, dtype=torch.long)
    proposer.confidence_threshold = confidence_threshold
    proposer._collect_confidence_diagnostics = collect_confidence_diagnostics
    proposer._export_draft_probs = export_draft_probs
    proposer._collect_position0_diagnostics = collect_position0_diagnostics

    draft_ids, logits, confidence = DSparkProposer._run_draft_from_buffers(
        proposer
    )

    assert observed_flags == [
        (expected_return_logits, expected_return_confidence, False)
    ]
    assert draft_ids.tolist() == [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7]]
    assert logits.numel() > 0 if expected_return_logits else logits.numel() == 0
    assert (
        confidence.numel() > 0
        if expected_return_confidence
        else confidence.numel() == 0
    )


def test_dspark_hardware_scheduler_skips_confidence_when_full_prefix_dominates() -> None:
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.num_speculative_tokens = 5
    proposer.confidence_scheduler = "hardware"
    proposer.confidence_threshold = 0.0
    proposer._forced_draft_length = None
    proposer._collect_confidence_diagnostics = False
    proposer._collect_position0_diagnostics = False
    proposer._collect_sts_calibration_diagnostics = False
    proposer._draft_active_batch_size = 1
    proposer._sps_curve = (
        (1, 0.038045),
        (2, 9.840034),
        (3, 10.400448),
        (4, 10.796546),
        (5, 10.508421),
        (6, 14.244464),
    )

    assert not DSparkProposer._should_observe_confidence(proposer)
    assert not DSparkProposer._needs_confidence(proposer)


def test_dspark_hardware_scheduler_keeps_confidence_when_shorter_width_can_win() -> None:
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.num_speculative_tokens = 5
    proposer.confidence_scheduler = "hardware"
    proposer.confidence_threshold = 0.0
    proposer._forced_draft_length = None
    proposer._collect_confidence_diagnostics = False
    proposer._collect_position0_diagnostics = False
    proposer._collect_sts_calibration_diagnostics = False
    proposer._draft_active_batch_size = 8
    proposer._sps_curve = (
        (8, 7.235357),
        (12, 7.749286),
        (16, 6.334372),
        (48, 5.396963),
    )

    assert DSparkProposer._should_observe_confidence(proposer)
    assert DSparkProposer._needs_confidence(proposer)


def test_dspark_forced_draft_length_skips_scheduler_confidence() -> None:
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.num_speculative_tokens = 5
    proposer.confidence_scheduler = "hardware"
    proposer.confidence_threshold = 0.0
    proposer._forced_draft_length = 3
    proposer._collect_confidence_diagnostics = False
    proposer._collect_position0_diagnostics = False
    proposer._collect_sts_calibration_diagnostics = False
    proposer._draft_active_batch_size = 8
    proposer._sps_curve = (
        (8, 7.235357),
        (12, 7.749286),
        (16, 6.334372),
        (48, 5.396963),
    )

    assert not DSparkProposer._should_observe_confidence(proposer)
    assert not DSparkProposer._needs_confidence(proposer)


def test_dspark_default_draft_length_does_not_cross_scheduler_bridge() -> None:
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.num_speculative_tokens = 5

    DSparkProposer._set_last_draft_lengths(proposer, [5, 5])

    assert DSparkProposer.take_last_draft_lengths(proposer) is None


def test_dspark_short_draft_length_crosses_scheduler_bridge() -> None:
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.num_speculative_tokens = 5

    DSparkProposer._set_last_draft_lengths(proposer, [5, 3])

    assert DSparkProposer.take_last_draft_lengths(proposer) == [5, 3]


def test_dspark_proposer_exposes_position0_confidence() -> None:
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.num_speculative_tokens = 5
    proposer._collect_position0_diagnostics = True
    proposer._last_confidence = None
    confidence = torch.tensor(
        [[0.9, 0.8, 0.7, 0.6, 0.5], [0.4, 0.3, 0.2, 0.1, 0.0]],
        dtype=torch.float32,
    )

    proposer._last_confidence = confidence.detach()

    observed = DSparkProposer.take_last_confidence(proposer)
    assert observed is not None
    torch.testing.assert_close(observed, confidence)
    assert DSparkProposer.take_last_confidence(proposer) is None


def test_dspark_draft_fast_path_can_return_confidence_without_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dspark_module = _dspark_model_module()

    monkeypatch.setattr(
        dspark_module,
        "_vocab_parallel_argmax",
        lambda local_logits, _lm_head: local_logits.argmax(dim=-1).to(torch.long),
    )

    class FakeEmbedTokens:

        def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
            return torch.zeros(*input_ids.shape, 3)

    class FakeQuantMethod:

        def apply(
            self,
            lm_head: object,
            normed: torch.Tensor,
            bias: object = None,
        ) -> torch.Tensor:
            del bias
            return lm_head.local_logits[: normed.shape[0]]

    class FakeLMHead:
        quant_method = FakeQuantMethod()

        def __init__(self) -> None:
            self.local_logits = torch.tensor(
                [
                    [0.0, 8.0, 0.0, 0.0],
                    [0.0, 0.0, 8.0, 0.0],
                ],
                dtype=torch.float32,
            )

    class FakeMarkovHead:

        def forward_local(
            self,
            token_ids: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            markov_embed = token_ids.float().unsqueeze(-1).repeat(1, 2)
            return torch.zeros(token_ids.shape[0], 4), markov_embed

    class FakeConfidenceHead:

        def __init__(self) -> None:
            self.markov_embed: torch.Tensor | None = None

        def __call__(
            self,
            dense: torch.Tensor,
            markov_embed: torch.Tensor,
        ) -> torch.Tensor:
            del dense
            self.markov_embed = markov_embed.detach().clone()
            return torch.zeros(markov_embed.shape[:2], dtype=torch.float32)

    class FakeFinalLayer:

        def __init__(self) -> None:
            self.markov_head = FakeMarkovHead()
            self.confidence_head = FakeConfidenceHead()
            self.norm = lambda x: x
            self.store_main_kv_flags: list[bool] = []
            self.draft_positions: torch.Tensor | None = None
            self.draft_input_ids: torch.Tensor | None = None
            self.main_x_shape: torch.Size | None = None

        def forward_dspark(
            self,
            x: torch.Tensor,
            positions: torch.Tensor,
            input_ids: torch.Tensor,
            **kwargs: object,
        ) -> torch.Tensor:
            self.draft_positions = positions.detach().clone()
            self.draft_input_ids = input_ids.detach().clone()
            self.main_x_shape = kwargs["main_x"].shape
            self.store_main_kv_flags.append(
                bool(kwargs.get("store_main_kv", True))
            )
            return x

        def forward_head(self, x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(1)

    final_layer = FakeFinalLayer()
    model = dspark_module.DeepSeekV4DSparkModel.__new__(
        dspark_module.DeepSeekV4DSparkModel
    )
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=3, hc_mult=1, vocab_size=4)
    model.block_size = 2
    model.noise_token_id = 0
    model._local_argmax = True
    model.stage_layer_keys = ["0"]
    model.layers = {"0": final_layer}
    model.embed_tokens = FakeEmbedTokens()
    project_main_calls: list[torch.Tensor] = []

    def fake_project_main(main_hidden: torch.Tensor) -> torch.Tensor:
        project_main_calls.append(main_hidden.detach().clone())
        return torch.zeros(main_hidden.shape[0], 3)

    model.project_main = fake_project_main

    draft_ids, logits, confidence = dspark_module.DeepSeekV4DSparkModel.draft(
        model,
        torch.tensor([3], dtype=torch.long),
        torch.zeros(1, 3),
        torch.tensor([9], dtype=torch.long),
        FakeLMHead(),
        logits_processor=None,
        return_logits=False,
        return_confidence=True,
        store_main_kv=False,
    )

    assert draft_ids.tolist() == [[1, 2]]
    assert logits.numel() == 0
    assert confidence.tolist() == [[0.5, 0.5]]
    assert final_layer.draft_input_ids is not None
    assert final_layer.draft_input_ids.tolist() == [3, 0]
    assert final_layer.draft_positions is not None
    assert final_layer.draft_positions.tolist() == [9, 10]
    assert final_layer.main_x_shape == torch.Size([1, 1, 3])
    assert len(project_main_calls) == 1
    assert project_main_calls[0].shape == torch.Size([1, 3])
    assert final_layer.store_main_kv_flags == [False]
    assert final_layer.confidence_head.markov_embed is not None
    assert final_layer.confidence_head.markov_embed[:, :, 0].tolist() == [
        [3.0, 1.0]
    ]


def test_dspark_draft_fast_path_can_use_fused_markov_argmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dspark_module = _dspark_model_module()
    fused_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fake_fused_markov_argmax(
        base_logits: torch.Tensor,
        markov_embed: torch.Tensor,
        markov_w2: object,
        lm_head: object,
    ) -> torch.Tensor:
        del markov_w2, lm_head
        fused_calls.append(
            (base_logits.detach().clone(), markov_embed.detach().clone())
        )
        return torch.full(
            (base_logits.shape[0],),
            len(fused_calls),
            dtype=torch.long,
            device=base_logits.device,
        )

    monkeypatch.setattr(
        dspark_module,
        "_vocab_parallel_markov_argmax",
        fake_fused_markov_argmax,
    )

    class FakeEmbedTokens:

        def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
            return torch.zeros(*input_ids.shape, 3)

    class FakeQuantMethod:

        def apply(
            self,
            lm_head: object,
            normed: torch.Tensor,
            bias: object = None,
        ) -> torch.Tensor:
            del bias
            return lm_head.local_logits[: normed.shape[0]]

    class FakeLMHead:
        quant_method = FakeQuantMethod()

        def __init__(self) -> None:
            self.local_logits = torch.tensor(
                [
                    [0.0, 8.0, 0.0, 0.0],
                    [0.0, 0.0, 8.0, 0.0],
                ],
                dtype=torch.float32,
            )

    class FakeMarkovHead:
        markov_w2 = object()

        def markov_w1(self, token_ids: torch.Tensor) -> torch.Tensor:
            return token_ids.float().unsqueeze(-1).repeat(1, 2)

        def forward_local(
            self,
            token_ids: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del token_ids
            raise AssertionError("fused fast path should not materialize logits")

    class FakeFinalLayer:

        def __init__(self) -> None:
            self.markov_head = FakeMarkovHead()
            self.norm = lambda x: x

        def forward_dspark(
            self,
            x: torch.Tensor,
            positions: torch.Tensor,
            input_ids: torch.Tensor,
            **kwargs: object,
        ) -> torch.Tensor:
            del positions, input_ids, kwargs
            return x

        def forward_head(self, x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(1)

    model = dspark_module.DeepSeekV4DSparkModel.__new__(
        dspark_module.DeepSeekV4DSparkModel
    )
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=3, hc_mult=1, vocab_size=4)
    model.block_size = 2
    model.noise_token_id = 0
    model._local_argmax = True
    model._fused_markov_argmax = True
    model.stage_layer_keys = ["0"]
    model.layers = {"0": FakeFinalLayer()}
    model.embed_tokens = FakeEmbedTokens()
    model.project_main = lambda main_hidden: torch.zeros(main_hidden.shape[0], 3)

    draft_ids, logits, confidence = dspark_module.DeepSeekV4DSparkModel.draft(
        model,
        torch.tensor([3], dtype=torch.long),
        torch.zeros(1, 3),
        torch.tensor([9], dtype=torch.long),
        FakeLMHead(),
        logits_processor=None,
        return_logits=False,
        return_confidence=False,
        store_main_kv=False,
    )

    assert draft_ids.tolist() == [[1, 2]]
    assert logits.numel() == 0
    assert confidence.numel() == 0
    assert len(fused_calls) == 2
    assert fused_calls[0][1].tolist() == [[3.0, 3.0]]
    assert fused_calls[1][1].tolist() == [[1.0, 1.0]]


def test_dspark_proposer_trims_rejected_target_context() -> None:

    class AttentionMetadataStub:
        query_start_loc_cpu = torch.tensor([0, 4, 8], dtype=torch.int32)
        query_start_loc = query_start_loc_cpu

    proposer = DSparkProposer.__new__(DSparkProposer)
    hidden = torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2)
    positions = torch.arange(8, dtype=torch.long)

    trimmed_hidden, trimmed_positions = (
        DSparkProposer._trim_rejected_target_context(
            proposer,
            hidden,
            positions,
            AttentionMetadataStub(),  # type: ignore[arg-type]
            torch.tensor([1, 1], dtype=torch.int32),
        )
    )

    assert trimmed_positions.tolist() == [0, 1, 2, 4, 5, 6]
    torch.testing.assert_close(
        trimmed_hidden,
        torch.cat([hidden[0:3], hidden[4:7]], dim=0),
    )


def test_dspark_proposer_rejects_non_uniform_trimmed_context() -> None:

    class AttentionMetadataStub:
        query_start_loc_cpu = torch.tensor([0, 4, 8], dtype=torch.int32)
        query_start_loc = query_start_loc_cpu

    proposer = DSparkProposer.__new__(DSparkProposer)

    with pytest.raises(ValueError, match="uniform effective"):
        DSparkProposer._trim_rejected_target_context(
            proposer,
            torch.zeros(8, 2),
            torch.arange(8),
            AttentionMetadataStub(),  # type: ignore[arg-type]
            torch.tensor([1, 2], dtype=torch.int32),
        )


def test_dspark_proposer_can_mask_rejected_context_on_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_rejected: list[torch.Tensor | None] = []

    class FakeDSparkModel:
        def prefill_main(
            self,
            _hidden_by_req: torch.Tensor,
            _positions_by_req: torch.Tensor,
            *,
            num_rejected_tokens: torch.Tensor | None = None,
        ) -> None:
            captured_rejected.append(num_rejected_tokens)

    def fail_cpu_trim(*_args, **_kwargs):
        raise AssertionError("GPU rejected-context mask should skip CPU trim")

    monkeypatch.setattr(
        DSparkProposer,
        "_trim_rejected_target_context",
        fail_cpu_trim,
    )

    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.device = torch.device("cpu")
    proposer.num_speculative_tokens = 5
    proposer.noise_token_id = 128799
    proposer._last_draft_probs = None
    proposer._last_confidence = None
    proposer._last_draft_lengths = None
    proposer._gpu_rejected_context_mask = True
    proposer._prefilled = False
    proposer.model = FakeDSparkModel()
    rejected = torch.tensor([2], dtype=torch.int32)

    draft_ids = DSparkProposer.propose(
        proposer,
        target_token_ids=torch.empty(1, dtype=torch.long),
        target_positions=torch.arange(4, dtype=torch.long),
        target_hidden_states=torch.zeros(4, 3),
        next_token_ids=torch.tensor([11], dtype=torch.int32),
        token_indices_to_sample=None,
        common_attn_metadata=None,  # type: ignore[arg-type]
        sampling_metadata=None,  # type: ignore[arg-type]
        num_rejected_tokens_gpu=rejected,
    )

    assert captured_rejected == [rejected]
    assert draft_ids.tolist() == [[128799] * 5]


def test_dspark_proposer_gpu_mask_anchors_on_last_non_rejected_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.spec_decode.dspark_proposer as dspark_proposer_module

    class FakeForwardContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    captured_prefill: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
    ] = []
    captured_draft: list[tuple[torch.Tensor, torch.Tensor]] = []

    class FakeDSparkModel:
        def prefill_main(
            self,
            hidden_by_req: torch.Tensor,
            positions_by_req: torch.Tensor,
            *,
            num_rejected_tokens: torch.Tensor | None = None,
            request_indices: torch.Tensor | None = None,
        ) -> None:
            captured_prefill.append(
                (
                    hidden_by_req.detach().clone(),
                    positions_by_req.detach().clone(),
                    num_rejected_tokens,
                    request_indices,
                )
            )

    def fail_cpu_trim(*_args, **_kwargs):
        raise AssertionError("GPU rejected-context mask should skip CPU trim")

    def capture_draft_buffers(
        _self: DSparkProposer,
        *,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        padded_batch_size: int,
    ) -> None:
        del input_ids, padded_batch_size
        captured_draft.append(
            (hidden_states.detach().clone(), positions.detach().clone())
        )

    monkeypatch.setattr(
        dspark_proposer_module,
        "set_forward_context",
        lambda *_args, **_kwargs: FakeForwardContext(),
    )
    monkeypatch.setattr(
        DSparkProposer,
        "_trim_rejected_target_context",
        fail_cpu_trim,
    )
    monkeypatch.setattr(
        DSparkProposer,
        "_determine_graph_batch",
        lambda _self, batch_size: (None, batch_size, None, None),
    )
    monkeypatch.setattr(
        DSparkProposer,
        "_prepare_draft_buffers",
        capture_draft_buffers,
    )
    monkeypatch.setattr(
        DSparkProposer,
        "_run_draft_for_current_context",
        lambda _self: (
            torch.full((1, 5), 7, dtype=torch.long),
            torch.empty(0, 0, 0),
            torch.empty(1, 0),
        ),
    )

    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.vllm_config = object()
    proposer.device = torch.device("cpu")
    proposer.num_speculative_tokens = 5
    proposer.confidence_threshold = 0.0
    proposer._collect_confidence_diagnostics = False
    proposer._collect_position0_diagnostics = False
    proposer._export_draft_probs = False
    proposer._last_draft_probs = None
    proposer._last_confidence = None
    proposer._last_draft_lengths = None
    proposer._gpu_rejected_context_mask = True
    proposer._prefilled = True
    proposer.model = FakeDSparkModel()

    hidden = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    positions = torch.tensor([10, 11, 12, 13], dtype=torch.long)
    rejected = torch.tensor([2], dtype=torch.int32)

    draft_ids = DSparkProposer.propose(
        proposer,
        target_token_ids=torch.empty(1, dtype=torch.long),
        target_positions=positions,
        target_hidden_states=hidden,
        next_token_ids=torch.tensor([21], dtype=torch.int32),
        token_indices_to_sample=None,
        common_attn_metadata=None,  # type: ignore[arg-type]
        sampling_metadata=None,  # type: ignore[arg-type]
        num_rejected_tokens_gpu=rejected,
    )

    assert len(captured_prefill) == 1
    prefill_hidden, prefill_positions, prefill_rejected, prefill_indices = (
        captured_prefill[0]
    )
    torch.testing.assert_close(prefill_hidden, hidden.view(1, 4, 3))
    torch.testing.assert_close(prefill_positions, positions.view(1, 4))
    assert prefill_rejected is rejected
    assert prefill_indices is None

    assert len(captured_draft) == 1
    draft_hidden, draft_positions = captured_draft[0]
    torch.testing.assert_close(draft_hidden, hidden[1:2])
    torch.testing.assert_close(draft_positions, torch.tensor([11]))
    assert draft_ids.tolist() == [[7, 7, 7, 7, 7]]
    assert DSparkProposer.take_last_draft_lengths(proposer) is None


def test_dspark_proposer_groups_mixed_prefill_and_decode_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.spec_decode.dspark_proposer as dspark_proposer_module

    class AttentionMetadataStub:
        query_start_loc_cpu = torch.tensor([0, 415, 421, 836], dtype=torch.int32)
        query_start_loc = query_start_loc_cpu

    class FakeForwardContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    captured_prefill: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
    ] = []
    captured_draft: list[tuple[torch.Tensor, torch.Tensor]] = []

    class FakeDSparkModel:
        def prefill_main(
            self,
            hidden_by_req: torch.Tensor,
            positions_by_req: torch.Tensor,
            *,
            num_rejected_tokens: torch.Tensor | None = None,
            request_indices: torch.Tensor | None = None,
        ) -> None:
            captured_prefill.append(
                (
                    hidden_by_req.detach().clone(),
                    positions_by_req.detach().clone(),
                    None
                    if num_rejected_tokens is None
                    else num_rejected_tokens.detach().clone(),
                    None
                    if request_indices is None
                    else request_indices.detach().clone(),
                )
            )

    def capture_draft_buffers(
        _self: DSparkProposer,
        *,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        padded_batch_size: int,
    ) -> None:
        del input_ids, padded_batch_size
        captured_draft.append(
            (hidden_states.detach().clone(), positions.detach().clone())
        )

    monkeypatch.setattr(
        dspark_proposer_module,
        "set_forward_context",
        lambda *_args, **_kwargs: FakeForwardContext(),
    )
    monkeypatch.setattr(
        DSparkProposer,
        "_determine_graph_batch",
        lambda _self, batch_size: (None, batch_size, None, None),
    )
    monkeypatch.setattr(
        DSparkProposer,
        "_prepare_draft_buffers",
        capture_draft_buffers,
    )
    monkeypatch.setattr(
        DSparkProposer,
        "_run_draft_for_current_context",
        lambda _self: (
            torch.full((3, 5), 9, dtype=torch.long),
            torch.empty(0, 0, 0),
            torch.empty(3, 0),
        ),
    )

    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.vllm_config = object()
    proposer.device = torch.device("cpu")
    proposer.num_speculative_tokens = 5
    proposer.confidence_threshold = 0.0
    proposer._collect_confidence_diagnostics = False
    proposer._collect_position0_diagnostics = False
    proposer._export_draft_probs = False
    proposer._last_draft_probs = None
    proposer._last_confidence = None
    proposer._last_draft_lengths = None
    proposer._gpu_rejected_context_mask = True
    proposer._multi_seq_pad = True
    proposer._prefilled = True
    proposer.model = FakeDSparkModel()

    hidden = torch.arange(836 * 3, dtype=torch.float32).reshape(836, 3)
    positions = torch.cat(
        [
            torch.arange(415, dtype=torch.long),
            torch.arange(669, 675, dtype=torch.long),
            torch.arange(900, 1315, dtype=torch.long),
        ]
    )

    draft_ids = DSparkProposer.propose(
        proposer,
        target_token_ids=torch.empty(3, dtype=torch.long),
        target_positions=positions,
        target_hidden_states=hidden,
        next_token_ids=torch.tensor([21, 22, 23], dtype=torch.int32),
        token_indices_to_sample=None,
        common_attn_metadata=AttentionMetadataStub(),  # type: ignore[arg-type]
        sampling_metadata=None,  # type: ignore[arg-type]
        num_rejected_tokens_gpu=torch.tensor([0, 0, 0], dtype=torch.int32),
    )

    assert len(captured_prefill) == 2
    long_hidden, long_positions, long_rejected, long_indices = captured_prefill[0]
    assert long_hidden.shape == (2, 415, 3)
    torch.testing.assert_close(long_hidden[0], hidden[:415])
    torch.testing.assert_close(long_hidden[1], hidden[421:836])
    torch.testing.assert_close(long_positions[0], positions[:415])
    torch.testing.assert_close(long_positions[1], positions[421:836])
    assert long_rejected is None
    assert long_indices is not None
    assert long_indices.tolist() == [0, 2]

    short_hidden, short_positions, short_rejected, short_indices = captured_prefill[1]
    assert short_hidden.shape == (1, 6, 3)
    torch.testing.assert_close(short_hidden[0], hidden[415:421])
    torch.testing.assert_close(short_positions[0], positions[415:421])
    assert short_rejected is None
    assert short_indices is not None
    assert short_indices.tolist() == [1]

    assert len(captured_draft) == 1
    draft_hidden, draft_positions = captured_draft[0]
    torch.testing.assert_close(
        draft_hidden,
        torch.stack([hidden[414], hidden[420], hidden[835]]),
    )
    torch.testing.assert_close(draft_positions, torch.tensor([414, 674, 1314]))
    assert draft_ids.tolist() == [
        [9, 9, 9, 9, 9],
        [9, 9, 9, 9, 9],
        [9, 9, 9, 9, 9],
    ]
    assert DSparkProposer.take_last_draft_lengths(proposer) is None


def test_dspark_attention_store_main_kv_can_skip_fully_masked_rows() -> None:
    dspark_module = _dspark_model_module()
    attn = dspark_module.DeepSeekV4DSparkAttention.__new__(
        dspark_module.DeepSeekV4DSparkAttention
    )
    torch.nn.Module.__init__(attn)
    attn.window_size = 4
    attn.hidden_size = 2
    attn.head_dim = 2
    original_cache = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 4, 2)
    attn.main_kv_cache = original_cache.clone()
    attn._project_kv = lambda hidden_states, _positions: hidden_states

    main_x = torch.tensor(
        [
            [[100.0, 101.0], [102.0, 103.0], [104.0, 105.0]],
            [[200.0, 201.0], [202.0, 203.0], [204.0, 205.0]],
        ]
    )
    main_positions = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)

    dspark_module.DeepSeekV4DSparkAttention.store_main_kv(
        attn,
        main_x,
        main_positions,
        num_rejected_tokens=torch.tensor([0, 3], dtype=torch.int32),
    )

    expected = original_cache.clone()
    expected[0, 0:3] = main_x[0]
    torch.testing.assert_close(attn.main_kv_cache, expected)


def test_dspark_attention_store_main_kv_can_update_selected_rows() -> None:
    dspark_module = _dspark_model_module()
    attn = dspark_module.DeepSeekV4DSparkAttention.__new__(
        dspark_module.DeepSeekV4DSparkAttention
    )
    torch.nn.Module.__init__(attn)
    attn.window_size = 4
    attn.hidden_size = 2
    attn.head_dim = 2
    original_cache = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
    attn.main_kv_cache = original_cache.clone()
    attn._project_kv = lambda hidden_states, _positions: hidden_states

    main_x = torch.tensor(
        [
            [[100.0, 101.0], [102.0, 103.0]],
            [[200.0, 201.0], [202.0, 203.0]],
        ]
    )
    main_positions = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)

    dspark_module.DeepSeekV4DSparkAttention.store_main_kv(
        attn,
        main_x,
        main_positions,
        request_indices=torch.tensor([2, 0], dtype=torch.long),
    )

    expected = original_cache.clone()
    expected[2, 0] = main_x[0, 0]
    expected[2, 2] = main_x[0, 1]
    expected[0, 1] = main_x[1, 0]
    expected[0, 3] = main_x[1, 1]
    torch.testing.assert_close(attn.main_kv_cache, expected)


def test_dspark_attention_store_main_kv_masks_rejected_selected_rows() -> None:
    dspark_module = _dspark_model_module()
    attn = dspark_module.DeepSeekV4DSparkAttention.__new__(
        dspark_module.DeepSeekV4DSparkAttention
    )
    torch.nn.Module.__init__(attn)
    attn.window_size = 4
    attn.hidden_size = 2
    attn.head_dim = 2
    original_cache = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
    attn.main_kv_cache = original_cache.clone()
    attn._project_kv = lambda hidden_states, _positions: hidden_states

    main_x = torch.tensor(
        [
            [[100.0, 101.0], [102.0, 103.0], [104.0, 105.0]],
            [[200.0, 201.0], [202.0, 203.0], [204.0, 205.0]],
        ]
    )
    main_positions = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    dspark_module.DeepSeekV4DSparkAttention.store_main_kv(
        attn,
        main_x,
        main_positions,
        num_rejected_tokens=torch.tensor([1, 2], dtype=torch.int32),
        request_indices=torch.tensor([2, 0], dtype=torch.long),
    )

    expected = original_cache.clone()
    expected[2, 0] = main_x[0, 0]
    expected[2, 1] = main_x[0, 1]
    expected[0, 1] = main_x[1, 0]
    torch.testing.assert_close(attn.main_kv_cache, expected)


def test_dspark_store_main_kv_torch_updates_selected_rows_without_copy_bridge() -> None:
    kernels = _dspark_kernels()
    main_kv_cache = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
    original = main_kv_cache.clone()
    flat_kv = torch.tensor(
        [
            [[100.0, 101.0], [102.0, 103.0], [104.0, 105.0]],
            [[200.0, 201.0], [202.0, 203.0], [204.0, 205.0]],
        ]
    )
    slots = torch.tensor([[0, 2, 3], [1, 2, 3]], dtype=torch.long)

    returned = kernels.dspark_store_main_kv_torch(
        main_kv_cache,
        flat_kv,
        slots,
        num_rejected_tokens=torch.tensor([1, 2], dtype=torch.int32),
        request_indices=torch.tensor([2, 0], dtype=torch.long),
    )

    assert returned is main_kv_cache
    expected = original.clone()
    expected[2, 0] = flat_kv[0, 0]
    expected[2, 2] = flat_kv[0, 1]
    expected[0, 1] = flat_kv[1, 0]
    torch.testing.assert_close(main_kv_cache, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dspark_store_main_kv_triton_matches_reference() -> None:
    kernels = _dspark_kernels()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(619)
    main_kv_cache = torch.randn(
        4,
        8,
        64,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    flat_kv = torch.randn(
        3,
        5,
        64,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    slots = torch.tensor(
        [[0, 2, 4, 6, 7], [1, 3, 5, 6, 7], [0, 1, 2, 3, 4]],
        device=device,
        dtype=torch.long,
    )
    rejected = torch.tensor([0, 2, 5], device=device, dtype=torch.int32)
    request_indices = torch.tensor([3, 0, 2], device=device, dtype=torch.long)

    expected = main_kv_cache.clone()
    kernels.dspark_store_main_kv_torch(
        expected,
        flat_kv,
        slots,
        num_rejected_tokens=rejected,
        request_indices=request_indices,
    )
    actual = main_kv_cache.clone()
    returned = kernels.dspark_store_main_kv(
        actual,
        flat_kv,
        slots,
        num_rejected_tokens=rejected,
        request_indices=request_indices,
    )
    torch.cuda.synchronize()

    assert returned is actual
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dspark_proposer_confidence_threshold_sets_prefix_lengths() -> None:
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.num_speculative_tokens = 5
    proposer.confidence_threshold = 0.73
    proposer._forced_draft_length = None
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


def test_dspark_proposer_reads_sts_temperatures(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_DSPARK_STS_TEMPERATURES", "2.0")
    assert DSparkProposer._read_sts_temperatures(5) == (2.0,)

    monkeypatch.setenv("VLLM_DSPARK_STS_TEMPERATURES", "1, 2, 3, 4, 5")
    assert DSparkProposer._read_sts_temperatures(5) == (1.0, 2.0, 3.0, 4.0, 5.0)

    monkeypatch.setenv("VLLM_DSPARK_STS_TEMPERATURES", "1, 2")
    with pytest.raises(ValueError, match="either one temperature or 5"):
        DSparkProposer._read_sts_temperatures(5)

    monkeypatch.setenv("VLLM_DSPARK_STS_TEMPERATURES", "0")
    with pytest.raises(ValueError, match="positive"):
        DSparkProposer._read_sts_temperatures(5)


def test_dspark_proposer_calibrates_confidence_without_mutating_raw() -> None:
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer._sts_temperatures = (2.0, 0.5)
    raw = torch.tensor([[0.80, 0.20]], dtype=torch.float32)

    calibrated = DSparkProposer._calibrate_confidence(proposer, raw)

    expected = torch.sigmoid(torch.logit(raw) / torch.tensor([[2.0, 0.5]]))
    torch.testing.assert_close(calibrated, expected)
    torch.testing.assert_close(raw, torch.tensor([[0.80, 0.20]], dtype=torch.float32))


def test_dspark_proposer_reuses_sts_temperature_tensor() -> None:
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer._sts_temperatures = (2.0, 0.5, 1.5)
    proposer._sts_temperature_tensor = DSparkProposer._make_sts_temperature_tensor(
        proposer._sts_temperatures,
        torch.device("cpu"),
    )
    cached = proposer._sts_temperature_tensor
    raw = torch.tensor([[0.80, 0.20]], dtype=torch.float32)

    DSparkProposer._calibrate_confidence(proposer, raw)

    assert proposer._sts_temperature_tensor is cached


def test_dspark_proposer_reports_confidence_calibration_metrics() -> None:
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.num_speculative_tokens = 2
    proposer.confidence_threshold = 0.0
    proposer._forced_draft_length = None
    proposer.diagnostics = DSparkDiagnostics(max_spec_tokens=2)

    lengths = DSparkProposer._observe_confidence(
        proposer,
        torch.tensor([[0.70, 0.30]], dtype=torch.float32),
        raw_confidence=torch.tensor([[0.80, 0.20]], dtype=torch.float32),
    )

    assert lengths == [2]
    snapshot = proposer.diagnostics.snapshot()
    assert snapshot.avg_confidence_per_pos == pytest.approx((0.70, 0.30))
    assert snapshot.avg_raw_confidence_per_pos == pytest.approx((0.80, 0.20))
    assert snapshot.avg_confidence_calibration_delta_per_pos == pytest.approx(
        (-0.10, 0.10)
    )


def test_dspark_proposer_reads_confidence_diagnostics_log_every(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_DSPARK_CONFIDENCE_DIAGNOSTICS_LOG_EVERY", "0")
    assert DSparkProposer._read_confidence_diagnostics_log_every() == 0

    monkeypatch.setenv("VLLM_DSPARK_CONFIDENCE_DIAGNOSTICS_LOG_EVERY", "17")
    assert DSparkProposer._read_confidence_diagnostics_log_every() == 17

    monkeypatch.setenv("VLLM_DSPARK_CONFIDENCE_DIAGNOSTICS_LOG_EVERY", "-1")
    with pytest.raises(ValueError, match="non-negative"):
        DSparkProposer._read_confidence_diagnostics_log_every()

    monkeypatch.setenv("VLLM_DSPARK_CONFIDENCE_DIAGNOSTICS_LOG_EVERY", "bad")
    with pytest.raises(ValueError, match="non-negative integer"):
        DSparkProposer._read_confidence_diagnostics_log_every()


def test_dspark_proposer_sts_calibration_diagnostics_requires_env(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VLLM_DSPARK_STS_CALIBRATION_DIAGNOSTICS", raising=False)
    assert not DSparkProposer._read_sts_calibration_diagnostics()

    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.confidence_threshold = 0.0
    proposer.confidence_scheduler = "off"
    proposer._collect_position0_diagnostics = False
    proposer._collect_sts_calibration_diagnostics = False
    assert not DSparkProposer._needs_confidence(proposer)

    monkeypatch.setenv("VLLM_DSPARK_STS_CALIBRATION_DIAGNOSTICS", "1")
    assert DSparkProposer._read_sts_calibration_diagnostics()
    proposer._collect_sts_calibration_diagnostics = True
    assert DSparkProposer._needs_confidence(proposer)


def test_dspark_proposer_logs_confidence_diagnostics(monkeypatch) -> None:
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.num_speculative_tokens = 2
    proposer.confidence_threshold = 0.0
    proposer.confidence_scheduler = "off"
    proposer._forced_draft_length = None
    proposer._sps_curve = ()
    proposer._confidence_diagnostics_log_every = 1
    proposer._confidence_diagnostics_log_next = 1
    proposer.diagnostics = DSparkDiagnostics(max_spec_tokens=2)

    messages: list[str] = []

    def fake_info(message: str, *args) -> None:
        messages.append(message % args)

    monkeypatch.setattr(
        "vllm.v1.spec_decode.dspark_proposer.logger.info",
        fake_info,
    )

    lengths = DSparkProposer._observe_confidence(
        proposer,
        torch.tensor([[0.70, 0.30]], dtype=torch.float32),
        raw_confidence=torch.tensor([[0.80, 0.20]], dtype=torch.float32),
    )

    assert lengths == [2]
    log_text = "\n".join(messages)
    assert "DSpark confidence diagnostics" in log_text
    assert "avg_scheduled_length=2.000" in log_text
    assert "calibration_delta=[-0.100, 0.100]" in log_text


def test_dspark_proposer_threshold_zero_keeps_full_block() -> None:
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.num_speculative_tokens = 5
    proposer.confidence_threshold = 0.0
    proposer._forced_draft_length = None

    lengths = DSparkProposer._draft_lengths_from_confidence(
        proposer,
        [[0.10, 0.10], [0.10, 0.10, 0.10, 0.10, 0.10]],
    )

    assert lengths == [2, 5]


def test_dspark_proposer_forced_draft_length_overrides_scheduler() -> None:
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.num_speculative_tokens = 5
    proposer.confidence_threshold = 0.99
    proposer.confidence_scheduler = "hardware"
    proposer._forced_draft_length = 2
    proposer._sps_curve = ()

    lengths = DSparkProposer._draft_lengths_from_confidence(
        proposer,
        [[0.10], [0.10, 0.10, 0.10, 0.10, 0.10]],
    )

    assert lengths == [1, 2]


def test_dspark_proposer_validates_forced_draft_length(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_DSPARK_FORCE_DRAFT_LENGTH", "3")
    assert DSparkProposer._read_forced_draft_length(5) == 3

    monkeypatch.setenv("VLLM_DSPARK_FORCE_DRAFT_LENGTH", "")
    assert DSparkProposer._read_forced_draft_length(5) is None

    monkeypatch.setenv("VLLM_DSPARK_FORCE_DRAFT_LENGTH", "6")
    with pytest.raises(ValueError, match=r"\[0, 5\]"):
        DSparkProposer._read_forced_draft_length(5)


def test_dspark_proposer_hardware_scheduler_uses_profiled_sps_curve() -> None:
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.num_speculative_tokens = 3
    proposer.confidence_threshold = 0.0
    proposer.confidence_scheduler = "hardware"
    proposer._forced_draft_length = None
    proposer._sps_curve = (
        (1, 100.0),
        (2, 100.0),
        (3, 40.0),
        (4, 30.0),
    )
    proposer._hardware_scheduler_early_stop = True

    schedule = DSparkProposer._schedule_from_confidence(
        proposer,
        [[0.95, 0.90, 0.90]],
    )

    assert schedule.lengths == (1,)
    assert schedule.batch_tokens == 2
    assert schedule.expected_tokens_per_second == pytest.approx(195.0)


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


def test_dspark_position0_diagnostics_accumulates_confidence_by_outcome() -> None:
    diagnostics = DSparkPosition0Diagnostics()

    diagnostics.observe([True, False, True], [0.9, 0.2, 0.7])
    snapshot = diagnostics.snapshot()

    assert snapshot.num_tokens == 3
    assert snapshot.num_matches == 2
    assert snapshot.match_rate == pytest.approx(2 / 3)
    assert snapshot.avg_confidence == pytest.approx(0.6)
    assert snapshot.avg_confidence_when_matched == pytest.approx(0.8)
    assert snapshot.avg_confidence_when_missed == pytest.approx(0.2)
    assert snapshot.num_confidence_logits_normalized == 0


def test_dspark_position0_diagnostics_normalizes_raw_confidence_logits() -> None:
    diagnostics = DSparkPosition0Diagnostics()

    diagnostics.observe([False], [-0.048030007630586624])
    snapshot = diagnostics.snapshot()

    assert snapshot.num_tokens == 1
    assert snapshot.num_matches == 0
    assert snapshot.avg_confidence == pytest.approx(
        torch.sigmoid(torch.tensor(-0.048030007630586624)).item()
    )
    assert snapshot.avg_confidence_when_matched is None
    assert snapshot.avg_confidence_when_missed == pytest.approx(
        snapshot.avg_confidence
    )
    assert snapshot.num_confidence_logits_normalized == 1


def test_dspark_position0_diagnostics_allows_missing_confidence() -> None:
    diagnostics = DSparkPosition0Diagnostics()

    diagnostics.observe([True, False])
    snapshot = diagnostics.snapshot()

    assert snapshot.num_tokens == 2
    assert snapshot.num_matches == 1
    assert snapshot.match_rate == pytest.approx(0.5)
    assert snapshot.avg_confidence is None
    assert snapshot.avg_confidence_when_matched is None
    assert snapshot.avg_confidence_when_missed is None
    assert snapshot.num_confidence_logits_normalized == 0


def test_gpu_model_runner_observes_dspark_position0_quality() -> None:
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["req-a", "req-b"])
    runner._draft_confidence = torch.tensor(
        [
            [0.9, 0.8, 0.7],
            [0.2, 0.1, 0.0],
        ],
        dtype=torch.float32,
    )
    runner._draft_confidence_req_ids = ["req-a", "req-b"]
    runner._dspark_position0_diagnostics = DSparkPosition0Diagnostics()
    runner._dspark_position0_log_next = 10_000

    metadata = SpecDecodeMetadata(
        draft_token_ids=torch.tensor([3, 4, 5], dtype=torch.int32),
        num_draft_tokens=[2, 1],
        cu_num_draft_tokens=torch.tensor([2, 3], dtype=torch.int32),
        cu_num_sampled_tokens=torch.tensor([3, 5], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1, 3], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([2, 4], dtype=torch.int32),
        logits_indices=torch.arange(5, dtype=torch.int32),
    )
    logits = torch.zeros(5, 8, dtype=torch.float32)
    logits[0, 3] = 10.0
    logits[3, 6] = 10.0

    GPUModelRunner._maybe_observe_dspark_position0_quality(
        runner,
        metadata,
        logits,
    )

    snapshot = runner._dspark_position0_diagnostics.snapshot()
    assert snapshot.num_tokens == 2
    assert snapshot.num_matches == 1
    assert snapshot.match_rate == pytest.approx(0.5)
    assert snapshot.avg_confidence == pytest.approx(0.55)
    assert snapshot.avg_confidence_when_matched == pytest.approx(0.9)
    assert snapshot.avg_confidence_when_missed == pytest.approx(0.2)
    assert snapshot.num_confidence_logits_normalized == 0


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
