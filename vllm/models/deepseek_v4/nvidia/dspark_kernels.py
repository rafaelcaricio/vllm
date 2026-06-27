# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

_NEG_INF = -3.4028234663852886e38


@triton.jit
def _dspark_sparse_scores_kernel(
    q_ptr,
    draft_kv_ptr,
    main_kv_ptr,
    valid_main_lengths_ptr,
    scores_ptr,
    softmax_scale: tl.constexpr,
    q_stride_b,
    q_stride_q,
    q_stride_h,
    q_stride_d,
    draft_stride_b,
    draft_stride_k,
    draft_stride_d,
    main_stride_b,
    main_stride_k,
    main_stride_d,
    scores_stride_b,
    scores_stride_q,
    scores_stride_h,
    scores_stride_k,
    BLOCK_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    KV_TOKENS: tl.constexpr,
    K_BLOCK: tl.constexpr,
    D_BLOCK: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    pid_bqh = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)

    h = pid_bqh % NUM_HEADS
    tmp = pid_bqh // NUM_HEADS
    q_idx = tmp % BLOCK_SIZE
    batch_idx = tmp // BLOCK_SIZE

    offs_k = pid_k * K_BLOCK + tl.arange(0, K_BLOCK)
    valid_main_len = tl.load(valid_main_lengths_ptr + batch_idx).to(tl.int64)
    is_main = offs_k < WINDOW_SIZE
    is_draft = (offs_k >= WINDOW_SIZE) & (offs_k < KV_TOKENS)
    is_valid = (is_main & (offs_k < valid_main_len)) | is_draft

    acc = tl.zeros((K_BLOCK,), dtype=tl.float32)
    for d_start in tl.static_range(0, HEAD_DIM, D_BLOCK):
        offs_d = d_start + tl.arange(0, D_BLOCK)
        q_vals = tl.load(
            q_ptr
            + batch_idx * q_stride_b
            + q_idx * q_stride_q
            + h * q_stride_h
            + offs_d * q_stride_d
        ).to(tl.float32)

        main_vals = tl.load(
            main_kv_ptr
            + batch_idx * main_stride_b
            + offs_k[:, None] * main_stride_k
            + offs_d[None, :] * main_stride_d,
            mask=(offs_k[:, None] < WINDOW_SIZE),
            other=0.0,
        )
        draft_k = offs_k - WINDOW_SIZE
        draft_vals = tl.load(
            draft_kv_ptr
            + batch_idx * draft_stride_b
            + draft_k[:, None] * draft_stride_k
            + offs_d[None, :] * draft_stride_d,
            mask=(draft_k[:, None] >= 0) & (draft_k[:, None] < BLOCK_SIZE),
            other=0.0,
        )
        kv_vals = tl.where(is_main[:, None], main_vals, draft_vals).to(tl.float32)
        acc += tl.sum(kv_vals * q_vals[None, :], axis=1)

    scores = acc * softmax_scale
    scores = tl.where(is_valid, scores, NEG_INF)
    tl.store(
        scores_ptr
        + batch_idx * scores_stride_b
        + q_idx * scores_stride_q
        + h * scores_stride_h
        + offs_k * scores_stride_k,
        scores,
        mask=offs_k < KV_TOKENS,
    )


@triton.jit
def _dspark_sparse_out_kernel(
    scores_ptr,
    draft_kv_ptr,
    main_kv_ptr,
    attn_sink_ptr,
    out_ptr,
    draft_stride_b,
    draft_stride_k,
    draft_stride_d,
    main_stride_b,
    main_stride_k,
    main_stride_d,
    scores_stride_b,
    scores_stride_q,
    scores_stride_h,
    scores_stride_k,
    out_stride_b,
    out_stride_q,
    out_stride_h,
    out_stride_d,
    BLOCK_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    KV_TOKENS: tl.constexpr,
    K_BLOCK: tl.constexpr,
    D_BLOCK: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    pid_bqh = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)

    h = pid_bqh % NUM_HEADS
    tmp = pid_bqh // NUM_HEADS
    q_idx = tmp % BLOCK_SIZE
    batch_idx = tmp // BLOCK_SIZE

    offs_k = tl.arange(0, K_BLOCK)
    scores = tl.load(
        scores_ptr
        + batch_idx * scores_stride_b
        + q_idx * scores_stride_q
        + h * scores_stride_h
        + offs_k * scores_stride_k,
        mask=offs_k < KV_TOKENS,
        other=NEG_INF,
    ).to(tl.float32)

    sink = tl.load(attn_sink_ptr + h).to(tl.float32)
    normalizer = tl.maximum(tl.max(scores, axis=0), sink)
    weights = tl.exp(scores - normalizer)
    denom = tl.sum(weights, axis=0) + tl.exp(sink - normalizer)

    offs_d = pid_d * D_BLOCK + tl.arange(0, D_BLOCK)
    main_vals = tl.load(
        main_kv_ptr
        + batch_idx * main_stride_b
        + offs_k[:, None] * main_stride_k
        + offs_d[None, :] * main_stride_d,
        mask=(offs_k[:, None] < WINDOW_SIZE) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    draft_k = offs_k - WINDOW_SIZE
    draft_vals = tl.load(
        draft_kv_ptr
        + batch_idx * draft_stride_b
        + draft_k[:, None] * draft_stride_k
        + offs_d[None, :] * draft_stride_d,
        mask=(
            (draft_k[:, None] >= 0)
            & (draft_k[:, None] < BLOCK_SIZE)
            & (offs_d[None, :] < HEAD_DIM)
        ),
        other=0.0,
    )
    vals = tl.where((offs_k < WINDOW_SIZE)[:, None], main_vals, draft_vals).to(
        tl.float32
    )
    out = tl.sum(weights[:, None] * vals, axis=0) / denom
    tl.store(
        out_ptr
        + batch_idx * out_stride_b
        + q_idx * out_stride_q
        + h * out_stride_h
        + offs_d * out_stride_d,
        out,
        mask=offs_d < HEAD_DIM,
    )


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def dspark_sparse_attention_torch(
    q: torch.Tensor,
    draft_kv: torch.Tensor,
    main_kv_cache: torch.Tensor,
    valid_main_lengths: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Reference DSpark sparse attention matching the CUDA kernel contract."""

    batch_size, block_size, num_heads, head_dim = q.shape
    window_size = main_kv_cache.shape[1]
    main_kv = main_kv_cache[:batch_size]
    kv = torch.cat([main_kv, draft_kv], dim=1)
    kv_tokens = window_size + block_size

    kv_idx = torch.arange(kv_tokens, device=q.device)
    valid_main = kv_idx.unsqueeze(0) < valid_main_lengths.to(torch.long).unsqueeze(1)
    valid = torch.where(
        kv_idx.unsqueeze(0) < window_size,
        valid_main,
        torch.ones((batch_size, kv_tokens), dtype=torch.bool, device=q.device),
    )

    scores = torch.einsum("bqhd,bkd->bqhk", q.float(), kv.float())
    scores.mul_(softmax_scale)
    scores.masked_fill_(~valid[:, None, None, :], _NEG_INF)
    normalizer = torch.maximum(
        scores.max(dim=-1, keepdim=True).values,
        attn_sink[:num_heads].view(1, 1, num_heads, 1),
    )
    weights = torch.exp(scores - normalizer)
    denom = weights.sum(dim=-1, keepdim=True) + torch.exp(
        attn_sink[:num_heads].view(1, 1, num_heads, 1) - normalizer
    )
    out = torch.einsum("bqhk,bkd->bqhd", weights.to(kv.dtype), kv) / denom.to(kv.dtype)
    return out.reshape(batch_size * block_size, num_heads, head_dim)


def dspark_sparse_attention(
    q: torch.Tensor,
    draft_kv: torch.Tensor,
    main_kv_cache: torch.Tensor,
    valid_main_lengths: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    scores_buffer: torch.Tensor,
) -> torch.Tensor:
    """Run DSpark sparse attention with a Triton CUDA kernel when available."""

    if not q.is_cuda or not HAS_TRITON:
        return dspark_sparse_attention_torch(
            q,
            draft_kv,
            main_kv_cache,
            valid_main_lengths,
            attn_sink,
            softmax_scale,
        )

    batch_size, block_size, num_heads, head_dim = q.shape
    window_size = main_kv_cache.shape[1]
    kv_tokens = window_size + block_size
    assert scores_buffer.shape[:4] == (batch_size, block_size, num_heads, kv_tokens)
    assert head_dim % 64 == 0

    scores = scores_buffer
    out = torch.empty_like(q)
    k_score_block = 8
    k_out_block = _next_power_of_2(kv_tokens)
    d_score_block = 64
    d_out_block = 32

    grid_scores = (
        batch_size * block_size * num_heads,
        triton.cdiv(kv_tokens, k_score_block),
    )
    _dspark_sparse_scores_kernel[grid_scores](
        q,
        draft_kv,
        main_kv_cache,
        valid_main_lengths,
        scores,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        draft_kv.stride(0),
        draft_kv.stride(1),
        draft_kv.stride(2),
        main_kv_cache.stride(0),
        main_kv_cache.stride(1),
        main_kv_cache.stride(2),
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        scores.stride(3),
        BLOCK_SIZE=block_size,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        WINDOW_SIZE=window_size,
        KV_TOKENS=kv_tokens,
        K_BLOCK=k_score_block,
        D_BLOCK=d_score_block,
        NEG_INF=_NEG_INF,
        num_warps=2,
        num_stages=4,
    )

    grid_out = (
        batch_size * block_size * num_heads,
        triton.cdiv(head_dim, d_out_block),
    )
    _dspark_sparse_out_kernel[grid_out](
        scores,
        draft_kv,
        main_kv_cache,
        attn_sink,
        out,
        draft_kv.stride(0),
        draft_kv.stride(1),
        draft_kv.stride(2),
        main_kv_cache.stride(0),
        main_kv_cache.stride(1),
        main_kv_cache.stride(2),
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        scores.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_SIZE=block_size,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        WINDOW_SIZE=window_size,
        KV_TOKENS=kv_tokens,
        K_BLOCK=k_out_block,
        D_BLOCK=d_out_block,
        NEG_INF=_NEG_INF,
        num_warps=8,
        num_stages=4,
    )
    return out.reshape(batch_size * block_size, num_heads, head_dim)
