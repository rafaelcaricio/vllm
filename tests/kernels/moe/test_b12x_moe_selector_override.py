# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import pytest

from vllm.model_executor.layers.fused_moe import b12x_moe


def _install_fake_w4a16_kernel(monkeypatch: pytest.MonkeyPatch):
    b12x = types.ModuleType("b12x")
    moe = types.ModuleType("b12x.moe")
    fused = types.ModuleType("b12x.moe.fused")
    w4a16 = types.ModuleType("b12x.moe.fused.w4a16")
    kernel = types.ModuleType("b12x.moe.fused.w4a16.kernel")

    def _select_tile_config(*_args, **_kwargs):
        return 64, 128, 128, 3

    def _covering_count(value, divisor):
        return (int(value) + int(divisor) - 1) // int(divisor)

    def _candidate_tile_fits(**_kwargs):
        return True

    def _determine_blocks_per_sm(**kwargs):
        return 2 if int(kwargs["tile_n"]) == 64 else 3

    kernel._select_tile_config = _select_tile_config
    kernel._covering_count = _covering_count
    kernel._candidate_tile_fits = _candidate_tile_fits
    kernel._determine_blocks_per_sm = _determine_blocks_per_sm
    w4a16.kernel = kernel
    fused.w4a16 = w4a16
    moe.fused = fused
    b12x.moe = moe

    monkeypatch.setitem(sys.modules, "b12x", b12x)
    monkeypatch.setitem(sys.modules, "b12x.moe", moe)
    monkeypatch.setitem(sys.modules, "b12x.moe.fused", fused)
    monkeypatch.setitem(sys.modules, "b12x.moe.fused.w4a16", w4a16)
    monkeypatch.setitem(sys.modules, "b12x.moe.fused.w4a16.kernel", kernel)
    return kernel


def test_b12x_w4a16_selector_override_preserves_selected_tile(monkeypatch):
    kernel = _install_fake_w4a16_kernel(monkeypatch)
    original_select_tile_config = kernel._select_tile_config
    monkeypatch.setenv("VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM", "4")

    b12x_moe._maybe_apply_b12x_w4a16_selector_override()

    assert kernel._select_tile_config(problem_m=6) == (64, 128, 128, 4)
    assert kernel._vllm_original_select_tile_config is original_select_tile_config


def test_b12x_w4a16_selector_override_is_idempotent(monkeypatch):
    kernel = _install_fake_w4a16_kernel(monkeypatch)
    monkeypatch.setenv("VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM", "4")

    b12x_moe._maybe_apply_b12x_w4a16_selector_override()
    wrapped_select_tile_config = kernel._select_tile_config
    b12x_moe._maybe_apply_b12x_w4a16_selector_override()

    assert kernel._select_tile_config is wrapped_select_tile_config
    assert kernel._select_tile_config(problem_m=6) == (64, 128, 128, 4)


def test_b12x_w4a16_selector_override_keeps_large_m_default(monkeypatch):
    kernel = _install_fake_w4a16_kernel(monkeypatch)
    monkeypatch.setenv("VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM", "4")
    monkeypatch.setenv("VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M", "16")

    b12x_moe._maybe_apply_b12x_w4a16_selector_override()

    assert kernel._select_tile_config(problem_m=17) == (64, 128, 128, 3)


def test_b12x_w4a16_selector_override_can_force_tile(monkeypatch):
    kernel = _install_fake_w4a16_kernel(monkeypatch)
    monkeypatch.setenv("VLLM_B12X_W4A16_FORCE_TILE_CONFIG", "128,64,128")
    monkeypatch.setenv("VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M", "16")

    b12x_moe._maybe_apply_b12x_w4a16_selector_override()

    assert kernel._select_tile_config(
        problem_m=6,
        problem_n=4096,
        problem_k=4096,
        top_k=6,
        moe_block_size=8,
        sms=48,
        max_shared_mem=101376,
    ) == (128, 64, 128, 2)


def test_b12x_w4a16_selector_override_disabled_by_default(monkeypatch):
    kernel = _install_fake_w4a16_kernel(monkeypatch)
    original_select_tile_config = kernel._select_tile_config
    monkeypatch.delenv("VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM", raising=False)

    b12x_moe._maybe_apply_b12x_w4a16_selector_override()

    assert kernel._select_tile_config is original_select_tile_config
    assert not hasattr(kernel, "_vllm_original_select_tile_config")
