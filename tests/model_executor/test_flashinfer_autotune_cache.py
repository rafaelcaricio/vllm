# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from hashlib import sha256
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

from vllm.model_executor.warmup import kernel_warmup


def test_resolve_flashinfer_autotune_file_default_layout(
    monkeypatch, tmp_path: Path
) -> None:
    fake_jit = SimpleNamespace(
        env=SimpleNamespace(
            FLASHINFER_WORKSPACE_DIR=Path("/flashinfer-cache/0.6.11.post2/103a")
        )
    )
    fake_flashinfer = SimpleNamespace(jit=fake_jit)
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.jit", fake_jit)
    monkeypatch.setattr(
        kernel_warmup, "aot_compile_hash_factors", lambda _: ["env-hash", "config-hash"]
    )
    monkeypatch.setattr(kernel_warmup.envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(kernel_warmup.envs, "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", None)

    runner = SimpleNamespace(vllm_config=SimpleNamespace())
    cache_hash = sha256(str(["env-hash", "config-hash"]).encode()).hexdigest()

    path = kernel_warmup._resolve_flashinfer_autotune_file(runner)

    assert path == (
        tmp_path
        / "flashinfer_autotune_cache"
        / "0.6.11.post2"
        / "103a"
        / cache_hash
        / "autotune_configs.json"
    )
    assert path.parent.is_dir()


def test_resolve_flashinfer_autotune_file_uses_override_dir(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        kernel_warmup.envs, "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path)
    )
    monkeypatch.setattr(
        kernel_warmup, "aot_compile_hash_factors", lambda _: ["env-hash", "config-hash"]
    )

    runner = SimpleNamespace(vllm_config=SimpleNamespace())
    cache_hash = sha256(str(["env-hash", "config-hash"]).encode()).hexdigest()

    path = kernel_warmup._resolve_flashinfer_autotune_file(runner)

    assert path == tmp_path / cache_hash / "autotune_configs.json"


def test_dspark_uniform_decode_autotune_kwargs_cover_logged_shapes() -> None:
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="dspark",
                num_speculative_tokens=5,
            )
        ),
        model_runner=SimpleNamespace(max_num_tokens=8192, max_model_len=262144),
    )

    kwargs = kernel_warmup._dspark_uniform_decode_autotune_kwargs(worker)

    assert kwargs == [
        {
            "num_tokens": 6,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "uniform_decode": True,
            "profile_seq_lens": 512,
        },
        {
            "num_tokens": 6,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "uniform_decode": True,
            "profile_seq_lens": 2048,
        },
    ]


def test_dspark_uniform_decode_autotune_kwargs_skip_non_dspark() -> None:
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="mtp",
                num_speculative_tokens=5,
            )
        ),
        model_runner=SimpleNamespace(max_num_tokens=8192, max_model_len=262144),
    )

    assert kernel_warmup._dspark_uniform_decode_autotune_kwargs(worker) == []


def test_dspark_warmup_request_counts_cover_single_and_capped_multi() -> None:
    worker = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=8))

    assert kernel_warmup._dspark_warmup_request_counts(worker) == (1, 4)


def test_dspark_store_main_kv_warmup_seq_lens_cover_pruned_and_prefill() -> None:
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="dspark",
                num_speculative_tokens=5,
            )
        ),
        model_runner=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(sliding_window=128)
            )
        ),
    )

    assert kernel_warmup._dspark_store_main_kv_warmup_seq_lens(worker) == (
        1,
        2,
        3,
        4,
        5,
        6,
        128,
    )


def test_dspark_store_main_kv_warmup_uses_pruned_and_prefill_shapes(
    monkeypatch,
) -> None:
    calls = []
    fake_module = ModuleType("vllm.models.deepseek_v4.nvidia.dspark_kernels")

    def fake_store(main_kv_cache, flat_kv, slots, **kwargs):
        calls.append(
            (
                tuple(main_kv_cache.shape),
                tuple(flat_kv.shape),
                tuple(slots.shape),
                kwargs.get("num_rejected_tokens") is not None,
                kwargs.get("request_indices") is not None,
            )
        )
        return main_kv_cache

    fake_module.dspark_store_main_kv = fake_store
    monkeypatch.setitem(
        sys.modules,
        "vllm.models.deepseek_v4.nvidia.dspark_kernels",
        fake_module,
    )

    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="dspark",
                num_speculative_tokens=5,
            )
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        model_runner=SimpleNamespace(
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(head_dim=512, sliding_window=128)
            ),
        ),
    )

    kernel_warmup._deepseek_v4_dspark_store_main_kv_warmup(worker)

    seq_lens = [call[1][1] for call in calls[0::4]]
    assert seq_lens == [1, 2, 3, 4, 5, 6, 128]
    assert all(call[0] == (1, 128, 512) for call in calls)
    assert all(call[2] == (1, call[1][1]) for call in calls)
    assert [(call[3], call[4]) for call in calls[:4]] == [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]


def test_spec_decode_padded_kernel_warmup_uses_dspark_query_width(
    monkeypatch,
) -> None:
    calls = []

    class FakeKernel:
        def __init__(self, name: str) -> None:
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((self.name, grid, args, kwargs))
                if self.name == "next":
                    args[4].fill_(5)

            return launch

    monkeypatch.setattr(
        kernel_warmup,
        "_spec_decode_padded_warmup_kernels",
        lambda: (FakeKernel("next"), FakeKernel("inputs")),
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="dspark",
                num_speculative_tokens=5,
            )
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        model_runner=SimpleNamespace(
            device=torch.device("cpu"),
            model_config=SimpleNamespace(get_vocab_size=lambda: 129280),
        ),
    )

    kernel_warmup._deepseek_v4_spec_decode_padded_kernel_warmup(worker)

    assert [call[0] for call in calls] == ["next", "inputs"]
    next_call = calls[0]
    assert next_call[1] == (1,)
    assert next_call[2][0].shape == (1, 6)
    assert next_call[2][0].tolist() == [[0, 0, 0, 0, 0, -1]]
    assert next_call[2][5] == 129280
    assert next_call[2][6] == 6
    assert next_call[3] == {"BLOCK_SIZE_TOKENS": 8}

    inputs_call = calls[1]
    assert inputs_call[1] == (1,)
    assert inputs_call[2][0].tolist() == [5]
    assert inputs_call[2][2].tolist() == [0, 6]
    assert inputs_call[2][5] == 1


def test_b12x_route_pack_warmup_covers_dspark_short_shapes(
    monkeypatch,
) -> None:
    calls = []

    host_mod = ModuleType("b12x.moe.fused.w4a16.host")
    host_mod.select_route_block_size_m = lambda tokens, topk, experts: 8

    kernel_mod = ModuleType("b12x.moe.fused.w4a16.kernel")

    def fake_pack(topk_ids, block_size, num_experts):
        calls.append((tuple(topk_ids.shape), block_size, num_experts))

    kernel_mod.pack_topk_routes_by_expert = fake_pack
    monkeypatch.setitem(sys.modules, "b12x.moe.fused.w4a16.host", host_mod)
    monkeypatch.setitem(sys.modules, "b12x.moe.fused.w4a16.kernel", kernel_mod)

    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="dspark",
                num_speculative_tokens=5,
            )
        ),
        model_runner=SimpleNamespace(
            device=torch.device("cpu"),
            max_num_tokens=8192,
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(
                    n_routed_experts=256,
                    num_experts_per_tok=6,
                )
            ),
        ),
    )

    kernel_warmup._deepseek_v4_b12x_route_pack_warmup(worker)

    assert calls == [
        ((6, 6), 8, 256),
        ((20, 6), 8, 256),
        ((32, 6), 8, 256),
        ((512, 6), 8, 256),
        ((513, 6), 8, 256),
        ((1024, 6), 8, 256),
    ]


def test_rejection_sampler_warmup_uses_dspark_draft_width(
    monkeypatch,
) -> None:
    calls = []

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return launch

    import vllm.v1.sample.rejection_sampler as rejection_sampler

    monkeypatch.setattr(
        rejection_sampler, "rejection_greedy_sample_kernel", FakeKernel()
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="dspark",
                num_speculative_tokens=5,
            )
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        model_runner=SimpleNamespace(device=torch.device("cpu")),
    )

    kernel_warmup._deepseek_v4_rejection_sampler_warmup(worker)

    assert len(calls) == 1
    grid, args, kwargs = calls[0]
    assert grid == (1,)
    assert args[0].shape == (1, 6)
    assert args[1].tolist() == [5]
    assert args[2].shape == (5,)
    assert args[3].dtype == torch.int64
    assert args[5] is None
    assert args[6] == 5
    assert args[7] is None
    assert args[8] is None
    assert kwargs == {"SYNTHETIC_MODE": False}
