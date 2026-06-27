#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm.v1.spec_decode.dspark import DSparkModelSpec  # noqa: E402


def _load_tensor(
    model_dir: Path, weight_map: dict[str, str], name: str
) -> torch.Tensor:
    filename = weight_map[name]
    with safe_open(model_dir / filename, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate vLLM DSpark assumptions against a real HF checkpoint."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to a DeepSeek-V4-Flash-DSpark Hugging Face snapshot.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    config = json.loads((model_dir / "config.json").read_text())
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    weight_names = tuple(weight_map)

    spec = DSparkModelSpec.from_hf_config({**config, "_weight_names": weight_names})
    assert spec.block_size == 5, spec
    assert spec.noise_token_id == 128799, spec
    assert spec.target_layer_ids == (40, 41, 42), spec
    assert spec.markov_rank == 256, spec
    assert spec.weight_prefix == "mtp.2", spec
    assert spec.num_draft_layers == 3, spec
    assert spec.confidence_input_dim(config["hidden_size"]) == 4352, spec

    required = [
        "mtp.0.main_proj.weight",
        "mtp.0.main_norm.weight",
        "mtp.1.attn.wq_a.weight",
        "mtp.2.norm.weight",
        "mtp.2.confidence_head.proj.weight",
        "mtp.2.markov_head.markov_w1.weight",
        "mtp.2.markov_head.markov_w2.weight",
    ]
    missing = [name for name in required if name not in weight_map]
    if missing:
        raise AssertionError(f"Missing expected DSpark tensors: {missing}")

    if "mtp.1.main_proj.weight" in weight_map or "mtp.2.main_proj.weight" in weight_map:
        raise AssertionError("Only mtp.0 should own DSpark main_proj weights")

    confidence_weight = _load_tensor(
        model_dir, weight_map, "mtp.2.confidence_head.proj.weight"
    ).float()
    markov_w1 = _load_tensor(
        model_dir, weight_map, "mtp.2.markov_head.markov_w1.weight"
    )
    markov_w2 = _load_tensor(
        model_dir, weight_map, "mtp.2.markov_head.markov_w2.weight"
    )

    assert tuple(confidence_weight.shape) == (1, 4352)
    assert tuple(markov_w1.shape) == (config["vocab_size"], spec.markov_rank)
    assert tuple(markov_w2.shape) == (config["vocab_size"], spec.markov_rank)

    token_ids = torch.tensor([0, spec.noise_token_id, config["eos_token_id"]])
    markov_embed = F.embedding(token_ids, markov_w1).float()
    markov_bias = F.linear(markov_embed, markov_w2.float())
    assert tuple(markov_bias.shape) == (token_ids.numel(), config["vocab_size"])
    if not torch.isfinite(markov_bias[:, :1024]).all():
        raise AssertionError("Markov bias contains non-finite values")

    hidden = torch.zeros(token_ids.numel(), config["hidden_size"], dtype=torch.float32)
    confidence_features = torch.cat([hidden, markov_embed], dim=-1)
    confidence_logits = F.linear(confidence_features, confidence_weight)
    confidence = confidence_logits.sigmoid()
    if not torch.isfinite(confidence).all():
        raise AssertionError("Confidence output contains non-finite values")

    print("DSpark real checkpoint smoke passed")
    print(f"model_dir={model_dir}")
    print(f"block_size={spec.block_size}")
    print(f"target_layer_ids={spec.target_layer_ids}")
    print(f"num_draft_layers={spec.num_draft_layers}")
    print(f"weight_prefix={spec.weight_prefix}")
    print(f"confidence_weight_shape={tuple(confidence_weight.shape)}")
    print(f"markov_w1_shape={tuple(markov_w1.shape)}")
    print(f"markov_w2_shape={tuple(markov_w2.shape)}")
    print(f"confidence_sample={confidence.flatten().tolist()}")


if __name__ == "__main__":
    main()
