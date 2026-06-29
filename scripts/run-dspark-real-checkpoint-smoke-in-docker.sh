#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${DSPARK_VLLM_DEV_IMAGE:-vllm-dspark-dev:local}"
MODEL_DIR="${DSPARK_MODEL_DIR:-/home/pieter/.cache/huggingface-dspark/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/snapshots/913f0657a874f76844e2e91cbe706dbcaceeb6d7}"

if [ ! -f "${MODEL_DIR}/config.json" ]; then
  echo "Missing model config at ${MODEL_DIR}/config.json" >&2
  exit 1
fi

docker build \
  -f "${ROOT_DIR}/docker/Dockerfile.dspark-dev" \
  -t "${IMAGE}" \
  "${ROOT_DIR}"

docker run --rm \
  --entrypoint /bin/bash \
  -e "DSPARK_MODEL_DIR=${MODEL_DIR}" \
  -e "PYTHONPATH=/workspace/vllm" \
  -v "${ROOT_DIR}:/workspace/vllm" \
  -v "/home/pieter/.cache/huggingface-dspark:/home/pieter/.cache/huggingface-dspark:ro" \
  -w /workspace/vllm \
  "${IMAGE}" \
  -lc 'uv run --active --no-sync python scripts/dspark-real-checkpoint-smoke.py --model-dir "$DSPARK_MODEL_DIR"'
