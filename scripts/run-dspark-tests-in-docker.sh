#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${DSPARK_VLLM_DEV_IMAGE:-vllm-dspark-dev:local}"

if [ "$#" -eq 0 ]; then
  set -- tests/v1/spec_decode/test_dspark.py -q
fi

docker build \
  -f "${ROOT_DIR}/docker/Dockerfile.dspark-dev" \
  -t "${IMAGE}" \
  "${ROOT_DIR}"

docker run --rm \
  --entrypoint /bin/bash \
  -v "${ROOT_DIR}:/workspace/vllm" \
  -w /workspace/vllm \
  "${IMAGE}" \
  -lc 'uv run --active --no-sync python -m pytest "$@"' \
  bash "$@"
