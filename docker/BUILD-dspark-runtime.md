# Building `vllm-dspark-runtime:clean`

The runtime image used to serve DeepSeek-V4-Flash DSpark (the validated best single-stream
config — see `bjk110_spark-vllm-docker/experiments/dspark-benchmarks/best_single_stream_config.md`).

`vllm-dspark-runtime:clean` is a **thin overlay** on top of the prebuilt
unholy-fusion base image. It does **not** compile vLLM — it only copies the DSpark
source overlay files into the installed package and `py_compile`s them. Build time is
minutes (not the hours a full vLLM/CUDA build takes).

## What it is

- **Base:** `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready` (prebuilt; pulled from ghcr).
- **Overlay:** `docker/Dockerfile.dspark-runtime-overlay` — `FROM` the base, then `COPY`s a
  fixed set of DSpark-modified vLLM source files over `/opt/env/lib/python3.12/site-packages/vllm/...`
  and runs `py_compile` on them.
- **Tag:** `vllm-dspark-runtime:clean` (label `org.bjk110.dspark.runtime="clean-overlay"`).
  A sibling `:local` tag is an older/local variant; **`:clean` is the validated one.**

The overlay files (the only fork sources baked into the image — changing any other file
does **not** affect the image unless you add it to the Dockerfile's `COPY` list):
`vllm/envs.py`, `vllm/config/speculative.py`, `vllm/model_executor/layers/fused_moe/b12x_moe.py`,
`vllm/model_executor/warmup/kernel_warmup.py`, `vllm/model_executor/models/registry.py`,
`vllm/v1/outputs.py`, `vllm/v1/core/sched/scheduler.py`,
`vllm/models/deepseek_v4/{__init__,nvidia/model,nvidia/sm120,nvidia/dspark,nvidia/dspark_kernels}.py`,
`vllm/v1/attention/backends/registry.py`, `vllm/v1/attention/backends/mla/b12x_mla_sparse.py`,
`vllm/v1/spec_decode/{dspark,dspark_proposer,metrics}.py`,
`vllm/v1/worker/gpu_model_runner.py`.

## Prerequisites (per node)

Build is done **on each node** (head `192.168.250.12` and worker `192.168.250.13`) — both must
end up with the identical image for TP=2. Each node needs:

1. This fork checkout at `/home/pieter/Code/vllm-dspark-unholy` (on the `codex/dspark-harness-integration`
   branch, same commit on both nodes).
2. Docker.
3. The base image present:
   ```bash
   docker pull ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready
   # (if the ghcr package is private, ensure you are authenticated: docker login ghcr.io -u <user> --password-stdin <PAT>)
   ```

## Build (run on EACH node, from the fork root)

```bash
cd /home/pieter/Code/vllm-dspark-unholy
docker build -f docker/Dockerfile.dspark-runtime-overlay -t vllm-dspark-runtime:clean .
```

- The build context is the fork root (`.`); the Dockerfile's `COPY vllm/...` paths resolve from there.
- `:clean` is the tag referenced by `.env.dspark-experiment` (`VLLM_IMAGE=vllm-dspark-runtime:clean`)
  and used by `docker-compose.yml`'s `${VLLM_IMAGE}`.

## Verify

```bash
docker images vllm-dspark-runtime:clean
docker inspect vllm-dspark-runtime:clean --format '{{json .Config.Labels}}'   # org.bjk110.dspark.runtime=clean-overlay
# confirm an overlay file is present in the image, e.g.:
docker run --rm --entrypoint /opt/env/bin/python vllm-dspark-runtime:clean -c "import vllm.model_executor.warmup.kernel_warmup as m; print('warmup ok', hasattr(m,'_deepseek_v4_b12x_route_pack_warmup'))"
```

## When to rebuild

Rebuild **on both nodes** whenever an overlay-tracked file changes in the fork (e.g. after a
`kernel_warmup.py`, `dspark*.py`, `b12x_moe.py`, `gpu_model_runner.py` edit). Files **not** in the
overlay `COPY` list are not baked in — if you need a new file in the image, add it to
`docker/Dockerfile.dspark-runtime-overlay` first.

After rebuilding, restart the stack **head-first** (master waits, worker connects):
```bash
ENV=/home/pieter/Code/bjk110_spark-vllm-docker/.env.dspark-experiment
DC=/home/pieter/Code/bjk110_spark-vllm-docker/docker-compose.yml
# head node:
docker compose --env-file "$ENV" -f "$DC" --profile head up -d   # recreates with new image
sleep 25
# worker node (ssh):
ssh 192.168.250.13 "docker compose --env-file $ENV -f $DC --profile worker up -d"
# poll until /health -> 200 (~5 min: model load + graph capture)
```

## Other images (do not confuse with the runtime)

- `vllm-dspark-dev:local` — built from `docker/Dockerfile.dspark-dev`; used only by
  `scripts/run-dspark-tests-in-docker.sh` and `run-dspark-real-checkpoint-smoke-in-docker.sh` for
  running pytest / the checkpoint smoke test. **Not** the serving image.
- The base `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready` is the full unholy-fusion vLLM
  build; the `:clean` overlay only swaps in the DSpark source files on top of it.
