# DSpark Experimental Shortcuts

This file tracks deliberate simplifications in the experimental DeepSeek V4
Flash DSpark integration. Clear or replace these before promoting the path to a
production branch.

## Current Shortcuts

- DSpark draft attention uses a PyTorch fallback over the small DSpark window
  instead of the reference TileLang sparse attention kernel or a vLLM-native
  optimized attention backend. This is acceptable for correctness bring-up, but
  it will limit benchmark speed.
- The PyTorch attention fallback does not reproduce the reference draft-side
  `act_quant` calls on KV activations. It should be compared numerically against
  the reference and replaced with a kernel path.
- The PyTorch attention/HC fallback explicitly casts intermediate DSpark draft
  activations back to the configured model dtype before FP8/FP4 quantized
  linears. This keeps the current runtime path valid, but final numerical
  parity should come from the reference DSpark kernels and quantization flow.
- DSpark draft KV is kept in an internal sliding-window cache on the draft model
  instead of participating in vLLM's normal KV-cache allocator. This matches the
  reference DSpark shape better, but it needs stronger reset/reorder handling for
  batching, preemption, and long-running mixed workloads.
- The proposer currently assumes uniform flattened per-request target features
  when reshaping target hidden states by batch. Single-stream benchmarking is
  covered; heterogeneous batches need a ragged path.
- The first post-prefill draft call warms DSpark's main-token cache from the
  prompt, then returns a synthetic draft tensor filled with the configured
  DSpark noise token. This keeps vLLM's async speculative path type-stable and
  avoids invalid `-1` probability indexing, but it is not real DSpark output and
  can skew the first verified decode step.
- Draft tokens are selected greedily from DSpark logits. Sampling-temperature
  aware DSpark draft sampling from the reference code is not wired yet.
- Confidence scores are collected into diagnostics, but confidence-based dynamic
  prefix pruning is not yet used to shorten the draft list returned to vLLM.
- Confidence diagnostics currently copy small tensors to CPU during draft
  observation. This is useful while bringing the path up, but it should be
  converted to an asynchronous or aggregated GPU-side path before production
  benchmarking.
- Draft probabilities are not returned for probabilistic rejection sampling.
  The current path targets greedy single-stream benchmarking first.
- DSpark's draft model is integrated as `method="dspark"` with no-op draft
  cudagraph key initialization, dummy-run, and draft attention setup. It should
  gain real capture/profiling support once correctness is established.
- DSpark draft execution wraps the internal draft block in a minimal
  no-attention `ForwardContext` from the proposer so fused MoE kernels can find
  their registered layer objects. This fixes bring-up, but the final integration
  should align the draft pass with vLLM's normal model-runner forward context
  lifecycle and capture/profiling metadata.
- `dspark_num_draft_layers` is inferred from config target-layer count during
  speculative config override. The real checkpoint has three `mtp.*` stages; a
  weight-index based check exists in the smoke harness and should become a load
  time validation.
- The target model captures DSpark target features by materializing `hc_post`
  at DSpark target layers. This follows the reference use of `h.mean(dim=2)`,
  but needs close numerical validation against the reference script.
- The draft attention fallback collapses the HC stream dimension with a mean
  before q/k RoPE so the PyTorch path has one query per token. The production
  path should use the reference DSpark attention/MHC handling instead of this
  approximation.
- The draft attention fallback still computes sparse attention in PyTorch, then
  borrows vLLM's production DeepSeek V4 FP8 inverse-RoPE/WO projection primitive
  for `wo_a`/`wo_b`. This avoids the generic DeepGEMM grouped-linear rank path,
  but final DSpark should use a dedicated reference-matched kernel sequence.
- Loader/embedding/lm-head sharing currently reuses the generic MTP sharing path.
  Keep an eye on logs and loaded-parameter coverage for DSpark-specific names.
- The Docker experiment path reuses the production unholy-fusion entrypoint by
  rewriting its speculative config from `method="mtp"` to `method="dspark"` at
  container startup. The env knob is still named `MTP_NUM_TOKENS`, set to
  DSpark's block size of 5, until the launch path gets a dedicated DSpark
  entrypoint.
- The first real-model benchmark harness estimates decode speed from OpenAI
  streaming chunks and local tokenizer counts. vLLM may coalesce multiple tokens
  per streamed chunk, so this should be replaced or supplemented with internal
  engine timing when comparing final decode throughput.
- The DSpark experiment currently relies on the existing FlashInfer sparse MLA
  autotune cache. Startup warned that some decode capture shapes fall outside
  the tuned bucket range and fall back to tactic `-1`; add those DSpark shapes to
  the tuning pass before treating throughput numbers as final.
