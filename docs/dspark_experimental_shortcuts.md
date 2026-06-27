# DSpark Experimental Shortcuts

This file tracks deliberate simplifications in the experimental DeepSeek V4
Flash DSpark integration. Clear or replace these before promoting the path to a
production branch.

## Current Shortcuts

- DSpark draft attention now uses a first-pass Triton sparse-attention kernel
  over the small DSpark window. It is graph-safe and covered by reference tests,
  but it is not the reference TileLang kernel and is not yet fused or tuned.
- The native sparse-attention path still does not reproduce the reference
  draft-side `act_quant` calls on KV activations. It should be compared
  numerically against the reference and folded into the final kernel path.
- The attention/HC path still uses a simplified intermediate dtype and projection
  sequence before FP8/FP4 quantized linears. This keeps the current runtime path
  valid, but final numerical parity should come from the reference DSpark
  kernels and quantization flow.
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
- DSpark's draft model is integrated as `method="dspark"` with first-pass
  CUDA-graph key initialization and dummy-run support. This is enough for the
  real server to capture graphs, but capture metadata and logging should be
  made more explicit before promoting the path.
- DSpark draft execution wraps the internal draft block in a minimal
  no-attention `ForwardContext` from the proposer so fused MoE kernels can find
  their registered layer objects. This fixes bring-up, but the final integration
  should align the draft pass more tightly with vLLM's normal model-runner
  forward context lifecycle and profiling metadata.
- `dspark_num_draft_layers` is inferred from config target-layer count during
  speculative config override. The real checkpoint has three `mtp.*` stages; a
  weight-index based check exists in the smoke harness and should become a load
  time validation.
- The target model captures DSpark target features by materializing `hc_post`
  at DSpark target layers. This follows the reference use of `h.mean(dim=2)`,
  but needs close numerical validation against the reference script.
- The draft attention path collapses the HC stream dimension with a mean before
  q/k RoPE so the current kernel has one query per token. The production path
  should use the reference DSpark attention/MHC handling instead of this
  approximation.
- The native sparse-attention kernel still materializes attention output before
  borrowing vLLM's production DeepSeek V4 FP8 inverse-RoPE/WO projection
  primitive for `wo_a`/`wo_b`. This avoids the generic DeepGEMM grouped-linear
  rank path, but final DSpark should use a dedicated reference-matched fused
  kernel sequence.
- Loader/embedding/lm-head sharing currently reuses the generic MTP sharing path.
  Keep an eye on logs and loaded-parameter coverage for DSpark-specific names.
- The Docker experiment path reuses the production unholy-fusion entrypoint by
  rewriting its speculative config from `method="mtp"` to `method="dspark"` at
  container startup. The env knob is still named `MTP_NUM_TOKENS`, set to
  DSpark's block size of 5, until the launch path gets a dedicated DSpark
  entrypoint.
- The benchmark harness records OpenAI streaming timings and now supplements
  local tokenizer counts with Prometheus counter deltas. Final throughput
  comparisons should prefer internal engine timing or server-side generation
  token counters over retokenized response text.
- The DSpark experiment currently relies on the existing FlashInfer sparse MLA
  autotune cache. Startup warned that some decode capture shapes fall outside
  the tuned bucket range and fall back to tactic `-1`; add those DSpark shapes to
  the tuning pass before treating throughput numbers as final.
- First interactive real-model runs exposed inference-time Triton JIT gaps for
  request preparation and rejection sampling. Warmup should cover
  `_build_prefill_chunk_metadata_kernel`, route packing,
  `eagle_prepare_next_token_padded_kernel`, `eagle_prepare_inputs_padded_kernel`,
  and `rejection_greedy_sample_kernel`.

## Runtime Validation Notes

- Real two-node DSpark server started with `max_model_len=262144`,
  `method="dspark"`, `num_speculative_tokens=5`, and CUDA graph capture enabled.
- Warm single-stream interactive run, with the 262144-token window available but
  a normal 512-token prompt, generated 256 server-counted tokens in 10.68 s
  end-to-end. Time to first content was 3.16 s and server-counter decode speed
  after first content was 33.94 tokens/s.
- The same warm run produced 153 drafts, 765 draft tokens, and 102 accepted
  draft tokens. Accepted tokens by draft position were `[56, 25, 14, 6, 1]`.
- Fixed-prompt salted repeatability profile on 2026-06-27 ran three times with
  `cache_salt` to avoid prefix-cache reuse. Mean server-counter decode speed was
  37.64 tokens/s with 10.02% CV. Mean accepted draft rate was 14.28% with
  24.03% CV, so acceptance/reference parity is a measured priority before
  treating fused-kernel-only work as the largest speed lever.

## Custom Kernel Opportunities

- TODO P0: continue repeated warm interactive decode profiles before and after
  each kernel change. Track TTFC, server-side generation tok/s, drafts, draft
  tokens, accepted tokens, acceptance by position, and inference-time JIT
  warnings. Initial fixed-prompt salted profile was captured on 2026-06-27.
- TODO P0: add DSpark-specific observability for draft execution stages so we
  can separate target verification time, `prefill_main`/main-KV update time,
  draft sparse attention/projection time, Markov/logit selection time, and
  rejection sampling time.
- TODO P1: add warmup coverage for inference-time JIT gaps observed on the real
  server: request-prep metadata, route packing, EAGLE-named speculative prep,
  and rejection greedy sampling.
- TODO P1: add FlashInfer sparse MLA tuning buckets for DSpark decode and graph
  capture shapes that currently fall back to tactic `-1`.
- TODO P1: implement a fused DSpark sparse-attention kernel that combines score
  calculation, sink-aware softmax denominator, and value accumulation without a
  materialized score buffer.
- TODO P1: fuse DSpark `store_main_kv()` for the single-token decode path across
  norm/projection/RoPE/cache-store work.
- TODO P2: restore reference-parity DSpark `act_quant`, attention/MHC handling,
  and sampling behavior to improve draft acceptance rate, not only draft speed.
- TODO P2: fuse sparse attention output with inverse RoPE, FP8 quantization, and
  `wo_a`/`wo_b` projection.
- TODO P2: fuse Markov-head logits addition, greedy draft token selection, and
  confidence aggregation across the fixed five-token DSpark block.
- TODO P3: move confidence diagnostics and prefix-pruning decisions to a
  GPU-side/asynchronous metrics path.

- Implemented first pass: replace `DeepSeekV4DSparkAttention`'s PyTorch
  sparse-attention loop with a graph-safe native kernel for the serving decode
  shape `[batch, dspark_block_size, local_heads, head_dim]` over DSpark's
  sliding main-KV window plus the same draft block. Next step is fusing and
  tuning this path.
- Implemented first pass: make DSpark draft execution CUDA-graph friendly by
  removing scalar `.item()` reads, Python per-request loops, and shape-varying
  allocations from the sparse-attention hot path. The Markov/logit-selection
  path still has Python-level structure to collapse.
- Near-term speed win: fuse DSpark's main-KV projection/store for the one-token
  decode path so `store_main_kv()` does not run as separate linear/norm/RoPE and
  scatter kernels per draft layer.
- Near-term speed win: add a DSpark-specific fused input preparation kernel that
  builds draft input IDs, draft positions, and warm/cache metadata in one launch.
- Near-term speed win: implement draft-side activation quantization in the kernel
  path to match the reference `act_quant` flow and reduce bandwidth into the
  FP8/FP4 projection sequence.
- Later speed win: fuse sparse attention, inverse RoPE, FP8 quantization, and the
  `wo_a` grouped projection interface so the draft attention output is not
  materialized in bf16 before projection.
- Later speed win: fuse Markov-head logits addition with greedy token selection
  across the fixed DSpark block, avoiding a per-position Python loop and repeated
  vocab-sized temporary handling.
- Later speed win: move confidence-head collection and prefix scheduling to a
  GPU-side reduction path, then export only aggregated metrics asynchronously.
- Later speed win: add dedicated FlashInfer/TileLang autotune buckets for DSpark
  draft and target verification shapes, especially the single-stream
  `block_size=5` decode case and long-context verification windows.
- Later speed win: add warmup coverage or native replacements for request-prep,
  route-packing, EAGLE-named speculative prep, and rejection-sampling kernels
  that still JIT during first interactive inference.
