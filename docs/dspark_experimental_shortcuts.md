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
- The proposer now handles ragged mixed prefill+decode target batches by
  grouping requests with equal target-context lengths and issuing compact
  selected-row DSpark main-KV updates. This fixes the former uniform-reshape
  crash and removes dummy placeholder projection work. The selected-row cache
  write now uses a direct DSpark main-KV store kernel instead of
  `index_select`/`index_copy_`, but the path still relies on Python grouping
  and small CPU metadata reads. A production path should move grouping and
  metadata handling closer to the scheduler/GPU execution layer.
- 2026-06-29 real-model c16 A/B: direct main-KV store measured
  `319.24 +/- 6.32` aggregate tok/s versus prior scheduler-off baseline
  `319.37 +/- 6.72` aggregate tok/s. This removes a correctness/overhead
  shortcut but does not move steady-state decode throughput; the bottleneck is
  still elsewhere in verification/sparse MLA/rejection plumbing.
- The experimental runtime overlay now installs `gcc` and `libc6-dev` because a
  clean container could not compile Triton's launcher for the new direct-store
  kernel. This keeps first-run JIT and container-side tests deterministic, but
  should be revisited once kernels are fully precompiled or shipped with an
  explicit cache.
- The ragged mixed-batch opt-in is still named `VLLM_DSPARK_MULTI_SEQ_PAD`
  from the earlier placeholder-row implementation. It now enables ragged
  grouping rather than padding, so rename it or leave an explicit compatibility
  comment before this path graduates from experiment status.
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` is not part of the current
  concurrent benchmark config. Add a clear guard before combining that GPU-mask
  mode with ragged mixed prefill+decode batches, so future tests fail with an
  explicit unsupported-mode message instead of a uniform-shape assertion.
- The first post-prefill draft call warms DSpark's main-token cache from the
  prompt, then returns a synthetic draft tensor filled with the configured
  DSpark noise token. This keeps vLLM's async speculative path type-stable and
  avoids invalid `-1` probability indexing, but it is not real DSpark output and
  can skew the first verified decode step.
- Draft tokens are selected greedily from DSpark logits. Sampling-temperature
  aware DSpark draft sampling from the reference code is not wired yet.
- The DeepSpec paper's offline evaluation uses sampling temperature `1.0`, and
  the released model card recommends `temperature=1.0, top_p=1.0`. Current
  performance runs are greedy `temperature=0.0`, so they intentionally exercise
  a narrower path.
- Confidence-scheduled verification has a first-pass static threshold knob
  (`VLLM_DSPARK_CONFIDENCE_THRESHOLD`) that prunes each request to the longest
  cumulative-survival prefix above the threshold before vLLM verification. This
  is not yet the full hardware-aware dynamic scheduler from the paper.
- Async scheduling now carries DSpark's per-request confidence prefix lengths
  through `ModelRunnerOutput.draft_token_lengths` so the next speculative
  placeholder list is variable length. This is a minimal bridge; it does not yet
  use a scheduler cost model, bucketing, or GPU-side prefix selection.
- Confidence diagnostics and prefix-length decisions currently copy small
  tensors to CPU during draft observation. This is useful while bringing the
  path up, but it should be converted to an asynchronous or aggregated GPU-side
  path before production benchmarking. Use
  `VLLM_DSPARK_CONFIDENCE_DIAGNOSTICS_LOG_EVERY` for dedicated calibration
  runs; keep it `0` during throughput gates.
- 2026-06-29 single-stream SPS-dominance guard: when
  `VLLM_DSPARK_CONFIDENCE_SCHEDULER=hardware` is enabled but the profiled local
  SPS curve is dominated by the full DSpark prefix, the proposer now skips the
  confidence head and CPU scheduler path and verifies the full prefix. Real
  1024-token A/B showed this correctly logged on both ranks and returned to
  baseline behavior (`61.75` tok/s mean vs `62.08` scheduler-off baseline);
  it is overhead hygiene, not a new speed lever. Keep pushing hot prefix
  decisions toward GPU-side reductions instead of expanding CPU policy.
- 2026-06-29 full-prefix draft lengths no longer cross the async scheduler
  bridge as a Python list. DSpark now returns `None` for draft lengths when
  every request uses the configured full prefix, leaving vLLM's normal fixed
  speculative placeholder path intact. Only shortened prefixes return explicit
  per-request lengths. A 3x 1024-token real-model run measured `61.64 +/- 1.86`
  tok/s versus the `62.08 +/- 0.70` scheduler-off baseline, so this removes
  unnecessary CPU plumbing but does not explain the remaining decode bottleneck.
- 2026-06-29 `VLLM_DSPARK_FUSED_MARKOV_ARGMAX=1` was A/B tested with 3x
  1024-token real-model single-stream runs. It measured `61.75 +/- 3.52`
  tok/s versus the `62.08 +/- 0.70` scheduler-off baseline. The fused local
  Markov argmax avoids materializing local Markov logits, but it still needs the
  per-position TP top-1 reduction because the base LM-head logits remain
  vocab-sharded.
- The DeepSpec reference Markov head uses a full-vocab `W2`, but in this TP=2
  vLLM integration the draft base logits come from the target model's
  vocab-parallel `lm_head`. Replicating only Markov `W2` does not remove the
  global argmax communication. A paper-faithful no-TP-reduce draft-output path
  would need a draft-local replicated output head, a specialized low-latency
  GPU reduction, or another way for each rank to see full corrected logits.
- `VLLM_DSPARK_STS_CALIBRATION_DIAGNOSTICS=1` adds an opt-in calibration-label
  stream for fitting STS temperatures from real acceptance outcomes. It copies
  only the current verified draft's raw confidence row, then collapses samples
  into fixed-size per-position confidence-bin counters before logging. Keep it
  disabled for throughput gates; do not add per-request or per-step Python
  history here because long server runs must stay memory bounded.
- Draft probabilities are not returned for probabilistic rejection sampling.
  The current path targets greedy single-stream benchmarking first.
- Confidence scheduling consumes sigmoided probabilities today. The paper's STS
  calibration is now available as `VLLM_DSPARK_STS_TEMPERATURES`, which
  reconstructs confidence logits from probabilities, applies one global or
  per-position temperature, and records raw-vs-calibrated diagnostics. The
  optional diagnostics log now reports scheduled-length histograms, expected
  acceptance, raw-vs-calibrated confidence, survival, and scheduled fractions.
  Per-position temperature tensors are preallocated on proposer init. Real
  calibration scalars still need to be fit from held-out local data before this
  should be promoted as a default.
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
- Speculative decode metrics now export
  `vllm:spec_decode_num_drafts_by_draft_length_total`, and the DSpark runtime
  overlay copies `vllm/v1/spec_decode/metrics.py` into the packaged image. Keep
  this generic metric in place; it is the direct signal for whether dynamic
  prefix scheduling is actually pruning verification length.
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
- First-pass confidence scheduling is enabled in the Docker experiment with
  `VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.50`. Repeated real-model benchmarks are
  still needed to determine whether this threshold improves single-stream
  interactive decode speed or only raises acceptance rate by pruning too
  aggressively.
- Corrected async-bridge post-JIT profile on 2026-06-27 with threshold `0.50`
  ran three salted single-stream repetitions. Mean server-counter decode speed
  was 38.20 tokens/s with 41.76% CV, roughly matching the 37.64 tokens/s
  fixed-length baseline but with much worse variance. Mean scheduled draft
  tokens dropped from 5.00 to 3.59 per draft and mean accepted draft rate rose
  from 14.28% to 30.06%. This proves pruning is active, but static threshold
  `0.50` is not a clear speed win yet.
- Goal gate: do not treat the confidence bridge as complete until repeated
  real-model single-stream decode speed is recorded and increases
  significantly. The current result explains why: pruning and acceptance can
  improve while variable-prefix scheduling, route packing, rejection sampling,
  or confidence miscalibration erase the throughput gain.

## Custom Kernel Opportunities

- TODO P0: continue repeated warm interactive decode profiles before and after
  each kernel change. Track TTFC, server-side generation tok/s, drafts, draft
  tokens, accepted tokens, acceptance by position, and inference-time JIT
  warnings. Initial fixed-prompt salted profile was captured on 2026-06-27.
- TODO P0: add DSpark-specific observability for draft execution stages so we
  can separate target verification time, `prefill_main`/main-KV update time,
  draft sparse attention/projection time, Markov/logit selection time, and
  rejection sampling time.
- TODO P0: profile the TP top-1 reduction inside the DSpark Markov loop. There
  are five sequential per-position reductions in the default `gamma=5` path.
  Because base logits are sharded, a `markov_w2` replication-only shortcut is
  insufficient; compare a draft-local replicated output-head prototype against
  a custom low-latency pair-reduction path before promoting either direction.
- TODO P0: validate opt-in FlashInfer allreduce on SM121/world_size=2 with an
  explicit `VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB` override. The
  upstream default table has no tuned GB10 entry, so keep this off by default
  until startup logs prove the backend is active and repeated benchmarks show a
  decode-speed gain.
- TODO P0: run a threshold sweep over `VLLM_DSPARK_CONFIDENCE_THRESHOLD`
  against the real model and record tok/s, scheduled length, prune rate,
  accepted tokens, and acceptance by position. Compare static thresholding with
  the paper's hardware-aware scheduler before promoting a default.
- TODO P0: run an STS calibration diagnostic pass with
  `VLLM_DSPARK_STS_CALIBRATION_DIAGNOSTICS=1`,
  `VLLM_DSPARK_STS_TEMPERATURES=` empty, and scheduler decisions otherwise
  unchanged. Fit per-position temperatures from the logged confidence-bin
  acceptance labels before enabling STS in speed gates.
- TODO P0: use the threshold sweep as the next speed gate. Test thresholds such
  as `0.20`, `0.35`, and `0.50` with three post-JIT salted repetitions each;
  keep only changes that materially beat the 37.64 tokens/s fixed-length
  baseline without exploding CV.
- TODO P0: profile the variable-prefix async bridge overhead. The corrected
  threshold `0.50` run reduced verified draft tokens but increased tok/s
  variance, so measure scheduler placeholder updates, ragged metadata creation,
  route packing, and rejection-sampling shape changes.
- TODO P0: implement the paper-aligned hardware-aware prefix scheduler using a
  profiled local SPS curve now that ragged mixed batches can preserve request
  row identity. Compare c=4/c=8 per-user and aggregate tok/s against the
  compact-ragged checkpoint before treating scheduler work as a win.
- TODO P0: reprofile the hardware scheduler in a regime where it can actually
  choose shorter prefixes. The first c=4/c=8 curve mostly scheduled full
  length (`{4: 1, 5: 121}` in a c=4 smoke), c=4 was flat, and c=8 regressed
  versus scheduler-off controls. Fill the B=24..48 SPS gap with c=8 forced
  lengths and test c=16 / `MAX_NUM_SEQS=16` before changing the policy.
- TODO P1: add warmup coverage for inference-time JIT gaps observed on the real
  server: request-prep metadata, route packing, EAGLE-named speculative prep,
  and rejection greedy sampling.
- TODO P1: replace the mixed prefill+decode Python grouping path and
  selected-row bridge with a scheduler/GPU-native ragged `prefill_main` update
  that preserves request row identity and avoids Python grouping. The
  selected-row cache write itself is now direct-kernel based; the remaining
  overhead is grouping, compact tensor construction, and launch fragmentation.
- DONE 2026-06-29: validate `_dspark_store_main_kv_kernel` warmup on first real
  inference. The warmup pre-JITs no-reject, reject, request-index, and
  reject+request-index flag combinations. A fresh server start with image
  `sha256:dd5d0877318f32a9004f9bd9f1c23d51f4c154a841afc8a055ab070036b41a06`
  served a first chat request without logging `_dspark_store_main_kv_kernel` or
  any Triton JIT warning during inference on either node.
- TODO P1: add FlashInfer sparse MLA tuning buckets for DSpark decode and graph
  capture shapes that currently fall back to tactic `-1`.
- TODO P1: implement a fused DSpark sparse-attention kernel that combines score
  calculation, sink-aware softmax denominator, and value accumulation without a
  materialized score buffer.
- TODO P1: fuse DSpark `store_main_kv()` for the single-token decode path across
  norm/projection/RoPE/cache-store work.
- TODO P2: restore reference-parity DSpark `act_quant`, attention/MHC handling,
  and sampling behavior to improve draft acceptance rate, not only draft speed.
- TODO P2: add a low-overhead first-token quality diagnostic. The latest
  paper-style conditional acceptance read shows healthy suffix acceptance in
  steady runs but large position-0 variance, so first-token draft quality is a
  prime acceptance suspect.
- TODO P2: fuse sparse attention output with inverse RoPE, FP8 quantization, and
  `wo_a`/`wo_b` projection.
- TODO P2: fuse Markov-head logits addition, greedy draft token selection, and
  confidence aggregation across the fixed five-token DSpark block.
- TODO P3: move confidence diagnostics and prefix-pruning decisions to a
  GPU-side/asynchronous metrics path.
- TODO P3: port the derived 1M context padded `nvfp4_ds_mla` path as source
  changes rather than Dockerfile text patches. Keep it as a separate launch
  profile from the current 262k fp8 DSpark scheduler work, and validate with
  `/v1/models` max length, KV-pool logs, short decode probes, and later a real
  long-context retrieval/correctness prompt.

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
  direct-store kernels per draft layer. The selected-row cache write is now a
  dedicated direct-store kernel, but projection and RoPE are still separate.
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
