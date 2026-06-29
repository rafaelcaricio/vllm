# DSpark Experimental Shortcuts

This file tracks deliberate simplifications in the experimental DeepSeek V4
Flash DSpark integration. Clear or replace these before promoting the path to a
production branch.

## Current Shortcuts

- DSpark draft attention now uses a first-pass Triton sparse-attention kernel
  over the small DSpark window. It is graph-safe and covered by reference tests,
  but it is not the reference TileLang kernel and is not yet fused or tuned.
- 2026-06-29 request-stable DSpark main-KV slots now cover both write and
  draft-read paths. Writes use the direct selected-row main-KV store kernel.
  Draft reads pass the persistent slot tensor into the sparse-attention Triton
  kernels, so graph-captured draft execution no longer needs a full
  `index_select` copy of the 200k/1M KV row. The remaining shortcut is the
  small CPU request-id to slot assignment map in the proposer; it is bounded by
  `max_batch_size` and stale request IDs are reclaimed every step, but a
  production scheduler could own this mapping more directly.
- The `nvfp4_ds_mla` long-context lane is source-ported as Stage C only:
  DeepSeek V4 keeps the padded 584-byte sparse-MLA envelope while routing
  through the `nvfp4_ds_mla` dtype. This is isolated from the default 262k fp8
  speed path and is not the unresolved true 416-byte compact NVFP4 kernel.
  Runtime profile validation on 2026-06-29:
  `1M/max_num_seqs=1` booted with `2,110,064` KV tokens and `2.01x`
  full-1M concurrency; `1M/max_num_seqs=6` booted with `2,068,655` KV
  tokens and `1.97x`; `200k/max_num_seqs=16` booted with `788,856` KV
  tokens and `3.94x` full-200k concurrency. The mseq16 lane is therefore an
  interactive-concurrency profile, not sixteen simultaneous full-window
  requests.
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
- The proposer now handles ragged mixed prefill+decode target batches with a
  flat ragged DSpark main-KV store path. It passes GPU-resident
  `query_start_loc`, valid lengths, flat positions, and persistent
  request-slot indices into a Triton store kernel instead of grouping equal
  lengths in Python. This clears the earlier dummy-padding and Python grouping
  shortcut. Remaining shortcuts are small scalar boundary synchronizations for
  validation, the CPU request-id to slot map, and a simple per-row request scan
  inside the ragged store kernel rather than a scheduler-owned compact row map.
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
- 2026-06-29 community pull-back repeat: the derived 1M/NVFP4 repo no longer
  has a major source change left to port wholesale. Its stable request-slot idea
  is represented in our write and draft-read paths, and its Stage C
  `nvfp4_ds_mla` launch lane is source-ported as isolated profiles. The latest
  real-model c8 smoke on the current 262k/fp8 lane passed static c8
  (`129.995` aggregate tok/s), staggered c8 (`126.810` aggregate tok/s), and a
  grounded condense/churn test where the victim payload matched byte-for-byte
  while `52` churn requests ran. The matched three-run single-stream baseline
  was `63.019 +/- 0.970` server tok/s with `73.31%` draft acceptance and
  `3.666` accepted tokens per draft. This proves the pull-back is correctness
  neutral for single-stream speed; it does not satisfy the speed goal.
- Warmup-route cleanup is not the next speed milestone. A narrow ragged
  main-KV warmup exists and is unit-tested, but first-request JIT cleanup should
  not displace target-verifier and draft graph/launch work unless a benchmark
  shows it affects steady-state throughput.
- 2026-06-29 draft-stream/bookkeeping overlap pass 1:
  `VLLM_DSPARK_DRAFT_STREAM=1` enqueues DSpark proposal work on a dedicated
  CUDA stream after rejection sampling and fences it with a CUDA event before
  KV-connector finalization and the next default-stream use. This is not
  draft/verify overlap. The next DSpark draft depends on the rejection
  sampler's committed token and accepted target hidden row, so a full draft
  graph cannot legally start before verifier logits/rejection without a
  double-buffered optimistic branch. The implemented stream path can only hide
  draft GPU work behind CPU bookkeeping/output work first; Nsight must verify
  the timeline before promoting it beyond the experiment lane.
- The same pass removes normal-path GPU scalar fences from DSpark target-context
  batching by using `CommonAttentionMetadata.query_start_loc_cpu` for shape and
  ragged/uniform decisions, while keeping GPU tensors for store kernels. GPU
  rejected-count validation remains synchronization-free in throughput mode;
  enable stage timing/diagnostics when debugging invalid metadata.
- DSpark iteration timing now splits `target_postprocess_logits` into
  `target_select_hidden` and `target_compute_logits`, and reports
  `draft_propose_enqueue` / `draft_propose_fence` when the draft stream is
  enabled. These timing modes still synchronize CUDA and must stay off for
  throughput gates; use NVTX/PyTorch profiling scopes for timeline evidence.
- 2026-06-29 DSpark flag-surface cleanup: the Docker control repo now has a
  reduced 262k canonical speed preset and a flag matrix that separates
  canonical flags, active experiments, diagnostics, settled-off toggles, and
  isolated 1M/NVFP4 lanes. The forced-length curve wrapper now inherits the
  env-file baseline instead of silently overriding `VLLM_USE_B12X_WO_PROJECTION`
  back to `0`.
- TODO P1: after overlap/scheduler experiments settle, fold more one-way
  decisions into code defaults or narrow presets. Keep diagnostic and rejected
  experiment flags available for reproduction, but stop carrying them in normal
  speed-lane env files.
- Keep `VLLM_DSPARK_DRAFT_STREAM=1` as the default DSpark evolution lane, not a
  cleanup target. Even though the first safe pass measured only a modest speed
  delta, it is the hook for the paper-aligned pipeline direction: earlier launch
  points, stream/event fencing, and optimistic double-buffered drafts. Do not
  call this production-stable until async lifecycle, request churn, and CUDA
  graph stream-safety validation pass.
- 2026-06-29 draft-stream/deferred-fence pass 2: async DSpark now leaves the
  draft stream event pending across the `sample_tokens()` return instead of
  fencing immediately, except for sync scheduling and KV-connector paths. The
  next default-stream GPU consumers fence lazily before reading
  `valid_sampled_token_count_gpu`, `prev_sampled_token_ids`, or `_draft_token_ids`;
  `sample_tokens()` also fences before clearing/reusing draft state if a prior
  draft was not consumed by the next input prep. This preserves tensor ordering
  while overlapping the draft graph with engine scheduling and next-step CPU
  preparation.
- 2026-06-29 async lifecycle coverage added: unit tests now assert that a
  pending DSpark draft event is waited before draft/proposer buffers are cleared
  for reuse, explicit event fencing does not clear a newer pending event, and
  next-iteration input-id paths fence before reading `_draft_token_ids` for
  unchanged single-stream and reordered/churned survivor batches. These tests
  are necessary but not sufficient for CUDA graph stream safety.
- CUDA graph stream-safety risk remains open. `vllm/compilation/cuda_graph.py`
  still uses the global graph pool and contains a TODO warning that shared graph
  pools may be unsafe with multiple streams. DSpark draft graph replay now runs
  on a dedicated stream, so keep Nsight stress validation on the gate before
  treating `VLLM_DSPARK_DRAFT_STREAM=1` as production-stable.
- The current target/draft dependency blocks full draft/verify overlap: the
  released checkpoint consumes target layers `40,41,42` from a `43`-layer
  target, and the integration exposes those rows after target forward. The
  deferred-capture flag can reduce capture overhead for earlier taps, but layer
  `42` still arrives at the verifier tail. Real overlap work therefore needs a
  target-graph split/event at the DSpark feature write or must pivot to verifier
  kernels/draft FULL-graph fusion for larger gains.
- 2026-06-29 target-tail timing probe: with
  `VLLM_DSPARK_ITER_TIMING=1` and `VLLM_DSPARK_STAGE_TIMING=1`, a real
  1024-token single-stream run measured the legal post-target tail at only
  `~3.0-3.1 ms` (`target_forward_done_to_draft_start`) while
  `draft_propose` was `~11.8-11.9 ms` and the draft graph itself was
  `~10.4-10.6 ms`. This confirms the current draft-stream path can only hide a
  small target-logits/rejection/state-update tail unless the verifier graph is
  split earlier at the DSpark feature write or an optimistic double-buffered
  branch is added without mutating canonical KV before commit. Keep calling the
  existing default path "draft stream / deferred fence / bookkeeping overlap",
  not full draft/verify overlap.
- 2026-06-29 single-stream input assembly direct-copy was tested and reverted.
  Six 1024-token real-model runs averaged slower than the lazy-fence scatter
  path, so the branch keeps the existing indexed assembly and only retains the
  event-lifecycle safety tests.
- 2026-06-29 request-stable slot pullback validation: focused clean-image
  pytest selected-row/condense tests passed (`11 passed`), static c8 1024-token
  smoke passed at `135.45` aggregate tok/s, staggered c8 1024-token smoke
  passed at `130.65` aggregate tok/s, and the grounded condense victim matched
  the expected payload byte-for-byte during churn. The three-run matched
  single-stream benchmark measured `61.50 +/- 2.81` server tok/s versus the
  prior fused-Markov checkpoint `61.75 +/- 3.52`, so the slot fix closes a
  correctness gap without a measurable single-stream speed win.
- 2026-06-29 flat ragged main-KV validation: focused clean-image pytest passed
  on both head and worker (`10 passed`) including CPU reference and Triton
  parity for `dspark_store_main_kv_ragged`. Real-model smoke with
  `max_tokens=1024` passed static c8 (`131.64` aggregate tok/s), staggered c8
  (`126.60` aggregate tok/s), and condense-victim churn (`byte_for_byte_match:
  true`). The matched single-stream benchmark measured `63.46 +/- 1.56`
  server tok/s, within the existing single-stream band. This removes Python
  ragged grouping and selected-row cache-copy shortcuts, but it is not a
  significant single-stream speed win.
- 2026-06-29 post-ragged timing diagnostic: with opt-in synchronization timing
  enabled, late cumulative single-stream averages over `230` iterations were
  `target_forward=72.97ms`, `draft_propose=12.34ms`, and
  `target_postprocess_logits=2.59ms` on the head rank; worker was effectively
  identical. Proposer-stage timing showed `prefill_main=0.79ms`,
  `draft=10.93ms`, and proposer `total=12.05ms`. This isolates target
  verification as the next blocking term and confirms the main-KV/ragged path
  is not the meaningful single-stream bottleneck.
- 2026-06-29 target graph torch profile: a 1024-token single-stream profile on
  the flat-ragged build collected both TP ranks. The largest kernel bucket was
  target-side B12X W4A16 MoE (`~250ms` self CUDA per rank over the profiled
  window), followed by dense fp8/fp4/BF16 GEMM buckets (`~58-93ms` each).
  FlashInfer sparse MLA (`~8.5ms`) and MHC (`~15ms`) were much smaller, so the
  next large single-stream custom-kernel target is verifier MoE/dense GEMM, not
  sparse MLA/MHC first. Claude's review also flagged the sharper speed lever:
  draft-stage wall time is about `12ms`, while the profiler shows much less
  draft GPU work, so PIECEWISE graph/Python/launch/NCCL gaps in the draft path
  need an Nsight Systems timeline before more draft micro-kernel work.

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
- DONE 2026-06-29: replace the mixed prefill+decode Python grouping path and
  selected-row bridge with a flat ragged `prefill_main` update. The proposer now
  forwards GPU `query_start_loc`, valid lengths, positions, and request slots
  to a Triton main-KV store kernel, and the draft-read path already consumes the
  same request-stable slots. Remaining cleanup is scheduler-owned slot metadata,
  eliminating small scalar validation syncs, and replacing the kernel's simple
  per-row request scan with a lower-overhead row-to-request map if profiling
  says it matters.
- DONE 2026-06-29: validate `_dspark_store_main_kv_kernel` warmup on first real
  inference. The warmup pre-JITs no-reject, reject, request-index, and
  reject+request-index flag combinations. A fresh server start with image
  `sha256:dd5d0877318f32a9004f9bd9f1c23d51f4c154a841afc8a055ab070036b41a06`
  served a first chat request without logging `_dspark_store_main_kv_kernel` or
  any Triton JIT warning during inference on either node.
- TODO P1: add FlashInfer sparse MLA tuning buckets for DSpark decode and graph
  capture shapes that currently fall back to tactic `-1`.
- DONE 2026-06-29: profile the captured target-verifier graph with torch
  profiler. Kernel rows point first at verifier MoE/dense GEMM buckets:
  B12X W4A16 MoE `~250ms`, DeepGEMM fp8/fp4 `~93ms`, CUTLASS BF16/s1616
  `~89ms`, and B12X dense GEMM `~58ms` per profiled window. Sparse MLA and MHC
  are secondary in the single-stream trace.
- DONE 2026-06-29: run an Nsight Systems CUDA-profiler-range timeline around
  the draft path and full decode loop. The timeline shows the `~12ms` draft
  term is mostly a real short CUDA graph replay, not just Python/index bridge:
  short graphs average `~10.6ms`, long verifier graphs average `~61.2ms`, and
  serialized graph pairs average `~71.8ms`. Graph launch gaps are only
  `~4ms p50`, so PIECEWISE-to-FULL consolidation alone is not a `>25%` lever.
  Artifacts are under
  `experiments/dspark-benchmarks/profiles/draft_timeline_nsys_20260629_135000`
  in the Docker/control repo.
- TODO P1: run NCU on the target-side B12X W4A16 MoE and dense GEMM kernels at
  the single-stream decode shape, then decide whether the next implementation
  is tile/tactic tuning, a shape-specific kernel, or overlap.
- TODO P1: prototype draft/verify overlap as the next parity-preserving
  single-stream speed lever. Updated nsys bound: fully hiding the short
  `~10.6ms` graph under the `~61.2ms` long verifier graph gives roughly a
  `15-17%` ceiling before secondary gaps, so this is useful but probably not
  enough by itself for the `>25%` speed gate.
- TODO P1: target the long verifier graph for the next large single-stream win.
  Nsys shows the long graph replay itself at `~61ms` per decode cycle, but its
  child kernels are not reliably nested under the graph-trace rows. The
  repeated CUTLASS BF16/s1616 kernel appears immediately after the long graph
  and is likely target-logits projection (`~2.3ms` per verification step).
  B12X/DeepGEMM buckets remain important from the torch profile, but do not
  treat the raw nsys kernel table as direct proof that every bucket is inside
  the decode long graph without an NCU graph-node follow-up.
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
- DONE 2026-06-29: source-port the derived 1M context padded
  `nvfp4_ds_mla` path instead of Dockerfile text patching. The long-context
  lane is isolated in the Docker/control presets and remains Stage C only:
  DeepSeek V4 uses the padded 584-byte sparse-MLA envelope, not the unresolved
  compact 416-byte NVFP4 kernel. Added focused tests that lock
  `nvfp4_ds_mla` to `KVQuantMode.NVFP4`, verify DeepSeek V4 MLA page size is
  `block_size * 584`, and verify the DeepSeek V4 FlashMLA cache shape is
  `(num_blocks, block_size, 584)`.
- DONE 2026-06-29: widen DSpark direct-store warmup request counts from
  `(1, min(max_num_seqs, 4))` to cover the validated concurrency lanes
  `(1, 4, 8, 16)` as available. This prevents the first c8/c16 request from
  paying a live Triton compile for `store_main_kv` flag combinations after a
  clean image rebuild.
- DONE 2026-06-29: add a direct draft-read regression test for the condense
  corruption mode. The test writes distinct persistent main-KV rows and asserts
  `dspark_sparse_attention(..., request_indices=[2, 0])` reads the selected
  request-stable rows rather than compact batch rows.
- DONE 2026-06-29: isolate DSpark draft CUDA graph replay from the target
  graph pool when `VLLM_DSPARK_DRAFT_STREAM=1`. The generic
  `CUDAGraphWrapper` now accepts an explicit graph-pool override; DSpark uses a
  dedicated pool only for draft-stream graph replay. This is correctness
  groundwork for multi-stream overlap, not a standalone speed win. Validation:
  focused graph-pool tests passed and full
  `tests/v1/spec_decode/test_dspark.py` passed in a disposable runtime
  container.
- DONE 2026-06-29: split DSpark target-context projection from canonical
  main-KV store. `DeepSeekV4DSpark.project_main_context()` is now a non-mutating
  projection boundary, while `prefill_main_projected()` and
  `prefill_main_ragged_projected()` perform the request-slot-stable KV writes.
  The proposer records this as `prefill_project` before `prefill_main` in
  stage timing and falls back to the old combined path if a model lacks the new
  methods. This does not move work earlier yet and is not a throughput win by
  itself; it creates the source-level boundary needed for a future stream/event
  experiment or target feature-ready split without changing acceptance or
  canonical cache semantics.
- REVERTED 2026-06-29: removed the DSpark target-context preprojection stream
  prototype after real-model A/B regressed single-stream throughput
  (`52.65 +/- 1.25` tok/s with `max_tokens=1024`, versus recent controls in
  the `~59-64` tok/s band). The non-mutating projection/store split remains,
  but there is no active preprojection stream flag or projected-context side
  path in the runner.

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
