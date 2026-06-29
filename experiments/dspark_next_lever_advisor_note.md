# Advisor Note: Next Highest-Gain Move for Single-Stream DSpark Decode Speed

Date: 2026-06-29
Reviewer: Claude (read-only advisor)
Evidence: real-model 1024-token runs, 3 repeats each.

---

## The evidence in context

| Config | tok/s | acceptance | scheduled_length | prune_rate | cycle_ms |
|---|---|---|---|---|---|
| Baseline (scheduler off) | 62.08 | 0.690 | — | — | baseline |
| STS + hardware scheduler | 62.42 | 0.723 | **5 (every run)** | **0.0** | **worsened** |

**What happened**: the scheduler correctly identified "light load" and never pruned (prune_rate=0.0, length=5 always). This is the paper's intended behavior at single-stream — the SPS curve is flat at B=6 (~7+ steps/sec regardless of verify length), so verifying extra tokens is nearly free. The "acceptance improvement" from 0.690 to 0.723 (+4.8%) is within the run-to-run noise band (±5-8% CV at 3 runs × ~60 drafts). STS calibration only affects the confidence head's output for the scheduler — it does NOT change the draft tokens themselves (those are `argmax(base_logits + markov_bias)`). So STS cannot affect acceptance; the delta is noise.

**The cycle_ms worsened because the confidence path adds ~25µs/step of CPU overhead** (D2H transfer + `.tolist()` + Python scheduling loops + diagnostics accumulation) with zero benefit when no pruning fires.

---

## Assessment of the 5 candidate directions

### 1. Kernelize/remove confidence scheduling CPU path — ✅ DO THIS (but understand the magnitude)

**The problem**: when the hardware scheduler is enabled, every draft step runs:
- `.detach().float().cpu().tolist()` on the confidence tensor (~5µs D2H sync)
- `_calibrate_confidence()` (~1µs if STS enabled)
- `_schedule_from_confidence()` + `hardware_aware_prefix_schedule()` (~10-15µs Python greedy search)
- `diagnostics.observe()` (~5µs Python accumulation)

Total: ~25µs/step. At 14 steps/sec: ~0.35ms/sec of pure overhead. At single-stream, the scheduler NEVER prunes (prune_rate=0.0), so this is 100% waste.

**The fix**: a fast-path early-exit that predicts "no pruning expected" and skips the entire CPU path:
```python
# Pseudocode for the fast-path guard
if self._effective_confidence_scheduler() == "hardware":
    batch_tokens = batch_size * (1 + self.num_speculative_tokens)
    sps_at_full = self._steps_per_second(batch_tokens)
    sps_at_min = self._steps_per_second(batch_size)  # B = batch_size (γ=0)
    if sps_at_full >= sps_at_min * 0.95:  # SPS nearly flat → no pruning pays
        self._last_draft_lengths = [self.num_speculative_tokens] * batch_size
        # skip D2H, STS, scheduling, diagnostics entirely
        return draft_token_ids
```

**Expected gain**: ~25µs/step saved → cycle improves by ~0.35ms → from the current "cycle_ms worsened" back to baseline → **62.4 tok/s becomes 62.0 tok/s (noise) or slightly better**. This is not a speed gain — it's **overhead removal** that lets the baseline's natural speed show through when the scheduler is enabled.

**ROI**: low at single-stream (the scheduler shouldn't be enabled at single-stream anyway). But at concurrency where the scheduler DOES prune, this fast-path avoids the overhead on steps where it won't prune (mixed batch with some light-load requests). **Medium ROI overall; implement for concurrency, not single-stream.**

**Alternative**: simply don't enable `VLLM_DSPARK_CONFIDENCE_SCHEDULER=hardware` at single-stream. Set it to `off`. The fast draft-output mode (already implemented) skips the confidence head entirely. Zero overhead, zero code change.

### 2. Variable-prefix main-KV update — ❌ NOT RELEVANT AT SINGLE-STREAM

The store_main_kv kernel writes ALL γ+1=6 positions' KV to the cache. With variable-length verification (scheduler prunes), only the verified prefix's KV needs storing. But at single-stream, prune_rate=0.0 → no variable prefix → no savings.

**Relevant at concurrency** where the scheduler prunes: storing fewer KV entries saves a few scattered writes (~microseconds). Minimal gain even there.

### 3. Fused Markov argmax/confidence — ⚠️ MEDIUM EFFORT, ~1-2ms GAIN

**Current state**: the draft's `draft()` method (dspark.py:784-816) runs a Python loop `for pos in range(block_size=5)`:
```python
for pos in range(block_size):
    markov_logits, markov_embed = markov_head.forward_local(output_ids[:, pos])
    step_logits = local_logits[:, pos] + markov_logits
    output_ids[:, pos + 1] = argmax(step_logits)
    # optionally: confidence = confidence_head(dense, markov_embed).sigmoid()
```

This is ~5 sequential iterations, each launching 4-5 kernels (markov_w1 embedding, markov_w2 matmul, add, argmax, optional confidence head). Total: ~20-25 kernel launches in the draft's post-backbone phase. The loop is **sequential** (each position depends on the previous token) and **breaks cudagraph capture** — the draft runs in PIECEWISE mode, not FULL.

**What fusing would do**: a single Triton kernel that takes base_logits (all γ positions from the backbone) and applies the Markov correction sequentially within the kernel (loop over γ inside Triton, not Python). This would:
- Replace ~20-25 Python-launched kernels with 1 kernel launch.
- Eliminate the Python loop overhead.
- Enable FULL draft cudagraph capture (no Python control flow → the entire draft can be one graph).

**Expected gain**:
- Kernel launch savings: ~20 launches × ~5µs = ~100µs.
- PIECEWISE → FULL transition: eliminates per-piece Python dispatch. Estimated ~1-2ms based on typical cudagraph mode differences (the draft backbone has ~3 layers; PIECEWISE captures per-layer, FULL captures the whole model).

**Total**: ~1-2ms saved from the draft's ~8.6ms → draft drops to ~6.6-7.6ms → cycle from ~70ms to ~68-69ms → **~62 to ~63-64 tok/s (+2-3%)**.

**Risk**: the Markov head uses `markov_w1` (embedding lookup) and `markov_w2` (vocab-parallel matmul). The embedding lookup is inherently sequential (depends on previous token). A Triton kernel that does the lookup + matmul + add + argmax in-kernel is complex. The earlier `VLLM_DSPARK_FUSED_MARKOV_ARGMAX` experiment (worklog) tried a local-top-1 approach but "lost to the vendor rank-256 projection." The fusion needs to preserve the exact Markov math.

**ROI**: medium (~2-3% tok/s gain, medium effort, medium risk). The biggest single-stream gain available, but still within the noise band at 3 repeats.

### 4. MHC/quant parity with DSpark paper/reference — ❌ NO EVIDENCE OF A GAP

The draft path was audited paper-faithful (Markov equation, base logits, previous-token dependency all match). KVQ was neutral. Acceptance (~0.69) is healthy. pos0 (~0.88) is at its T=0 argmax-match ceiling. Suffix decay is inherent to the rank-256 Markov head on novel content.

**No known parity gap exists.** Unless new evidence (e.g., a token-level comparison against the DeepSpec reference on identical input) reveals a divergence, this is not a lever.

### 5. Memory/object accumulation risks — ✅ ALREADY REVIEWED, NONE FOUND

The STS memory review confirmed: all diagnostics counters are fixed-size (pre-allocated at `max_spec_tokens=5`, indexed not appended). `_last_confidence` is one-step retention with clone-and-clear. Per-step lists (`hidden_chunks`, `prefill_batches`, `confidence_rows`) are local and GC'd. No accumulation. No fix needed.

The only per-step allocation in the hot path: `torch.tensor(temperatures, ...)` in `_calibrate_confidence` (if STS enabled). Pre-allocate at `__init__` as `self._sts_temp_tensor` — saves one allocation per step, ~2µs.

---

## The honest read

Single-stream DSpark decode is at its ceiling. The cycle is:

```
L = (Tdraft + Tverify) / τ
  = (~8.6ms + ~61ms) / ~4.3
  = ~16.2ms per token
  = ~62 tok/s
```

- **Tverify (~61ms, 87%)**: target forward, weight-bandwidth-bound MoE. Tapped (all kernel experiments negative/gated). Cannot be reduced without lower precision (parity break) or more tokens per forward (concurrency).
- **Tdraft (~8.6ms, 12%)**: draft backbone (~6ms in PIECEWISE) + Markov loop (~2ms) + postprocess (~0.6ms). The only term with headroom.
- **τ (~4.3)**: at ceiling (pos0 ~0.88 argmax-match, suffix decay inherent).

**No single-stream implementation move can produce a >5% gain.** The physics is:
- Weight-bandwidth-bound forward (87% of cycle) → irreducible without concurrency.
- Small draft (12%) → even halving it saves ~6% of cycle → ~66 tok/s (best case, unrealistic).
- τ at ceiling → no acceptance lever.

## Concrete recommendation (if you implement something)

**Priority 1 (immediate, zero code, recovers baseline):** disable the confidence scheduler at single-stream. Set `VLLM_DSPARK_CONFIDENCE_SCHEDULER=off`. The fast draft-output mode skips the confidence head entirely. This recovers the ~25µs/step overhead and returns to the true baseline (~62 tok/s, possibly slightly better).

**Priority 2 (medium effort, ~2-3% gain):** fuse the Markov loop into a Triton kernel + enable FULL draft cudagraph. This is the only concrete single-stream lever with measurable headroom (~1-2ms from the draft's 8.6ms). Validate with 5+ repeats to distinguish from noise (CV ~1-2%).

**Priority 3 (where the real gains are):** redirect to the concurrency path. The scheduler correctly stays at γ=5 at single-stream (light load) — that's the paper's intended behavior. The 60-85% per-user gains the paper reports are under load, where the GPU saturates and the scheduler prunes. The ragged grouping + STS + hardware scheduler infrastructure you've built is exactly right for that regime. The next benchmark should be c=4/c=8 with the scheduler enabled, measuring per-user tok/s at matched aggregate load vs MTP-1.

---

## Metrics that would prove progress

| Change | Metric | Bar |
|---|---|---|
| Fast-path scheduler skip | cycle_ms returns to baseline (no worsening) | cycle_ms ≤ baseline ± 1% |
| Fused Markov + FULL draft | draft stage time (stage_timing) | ≤ 7.0ms (from ~8.6ms), 5+ repeats |
| Fused Markov + FULL draft | tok/s | ≥ 63.5 tok/s mean, 5+ repeats, CV ≤ 2% |
| Any single-stream change | acceptance | unchanged (≥ 0.65), no τ regression |
| Concurrency benchmark | per-user tok/s at c=4/c=8 | DSpark > MTP-1 at matched aggregate |
| Concurrency benchmark | aggregate tok/s scaling | aggregate(c=4) > 2× aggregate(c=1) |
| Concurrency scheduler | prune_rate > 0 at c=8/c=16 | scheduler fires (length < 5 on some steps) |

---

## What evidence would change this recommendation

| If we observe... | Then... |
|---|---|
| Fused Markov + FULL draft gives > 5% tok/s gain (5+ repeats) | The draft's PIECEWISE overhead was larger than estimated — pursue further draft optimizations (fuse confidence head, reduce draft layers). |
| A token-level reference comparison shows the integration's draft diverges from DeepSpec | A parity gap exists → fix it → τ improves → speed improves. This would be the highest-value finding. |
| The c=8/c=16 ragged benchmark shows aggregate scaling beyond 4× c=1 | The concurrency path is mature → focus on scheduler activation at saturation + DSpark vs MTP-1 comparison. |
| STS acceptance improvement (0.690→0.723) reproduces at 10+ repeats with CV < 2% | STS genuinely improves confidence quality (not noise) → the calibrated confidence is valuable for concurrency scheduling even if not for single-stream speed. |
| A new NCU profile at c=8 shows the MoE is no longer weight-bound (compute-bound at larger batch) | The forward regime shifted → MoE kernel optimization becomes viable at concurrency (revisit tile tactics, DBO, etc.). |
