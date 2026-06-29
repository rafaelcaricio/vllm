# Advisor Note: Next GPU-Resident Implementation Target for DSpark Decode Speed

Date: 2026-06-29
Reviewer: Claude (read-only advisor)
Evidence: 62.08 tok/s baseline, scheduler overhead confirmed as pure waste at single-stream (prune_rate=0.0). User preference: GPU-resident/fused, avoid CPU.

Codex follow-up: this note correctly identifies the Markov loop as a
GPU-resident optimization target, but the `markov_w2` replication step is not
sufficient by itself in the current TP=2 vLLM path. The draft base logits come
from the target model's vocab-parallel `lm_head`, so a rank still cannot know
the global corrected argmax without either a per-position TP top-1 reduction, a
draft-local replicated output head, or another full-corrected-logits path. Treat
W2 replication as part of a larger output-head/argmax redesign, not as a
standalone comm-elimination.

---

## The target: eliminate the Markov loop's NCCL serialization, then fuse it for FULL draft graph

### Why this is the highest-gain GPU-resident move

The draft currently runs in **PIECEWISE** cudagraph mode (dspark_proposer.py:610). The draft backbone layers are captured as graph pieces. But the **Markov head loop** runs eagerly — 5 Python iterations, each with a **TP all-gather** to find the global argmax across ranks.

Each Markov loop iteration:
1. `markov_w1(token)` — GPU embedding lookup (replicated, no comm) ✓
2. `markov_w2(embed)` → local logits — GPU matmul (vocab-parallel, sharded) 
3. `base + markov_logits` → step_logits — GPU elementwise
4. `_vocab_parallel_argmax(step_logits, lm_head)` → **`tensor_model_parallel_all_gather`** — NCCL sync (~40µs RDMA per call)
5. Global argmax from gathered tensor — GPU

**5 iterations × 1 NCCL all-gather each = 5 CPU-blocking synchronization points in the Markov loop.** The CPU must wait for each all-gather to complete before issuing the next iteration's kernels. The GPU pipeline stalls between iterations.

At ~40µs RDMA latency per all-gather: 5 × 40µs = **~200µs of NCCL serialization** in the Markov loop. Plus the CPU cannot issue all 5 iterations' kernels back-to-back (each waits for the previous all_gather). Total wall time: ~225µs for the loop (vs ~25µs if it were pure GPU pipelined).

### The 3-step fix (all GPU-resident)

#### Step 1: Replicate markov_w2 (trivial, eliminates all NCCL from the Markov loop)

`markov_w1` is already replicated (`VLLM_DSPARK_REPLICATE_MARKOV_W1=1`). **Replicate `markov_w2` too**: each rank holds the full [256, vocab] projection → each rank computes full-vocab logits → local argmax is the global argmax → **no all_gather needed**.

- Memory: markov_w2 goes from [256, vocab/2] to [256, vocab] per rank = +33MB per rank (at vocab~129k, bf16). Negligible on GB10 (128GB).
- Compute: 2× the matmul (256 × 129k = ~33M FLOPs vs ~16M). At batch 1, γ=5: ~0.16 GFLOP → sub-microsecond. Negligible.
- Saved: 5 NCCL all_gathers × ~40µs = **~200µs per draft step** + eliminates the CPU serialization between loop iterations.

Implementation: mirror the existing markov_w1 replication pattern:
```python
if self._replicated_w2:
    self.markov_w2 = nn.Linear(config.dspark_markov_rank, config.vocab_size, bias=False)
    self.markov_w2.weight.requires_grad_(False)
else:
    self.markov_w2 = ParallelLMHead(config.vocab_size, config.dspark_markov_rank, ...)
```
With replicated W2, `_vocab_parallel_argmax` becomes a plain `.argmax(dim=-1)` — no all_gather, no `_vocab_parallel_argmax_from_local`.

Flag: `VLLM_DSPARK_REPLICATE_MARKOV_W2=1` (default off, opt-in).

#### Step 2: Fuse the Markov loop into a single Triton kernel (medium effort)

With replicated W1 + W2, each Markov loop iteration is pure GPU:
```
embed = W1[prev_token]        # embedding lookup
logits = W2 @ embed           # matmul (full vocab, local)
step = base_logits[pos] + logits  # elementwise add
next_token = argmax(step)     # local argmax
```

A single Triton kernel can do all 5 positions sequentially (position k depends on k-1's argmax, which is available in the same kernel via a register). Launch: 1 kernel instead of ~25 Python-issued kernels.

The existing `dspark_markov_argmax` kernel already fuses the local argmax (base + W1[x]·W2). Extend it to loop over γ positions in-kernel:
- Input: base_logits [batch, γ, local_vocab], W1 (embedding table), W2 [rank, vocab], input_ids [batch]
- Output: output_ids [batch, γ]
- Loop: for pos in range(γ): embed = W1[output_ids[pos]]; logits = W2 @ embed + base[pos]; output_ids[pos+1] = argmax(logits)

This eliminates the Python loop entirely → **no Python control flow in the draft's post-backbone path**.

#### Step 3: Switch draft graph from PIECEWISE to FULL (config change, requires steps 1+2)

With the Markov loop fused into a kernel (no Python control flow, no NCCL syncs), the entire draft path (backbone layers + forward_head + norm + lm_head + fused Markov kernel) is a fixed sequence of GPU ops → **can be captured as a FULL cudagraph**.

Change `dspark_proposer.py:610`:
```python
dspark_cudagraph_mode = CUDAGraphMode.FULL  # was PIECEWISE
```

FULL captures the entire draft as one graph → **eliminates per-piece Python dispatch overhead** (~1-2ms based on typical PIECEWISE→FULL transitions).

### Expected gain

| Step | Saved per draft step | Source |
|---|---|---|
| Step 1 (replicate W2) | ~200µs | 5 NCCL all_gathers eliminated |
| Step 2 (fuse Markov loop) | ~75µs | ~20 Python-issued kernel launches → 1 |
| Step 3 (FULL draft graph) | ~1-2ms | PIECEWISE→FULL dispatch elimination |
| **Total** | **~1.3-2.3ms** | **Draft from ~8.6ms to ~6.3-7.3ms** |

Cycle: ~70ms → ~68-69ms → **~62 tok/s → ~63-64 tok/s (+2-3%)**.

Requires 5+ repeats to distinguish from noise (CV ~1-2%).

### Why not the other candidates

| Candidate | Verdict |
|---|---|
| Kernelize confidence scheduling CPU path | The scheduler never prunes at single-stream (prune_rate=0.0). Skipping it entirely (scheduler=off) is simpler and already recovers baseline. Only relevant at concurrency. |
| Variable-prefix main-KV update | Irrelevant at single-stream (no pruning). Store_main_kv already fused (Triton kernel). |
| Fused Markov argmax/confidence (without W2 replication) | The worklog already tried `VLLM_DSPARK_FUSED_MARKOV_ARGMAX` — "lost to the vendor rank-256 projection." The fused local argmax avoids materializing full Markov logits but still does the all_gather. Without W2 replication, the NCCL serialization remains. **W2 replication is the prerequisite that makes the fusion worthwhile.** |
| MHC/quant parity | No evidence of a gap. Acceptance is healthy. Draft path audited paper-faithful. |

### Memory/object accumulation risks

**None.** The replicated markov_w2 is a fixed tensor allocated at `__init__` (~33MB, constant). The fused Triton kernel is stateless (input→output, no internal state). The FULL draft graph captures fixed-size buffers (the draft input/hidden/position buffers already pre-allocated). No growing lists, no per-step accumulators, no Python objects retained between steps.

### Metrics that would prove progress

| Metric | How to measure | Bar |
|---|---|---|
| Draft stage time | `VLLM_DSPARK_STAGE_TIMING=1`, 5 repeats | ≤ 7.0ms (from ~8.6ms) |
| NCCL calls per draft | NCU or nsys: count `ncclDevKernel_AllReduce` in the draft window | 0 in the Markov loop (from 5) |
| Draft graph mode | Startup log | FULL (from PIECEWISE) |
| Server decode tok/s | `code_completion` ×5 repeats | ≥ 63.5 mean, CV ≤ 2% |
| Acceptance | spec-decode metrics | Unchanged (≥ 0.65) — the draft tokens are bit-identical (same math, different execution order) |

### Bit-equivalence concern

Steps 1-3 change the execution order of the Markov head's matmul and argmax, but **not the math**:
- Step 1: replicated W2 computes the SAME logits as the sharded version (full vocab vs gathered shards → same values).
- Step 2: the fused kernel does the SAME arithmetic (W1[x]·W2 + base → argmax), just in-kernel rather than Python-issued.
- Step 3: FULL graph replays the same operations as PIECEWISE, just without Python between pieces.

The draft tokens should be **bit-identical** to the current path. Validate: run a single-stream benchmark with the old path and the new path, diff the first 256 output tokens. They must match exactly (greedy decode → deterministic).

### Recommended implementation order

1. **Step 1 first** (replicate W2) — trivial, testable independently. Measure: does removing the 5 NCCL all_gathers from the Markov loop measurably reduce draft stage time? If yes (~200µs visible in stage timing), proceed.
2. **Step 2** (fuse Markov loop) — medium effort. Depends on step 1 (no point fusing if the NCCL syncs remain). Measure: does the fused kernel reduce draft stage time further?
3. **Step 3** (FULL graph) — config change. Depends on steps 1+2 (FULL requires no Python control flow in the captured path). Measure: PIECEWISE→FULL transition gain.

Each step is independently measurable (stage timing). The gains compound. But each is also independently revertible (flags: `VLLM_DSPARK_REPLICATE_MARKOV_W2=1`, `VLLM_DSPARK_FUSED_MARKOV_LOOP=1`, draft graph mode).

---

## The concurrency reminder

This optimization benefits **both** single-stream and concurrent decode (the Markov loop runs at every batch size). At c=4/c=8, the NCCL savings multiply (more requests = larger all_gather payload = higher per-call latency). The FULL draft graph also helps at concurrency (the draft runs per-cycle regardless of batch size).

But the **concurrency-specific** gain (the scheduler pruning at saturation) requires the CPU confidence path — which the user wants to avoid. The resolution: the CPU confidence path only fires when the scheduler is enabled (opt-in). At single-stream, it's off. At concurrency where pruning pays, it's on. The two modes are cleanly separated by the `VLLM_DSPARK_CONFIDENCE_SCHEDULER` flag.

**The GPU-resident draft optimization (steps 1-3) is orthogonal to the scheduler.** It speeds up the draft regardless of whether the scheduler is on or off. This is the right priority: make the draft as fast as possible (GPU-resident), then layer the scheduler on top for concurrency.
