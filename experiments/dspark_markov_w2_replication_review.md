# Markov W2 Replication: Implementation Gotchas Review

Date: 2026-06-29
Reviewer: Claude (read-only advisor)
Scope: nuanced implementation review of replicating `markov_w2` under TP=2, including the critical discovery that W2 replication alone does NOT eliminate the per-position NCCL all-gather.

---

## ⚠️ Critical finding: W2 replication alone does NOT eliminate the NCCL all-gather

Codex's follow-up note on the advisor doc was correct. I was wrong in my original recommendation. Here is the precise reason:

### The base logits are ALSO vocab-parallel

In the draft's fast path (`_local_argmax=True`, the current default), `draft()` at dspark.py:776:

```python
local_logits = lm_head.quant_method.apply(lm_head, normed, bias=None)
# → [batch, γ, local_vocab]  — LOCAL (sharded) on each TP rank
```

`lm_head` is the **target model's vocab-parallel output head** — it is sharded across TP ranks. Each rank sees only `vocab_size / tp_size` columns. This is passed into `draft()` from the proposer (dspark_proposer.py:686: `self.model.draft_with_confidence(..., self.lm_head, ...)`).

Then in the Markov loop (dspark.py:787-791):
```python
markov_logits, markov_embed = final_layer.markov_head.forward_local(output_ids[:, pos])
# markov_logits: [batch, local_vocab]  — LOCAL (sharded via ParallelLMHead markov_w2)

step_logits = local_logits[:, pos] + markov_logits
# step_logits: [batch, local_vocab]  — LOCAL (sum of two sharded tensors)

output_ids[:, pos + 1] = _vocab_parallel_argmax(step_logits, lm_head)
# → all_gather of (local_max_val, global_index) → global argmax
```

**Even if markov_w2 is replicated (full vocab on each rank)**: `local_logits[:, pos]` is still sharded (from the vocab-parallel lm_head). Adding a full-vocab markov_logits to a sharded local_logits → **shape mismatch** (`[batch, full_vocab]` vs `[batch, local_vocab]`). The add cannot be performed.

### What W2 replication WOULD achieve (and what it wouldn't)

| With replicated W2 | Effect |
|---|---|
| `markov_w2` returns full-vocab logits | markov_logits is `[batch, full_vocab]` |
| `local_logits` is still sharded | `[batch, local_vocab]` |
| `step_logits = local_logits + markov_logits` | ❌ shape mismatch — cannot add |
| NCCL all_gather in `_vocab_parallel_argmax` | still needed (for base logits) |

**W2 replication alone changes nothing about the NCCL calls.** The all_gather is required by the base logits' vocab-parallelism, not by markov_w2.

### What WOULD eliminate the NCCL calls

To eliminate all 5 per-position all_gathers, you need BOTH:
1. **Replicate markov_w2** → full-vocab markov logits. (~+33MB per rank.)
2. **Replicate lm_head** (the target's output head) → full-vocab base logits. (~+925MB per rank at bf16, [7168, 129280] × 2 bytes.)

Then `step_logits` is full-vocab on both ranks → local argmax is the global argmax → **no all_gather needed**.

**Memory cost**: ~958MB extra per rank. On GB10 (128GB unified memory), the current model + KV cache at TP=2 uses ~80-100GB. Adding ~1GB per rank is feasible but eats into KV cache budget. This is a **deployment tradeoff**, not a code issue.

---

## Implementation gotchas (if you proceed with W2 replication + lm_head replication)

### Gotcha 1: Weight loading — two different mechanisms for the same checkpoint name

**Checkpoint name**: `mtp.{stage}.markov_head.markov_w2.weight` → remapped to `model.layers.{vid}.markov_head.markov_w2.weight`.

| W2 type | Param class | `weight_loader` | Loading behavior |
|---|---|---|---|
| `ParallelLMHead` (current) | `VocabParallelEmbedding` subclass | Yes — shards across TP ranks | Each rank loads `vocab/TP` rows |
| `nn.Linear` (replicated) | `nn.Parameter` | No (plain Parameter) | `default_weight_loader` copies FULL checkpoint weight to each rank |

The load_weights fallback at dspark.py:1035:
```python
weight_loader = getattr(param, "weight_loader", default_weight_loader)
weight_loader(param, loaded_weight)
```

For `nn.Linear`: `getattr(param, "weight_loader", default_weight_loader)` → `default_weight_loader` → `param.data.copy_(loaded_weight)` → full weight copied. ✅ Correct — each rank gets the full weight.

For the target's `lm_head` (if replicated): same pattern. The checkpoint has the FULL vocab weight. `default_weight_loader` copies it to each rank. ✅ Correct.

**Risk**: if the `lm_head` is replicated as `nn.Linear` but the checkpoint weight is already sharded (unlikely — checkpoints store full weights), the copy would be wrong. vLLM checkpoints store full weights and rely on the param's `weight_loader` to shard. For nn.Linear (no weight_loader), the full weight is copied as-is. ✅ Correct.

### Gotcha 2: LogitsProcessor and forward() break

The MarkovHead's `forward()` method (dspark.py:430-432):
```python
def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    markov_embed = self.markov_w1(token_ids)
    markov_logits = self.logits_processor(self.markov_w2, markov_embed)
    return markov_logits, markov_embed
```

`self.logits_processor` is a `LogitsProcessor(config.vocab_size)`. Its `__call__` does:
```python
logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
```

This expects `lm_head` (markov_w2) to have `.quant_method` and `.shard_indices`. An `nn.Linear` has NEITHER → **crash**.

**Fix**: when W2 is replicated, `forward()` should bypass the LogitsProcessor:
```python
if self._replicated_w2:
    markov_logits = torch.nn.functional.linear(markov_embed, self.markov_w2.weight)
else:
    markov_logits = self.logits_processor(self.markov_w2, markov_embed)
```

### Gotcha 3: forward_local() and quant_method.apply

`forward_local()` (dspark.py:435-442):
```python
markov_logits = self.markov_w2.quant_method.apply(self.markov_w2, markov_embed, bias=None)
```

An `nn.Linear` has no `.quant_method` → **crash**.

**Fix**: when W2 is replicated:
```python
if self._replicated_w2:
    markov_logits = torch.nn.functional.linear(markov_embed, self.markov_w2.weight)
else:
    markov_logits = self.markov_w2.quant_method.apply(self.markov_w2, markov_embed, bias=None)
```

### Gotcha 4: _vocab_parallel_argmax expects shard_indices

`_vocab_parallel_argmax(step_logits, lm_head)` (dspark.py:85-96):
```python
num_pad = lm_head.shard_indices.num_org_vocab_padding
...
global_indices = local_max_indices + lm_head.shard_indices.org_vocab_start_index
return _vocab_parallel_argmax_from_local(local_max_vals, global_indices)
```

This accesses `lm_head.shard_indices` — if `lm_head` is replicated (nn.Linear), it has no `shard_indices` → **crash**.

**Fix**: when both W2 and lm_head are replicated, bypass `_vocab_parallel_argmax` entirely:
```python
if self._replicated_w2 and self._replicated_lm_head:
    output_ids[:, pos + 1] = step_logits.argmax(dim=-1)
else:
    output_ids[:, pos + 1] = _vocab_parallel_argmax(step_logits, lm_head)
```

### Gotcha 5: _vocab_parallel_markov_argmax (fused path) uses lm_head.shard_indices

The fused Markov argmax kernel path (dspark.py:796-803):
```python
output_ids[:, pos + 1] = _vocab_parallel_markov_argmax(
    local_logits[:, pos],
    markov_embed,
    final_layer.markov_head.markov_w2,
    lm_head,
)
```

`_vocab_parallel_markov_argmax` (dspark.py:120-136) accesses:
- `markov_w2.weight` — exists for nn.Linear ✅
- `markov_w2.shard_indices` — ❌ doesn't exist for nn.Linear
- `lm_head.shard_indices.num_org_vocab_padding` — ❌ if lm_head is replicated

**Fix**: the fused path should be bypassed entirely when W2 is replicated (the fused kernel was designed for the sharded path; with replicated W2, a simpler kernel or a plain argmax is used).

### Gotcha 6: lm_head replication affects the ENTIRE draft, not just markov_w2

The draft's `draft()` method receives `lm_head` as a parameter. If `lm_head` is replicated, it must be replicated for BOTH:
- The base logits computation (line 776: `lm_head.quant_method.apply(lm_head, normed)`)
- The Markov loop's argmax (via `_vocab_parallel_argmax`)

But `lm_head` is the TARGET model's output head — it's shared between the target forward and the draft. Replicating it for the draft means the target's logits are also computed with the full head → the target's output would change too (or need a separate replicated copy).

**Option A**: replicate a COPY of lm_head for the draft only. The target keeps its sharded lm_head. The draft uses the replicated copy. Memory: +925MB per rank for the copy. The copy is loaded from the same checkpoint weights.

**Option B**: replicate lm_head for both target and draft. The target's logits computation would produce full-vocab logits → changes the target's sampling path → may require changes to the sampler. More invasive.

**Recommendation**: Option A (draft-only replicated lm_head copy). The target is unchanged. The draft uses the replicated copy for base logits + argmax. The copy is `requires_grad_(False)`.

### Gotcha 7: Bit-equivalence — is it preserved?

With replicated lm_head + replicated markov_w2:
- Base logits: `full_weight @ hidden` vs `(shard_0 @ hidden)` gathered. The per-element computation is `sum_k hidden[k] * weight[k, v]` — identical regardless of sharding (the shard just selects a subset of v). Same rounding. ✅
- Markov logits: same argument. ✅
- step_logits = base + markov: same values. ✅
- argmax: same result (argmax of the same values). ✅

**Bit-equivalent.** The draft tokens are identical. Validate: diff output tokens old vs new path at T=0.

### Gotcha 8: Env flag naming and defaults

Mirror the W1 pattern:
```python
VLLM_DSPARK_REPLICATE_MARKOV_W2  # default 0 (off), opt-in
VLLM_DSPARK_REPLICATE_LM_HEAD    # default 0 (off), opt-in (for the draft-only copy)
```

Both must be ON for the NCCL-elimination path. If only W2 is on (without lm_head), the code falls back to the current `_vocab_parallel_argmax` path (the W2 replication is wasted but harmless — the markov logits are full-vocab but get sharded by the argmax path).

**Safe default**: both off → identical to current behavior. No risk.

### Gotcha 9: Test coverage

| Test | What to verify |
|---|---|
| **Weight loading** | Replicated W2 loads the FULL checkpoint weight on each rank (not a shard). Verify: `markov_w2.weight.shape[1] == vocab_size` (not `vocab_size / tp_size`). |
| **forward_local** | Returns full-vocab logits (shape `[batch, vocab_size]`, not `[batch, vocab_size/tp_size]`). |
| **Draft tokens bit-equivalence** | Run the draft with replicated W2+lm_head vs sharded, diff the output tokens. Must be bit-identical at greedy T=0. |
| **NCCL call count** | nsys profile: 0 `ncclDevKernel_AllReduce` in the draft window (from 5). |
| **Memory** | Container RSS: +~1GB per rank when both flags are on. |

---

## The strategic question: is this worth it?

| Factor | Assessment |
|---|---|
| NCCL calls eliminated | 5 per draft step → 0 |
| Latency saved | ~200µs per draft step (~40µs × 5 RDMA round-trips) |
| Draft stage improvement | ~8.6ms → ~8.4ms (~2.3%) |
| Cycle improvement | ~70ms → ~69.8ms (~0.3%) |
| tok/s improvement | ~62 → ~62.2 (within noise) |
| Memory cost | +958MB per rank (W2 + lm_head copy) |
| Implementation effort | Medium (8 gotchas to handle) |
| Enables FULL draft graph? | Yes — with no NCCL in the Markov loop, the Python loop can be fused and the entire draft captured as FULL. The FULL transition is where the real gain lives (~1-2ms from PIECEWISE dispatch elimination). |

**W2 + lm_head replication alone saves ~200µs** (within noise). **The real value is enabling the PIECEWISE → FULL transition** (saves ~1-2ms → ~2-3% tok/s). But the FULL transition ALSO requires fusing the Python Markov loop (step 2 of the 3-step plan). So the sequence is:

1. Replicate W2 + lm_head (this doc) — eliminates NCCL, enables the next steps.
2. Fuse the Markov loop into a Triton kernel — eliminates Python control flow.
3. Switch to FULL draft graph — eliminates per-piece dispatch.

Steps 1-3 together: ~1.2-2.2ms → ~62 → ~63-64 tok/s. Step 1 alone: ~200µs (noise).

**Recommendation**: proceed with the full 3-step plan if the ~2-3% gain is worth the effort + ~1GB memory. If not, the single-stream ceiling (~62 tok/s) stands, and the effort should redirect to concurrency (where the scheduler prunes and the batch amortizes).

---

## Memory/object accumulation risks

**None.** The replicated W2 and lm_head copy are fixed tensors allocated at `__init__`. The draft graph captures pre-allocated buffers. No growing lists, no per-step accumulators. The NCCL call elimination removes the only sync points in the Markov loop, but doesn't create any new data structures.

---

## Summary for Codex

1. **W2 replication alone does NOT eliminate the NCCL calls.** The base logits from the vocab-parallel `lm_head` also need replication. Both must be on for the comm-elimination path.
2. **8 implementation gotchas** (weight loading, LogitsProcessor, forward_local, _vocab_parallel_argmax, _vocab_parallel_markov_argmax, lm_head scope, bit-equiv, env flags). All have clean fixes listed above.
3. **Bit-equivalent** — the draft tokens are identical (same arithmetic, different execution path).
4. **The real gain is the FULL graph transition** (~1-2ms), not the NCCL savings alone (~200µs). The replication is the prerequisite, not the payoff.
5. **Memory cost**: ~958MB per rank (W2 ~33MB + lm_head copy ~925MB). Feasible on GB10 but reduces KV cache budget.
6. **If the 2-3% single-stream gain is worth it**: proceed with all 3 steps (replicate → fuse → FULL). If not: accept the single-stream ceiling and redirect to concurrency.
