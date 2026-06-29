# STS Calibration Diagnostics — Memory Safety Review

Date: 2026-06-29
Reviewer: Claude (read-only advisor — no code changes)
Scope: memory safety audit of the STS calibration + diagnostics patch for long-running server processes.
Design intent verified against: fixed-size counters only, diagnostics off by default, one-step raw confidence handoff.

Codex follow-up after this review: the final patch does include
`ModelRunnerOutput.dspark_confidence`, a per-step `dict[str, tuple[float, ...]]`
that carries only the currently verified draft's raw confidence labels into the
scheduler. It is not stored by the scheduler; it is folded into
`SpecDecodingStats` and then released with the output object. The final patch
also changed the calibration matrices to lazy allocation, so normal
spec-decoding stats allocate no DSpark calibration matrices when
`dspark_confidence` is absent.

---

## Verdict: memory-safe. No leaks found.

Every data structure in the patch is either fixed-size (pre-allocated, indexed), ephemeral (created per-step, GC'd after use), or one-step retention (set once, cleared by take/reset). No list, dict, or tensor grows unbounded across steps. The design intent is met.

---

## What I checked (every accumulation point)

### 1. DSparkDiagnostics counters — ✅ FIXED-SIZE

All lists are pre-allocated at `max_spec_tokens` length in `__post_init__` and only **indexed** (`[position] += value`), never appended:

| Field | Init | Growth pattern | Safe? |
|---|---|---|---|
| `scheduled_length_histogram` | `[0] * (max_spec_tokens + 1)` = 6 elements | `[scheduled_length] += 1` | ✅ |
| `confidence_sums` | `[0.0] * max_spec_tokens` = 5 | `[position] += float(confidence)` | ✅ |
| `confidence_counts` | `[0] * max_spec_tokens` = 5 | `[position] += 1` | ✅ |
| `raw_confidence_sums` | `[0.0] * max_spec_tokens` = 5 | `[position] += raw_confidence` | ✅ |
| `calibration_delta_sums` | `[0.0] * max_spec_tokens` = 5 | `[position] += (cal - raw)` | ✅ |
| `calibration_counts` | `[0] * max_spec_tokens` = 5 | `[position] += 1` | ✅ |
| `survival_sums` | `[0.0] * max_spec_tokens` = 5 | `[position] += survival` | ✅ |
| `survival_counts` | `[0] * max_spec_tokens` = 5 | `[position] += 1` | ✅ |
| `scheduled_counts` | `[0] * max_spec_tokens` = 5 | `[position] += 1` | ✅ |

**Note**: the dataclass fields declare `default_factory=list` (empty list at construction), but `__post_init__` immediately replaces them with fixed-length lists. The `default_factory=list` is never exposed to callers — it's a dataclass artifact, not a growth risk.

Scalar counters (`num_steps`, `num_requests`, `num_possible_draft_tokens`, etc.) are Python ints/floats. Fixed 8-byte (float) or word-size (int). No growth. ✅

### 2. DSparkDiagnosticsSnapshot — ✅ EPHEMERAL

`snapshot()` creates a `DSparkDiagnosticsSnapshot` (named tuple with tuples derived from the fixed lists). Returned to `_maybe_log_confidence_diagnostics`, used for `logger.info(...)` formatting, then GC'd. **No retention.** ✅

### 3. _last_confidence / _last_raw_confidence — ✅ ONE-STEP RETENTION

Lifecycle:
1. `propose()` starts → line 1210-1211: `self._last_confidence = None; self._last_raw_confidence = None` (clears previous step's tensor).
2. `postprocess()` → line 1309-1313: `self._last_raw_confidence = raw.detach().clone(); self._last_confidence = calibrated.detach().clone()` (sets current step's tensor — a clone, no aliasing to the draft model's internal buffer).
3. `take_last_confidence()` / `take_last_raw_confidence()` → returns the tensor, sets reference to None.
4. If `take_last_*()` is never called between steps → the tensor persists until step 2 of the NEXT propose() clears it. **At most one tensor retained.** ✅

The `.detach().clone()` at line 1309/1313 is important: it creates a NEW tensor (not a view into the draft model's output buffer). The draft model's buffer may be reused before the consumer reads `_last_confidence` — the clone prevents aliasing corruption. This is the same fix pattern as the earlier position-0 confidence aliasing bug (the "logit-like 0.31 confidences" from the worklog). ✅

### 4. confidence_rows / raw_confidence_rows (the .tolist() path) — ✅ EPHEMERAL

In `_observe_confidence()`:
```python
confidence_rows = confidence.detach().float().cpu().tolist()  # creates list[list[float]]
```
Used for the scheduler + diagnostics.observe(). Goes out of scope when `_observe_confidence` returns. **No retention.** ✅

The `.cpu()` call is a D2H copy (small: batch × γ = 8 × 5 = 40 floats). The `.tolist()` creates a Python list of lists. Both are ephemeral. ✅

### 5. _calibrate_confidence output — ✅ EPHEMERAL

```python
confidence_for_batch = self._calibrate_confidence(raw_confidence_for_batch)
```

Creates a new tensor (via `.float()`, `.logit()`, division, `.sigmoid()`). Used for the scheduler. Goes out of scope when `postprocess()` returns. **No retention.** ✅

### 6. prefill_batches (ragged grouping) — ✅ EPHEMERAL

```python
prefill_batches: list[tuple[torch.Tensor, ...]] = []
for group_len in grouped_lengths:
    ...
    prefill_batches.append((hidden_by_req, positions_by_req, rejected, indices))
```

Built per `propose()` call. Contains tensors that are `torch.stack` of slices of target hidden states (new allocations, not views). Used for `prefill_main()` calls. Goes out of scope when `propose()` returns. **No retention.** ✅

### 7. Pre-allocated buffers — ✅ FIXED-SIZE

```python
self._draft_input_ids_buffer = torch.zeros(self.max_batch_size, ...)
self._draft_hidden_buffer = torch.zeros(self.max_batch_size, ...)
self._draft_positions_buffer = torch.zeros(self.max_batch_size, ...)
```

Allocated once at `__init__`. Reused (overwritten in-place) each draft step. **No growth.** ✅

### 8. Per-step `.append()` calls — ✅ ALL EPHEMERAL

Every `.append()` in the hot path is on a per-`propose()` local list that goes out of scope:
- `hidden_chunks.append(chunk)` — local, cleared after grouping.
- `position_chunks.append(positions)` — same.
- `chunk_lengths.append(chunk_len)` — same.
- `prefill_batches.append(...)` — same.
- `grouped_lengths.append(chunk_len)` — same.
- `parts.append(...)` — stage timing format, ephemeral.

**No cross-step accumulation via append.** ✅

---

## Non-blocking observations (not memory leaks)

### 1. Diagnostics are cumulative (no reset/decay) — statistical staleness

The diagnostics counters accumulate from server start. Over a long run (weeks), the averages converge to the long-term mean and become insensitive to recent changes (e.g., after a config change or workload shift).

**Impact**: if the diagnostics are used to derive STS temperatures from a short benchmark run (~200 steps), cumulative is fine (all steps are recent). For long-running production diagnostics, the averages become stale.

**Not a memory issue** (the counters are fixed-size scalars). But if long-running diagnostics are desired, consider adding a `reset()` method (callable via env or signal) or an exponential decay factor on the sums.

### 2. Per-step `torch.tensor(temperatures, ...)` allocation in _calibrate_confidence

```python
temp_tensor = torch.tensor(
    temperatures[: confidence.shape[1]],
    dtype=torch.float32,
    device=confidence.device,
).view(1, -1)
```

Creates a new GPU tensor (5 floats = 40 bytes) per draft step when per-position STS is used. The CUDA caching allocator reuses the memory (no leak), but it's a per-step allocation that could be pre-allocated.

**Not a memory leak** (the tensor is GC'd after `_calibrate_confidence` returns). Just a minor hot-path overhead. Pre-allocate at `__init__` as `self._sts_temp_tensor` to eliminate. (Already flagged in the patch review.)

### 3. The caller of `take_last_confidence()` must not accumulate

The proposer correctly retains only one tensor (cleared by take or next-propose). But if the CALLER (the runner's position-0 diagnostic path, or any external hook) stores the returned tensors in a growing list, that's a leak in the caller — not in the proposer.

**Action**: verify that every caller of `take_last_confidence()` / `take_last_raw_confidence()` uses the tensor immediately (logs it, computes a scalar, etc.) and does NOT append it to a list. The proposer is safe regardless; the caller's discipline is the question.

---

## Confirmation: design intent met

| Intent | Status | Evidence |
|---|---|---|
| Fixed-size counters only | ✅ | All diagnostics lists pre-allocated at `max_spec_tokens` (5 elements), indexed not appended |
| Diagnostics off by default | ✅ | `VLLM_DSPARK_CONFIDENCE_DIAGNOSTICS_LOG_EVERY=0` default → `_maybe_log_confidence_diagnostics` early-returns |
| One-step raw confidence handoff | ✅ | `_last_raw_confidence` set in `postprocess()`, cleared at next `propose()` start or by `take_last_raw_confidence()` |
| No Python dicts/lists that grow across steps | ✅ | All per-step lists (hidden_chunks, prefill_batches, etc.) are local → GC'd after propose() |
| No ModelRunnerOutput payloads retained | ✅ | Final patch has a per-step `ModelRunnerOutput.dspark_confidence` payload, folded immediately into scheduler stats and not retained |
| No torch tensors that accumulate | ✅ | All per-step tensors (calibrated confidence, stacked chunks) are ephemeral; _last_* is one-step with explicit clearing |

---

## Bottom line

**No memory leaks. No concrete fixes needed.** The patch is memory-safe for long-running server processes. The diagnostics are fixed-size, the STS calibration is ephemeral, and the one-step confidence handoff is correctly implemented with clone-and-clear semantics. The three non-blocking observations (cumulative counters, per-step tensor allocation, caller discipline) are minor design notes, not defects.
