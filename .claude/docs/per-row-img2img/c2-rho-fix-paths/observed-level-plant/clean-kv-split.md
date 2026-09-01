<!-- provenance: confirmed (GPU-CONFIRMED 2026-08-31 — clean-anchored two-forward is the working mechanism) -->
<!-- verified: GPU-CONFIRMED 2026-08-31 — static-clean-anchored clean-K/V two-forward; Finding 1 + Finding 3 both durable (prev_denoised-anchored build regressed first; static clean anchor fixed it) -->
# Clean-sourced K/V inside observer split — replacement for Option I

Parent: [../observed-level-plant.md](../observed-level-plant.md).
Supersedes: [second-stream.md](second-stream.md) (Option I second-stream, OVERTURNED 2026-08-31).

**Durable (branch `clean-kv-observer-splice`):** this clean-K/V euler splice is now the DURABLE
(non-proto) implementation. Two experiments from `proto-observed-level-inject-noise` did NOT ship and
remain PROTO-ONLY: (a) the observed-level ANCESTRAL renoise change to `_euler_ancestral_rf_step`
(reverted here to main's stock per-row RF-ancestral — `σ_obs = m·σ_g` is used ONLY as the euler
`x_obs` anchor, not for ancestral renoise); (b) the `H3RescaleNoiseMask` node / `mask_rescale.py`
discriminator ([../stock-mask-remap-port.md](../stock-mask-remap-port.md)).

## Finding 1: observer-split governs self-reception, not just broadcast (durable)

In `observer_split.py` the block patch overwrites fractional-row K/V with observer-label
(`σ_obs = m·σ_g`) K/V while leaving Q truthful (`σ_row`). Because self-attention reads a
token's OWN K/V, this relabel governs BOTH what neighbours read of the frame AND what the
frame reads of ITSELF.

Self-reception at label `m` makes the frame perceive itself as the low-noise anchor while its
own velocity (Q/gate/MLP) runs at `σ_row` → correct denoise strength, no ghost. Previously
this was treated as "what neighbours read" only. The Option I GPU regression (split off →
frame self-received at `σ_row` → over-change) confirms the self-reception role.

## Finding 2 summary

See [second-stream.md §GPU result + overturn](second-stream.md) for the full decomposition.
Short: split OFF → self-reception at `σ_row` → over-change; clean-anchored `x_obs` → weak blend.

## Finding 3: the replacement mechanism (GPU-CONFIRMED — static clean anchor)

> **Confirmed.** The clean-K/V two-forward IS the working mechanism, once `x_obs` is anchored on
> the STATIC `clean` inject rather than the previous step's `denoised`. The initial
> prev_denoised-anchored build regressed (drift); the static-clean-anchored build is
> GPU-CONFIRMED — "the most solid version so far." See "## GPU result (2026-08-31)" below.

**Principle:** keep the split architecture (Q/velocity truthful at `σ_row`; fractional K/V at
observer label `m·σ_g` for BOTH broadcast AND self-reception). Change only the K/V CONTENT
SOURCE: source from genuinely-clean content re-noised to the observer level. The CURRENT anchor
is the STATIC `clean` inject:

    x_obs = clean + (σ_obs/σ_row)·(x_prev − clean),   σ_obs = m·σ_g

**KEY INSIGHT — `clean` IS the exact x0 of the fractional rows.** For an injected keyframe the
`clean` inject is, by construction, the ground-truth x0 of the fractional rows. So `clean`
re-noised to `σ_obs` is the EXACT observer content, not an estimate. `prev_denoised` was a
noisier, drift-prone ESTIMATE of a quantity we already know exactly. Anchoring on the static
`clean` is therefore simultaneously MORE STABLE and HIGHER QUALITY — and it is why cross-stream
incoherence never bit us in practice: the captured K/V are anchored on true content, not a
wandering estimate.

**Parameterization invariance:** H3 uses rectified flow; `σ_obs·clean` terms cancel between
the x0 prediction and the re-noise formula. Verified analytically (Fable review).

**Circularity resolution:** anchor on the STATIC `clean` inject (= exact x0 for injected
keyframes), NOT the previous step's `denoised`. There is no same-step forward dependency and no
cross-step loop — `clean` is constant across steps. Step 0 and every later step use one code
path: `clean + m·(x_prev − clean)` re-noised to `m·σ_g`; no branch.

**Implementation — two forwards per step, split ON:**
1. CAPTURE forward on `x_obs` (built from static `clean`): run model, cache each
   fractional block's `k[pos]/v[pos]` (raw pre-rope `qkv_proj` output).
2. SELF forward on `x_prev`: run model normally; observer_split consumes the cache (Q
   untouched). This forward's denoised feeds the euler step.

Cost: 2 forwards/step when fractional rows exist (same as Option I), correct architecture.

**Audio extension:** packed per-element `σ_obs`/`σ_row` on the audio-shifted schedule
(`time_shift_sigma`); no separate code path needed.

**Option II (single-forward) — correctness vs performance.** For CORRECTNESS the clean-anchored
two-forward is complete; Option II is NOT required to make results right. It IS being pursued as a
durable PERFORMANCE path: an EXACT single-forward that collapses the 2 forwards/step to ~1 by
carrying a fractional side-stream, reproducing this mechanism bit-for-bit. See
[option-ii-single-forward.md](option-ii-single-forward.md).

## GPU result (2026-08-31)

Two builds, in order:

**(1) prev_denoised-anchored — REGRESSED (history).** The first clean-K/V two-forward anchored
`x_obs` on the PREVIOUS step's `denoised`. GPU-tested 2026-08-31, it regressed:

- **Fades:** random brightening and color shifts.
- **Single-frame injects:** poorly blended, with inappropriate color and lighting shifts.
- Per-frame denoising was the correct AMOUNT but denoised into the same poor blends.

Diagnosed cause: a cross-step feedback loop — splice denoised → `x_obs` → cached K/V → next
splice denoised. DC/color bias amplified across steps → progressive brightening/tint.

**(2) static-clean-anchored — GPU-CONFIRMED (current mechanism).** Changing the single anchor
from `prev_denoised` to the static `clean` inject fixed it. The user ran several different gens
and ALL came out great — "the most solid version so far." Fades and single-frame injects both
clean: no brightening, no colour/lighting shift, coherent blends.

**Diagnosis confirmed:**
- The cross-step feedback loop (Cause 2) WAS the source of the brightening/colour drift.
  Removing it via the static anchor eliminates the drift.
- Cross-stream incoherence (Cause 1) is NOT a practical problem — blends are coherent. The
  captured K/V are anchored on true content (exact x0), so the two streams stay consistent.

## Status and cleanup (2026-08-31)

Implemented now as the SOLE always-on mechanism (no env toggles), anchored on static `clean`.
Deleted: `_second_stream_denoised` and `ss_*` state variables;
`H3_SECOND_STREAM` / `H3_SS_CLEAN_ANCHOR` env gates;
DC-debias env path (`H3_DC_DEBIAS` / `H3_DC_LOWPASS`, was default-ON);
plant-magnitude diagnostic.

Option I ([second-stream.md](second-stream.md)) is OVERTURNED/superseded — retained as history.
