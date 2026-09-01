<!-- provenance: theory (UNVERIFIED — CPU-designed & CPU-tested, plant-axis fix SHIPPED PR #32; GPU pending) -->
<!-- verified: 2026-09-01 (branch fix-euler-ancestral-per-row-renoise) · plant-axis shipped 3e82dba; GPU pending -->
# euler_ancestral per-row fix design — combined clean-K/V wiring + σ_v-axis + plant-axis fix

Design and shipped record for the combined fix that clears both `euler_ancestral` per-row
artifacts. Neither half alone is sufficient — that is why every prior single-sided attempt failed.
The plant-axis init fix (PR #32) is CPU-tested and shipped; GPU verification is PENDING.

Read alongside [bugs.md](bugs.md) Bug C (axis) + Bug F (clean-K/V gap) and
[audio-axis-verdict.md](audio-axis-verdict.md).

## Two artifacts under euler_ancestral (both on main; both absent under euler)

Stock H3 handles `euler_ancestral` cleanly, so OUR per-row machinery is the cause. A same-branch,
same-prompt sampler swap isolates it: `euler` shows NEITHER artifact; `euler_ancestral` shows BOTH.

1. **VIDEO ghost (Bug F).** `_euler_ancestral_rf_step` (sampler.py:412) calls `ctx.model(...)`
   directly at sampler.py:467 with NO observer/frac gate. `_euler_step` instead routes through the
   clean-K/V splice `_single_forward_denoised` (sampler.py:391). The loop arms observer/frac_mask
   state unconditionally, but the ancestral step never fires it → fractional rows (video AND audio)
   receive the ghost-contaminated denoised.
2. **AUDIO noise (Bug C).** NOT a release-schedule problem — that hypothesis is REFUTED: the per-row
   ancestral terms already track each row's schedule, and `renoise_coeff` vanishes branch-free at
   the terminal step. Real cause = AXIS INCOHERENCE: the ancestral integration uses `ctx.sig_row`,
   which for audio rows sits on the σ_a-shifted schedule, while stock steps the whole packed latent
   on the σ_v trajectory. Renoise is injected mis-scaled for the wrong sigma level every step.

## THE FIX = both halves together

Apply BOTH; neither alone clears both artifacts.

1. **Wire the clean-K/V gate into `_euler_ancestral_rf_step`.** Add the same observer/frac gate that
   `_euler_step` uses so the forward routes through `_single_forward_denoised`. The ancestral renoise
   still runs AFTER, unchanged.
2. **Re-apply the σ_v-axis integration (PR #31's change).** Add `sig_row_v`/`sig_row_v_next` to
   `_StepContext` and a `row_sigma_v` closure = per-row sigma on the RAW video schedule for ALL rows.
   Use σ_v-axis tensors for `denoised_r` and every ancestral term (`downstep_ratio`, `sigma_down`,
   `alpha`, `renoise_coeff`, `ratio`).

**σ_a stays load-bearing for the LABEL only** — per-row `w`, pooled conds, observer-split labels.
The σ_a-label proof is untouched ([audio-axis-verdict/sigma-a-label-proof.md](audio-axis-verdict/sigma-a-label-proof.md)).

**Do NOT apply the `/sig_g` velocity change from the #76 thread** — that was the abandoned
σ_a-coherent formulation and conflicts with σ_v coherence.

## Attribution — the σ_v axis was NECESSARY, not wrong

The σ_v axis is a NECESSARY half, GPU-validated for m=1 free audio 2026-08-28. PR #31 failed on
FRACTIONAL audio only because the splice bypass (half 1) still fed it a contaminated denoised. So
the axis was NOT "the wrong cause" — the earlier falsification framing is corrected to **necessary
but insufficient alone**. The audio noise needs BOTH the clean-K/V wiring AND the σ_v axis.

## GPU verification PENDING (all items below are CPU-tested only)

- 0.3 fractional-video ghost cleared.
- Fractional-audio noise (fade hiss) cleared.
- m=1 free audio still clean.
- `euler` unchanged.
- Hiss loudest for mid-m ramp rows (falsifiable: where σ_row_v − σ_row_a peaks).

## Fallback (diagnostic only)

If fractional-audio noise persists after both halves, try elementwise eta-gating: `eta_row =
eta·(m>=1)` so fractional rows take the deterministic RF sub-path. Diagnostic, not the intended fix.

## Stale-thread reconciliation — PR #20 was a PHANTOM code fix

PR #20 (commit b87535c, "Fix #76 audio hiss: divide velocity recovery by per-modality sig_g")
claims a code fix in its message, but its diff is **WIKI-ONLY** — 4 doc files, ZERO code changed
(verified via `git show --stat b87535c`). The advertised `/sig_g` velocity-recovery code change was
NEVER landed. That is fine: the `/sig_g` (σ_a-coherent) formulation is the abandoned direction this
fix explicitly rejects. Flagged so the #76 thread is not mistaken for a shipped code fix.

## SHIPPED FIX (PR #32, commits 3e82dba + e3ec742) — plant-axis; GPU PENDING

Fade-region audio hiss under `euler_ancestral` is fixed by planting fractional rows on the
σ_v INTEGRATION axis, NOT the σ_a label axis. In `sampler.py` the init-plant at loop `i==0`
(`x_cur = w_plant·x_cur + (1−w_plant)·clean`) uses `w_plant = row_sigma_v(0)/sig_v[0]`
for the ancestral step. The model LABEL/pooled `w` stays σ_a (label proof untouched).

Keyed per step-fn: `DEFAULT_PLANT_AXIS = "row"` (module const) +
`_euler_ancestral_rf_step.PLANT_AXIS = "v"`. `euler` keeps the σ_a-ratio plant (GPU-validated).

CPU-verified properties (`tests/test_sampler.py::TestAncestralPlantAxis`):
- Video rows byte-identical (sig_row_v == sig_row for video).
- m=1 → w_plant=1 → no-op → stock bit-identity + noise-draw sequence preserved.
- m=0 → w_plant=0 → exact-preserve intact.
- Terminal flush unchanged.

## Bug B mechanism REFINEMENT (corrects "not scale-invariant" algebra framing)

The per-row ancestral algebra is EXACT and level-preserving with the current `renoise_coeff`:
given accurate model velocity v̂, the deterministic sub-step + affine (α_ip1/α_down) rescale +
coeff keep each fractional row at its nominal level σ_{i+1}, flush at the terminal step, and
reduce to stock bit-exact at m=1. **The retention was NEVER a coeff/algebra defect.**

Real channel: VELOCITY-ESTIMATION ERROR (`x0̂ = x0 + σ_row·(v − v̂)`) re-excited every step by
fresh ancestral injection. Deterministic euler makes the same v̂ error but never re-excites it —
that is the euler/ancestral asymmetry. For fractional AUDIO specifically, the systematic v̂ error
source was the init-plant axis incoherence (see §below).

## Root cause: init-plant axis incoherence (closed-form)

Trained contract (Fix A GPU result at m=1): packed audio CONTENT carries noise on the σ_v axis;
LABEL is σ_a = shift(σ_v). Post-Fix-A, label (σ_a) and integration (σ_v) are coherent — but the
plant used the σ_a ratio `w = sig_row/sig_g`, dropping fractional audio content at RF level
σ_row_a ≈ σ_row_v/4 (shift compresses ~4× for small σ; gap peaks mid-grid). From step 0 the
model is told (via label) to expect σ_v-sized noise in a row carrying σ_a-sized noise →
systematic v̂ mis-estimate on exactly the fade-band audio ticks, re-excited each step.

Falsifiable GPU prediction: hiss loudest for mid-m ramp rows (where σ_row_v − σ_row_a peaks),
fading toward both ramp ends.

## Rejected alternatives

(a) Rescaling `renoise_coeff` — provably inexact: current coeff is the unique level-preserving
value; smaller under-noises (coeff→0 IS the rejected eta-gate), larger over-noises.
(b) Reprojecting noise onto carrier axis — injects carrier-sized noise into a σ_row-level row,
strictly worse.
(c) Eta-gating fractional rows to deterministic — user-rejected (disables stochasticity).
(d) Explicit-noise bookkeeping (subtract known injected ε) — denatures ancestral sampling.

## Co-location verdict (resolves earlier mis-attribution)

- Bug C (audio σ_a integration mis-scale) is TIMELINE-WIDE: all audio rows are m=1 in drop mode
  (audio_denoise=1.0); cannot localize to a single inject. "Single-frame drop-mode audio noise =
  Bug C" was WRONG.
- The single-frame co-located audio noise was Bug F (video→audio JOINT-ATTENTION COUPLING):
  pre-#32 the ancestral path bypassed the clean-K/V splice → fractional VIDEO band denoised
  ghost-contaminated → H3 shared A/V attention imprinted on co-located audio. ATTRIBUTED
  (consistent with Bug E audio-tracks-visual GPU precedent). NOT A/B-isolated — post-#32 both
  symptoms gone (retrodiction).
- Fade hiss = plant-axis mechanism above, inject-local because audio is fractional ONLY in fade
  ramps (schedule.py: fade audio follows the video envelope).
- Bug C stays REAL, just timeline-wide (not the co-located cause).
