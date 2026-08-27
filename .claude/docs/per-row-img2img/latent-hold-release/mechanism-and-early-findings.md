<!-- provenance: status (prototype design + early GPU debug findings; route-1 latent hold-and-release) -->
<!-- verified: 2026-08-25 · GPU runs + Fable audit + comfy-ref source -->
# Mechanism & early findings (1–3)

Index: [index](index.md). Current live thread: [attraction-and-envelope](attraction-and-envelope.md).

## Design (v1 simplifications)

- **k_sw is MANUAL**, not auto-calibrated `k_comp` — one `H3InjectSampler` widget `latent_hold_frac`
  (0=off; currently a fraction of STEPS — see Finding 1, this is the wrong unit). k_comp instrument
  NOT ported (tune by eye).
- **m′ = the anchor's configured `min_denoise`** (release at the same fractional rate), not a
  separate 0.3–0.5-of-remaining knob.
- **Anchor rows = fractional keyframe rows** (`0<m<1`); `m==0` preserve / `m==1` gen rows untouched.
  (See Finding 2 — this over-selects fade-ramp rows.)
- **Hold**: anchor `m→0` (clean content + `t_row=0.999` = trusted, held) for `σ>σ_sw`.
- **Release** at `σ_sw`: re-noise anchor to `m′·σ_sw·eps + (1−m′·σ_sw)·clean` (CORRELATED eps from
  `x_global`), then existing per-row img2img + denoised correction runs the tail. Re-noise is
  REQUIRED — dropping it recreates the clean-content/noisy-label contagion
  ([crux-and-mechanism](../highres-underdenoise-model/crux-and-mechanism.md) §switched-mode).
- **Mechanism**: two-phase `base_fn` split at `k_sw` in `build_per_row_sampler_function`; a
  `"release"` flag on a shared `hold_release` dict tells `build_conditioning_wrapper` which pooled
  `t_row`/`m` set (hold vs release) to inject each step.
- Deferred: auto k_comp calibration, continuous small-λ spring, coupled k_sw↔d starvation fix.

## Finding 1 — step-fraction `latent_hold_frac` is the WRONG knob (sigma-shift)

H3's video sigma-shift (~12) back-loads the schedule: `time_snr_shift(12,·)` reaches σ≈0.5 only at
t≈0.077 (~92% through a linear-step schedule). Observed schedule tail: `18:0.347, 19:0.183,
20:0.000`. So `latent_hold_frac=0.6` released at **step 12/20, σ=0.9320** — 60% of step-space ≈ 7%
of σ-space, i.e. **~93% of σ-space is POST-release** (this fact is central to the corrected release
model in [attraction-and-envelope](attraction-and-envelope.md) Finding 6). The knob must target a
**SIGMA** (release when σ first drops below σ_target), not a step fraction. Workaround until reparam:
sweep `latent_hold_frac` HIGH (0.9/0.92/0.95).

## Finding 2 — hold-residency CONFIRMED; red herrings + one real 2nd bug

- **Residency proven** (GPU log, `min_denoise=0.5`, `frac=0.6`): `anchor |clean|=0.8361 |x0|=0.8361`,
  post-hold `|x_mid|=0.8361` — anchor starts at clean and is bit-identical clean through all 12 hold
  steps. Trace: `x0[anchor]=clean` (`per_row_init_lerp` m=0), frozen through hold (wrapper correction
  returns `inp`, euler d=0).
- **Preview red herring:** the STOCK H3 latent preview decodes only frame 0, but the user runs a
  CUSTOM all-rows previewer (animated looped clip), so their visual observations are valid and frame-0
  did NOT cause what they saw. (Detail: memory `h3-preview-frame0-only`.)
- **Display bug (fixed):** the re-noise `level=` print used global `m.max()`=1.0; actual anchor level
  is `m′·σ_sw=0.466` (matches measured `|x|=0.6884`). Now prints `level[anchor].max()`.
- **2nd real bug (over-broad anchor, `nodes.py:251`):** `anchor_mask=(m_packed>0)&(m_packed<1)` grabs
  ALL fractional rows incl. fade-RAMP rows with NO inject content (clean ref = empty target latent).
  Those freeze at ~gray with a trusted label → neighbors compose against "trusted gray." **NOT
  harmless even for a "1-frame" inject** — Fable showed a single-frame inject with non-degenerate fade
  markers still produces a fractional shoulder row (see [attraction-and-envelope](attraction-and-envelope.md)
  Finding 6). Fix = restrict anchor to inject-backed rows (`inject is not None and 0<denoise<1`).

## Finding 3 — early-diagnosis corrections (superseded framing)

The hold-ON run shows the keyframe present at f136 and ~intact, but **frames before don't blend toward
it and frames after ignore it → HARD CUT**. Two of my earlier explanations were FALSIFIED and should
not be revived:
- **"windows never overlap / composition forms only at low σ":** FALSE — structure/composition
  coalesces EARLY (user's all-rows preview), even under `linear_quadratic`. Composition happened DURING
  the hold while the anchor was held clean.
- **"only cond attracts / latent can't attract":** FALSE — H3 rows attend each other via temporal
  self-attention every step; a clean anchor is always attendable. Distinguish *attend* (always) from
  *attract* (the content/motion pull we want).
- **No sign bug** (verified `grid.py:63-66`, comfy `model.py` `_forward` ~589): held `m=0` anchor is
  pinned to `VISUAL_COND_TIMESTEP=0.999` = clean/trusted end (H3: t≈1=data, t≈0=noise, inverse of
  sigma). The anchor IS presented as a clean trusted keyframe.

The live analysis of WHY the cut happened continues in
[attraction-and-envelope](attraction-and-envelope.md).
