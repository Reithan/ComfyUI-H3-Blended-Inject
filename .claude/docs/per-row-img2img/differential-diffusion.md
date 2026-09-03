<!-- provenance: reference + confirmed (source-verified file:line map and formula derivations;
     GPU-observed crack signature confirmed; duality argument analytical,
     corroborated by source analysis and memory h3-scale-latent-inpaint-override) -->
<!-- verified: 2026-08-23 · comfy-ref @b78cec87 -->
# Differential Diffusion & the ghost: why native mask paths fail on H3

Read this when considering any mask-based / DD / inpaint approach to fractional denoise on H3, or
to understand the original ghosting bug. Bottom line: **DD is the general "img2img via a mask"
primitive and it works, but it is the EXACT DUAL of [our approach](our-architecture.md) on H3**
(DD covers stochastic samplers, cracks deterministic; ours covers deterministic, breaks
stochastic). Neither native path gives free fractional img2img on H3.

## The three relevant native mechanisms

1. **Global img2img** = schedule truncation, no mask. Start at `sigma ≈ denoise·sigma_max`,
   noise the whole latent, sample to 0. One strength for the whole frame. Every sampler works.
2. **Inpaint via mask** (`denoise_mask` → `KSamplerX0Inpaint`) = freeze-and-composite. On H3
   this is *conditioning injection*, not noised-trajectory freeze (below). A raw FRACTIONAL mask
   into the composite ⇒ **double-exposure ghost** (the ORIGINAL project bug).
3. **Differential Diffusion** = the general img2img-via-mask primitive, built ON TOP of #2 via
   the `denoise_mask_function` hook. Great on SD/Flux; cracks on H3.

## Why native inpaint ghosts (the math)

Fractional row, `denoise_mask = m` (0<m<1). At the final step σ→0 the POST composite[^post]
gives `x_final ≈ m·D_final + (1−m)·clean`: a convex blend of two DIFFERENT coherent latents ⇒
double-exposure. For m=0.3 that's 70% inject + 30% new overlaid = ghost. Inpaint semantics
("freeze the known region, blend it every step") are the wrong operation for "denoise this row
from a partially-noised start." Not tunable; inherent to compositing. Our per-row init-lerp
injects clean **once at the start** (correct img2img), so no composite, no ghost.

**Scope of the ghost (corrected, see [motion-context-comparison](motion-context-comparison.md)):**
the composite double-exposes on any FRACTIONAL (0<m<1) row, but it's only *perceptible* when the
fraction is SUSTAINED on coherent held content, i.e. an anchor keyframe at `0<min_denoise<1`.
Transient temporal fades double-expose too but motion hides it (Motion Context's faded masks look
great, including on stochastic; MC is NOT binary-only). The visible bug is specifically
**fractional denoise of the keyframe**, exactly what this repo exists to fix.

## Differential Diffusion: mechanism

DD source[^ddsrc]. `DifferentialDiffusion.execute` calls `model.set_model_denoise_mask_function(
forward)`. `forward(sigma, denoise_mask, extra_options, strength)`:
- `threshold = (current_ts − ts_to)/(ts_from − ts_to)`: schedule-progress, falls ~1→0.
- `binary_mask = (denoise_mask >= threshold)`; with `strength<1`, blends
  `strength·binary + (1−strength)·continuous`.
- Effect: continuous mask value `v` becomes a per-STEP BINARY mask; region `v` activates once
  `threshold` drops below `v` (high-strength early/high-σ, low-strength late/low-σ). Binary each
  step ⇒ POST composite is a hard freeze/reveal ⇒ **no ghost**; sampler always runs at FULL σ
  (no per-row compression) ⇒ **stochastic samplers work for free**. This IS img2img via a mask.

**Only hook available**: `denoise_mask_function`[^hook], applied ONCE to the mask and reused for
BOTH the PRE and POST composites; there is NO hook to disable/override POST alone (would need
monkeypatching). `noise_mask=None` disables BOTH composites. `KSamplerX0Inpaint` is always
constructed but its PRE/POST are gated on `denoise_mask is not None`.

## Why DD cracks on H3 (source-level)

- DD assumes the SD/Flux model: a frozen region sits on the **σ-noised trajectory**
  `x_σ = σ·noise + (1−σ)·clean`, and when its threshold is crossed it is *revealed* and continues
  denoising seamlessly.
- H3's masked path is a **DIFFERENT mechanism: conditioning-image injection.**
  `scale_latent_inpaint`[^sli] freezes preserved regions to a FIXED cond-timestep image:
  `cleans[0] = aug·clean + (1−aug)·noise`, `aug = VISUAL_COND_TIMESTEP = 0.999` ⇒ ≈clean,
  INDEPENDENT of σ. The code comment says it outright: *"preserved regions run at the cond
  timestep, inject them at cond strength."* The DiT reinforces this by PINNING preserved rows'
  timestep to `VISUAL_COND_TIMESTEP`[^dit]. Preserved rows are permanent conditioning; never on
  the noised trajectory.
- So when DD flips a row frozen→active mid-schedule: H3 was holding x≈clean and pinning t=0.999;
  the row goes active at σ_activation with a clean (off-distribution) x and the pin released ⇒ the
  model must denoise an already-clean latent as if at noise level σ_activation ⇒ discontinuity =
  crack. Fundamental to H3's design, not a tunable bug.

## GPU-observed signature (euler, deterministic)

Live test: generation looks perfect through ~step 7/10, then accumulates noise/cracks 7→10.
Mechanism: anchor/fade rows (low & mid mask values) stay frozen-at-clean early (looks fine) and
high-m rows generate normally; the fractional fade rows RELEASE late (their small mask values
cross the falling threshold ~step 7). At release each is off-distribution; euler being
deterministic can't re-project, so error compounds every remaining step. **euler_a survives** the
same case because ancestral re-noising re-projects the released row onto the manifold each step.

## The duality (the real reason "this isn't simple")

- [Our per-row compression](our-architecture.md): correct on DETERMINISTIC, breaks on stochastic
  (RF renoise not scale-invariant, see [k-diffusion-samplers](native-h3-mechanism/k-diffusion-samplers.md)
  and [bugs · Bug B](bugs.md#bug-b)).
- DD / native inpaint: correct on STOCHASTIC (ancestral), cracks on deterministic.

**Scope note (AUDIO):** the crack/ghost objection is VIDEO-only — it comes from the cond-timestep
pin on `cleans[0]`. The audio branch of `scale_latent_inpaint` is a plain rescale with NO pin, so
audio fades CAN use the official composite. See [audio-native-composite](audio-native-composite.md).

Each covers exactly the class the other fails; no SINGLE *native* mechanism covers both on H3.
Supporting both *via native paths* = two denoise engines selected by sampler type = the
special-casing smell. ⇒ pick ONE class. Deterministic (our path) is the standard/correct choice
for flow models. **Caveat (revises this section):** the "two engines" framing applies to native
paths; a per-row ancestral step may recover stochastic *inside our one engine*; see
[stochastic-recovery-theory](stochastic-recovery-theory.md).
Making DD ALSO work on deterministic would require overriding BOTH `scale_latent_inpaint` (freeze
to σ-noised trajectory) AND the DiT row-pinning, reimplementing SD-style DD by hand on H3's
plain forward. Not simpler than ours. Confirmed independently in memory
`h3-scale-latent-inpaint-override`.

[^post]: `comfy/samplers.py` 642 (POST composite in `KSamplerX0Inpaint.__call__`, 630-643).
[^ddsrc]: `comfy_extras/nodes_differential_diffusion.py` (added to sparse checkout).
[^hook]: `comfy/samplers.py` 636-637 (`denoise_mask_function`); disables-both via `noise_mask=None`.
[^sli]: `comfy/model_base.py` 2248-2272 (aug at 2255-2256, comment at 2249).
[^dit]: `comfy/ldm/minimax/model.py` `_forward` 589-609 (`t_pin_v = max(t_v, VISUAL_COND_TIMESTEP)` at 589; per-row clamp applied 593-609).
