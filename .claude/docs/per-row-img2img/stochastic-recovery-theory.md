<!-- provenance: theory/confirmed (CONFIRMED for single-eval RF-ancestral 2026-08-27; multistep and 2nd-eval generalizations UNVERIFIED) -->
<!-- verified: 2026-08-27 · PR2 GPU pass task #68; repo @ede2d8c -->
# THEORY: recovering stochastic samplers via a per-row ancestral step

**Status: CONFIRMED for single-eval RF-ancestral (PR2 GPU-validated, task #68, 2026-08-27).**
Multistep (PR3) and DPM++ SDE (PR4) reuse the same identity but add history / a second in-step
eval — those remain UNVERIFIED and need their own GPU gates (tasks #70, #73). The theory carries
over to the shipped schedule-tail remap in simplified form — the remap loop already owns per-row σ
tensors and truthful labels, so the corrected-denoised identity is not needed; see
[sampler-class-support.md](sampler-class-support.md) for the remap-era design and the
SOURCE-CONFIRMED finding that multistep samplers currently run first-order under the remap loop.
This revises the earlier
"stochastic is a fundamental dead end / would need two engines" verdict in
[differential-diffusion · duality](differential-diffusion.md#the-duality-the-real-reason-this-isnt-simple)
and memory `prototype-goal-fade-mask-parity`. That verdict was correct **about the magnitude
shim** ([bugs · Bug B](bugs.md#bug-b)) but too strong about the whole idea.

## Claim

Stochastic (ancestral / SDE) samplers are recoverable **within our single engine** (no second
DD-style engine, no sampler-type special-casing) by making the sampler's ancestral arithmetic
per-row-aware instead of only rescaling injected noise magnitude.

## The linchpin identity

Our `corrected` denoised is **already** the exact per-row denoised. For a CONST (rectified-flow)
model, the network under our conditioning wrapper sees row `r` at compressed timestep `m_r·t`, so
it returns the velocity `v_r` for noise level `σ_r = m_r·σ`. Then (see lever 3 in
[our-architecture](our-architecture.md#the-three-levers)):

```
corrected_r = m_r·(x − σ·v_r) + (1−m_r)·x = x − m_r·σ·v_r = x − σ_r·v_r  ==  denoised_r
```

i.e. exactly what a genuine img2img run on the compressed schedule `σ_r` computes at that step,
sampler-independent. This is *why* deterministic euler works for free: euler's update is
homogeneous, so feeding it `corrected` at global σ is algebraically identical to running it at
`σ_r`.

## Why euler works but RF-ancestral cracks, and the fix

`sample_euler_ancestral_RF` has two sequential sub-steps per iteration (see
[k-diffusion-samplers](native-h3-mechanism/k-diffusion-samplers.md)):

1. **Euler step** (deterministic): `x ← (σ_down/σ_i)·x + (1 − σ_down/σ_i)·denoised`
2. **Renoise step**: `x ← (α_{i+1}/α_down)·x + noise·s_noise·renoise_coeff`

where `α = 1 − σ`, `σ_down = σ_{i+1}·downstep_ratio`, and
`renoise_coeff = (σ_{i+1}² − σ_down²·α_{i+1}²/α_down²)^½`.

The Euler sub-step **is** scale-invariant: `σ_down/σ_i` is unchanged when both scale by m_r.
The renoise sub-step cracks in **two places**: the x-coefficient `α_{i+1}/α_down =
(1−σ_{i+1})/(1−σ_down)` is affine-not-linear in σ and does NOT scale correctly under
`σ → m_r·σ`; AND `renoise_coeff` is also non-linear in m_r for the same reason; so even the
noise magnitude the shim provides is wrong, not just the x-rescale. The dead magnitude shim
(`make_per_row_noise_sampler`) pre-scales noise by m_r and fixes neither crack.

Fix: we already own `sampler_function` (`sampler.py::build_per_row_sampler_function`). Carry
`σ_r = m_r·σ` as a **per-row tensor** through the ENTIRE ancestral step (`sigma_down`/`sigma_up`,
both `alpha` terms, `renoise_coeff`) and hand it our `corrected` as denoised. Every op is
elementwise over the row dimension, so each row runs its own schedule in one outer loop:
- m=1 rows → standard full ancestral noise;
- m=0 rows → `σ_r=0` ⇒ `renoise_coeff=0` ⇒ frozen;
- 0<m<1 rows → correct intermediate.

No composite of two coherent latents anywhere ⇒ **no ghost**. Init already matches: the init-lerp
produces `x_r = σ_{r,max}·ε + (1−σ_{r,max})·clean`, exactly CONST noise-scaling at `m_r·σ_max`.

So it is **one engine** (per-row schedule compression via `σ_r`); deterministic is the special
case where the renoise term vanishes. This is a real distinction from "special-case by sampler
type."

## Cost / risks (why this is not free)

- Implementation: reimplement each stochastic step we want (start with
  `sample_euler_ancestral_RF`) as a per-row loop; mechanical scalar-σ → broadcast-`σ_r`
  substitution, one small reimpl per variant. Note: RF ancestral does NOT use `get_ancestral_step`
  (that function is for Karras-style ancestral); it has its own `downstep_ratio`/alpha formulas.
- **m=0 row special-casing required:** when σ_r = 0, the Euler sub-step computes
  `σ_down/σ_i = 0/0` (NaN). m=0 rows must be detected and short-circuited before the
  ancestral step (e.g., skip or clamp; the denoised correction already freezes them at
  input, so the step can be skipped entirely).
- **Verify exact formulas first:** read `comfy/k_diffusion/sampling.py` `sample_euler_ancestral_RF`
  precisely (our notes approximate them); the two-crack structure above is derived analytically.
- **Audio stream:** the A/V carry (`sigma_v/sigma_a`, see
  [dit-forward](native-h3-mechanism/dit-forward.md)) interacts with per-row `m_a`; video is the
  clean proof, audio needs the same treatment carried through.
- Leak risk: if the model has internal σ-dependence beyond the compressed embedding (e.g.
  `scale_latent_inpaint`, audio carry) that our path doesn't neutralize, the identity breaks in
  practice; that is exactly what a spike would expose.

## Cross-check: Motion Context confirms the diagnosis

[motion-context-comparison](motion-context-comparison.md) shows MC is stochastic-robust for the
exact reason this theory predicts: MC **never compresses the x-space schedule**. It runs the stock
sampler at global σ, uses our lever 2 (adaln label `1 − m·σ`) ONLY, and freezes preserved rows via
the native composite. So no ancestral term is ever evaluated at a compressed σ. Our path breaks
stochastic precisely because lever 3 pushes `m·σ` into x-space.

Two ways to reconcile:
- **This theory (keep our engine):** make the ancestral step itself per-row `σ_r`-aware, so the
  compressed schedule is honored *and* the affine renoise terms see `σ_r`. Preserves true
  fractional VIDEO denoise.
- **MC-style (drop lever 3):** run the sampler at global σ + native composite, don't compress
  x-space. Stochastic-free, and fractional/temporal-fade masks work, but fractionality is
  composite-blend, which double-exposes. That's invisible on transient fades yet a visible ghost on
  a **sustained keyframe at 0<min_denoise<1** (see
  [motion-context-comparison](motion-context-comparison.md)), re-introducing the exact bug
  this repo exists to fix.

The theory is the only route that keeps BOTH clean fractional-keyframe denoise AND stochastic.

## Spike result (PASSED, 2026-08-27)

The spike was run as PR2 (`add-per-row-rf-ancestral-step`, merged ede2d8c). Config: 0.2MP, two
injects at fractional min_denoise d=0.20 and d=0.15, fade-in at start, 20 steps; both `euler`
(deterministic baseline) and `euler_ancestral` (new RF-ancestral path) tested.
Result: no grey static, no corruption on fractional or preserved rows, no ghosting;
euler_ancestral quality matches euler.
The linchpin identity `v = (x − denoised)/σ_carrier`, `denoised_r = x − σ_row·v` holds on the
real H3 model for the single-eval RF-ancestral path — **no hidden σ-dependence detected beyond
the label channel**. Bug B is killed for euler_ancestral.
Generalizing: PR3 (multistep) and PR4 (DPM++ SDE) reuse the same recovery identity but add
multistep history / a second in-step eval — de-risked by this result, but their own GPU gates
are still required (tasks #70, #73).
