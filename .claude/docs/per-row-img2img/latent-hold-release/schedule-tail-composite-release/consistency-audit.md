<!-- provenance: theory (source-verified consistency audit of 'rescheduled'; audio finding EXERCISED on GPU and observed MILD, mildness-explanation still theory) -->
<!-- verified: 2026-08-27 · comfy-ref source read (ldm/minimax/model.py, model_sampling.py, k_diffusion/sampling.py, samplers.py, model_base.py); code @34a5925 -->
# Consistency audit of mode `rescheduled`

Child of [schedule-tail-composite-release](../schedule-tail-composite-release.md); audits the
mechanism described in [design-and-mechanism](design-and-mechanism.md).

Question asked: is `rescheduled` self-consistent across its three channels — the LABEL the model
is told, the CONTENT the tensor actually holds, and the STEP each row integrates? Answer: exact
for the video stream under Euler; one real break on the audio stream at fractional ticks, already
exercised on GPU and observed to be mild.

## Verified self-consistent (video stream, Euler)

- **Label channel exact.** The model computes per-row `t = (1 − m·σ_v).clamp(max=t_pin)` at
  comfy-ref `comfy/ldm/minimax/model.py:588-600` (`mask_row_values` :77-86 pools the mask per
  2x2-patch-row). Our back-solved `w = σ_row/σ_glob` therefore yields `t_row = 1 − σ_row` exactly.
- **Content channel exact.** RF init is `CONST.noise_scaling = σ·noise + (1−σ)·clean` (class
  `CONST` in `comfy/model_sampling.py`), so the i=0 composite `w·x + (1−w)·clean` lands the row
  exactly on its own noise-line at `σ_row(0)`, for any `σ_max ≤ 1`.
- **Step channel exact for Euler.** `CONST.calculate_input` is the identity, so there is no `c_in`
  mis-scaling of rows sitting off the global level, and `to_d` recovers pure velocity with the
  global σ cancelling: `d = (x − (x − σ·v))/σ = v`. The r-lerp `x_prev + r·(x_cur − x_prev)` is
  thus `x_prev + Δσ_row·v`, the row's true Euler step
  (`comfy/k_diffusion/sampling.py:190-213`). Note the official step size is a single global scalar
  `dt`; no per-row step exists anywhere in stock comfy, where the per-step composites in
  `KSamplerX0Inpaint` (`comfy/samplers.py:630-643`) are the official substitute.
- **d=0 rows are triple-protected.** The label pins to the cond timestep (official behavior),
  `r = 0` (since `σ_row ≡ 0`) freezes content every step, and the final `never` restore backstops.
- **w ≤ 1 structurally**, since row position `k_d + i·span ≥ i` and sigma is monotone. And
  heterogeneous per-row noise levels within one forward pass are the model's native contract —
  official mask and cond rows do exactly the same thing.

## Inconsistencies found (ranked)

### A. Audio fractional ticks — a three-way break (REAL, empirically MILD)

This is the real finding. The audio stream runs in a CARRIED coordinate: the sampler carries audio
as `(σ_v/σ_a)·x_a`, which the model undoes with `carry = σ_a/σ_v` at
`comfy/ldm/minimax/model.py:527-551`, together with a `d/dσ_v` output correction. All of that is
keyed to the GLOBAL video sigma and the shift map `σ_a = time_shift_sigma(σ_v)`. Audio labels use
the SAME mask value against the shifted sigma: `rows_t = 1 − m·σ_a_glob` (`model.py:601-610`).

Our `w` is back-solved in VIDEO σ-space, and both the composite and the r-lerp treat the packed
tensor uniformly. For a fractional AUDIO tick that breaks three ways at once:

1. The implied label level `w·σ_a_glob` is not the stretched-audio-tail level, because the shift
   map is non-linear: `σ_a(w·σ_v) ≠ w·σ_a(σ_v)`.
2. The convex composite is taken in carried coordinates, where it is not the native-audio convex
   combination.
3. The ratio should be `Δσ_a_row/Δσ_a_glob`, not the video ratio.

**Empirical status (user, 2026-08-27): this path HAS been exercised on GPU and the impact is
mild.** An earlier draft of this audit claimed the fractional-audio path was untested because the
runs used binary audio masks — that was wrong. The test video's opening fade-in carries a
FRACTIONAL audio fade mask, so `rescheduled` has been running the broken path all along, producing
"some artifacts, but so far audio has been much less error-prone than video."

Why it stays mild (analysis, theory — not separately verified): the errors are smooth level
mislabels bounded by how far the shift map departs from linearity. They vanish at the endpoints
`w ∈ {0,1}` and again at `σ_v = σ_max = 1` (time_shift fixes σ=1), they are transient over a short
fade ramp, and critically they introduce no step discontinuity — unlike the label jump that
falsified `mask-drop` in STR-8 (see [gpu-results](gpu-results.md)).

**Planned direction (user decision, 2026-08-27):** settle the video methodology first, then extend
it to audio. The extension is cheap in principle — `σ_a_row(i) = time_shift_sigma(σ_row(i),
shift_v, shift_a)`, since a pointwise shift of the existing dense grid commutes with the
tail-stretch — followed by a per-stream back-solve `m_a = σ_a_row/σ_a_glob` and per-stream step
ratio `r_a = Δσ_a_row/Δσ_a_glob`. The carried-coordinate composite algebra still needs its own
derivation pass before any of that is implemented.

### B. Label quantization (negligible)

`_token_grid_masks` ceil-quantizes masks to 1/256 (`comfy/model_base.py:2230-2232`), so labels run
up to ~0.004σ noisier than content and step, with a ceil bias.

### C. t_pin saturation (negligible)

Labels clamp at `VISUAL_COND_TIMESTEP` 0.999 once `σ_row < 0.001` — the very tail end, truthful to
within 0.001.

### D. m=0 rows get PURE clean

Official injects `0.999·clean + 0.001·noise` augmentation (H3 `scale_latent_inpaint` in
`comfy/model_base.py`). Known deliberate divergence, not a defect.

### E. r-lerp is exact only for Euler

Multistep samplers (`dpmpp_2m`, `res_multistep`) keep internal history keyed to the global sigmas,
so the per-row step is a first-order approximation there.

### F. Preview/callback `denoised` uses global σ

Fractional-row preview is therefore slightly off. Cosmetic only.
