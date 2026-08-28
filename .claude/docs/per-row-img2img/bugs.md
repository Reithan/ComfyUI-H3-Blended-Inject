<!-- provenance: bug (A: fixed; B: open/deferred) -->
<!-- verified: 2026-08-23 · repo @72b61c6 -->
# Bugs: audio scale (A, fixed) & stochastic samplers (B)

Read this when debugging fractional-region artifacts (audio garble, grey/reverse noise). The code
these bugs live in is described in [our-architecture](our-architecture.md).

## Bug A

**Fractional AUDIO garbled under deterministic euler (FIXED).**

**Symptom** (user, GPU): "39f inject f0 denoise 0.01, audio fade, fade 0/0/17/39, euler: video
perfect, audio garbled/staticky in the inject portion." Video fine, m==0 audio fine, only
fractional (0<m<1) audio corrupted.

**Root cause:** comfy applies `MiniMaxH3.process_latent_in`[^plin] to `sample_custom`'s
latent_image, multiplying the AUDIO slice by `audio_scale = 4.0` (video ×1.0). So the `x_global`
the sampler holds has audio ×4. But our init-lerp clean term (`clean_packed`) was packed RAW
(audio ×1). The lerp `m·x_global + (1−m)·clean_packed` mixed 4× and 1× audio ⇒ fractional audio
img2img'd from a 4×-mismatched reference ⇒ static. m=1 drops the clean term (fine); m=0 restored
post-sampling (fine) ⇒ only 0<m<1 audio affected.

**Fix** (commit "fix fractional-audio garble: scale init-lerp clean term by audio_scale"):
- `sampler.py::scale_packed_audio(packed, video_element_count, audio_scale)`: scales the audio
  tail in place; no-op when scale==1.0 or no audio tail.
- `nodes.py::_run_sampler`: after `pack_latents(clean_components)`, compute
  `n_video_elems = prod(latent_shapes[0][1:])`, read `audio_scale`, call `scale_packed_audio`.
- Regression: `tests/test_sampler.py::TestScalePackedAudio` (4 tests). Suite 522 pass, ruff clean.

**Scale is NOT fixed at ×4:** `audio_scale = shift_video/shift_audio` (S), set by the user via the
"ModelSamplingMiniMaxH3 / MiniMax H3 Sigma Shift" node. Official guidance: `shift_video` permissive
(~10–14+, tune per movement speed/detail); `shift_audio` ≈ 3±1, leave at default. S = 4.0 only at
defaults 12/3. `_run_sampler` reads it dynamically from `model_sampling.audio_scale`; correct.

**CAVEAT: carry IS active in our path (source-verified, corrected 2026-08-23).** A prior version
of this doc claimed our path "does NOT set audio_scale in the payload (we bypass native
extra_conds)". **FALSE**: `sample_custom → process_conds`[^pconds] calls `model.extra_conds`,
which unconditionally sets `payload["audio_scale"]`[^ec], so the per-step `sigma_v/sigma_a` carry
in `forward`[^fwd] runs every step. The [audio-carry identity](audio-carry-identity.md) shows this
is exactly what makes the constant ×S clean-term fix correct at the m=1/global level; the packed
audio trajectory is plain CONST with clean = S·A. **BUT** the same derivation exposes a residual
per-row mismatch for 0≤m<1 rows (clean coeff error up to ×S at early steps), the prime suspect if
fractional audio ever artifacts again. **GPU retest 2026-08-23 (user): fractional audio CLEAN**
(39f fade, euler, incl. non-default shift): Bug A fix CONFIRMED; the per-row mismatch is not
perceptible in practice so far. A wrapper-side affine compensation is derivable if it ever shows. `scale_latent_inpaint`[^sli] (factor `(sigma_v/sigma_a)/S`) is the NATIVE counterpart of that
compensation; unused in our path (`noise_mask=None`).

## Bug B

**Stochastic samplers (euler_ancestral etc.) corrupt fractional rows.**

**Symptom** (user, GPU): under `euler_ancestral`, the fractional/0.0-denoise section "ran in
reverse" (started clear, ended grey static); decode: fade frames grey noise + static audio.

**Root cause:** euler_ancestral → `sample_euler_ancestral_RF` for CONST models. Its renoise uses
affine `alpha = 1−sigma` terms (see
[k-diffusion-samplers](native-h3-mechanism/k-diffusion-samplers.md)) that are NOT scale-invariant,
so per-row compression can't be reproduced by scaling the injected noise. The old
`make_per_row_noise_sampler` shim only scales noise magnitude; insufficient.

**Possible recovery (THEORY, unverified):** the magnitude shim is insufficient, but a full per-row
ancestral step driven by `σ_r = m_r·σ` may fix this inside our single engine; see
[stochastic-recovery-theory](stochastic-recovery-theory.md). Under the shipped schedule-tail remap
Bug B persists in a new form (r-scaling linearly rescales a displacement that contains the non-linear
renoise term); the current-architecture per-row step-function design to recover it is in
[sampler-class-support.md](sampler-class-support.md).

**Status: deterministic-only (prototype).** DD (the native img2img-via-mask primitive) is the
exact dual; it covers stochastic but cracks deterministic on H3 (see
[differential-diffusion](differential-diffusion.md)), so it can't replace our path, only ADD a
second sampler-type-selected mechanism (special-casing smell). Decisions (memory
`prototype-goal-fade-mask-parity`): stochastic-sampler gate DEFERRED; stochastic shim
(`make_per_row_noise_sampler`, `scale_stochastic_noise`) may be left DEAD or deleted; the user
doesn't care. Supported path = deterministic (euler / res_multistep / dpmpp_2m).

[^plin]: `comfy/model_base.py` 2158-2159 (`process_latent_in`, audio ×S).
[^pconds]: `comfy/samplers.py` 1046-1048 (`process_conds` → `extra_conds`).
[^ec]: `comfy/model_base.py` ~2199 (`extra_conds` sets `audio_scale`), 2140-2144 (`audio_scale()`).
[^fwd]: `comfy/ldm/minimax/model.py` `forward` 527-551 (carry 538/549-550).
[^sli]: `comfy/model_base.py` 2248-2272 (audio factor 2264).
