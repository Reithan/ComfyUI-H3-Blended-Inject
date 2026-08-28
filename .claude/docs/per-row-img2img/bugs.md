<!-- provenance: bug (A: fixed; B: open/deferred; C free-audio ancestral axis: FIXED by Fix A,
     GPU-validated 2026-08-28; C-remaining: H2 carry-contract renoise ROOT CAUSE CONFIRMED
     2026-08-28, fix designed, awaiting GPU; D optional inject_list: fixed-pending-merge) -->
<!-- verified: 2026-08-28 · controlled GPU A/B (user, branch fix-audio-ancestral-axis-mismatch, no fractional injects) reinstates Fix A; supersedes 94b1597 · repo @72b61c6 · D added 2026-08-28 -->
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
Separately, our sampler.py ran audio's ancestral integration (denoised_r, si/sigma_down/ratio/
renoise_coeff) on σ_a instead of σ_v — a real bug for FREE audio, now FIXED (see Bug C).

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

## Bug C

**Free-audio euler_ancestral distortion — an our-node axis bug, FIXED by Fix A (GPU-validated
2026-08-28). Retracts the earlier "sampler-independent noise floor" framing.**

A prior wiki commit (94b1597) called this a persistent, sampler-INDEPENDENT noise floor present
even under deterministic euler on free (m=1) audio, and marked the axis verdict FALSIFIED. That run
used FRACTIONAL injects, conflating two phenomena. **Retracted.** Controlled GPU A/B (user,
2026-08-28: same prompt, NO fractional injects, minimal graph) shows:

- STOCK KSampler (our node OUT): euler CLEAN, euler_ancestral CLEAN.
- OUR node, `main`, free audio (m=1): euler CLEAN, euler_ancestral TINNY/REVERB/NOISY.
- OUR node, Fix A branch, free audio: euler CLEAN, euler_ancestral **CLEAN**.

So free-audio `euler` is CLEAN (no floor); the distortion was an OUR-NODE `euler_ancestral` bug.

**Root cause:** on main, audio rows computed the ancestral RENOISE terms on the σ_a schedule
(`sig_row`) while the packed audio lives on the σ_v trajectory → mis-scaled renoise injected every
step → accumulating tinny/reverb noise. euler has no renoise, so it was clean both ways. **FIXED
by Fix A** (move denoised_r + si/sigma_down/ratio/renoise_coeff to the σ_v axis); m=1 audio now
bit-exact vs stock ancestral, video byte-identical. σ_a stays load-bearing for the LABEL
(model-contract proof still valid). Full verdict: [audio-axis-verdict.md](audio-axis-verdict.md).

## Bug C-remaining (H2 carry-contract renoise mis-scale — ROOT CAUSE CONFIRMED; fix designed, awaiting GPU)

**Symptom** (user, GPU 2026-08-28, Fix A branch): with a VIDEO fade-in inject, audio distorts +
loud noise IN THE FADE / fractional region only. euler+fade is CLEAN; only euler_ancestral+fade
is noisy.

**Root cause (H2, confirmed by discriminator matrix 2026-08-28):** the H3 model applies a GLOBAL
carry = σ_a/σ_v to audio every step (comfy-ref/comfy/ldm/minimax/model.py:528-538). The sampler's
packed audio lives at scaled amplitude σ̃ = (σ_v/σ_a)·sig_row. Fix A renoises on sig_row_v = σ̃
correctly only when carry=1 (video) or m=1 (w→1). For fractional audio (0<m<1), σ̃ ≠ sig_row_v;
the ancestral renoise (sigma_down/ratio/renoise_coeff on sig_row_v) injects a mis-scaled noise
magnitude every step → accumulates → audible noise. euler has no renoise → clean with identical ρ
mis-scale, which RULES OUT Consequence 2 ρ as the audible cause (real but inaudible for now).

**Fix design (H2 fix, on top of Fix A):** swap sig_row_v → carry-consistent σ̃ for ALL ancestral
terms in `_euler_ancestral_rf_step` (denoised_r, si, sip1, sigma_down, alpha, renoise_coeff,
ratio). Full fix spec + falsifiable prediction: [audio-axis-verdict.md](audio-axis-verdict.md).

**Status: fix designed + implemented on branch fix-audio-ancestral-axis-mismatch; awaiting GPU
confirmation (task #77). Consequence 2 ρ is next suspect only if H2 fix leaves a residual.**

## Bug D

**`H3InjectSampler` required `inject_list` → "Missing Connection" on zero injects (FIXED, pending merge).**

**Symptom:** the node declared `inject_list` as a required input, so wiring zero injects — or
bypassing all of them — raised a ComfyUI "Missing Connection" error, blocking a plain
passthrough/no-inject run.

**Fix** (branch `fix-optional-inject-list`): make `inject_list` optional so zero injects ==
passthrough (no-op inject). **Status: FIXED, pending merge.**

[^plin]: `comfy/model_base.py` 2158-2159 (`process_latent_in`, audio ×S).
[^pconds]: `comfy/samplers.py` 1046-1048 (`process_conds` → `extra_conds`).
[^ec]: `comfy/model_base.py` ~2199 (`extra_conds` sets `audio_scale`), 2140-2144 (`audio_scale()`).
[^fwd]: `comfy/ldm/minimax/model.py` `forward` 527-551 (carry 538/549-550).
[^sli]: `comfy/model_base.py` 2248-2272 (audio factor 2264).
