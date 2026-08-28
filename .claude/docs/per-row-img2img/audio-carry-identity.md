<!-- provenance: theory (source-derived math; Consequence 2 GPU-OBSERVED 2026-08-27, isolation pending) -->
<!-- verified: 2026-08-27 · comfy-ref @b78cec87 · repo @06c6bda; GPU obs added, math unchanged -->
# Audio carry identity: why ×S is exact globally but leaks per-row

Derived 2026-08-23 from comfy-ref source. Read when reasoning about fractional-AUDIO artifacts
([bugs · Bug A caveat](bugs.md#bug-a)) or the `forward` carry
([dit-forward](native-h3-mechanism/dit-forward.md)).

## The identity

Both streams' sigmas come from the same base schedule through `time_shift_sigma` with shifts
`shift_v`, `shift_a`; let `S = shift_v/shift_a` (= `audio_scale`, workflow-configurable). Shift
composition gives, at every step:

```
1/σ_a − 1 = S · (1/σ_v − 1)          (exact, any base sigma)
⇒  S·(1−σ_v) ≡ (σ_v/σ_a)·(1−σ_a)     (multiply both sides by σ_a·…; algebra)
```

## Consequence 1: the ×S clean-term fix is exactly right (global/m=1 level)

`process_latent_in` multiplies audio by S once at entry; `forward`'s per-step carry multiplies the
audio input by `carry = σ_a/σ_v`. Substituting the identity: the packed-space audio the sampler
holds follows a **plain CONST trajectory with clean = S·A**: `x_audio = σ_v·ε + (1−σ_v)·(S·A)`.
So Bug A's fix (scale the init-lerp clean term by S) is not an approximation; it is the exact
clean reference for the packed trajectory. Video works identically with S=1.

## Consequence 2: per-row compression breaks the carry (0 ≤ m < 1)

For a row compressed to level `m`, the DiT labels it at `m·σ_a` (audio), but the *input* the
carry produces has coefficients belonging to the global trajectory, not the compressed one. With
`k = σ_a/σ_v` (the carry): the row's model input has noise coefficient exactly `m·σ_a` (right!),
but clean coefficient `k·S·(1−m·σ_v)` where a true img2img row at `m·σ_a` needs `(1−m·σ_a)`.
Error ratio `ρ = k·S·(1−m·σ_v)/(1−m·σ_a)`:

- `m = 1`: ρ = 1 (identity ⇒ no error, consistent with Bug A fix being exact globally).
- `σ → 0` (late steps): ρ → 1 for all m.
- `σ → 1` (early steps): ρ → S for any m < 1; e.g. m=0 rows present up-to-×S-too-loud audio
  context to the DiT early in sampling.

So fractional/preserved AUDIO rows see a mis-scaled clean component early; perceptibility is
**GPU-observed 2026-08-27** (attribution pending — see "GPU observation" section below).

## Consequence 3 / #76 root cause: velocity recovery must use per-modality σ_g

Line 24 above: packed audio lives on the σ_v trajectory (`x_audio = σ_v·ε + (1−σ_v)·S·A`), so
comfy's `denoised` (formed with σ_v) is correct. But per-row velocity recovery forms
`denoised_r = x − σ_row·v` where `σ_row` is at σ_a scale for audio rows. If the recovery divides
by σ_v (`ctx.sigmas[i]`, the video carrier) instead of `ctx.sig_g` (per-modality global), the
effective lerp weight becomes `σ_a/σ_v = w/S ≈ w/4`: audio rows under-denoise ~4×; at m=1 ~75%
of pre-final noise survives → the euler_a hiss (task #76, CONFIRMED analytically 2026-08-27).
Fix: divide by `ctx.sig_g`. Video is a no-op (`sig_g == sigmas[i]` for video), so this cannot
regress the #68 GPU pass. GPU perceptual confirm pending (task #76 verify).

## GPU observation 2026-08-27 (Consequence 2 candidate; attribution pending)

**Build tested:** `wiki-dpmpp-spine @ b21dd87` — `sampler.py` is stock `main`.

**Confound:** the #76 carrier bug (`/σ_v` instead of `/ctx.sig_g`) is LIVE in this build.
The observation is therefore NOT yet cleanly attributed to either cause.

**Symptom:** a "ringing feedback" audio artifact originating specifically in fractional-denoise
(0<m<1) frames — the fade-out of inject 1 and the blend boundaries of keyframe injects 1 & 2.
The artifact then echoes and resolves diegetically forward into following frames, including regions
with no injected audio.

**Leading hypothesis:** the symptom matches Consequence 2 (×ρ mis-scaling worst early, σ→1,
localised to fractional rows) far better than the #76 carrier bug (which is broadband
under-denoise, uniform across the run, worst at m=1).
The forward propagation is consistent with H3's overlapping VAE decode windows
(memory `h3-vae-decode-overlap`) plus the model carrying excess energy until it decays.

**Decisive retest PENDING:** same workflow on `fix-audio-carrier-recovery` (= `main` + `/sig_g` only).

- Ringing persists → Consequence 2 **CONFIRMED**; build the 1/ρ wrapper compensation (next section).
- Ringing vanishes → symptom was the #76 carrier bug; Consequence 2 still unconfirmed perceptually.

## If GPU retest shows fractional-audio artifacts

A wrapper-side per-row affine compensation is derivable: adjust the audio slice of the wrapper's
input per row by `1/ρ` on the clean component (requires `clean_packed`, per-step σ_v/σ_a, and
inverting `forward`'s output transform `out[1] = (1−S)·(A·k) + (1+(S−1)·σ_a)·out[1]`). Native
precedent: `scale_latent_inpaint` pre-divides injected clean audio by `(σ_v/σ_a)/S` for exactly
this reason. Do NOT build this until an artifact is actually observed.
