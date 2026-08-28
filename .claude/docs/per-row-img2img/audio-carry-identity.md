<!-- provenance: confirmed C1/C2 (source-derived); C3 single-cause axis verdict FALSIFIED by GPU 2026-08-28 (see audio-axis-verdict.md); σ_a-load-bearing-for-LABEL proof still valid; Fix A a local improvement, NOT the #76 cure -->
<!-- verified: 2026-08-28 · source-derived algebra (C1/C2); C3 axis verdict FALSIFIED, true cause under investigation — audio-axis-verdict.md;
     source base comfy-ref @b78cec87 · repo @06c6bda -->
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

So fractional/preserved AUDIO rows see a mis-scaled clean component early. Note the m=1-free noise
floor (below) rules OUT Consequence 2 as the floor's cause: at m=1, ρ = 1 exactly (no error), yet
the floor is still audible there — so ρ is present but is NOT what produces the floor.

## Consequence 3 — single-cause axis verdict FALSIFIED (GPU 2026-08-28); Fix A a local improvement

Full detail, the falsification, the surviving σ_a-load-bearing-for-LABEL proof, and Fix A/B status
are in the child doc: **[audio-axis-verdict.md](audio-axis-verdict.md)**.

Short summary: the earlier verdict named the ancestral axis mismatch as the SINGLE root cause of
both fractional ringing (FACT 1) and FACT 2, with "euler is clean" as the discriminator. GPU test
of Fix A **FALSIFIED** this: (1) the fractional squeak persists under euler_ancestral; (2) a
persistent low noise floor runs through the m=1 FREE timeline; (3) that floor is present under
deterministic euler too — whose code Fix A never touches (byte-identical main↔branch). So the
floor is sampler-INDEPENDENT and pre-existing; the axis mismatch is not its cause, nor the squeak's
sole cause. The floor is NOT Consequence 2 either (ρ=1 at m=1). σ_a remains LOAD-BEARING for the
LABEL (model-contract proof, still valid); Fix A is a correct LOCAL improvement (m=1 audio
bit-exact vs stock ancestral, video byte-identical) but NOT the #76 cure. True cause under
investigation — see [bugs.md](bugs.md) noise-floor bug.

## If artifacts survive the axis fix — Consequence-2 (ρ) compensation

A wrapper-side per-row affine compensation is derivable: adjust the audio slice of the wrapper's
input per row by `1/ρ` on the clean component (requires `clean_packed`, per-step σ_v/σ_a, and
inverting `forward`'s output transform). Native precedent: `scale_latent_inpaint` pre-divides
injected clean audio by `(σ_v/σ_a)/S` for this reason. Do NOT build until an artifact is
actually observed on pure euler after the axis fix is applied.
