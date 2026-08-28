<!-- provenance: confirmed (C1/C2 source-derived; C3 axis mismatch DEFINITE — see audio-axis-verdict.md; σ_a load-bearing for label by model-contract proof; Fix A confirmed; Fix B REJECTED with proof 2026-08-28) -->
<!-- verified: 2026-08-28 · source-derived algebra (C1/C2); axis-mismatch verdict + σ_a-label proof in audio-axis-verdict.md;
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

So fractional/preserved AUDIO rows see a mis-scaled clean component early; the euler A/B (2026-08-28) shows ρ is present but perceptually negligible — see [audio-axis-verdict.md](audio-axis-verdict.md).

## Consequence 3 — velocity/ancestral axis mismatch (DEFINITE; Fix A confirmed; Fix B rejected)

Full verdict, σ_a-load-bearing proof, Fix A/B design, and test order are in the child doc:
**[audio-axis-verdict.md](audio-axis-verdict.md)**.

Short summary: stock ancestral integration uses pure σ_v (no AV branching); our sampler wrongly
runs audio's `denoised_r` + ancestral terms on σ_a. σ_a is LOAD-BEARING for the label (proven
by model contract, 2026-08-28) but NOT for the integration. Fix B (unify both to σ_v) is
**REJECTED with proof**. Fix A (move only ancestral integration to σ_v; leave label + r-scaling
on σ_a) is confirmed. Code change still UNVERIFIED on GPU; needs fail-then-pass regression test.

## If artifacts survive the axis fix — Consequence-2 (ρ) compensation

A wrapper-side per-row affine compensation is derivable: adjust the audio slice of the wrapper's
input per row by `1/ρ` on the clean component (requires `clean_packed`, per-step σ_v/σ_a, and
inverting `forward`'s output transform). Native precedent: `scale_latent_inpaint` pre-divides
injected clean audio by `(σ_v/σ_a)/S` for this reason. Do NOT build until an artifact is
actually observed on pure euler after the axis fix is applied.
