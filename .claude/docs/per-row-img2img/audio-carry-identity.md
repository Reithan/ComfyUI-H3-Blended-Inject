<!-- provenance: confirmed C1/C2 (source-derived); C3 free-audio ancestral axis fix VALIDATED,
     GPU 2026-08-28; σ_a-LABEL proof valid; C2 status AMBIGUOUS (H2 FALSIFIED as fade-length confound;
     primary audible artifact is video-primary long-fade interference, not C2 and not H2 renoise axis) -->
<!-- verified: 2026-08-28 · source-derived algebra (C1/C2); C3 Fix A validated (free audio), earlier falsification retracted — audio-axis-verdict.md;
     H2 subsequently falsified by fade-length GPU data (same date, late); source base comfy-ref @b78cec87 · repo @06c6bda -->
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

So fractional/preserved AUDIO rows see a mis-scaled clean component early. **Consequence 2 is a
real packed-clean error; its audibility is AMBIGUOUS.** The earlier euler+fade=CLEAN claim that
"ruled out" ρ rested on a confounded discriminator (fade length was uncontrolled — see
[audio-axis-verdict.md](audio-axis-verdict.md)). The H2 renoise hypothesis that replaced it is
also REJECTED — the primary audible artifact is a video-primary long-fade interference bug
(moiré/streamers, sampler-independent; see [bugs.md](bugs.md) Bug E). C2 ρ is NOT that bug.
Whether a faint C2 residual is audible after the primary bug is resolved remains unknown.
C2 does NOT touch free (m=1) audio, where ρ = 1 exactly.

## Consequence 3 — free-audio ancestral axis fix VALIDATED (controlled GPU 2026-08-28)

Full detail, the retraction of the earlier falsification, the σ_a-load-bearing-for-LABEL proof, and
Fix A/B status are in the child doc: **[audio-axis-verdict.md](audio-axis-verdict.md)**.

Short summary: on `main`, our `euler_ancestral` distorted FREE (m=1) audio (tinny/reverb) while
stock KSampler ancestral was clean — an OUR-NODE bug. Root cause: audio rows ran the ancestral
RENOISE terms on the σ_a schedule while the packed audio lives on the σ_v trajectory. **Fix A**
(move the ancestral integration to σ_v) makes free-audio euler_ancestral CLEAN, matching stock;
m=1 audio bit-exact, video byte-identical. Controlled GPU A/B (2026-08-28, no fractional injects)
VALIDATED this. An earlier commit (94b1597) had called it FALSIFIED / "not the cure" based on a run
WITH fractional injects that conflated two phenomena — RETRACTED as premature. σ_a remains
LOAD-BEARING for the LABEL (model-contract proof, still valid). The primary open issue — long-fade VIDEO-latent interference (moiré/streamers, ~60f noisy vs ~30f
clean, sampler-independent, present on main) — is VIDEO-PRIMARY, not a C2 or H2 audio bug.
H2 (carry-contract renoise) is REJECTED as the cause. RCA in progress; see
[audio-axis-verdict.md](audio-axis-verdict.md) and [bugs.md](bugs.md) Bug E.

## If artifacts survive the axis fix — Consequence-2 (ρ) compensation

A wrapper-side per-row affine compensation is derivable: adjust the audio slice of the wrapper's
input per row by `1/ρ` on the clean component (requires `clean_packed`, per-step σ_v/σ_a, and
inverting `forward`'s output transform). Native precedent: `scale_latent_inpaint` pre-divides
injected clean audio by `(σ_v/σ_a)/S` for this reason. Do NOT build until an artifact is
actually observed on pure euler after the axis fix is applied.
