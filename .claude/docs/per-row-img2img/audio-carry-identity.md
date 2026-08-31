<!-- provenance: confirmed C1/C2 (source-derived); C3 free-audio ancestral axis fix VALIDATED,
     GPU 2026-08-28; σ_a-LABEL proof valid; C2 ρ RESOLVED = REAL and ancestral-amplified (GPU discriminator
     0/0/49/73, 2026-08-29: euler CLEAN, euler_ancestral soft buzz on fade-out ramp); primary long-fade
     artifact is a SEPARATE video-primary interference bug (Bug E, needs ramp ≥ 51), not C2 and not H2;
     velocity-recovery divisor /sig_g GPU-FALSIFIED 2026-08-27 (/carrier load-bearing, corroborates Fix A);
     C2 ρ integration axis CORRECTED σ_v→σ_c (2026-08-29): a fractional audio row sits at
     σ_c = sig_row·carrier/sig_g, NOT m·σ_v (old form UNDERSHOT ~2.5 vs true ~4.0); input-side σ_c pre-comp
     (63b291e) GPU-FALSIFIED (buzz LOUDER) + ROOT-CAUSED (FALSE-REFERENCE: clean_packed=S·A truthful only @step 0);
     v3 denoised_r-only σ_c-projection ÷ρ_true (f06a84a) GPU quieter than B; v4 (12ea3b6) subtracts EXACT residual
     (S−1)(sig_row−sig_g)·x_prev, GPU MUCH-QUIETER = CURRENT BEST; v5 init-composite bootstrap (7436165)
     GPU-FALSIFIED + REVERTED @8c8cb90 → residual NOT init-driven; v6 m=0 audio-context heat (02fee22)
     GPU-CONFIRMED further improvement (2026-08-31: very-quiet metallic hum, buzz→metallic), native-confirmed (scale_latent_inpaint 1/(S·k));
     ⚠ OVERTURN (GPU 2026-08-31): the "euler CLEAN" premise below is FALSE — a DETERMINISTIC injection noise exists in BOTH the audio AND visual fade,
     isolated to OUR node (present in our euler @5 steps, absent in official euler @5); the many-step model NATURALIZES it into diegetic sound (read
     "clean" @20). C2 fixes (v3/v4/v6) live ONLY in _euler_ancestral_rf_step; _euler_step (sampler.py:305-335) applies NO C2 correction. Ancestral
     renoise AMPLIFIES, does not create it. Video noise ≠ C2 (S=1). See c2-rho-fix-paths/residual-accounting.md (Fable endgame re-derivation in progress) -->
<!-- verified: 2026-08-29 · source-derived algebra (C1/C2); C3 Fix A validated (free audio), earlier falsification retracted — audio-axis-verdict.md;
     C2 ρ ancestral-amplification discriminator on 0/0/49/73 (2026-08-29); H2 falsified by fade-length GPU data (2026-08-28);
     /sig_g A/B branch fix-audio-carrier-recovery @2483914; input-side pre-comp @63b291e GPU-FALSIFIED + root-caused (false-reference);
     v3 denoised_r-only σ_c ÷ρ_true @f06a84a GPU-CONFIRMED reduced; v4 exact-residual cancel @12ea3b6 GPU-CONFIRMED much-quieter (C2 ρ/residual COMPLETE);
     source base comfy-ref @b78cec87 (model.py:535-550, model_sampling.py:90-92) · repo @06c6bda -->
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
but a clean coefficient that a true img2img row at the row's own level needs to match.

**Integration axis — corrected σ_v→σ_c (2026-08-29).** The true σ_v-AXIS position of a fractional
AUDIO row is NOT `sig_row_v = m·σ_v`; it is `σ_c = sig_row·carrier/sig_g`, where `sig_row = m·σ_a`
(the per-row/label sigma), `carrier = sigmas[i]` (global σ_v at this step), `sig_g` (global σ_a).
So the true clean-coeff error ratio is `ρ_true = S·k·(1−σ_c)/(1−sig_row)` (with `k = σ_a/σ_v`):

- `m = 1`: ρ = 1 (identity ⇒ no error, consistent with Bug A fix being exact globally).
- `σ → 0` (late steps): ρ → 1 for all m.
- `σ → 1` (early steps): ρ → S for any m < 1; e.g. m=0 rows present up-to-×S-too-loud audio
  context to the DiT early in sampling.

**Why the old `ρ = k·S·(1−m·σ_v)/(1−m·σ_a)` form UNDERSHOT.** It placed the row at `sig_row_v =
m·σ_v` on the σ_v axis, but the carry (σ_a/σ_v) means an audio row at label `sig_row = m·σ_a`
actually sits at `σ_c = sig_row·carrier/sig_g` — farther out. The undershoot was ~2.5 vs the true
~4.0 at the worst point of the fade-out ramp, which is why prototype B (the `denoised_r`/ρ hack,
commit 9877350) only *Reduced* the GPU buzz rather than eliminating it. The `σ_c` axis UNIFIES both
fix paths (A renoise integration sigma, B clean correction): both must sit on σ_c, not σ_v.

So fractional/preserved AUDIO rows see a mis-scaled clean component early. **Consequence 2 is a
real packed-clean error, ancestral-amplified but NOT ancestral-specific** (⚠ overturn 2026-08-31:
the deterministic euler render is also affected — a per-row injection noise exists in both
modalities, see the header stamp + [c2-rho-fix-paths/residual-accounting.md](c2-rho-fix-paths/residual-accounting.md)).
C2 is the audio-only EXTRA on top of that deterministic error. The primary long-fade artifact is a SEPARATE
video-primary interference bug (moiré/streamers, sampler-independent, needs ramp ≥ 51; see
[bugs.md](bugs.md) Bug E). C2 ρ is NOT that bug.
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
LOAD-BEARING for the LABEL (model-contract proof, still valid). The primary open issue — long-fade
VIDEO-latent interference (moiré/streamers, sampler-independent, on main) — is VIDEO-PRIMARY, not a
C2 or H2 audio bug; H2 (carry-contract renoise) is REJECTED as the cause. RCA in progress; see
[audio-axis-verdict.md](audio-axis-verdict.md) and [bugs.md](bugs.md) Bug E.

### Corroborating evidence — velocity-recovery divisor A/B (/sig_g FALSIFIED, 2026-08-27)

A separate 2026-08-27 GPU A/B (branch `fix-audio-carrier-recovery` @2483914) tested the
velocity-recovery divisor. Only change vs main: `/carrier` (σ_v) → `/sig_g` (σ_a for audio). Free
(m=1) audio then produced LOUD FLUCTUATING NOISE — worse than main's residual.
So `/sig_g` is **FALSIFIED** and `/carrier` (σ_v) is **load-bearing**, corroborating Fix A's σ_v
axis choice (video a no-op). Unverified mechanism: `/carrier` keeps a residual
`denoised_r = C + (σ_v−σ_a)(ε−C)` that offsets the mismatch; `/sig_g` drops it. The standalone
`/sig_g` fix (PR #20) is abandoned.

The #68 GPU run (euler_a "slight microphone feedback"; euler clean) first sighted this — now
attributed to the σ_a-axis renoise bug and FIXED by Fix A ([bugs.md](bugs.md) Bug C); its "OPEN"
status is superseded.

## Consequence-2 (ρ) — GPU discriminator 2026-08-29 (⚠ euler-clean premise OVERTURNED 2026-08-31)

**SUPERSEDED framing.** This section read `euler` as CLEAN and concluded C2 was ancestral-specific.
GPU 2026-08-31 overturns that: euler is NOT clean, and a deterministic injection noise exists in
BOTH modalities, isolated to OUR node (see the header stamp and
[c2-rho-fix-paths/residual-accounting.md](c2-rho-fix-paths/residual-accounting.md)). C2 is now an
audio-only EXTRA on top of that deterministic error; ancestral renoise AMPLIFIES rather than creates
it. The ρ math below is still valid; only the "ancestral-specific / euler-clean" attribution is void.

**Config:** video inject at frame 0, fade markers `0/0/49/73` (held=49, fade-out ramp=24f),
ease_in_out, audio_mode=`fade`, min_denoise=0.0. **Original 2026-08-29 read (SUPERSEDED):** `euler`
appeared clean; `euler_ancestral` buzzed for ~2.5s on the fade-OUT — the fractional-audio ramp rows
[49,73). The carry (σ_a/σ_v) gives a fractional audio row `denoised_r` clean-coeff error
`ρ_true = S·k·(1−σ_c)/(1−sig_row)` (σ_c-axis, Consequence 2), which `_euler_ancestral_rf_step`
(sampler.py:418-422) re-noises against every step. The deterministic component that was mistaken for
"euler-clean" is what the overturn now surfaces.

## Fix paths — Consequence-2 (ρ) compensation

Full detail (fix-path options, the falsified + root-caused input-side pre-comp at 63b291e, and the
v3/v4/v5 fix chain) lives in the child folder: **[c2-rho-fix-paths/index.md](c2-rho-fix-paths/index.md)**.

Short summary: the input-side σ_c pre-comp (63b291e) was GPU-FALSIFIED 2026-08-29 (buzz LOUDER) and
ROOT-CAUSED as a FALSE-REFERENCE error (`clean_packed = S·A` is truthful only at step 0); input-side
pre-compensation is fundamentally unsound (needs the model's own evolving estimate — full detail in
the child folder). The **v3 fix (commit f06a84a)** drops all input perturbation: `denoised_r`-ONLY on
a **σ_c projection ÷`ρ_true`** with σ_c renoise axis, keeping the state on the truthful carried
trajectory so `x_prev` is already truthful. m=1 bit-exact,
video byte-identical (σ_v nuance in the child folder). The **v4 fix (commit 12ea3b6)** subtracts the
EXACT residual `(S−1)(sig_row−sig_g)·x_prev` (denoised_r-only): GPU-CONFIRMED much-quieter than v3
(faint residual, not zero) — **C2 ρ/residual work COMPLETE; v4 is the CURRENT BEST**. **v5 (commit
7436165)**, an init-only composite bootstrap fix, was **GPU-FALSIFIED (NO CHANGE vs v4) + REVERTED
(8c8cb90)** → the faint residual is NOT init-driven (the DiT self-corrects init within the first
steps). The last per-step lever, **v6 (commit 02fee22) — m=0 audio-context heat — is GPU-CONFIRMED a
further improvement** over v4: frozen (m=0) audio rows are presented to the model at the truthful
`clean/(1+(S−1)·sig_g)` instead of `S·A` (native-confirmed by `scale_latent_inpaint`'s `1/(S·k)`
rescale). GPU (2026-08-31): residual now a VERY-QUIET hum, quieter than v4 — the LEVEL drop is the
reliable win; the buzz→metallic timbre shift is LOW-INFO / likely a red herring (interference peaks
in the first ~1/3 of the gen and is reshaped downstream), NOT a different-mechanism signal. Source
accounting pending (Fable).
See [c2-rho-fix-paths/index.md](c2-rho-fix-paths/index.md).
