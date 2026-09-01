<!-- provenance: bug (A: fixed; B: open/deferred; C free-audio ancestral axis: FIXED by Fix A,
     GPU-validated 2026-08-28; C-remaining: H2 REJECTED as fade-length confound 2026-08-28;
     D optional inject_list: fixed-pending-merge; E long-fade video interference: OPEN, DECOUPLED
     2026-08-28 — M-B (held ≥ ~28 AND ramp ≥ 51) unique survivor, M-A/M-C/M-D/M-E refuted;
     F euler_ancestral clean-K/V wiring gap → fractional video ghost: OPEN, GPU 2026-09-01) -->
<!-- verified: 2026-08-28 · Fix A: no-fractional-injects GPU A/B VALIDATED; H2 falsified by fade-length GPU data (same date, late); Bug E decoupling matrix on 0/0/39/90 · repo @72b61c6 -->
# Bugs: audio scale (A, fixed) & stochastic samplers (B)

Read this when debugging fractional-region artifacts (audio garble, grey/reverse noise). The code
these bugs live in is described in [our-architecture](our-architecture.md).

## Bug A

**Fractional AUDIO garbled under deterministic euler (FIXED).** Full record carved out to
[bugs/bug-a-audio-scale.md](bugs/bug-a-audio-scale.md) (char budget). Short: our init-lerp clean
term was packed RAW (audio ×1) while comfy's `x_global` carries audio ×S → fractional (0<m<1) audio
img2img'd from a mismatched reference ⇒ static. Fixed by `scale_packed_audio` (scale = dynamic
`audio_scale = shift_video/shift_audio`). Carry IS active in our path (source-verified); GPU retest
2026-08-23 fractional audio CLEAN. See the child for the full derivation, the carry caveat, and the
`scale_latent_inpaint` native counterpart.

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

**GPU-CONFIRMED umbrella (2026-09-01, PR #31):** a same-branch/same-prompt sampler swap shows
`euler` produces NEITHER the fractional audio noise NOR the video ghost; `euler_ancestral` produces
BOTH. The FRACTIONAL-INJECT audio noise (distinct from Bug C's m=1 free audio) is a symptom of THIS
bug — `eta>0` renoise breaks per-row scale-invariance. The co-occurring video ghost is the separate
clean-K/V wiring gap (Bug F). PR #31's σ_v-axis re-extraction did NOT resolve the audio noise (wrong
cause — axis, not stochastic); see [audio-axis-verdict.md](audio-axis-verdict.md).

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

## Bug C-remaining — H2 REJECTED (fade-length confound); see Bug E for the primary open bug

**What H2 was:** euler+fade=CLEAN vs euler_ancestral+fade=NOISY was read as a sampler discriminator
pointing to the ancestral-renoise mis-scale (σ̃ ≠ sig_row_v for fractional audio) as root cause.

**Why H2 is REJECTED:** new controlled data (user, 2026-08-28 late) shows the original test pair
used different fade lengths (~30f clean vs ~60f noisy). Fade length, not sampler, was the
independent variable. euler+LONG fade also artifacts. The discriminator is spurious; H2 is not
the root cause.

**σ̃ implementation status:** video-byte-identical and m=1 bit-exact (harmless), but UNVALIDATED
as a fix for the audible artifact. Do not present it as the fix. The mergeable branch
`fix-audio-ancestral-sigma-v-axis` is Fix-A-only — σ̃/`sig_row_c` is dropped, not shipped.

**The primary open bug is Bug E below.** Consequence-2 ρ status: **REAL and ancestral-amplified**
— audible on `euler_ancestral` fade ramps, silent on deterministic `euler` (GPU discriminator
`0/0/49/73`, 2026-08-29). Canonical mechanism + fix paths:
[audio-carry-identity.md](audio-carry-identity.md).

## Bug D

**`H3InjectSampler` required `inject_list` → "Missing Connection" on zero injects (FIXED, pending merge).**

**Symptom:** the node declared `inject_list` as a required input, so wiring zero injects — or
bypassing all of them — raised a ComfyUI "Missing Connection" error, blocking a plain
passthrough/no-inject run.

**Fix** (branch `fix-optional-inject-list`): make `inject_list` optional so zero injects ==
passthrough (no-op inject). **Status: FIXED, pending merge.**

## Bug E — OPEN: Long-fade video-latent interference (moiré/streamers), sampler-independent

**Symptom** (user, GPU 2026-08-28): moiré / streamers / electric patterns when a substantial
frozen held block coexists with a long fade ramp; sampler-independent; audio tracks via joint
attention; present on main before Fix A or σ̃.

**DECOUPLED 2026-08-28 — M-B is the unique surviving model.** A GPU single-variable perturbation
matrix on `0/0/39/90` plus the earlier L=55 shift series isolate the three-way confound. Define
held = ekf−skf, ramp = efo−ekf. **ERROR ⟺ held ≥ ~28 AND ramp ≥ 51**, both terms independently
necessary: ramp-necessity from `0/0/39/85` (CLEAN) vs `0/0/39/90` (ERROR) at fixed held;
held-necessity from `0/0/25/80` (CLEAN) vs `0/0/30/85` (ERROR) at fixed ramp=55. The competing
single-factor models are all REFUTED — pure-ramp band (M-A), held-alone (M-E), midpoint-position
(M-D), and trailing-free-heals (M-C). The c=39→40 flip is just ramp crossing 51→50 (a continuous
frame-length threshold, not grid-quantized). S2 cell-alignment was GPU-FALSIFIED earlier the same day.

**REFINED** ([long-fade-grid-beat/two-stage-heal.md](long-fade-grid-beat/two-stage-heal.md)): M-B's
single rule decomposes into FORMATION (ramp ≥ 51, monotone, no upper edge through 68) ∧ NOT-HEALED
(held ≥ ~28, a soft healing boundary; held≈29 → MIXED); numeric predictions unchanged. Full data
table, decoupling matrix, and the mechanism (KV/observer curvature seed + seam-attention SNR-trough,
theory) live in [long-fade-grid-beat.md](long-fade-grid-beat.md).

## Bug F — OPEN: euler_ancestral not wired to clean-K/V single-forward → fractional video keyframes ghost

**Symptom** (user, GPU 2026-09-01, PR #31): under `euler_ancestral`, a single-frame fractional
inject at **0.3** denoise showed a distorted ghost/noise frame; the same at **0.6** was clean —
low-fractional-worst, matching the core `0 < min_denoise < 1` keyframe-ghost signature.

**Root cause (source-confirmed):** the clean-K/V observer-splice `_single_forward_denoised` (the
fade-ghost fix from PR #28, [clean-kv-split.md](c2-rho-fix-paths/observed-level-plant/clean-kv-split.md))
is wired ONLY into `_euler_step` (sampler.py:395). `_euler_ancestral_rf_step` calls `ctx.model(...)`
directly at the carrier sigma (sampler.py:476), so fractional rows under `euler_ancestral` bypass
the ghost fix and ghost.

**Not introduced by PR #31** — the gap exists on `main`; PR #31 is a no-op on video. Stochastic
ancestral noise can make it vary run-to-run.

**Proposed fix (NOT implemented):** route `_euler_ancestral_rf_step`'s per-row denoised through
`_single_forward_denoised` as `_euler_step` does. Distinct from Bug B's stochastic renoise (which
also hits euler_ancestral); both vanish under `euler`.

Source footnotes for Bug A (`^plin`/`^pconds`/`^ec`/`^fwd`/`^sli`) live with its full record in
[bugs/bug-a-audio-scale.md](bugs/bug-a-audio-scale.md).
