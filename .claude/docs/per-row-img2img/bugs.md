<!-- provenance: bug (A: fixed; B: open/deferred, mechanism refined 2026-09-01 (v-hat error, not coeff);
     C free-audio observer content-axis: plant-axis FALSIFIED GPU 2026-09-01; content-axis fix
     SHIPPED PR #32 revised 4644fcf+e4a9940 GPU PENDING; C full record in bugs/bug-c-audio-axis.md;
     D optional inject_list: fixed-pending-merge; E long-fade video interference: OPEN, DECOUPLED
     2026-08-28 — M-B (held ≥ ~28 AND ramp ≥ 51) unique survivor, M-A/M-C/M-D/M-E refuted;
     F euler_ancestral clean-K/V wiring gap: ATTRIBUTED retrodiction post-#32 2026-09-01) -->
<!-- verified: 2026-09-01 · plant-axis FALSIFIED GPU; content-axis fix CPU-tested shipped; Fix A GPU 2026-08-28; H2 falsified fade-length GPU; Bug E decoupling matrix on 0/0/39/90 @72b61c6 -->
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

**Mechanism refinement (2026-09-01, PR #32 design):** the per-row ancestral algebra
(`renoise_coeff`) is EXACT and level-preserving given accurate v̂ — **not a coeff defect**.
Retention = VELOCITY-ESTIMATION ERROR (`x0̂ = x0 + σ_row·(v − v̂)`) re-excited each step by
fresh ancestral injection; deterministic euler makes the same error but never re-excites it. For
fractional AUDIO specifically, the systematic v̂ error source was init-plant axis incoherence
(plant used σ_a ratio for audio content sitting on σ_v); fixed by PR #32 plant-axis fix — see
[euler-ancestral-per-row-fix/plant-axis.md](euler-ancestral-per-row-fix/plant-axis.md) §"Bug B mechanism refinement".

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

**Free-audio euler_ancestral distortion — audio observer band K/V content wired to wrong axis.**
Full record, co-location note, H2 rejection, and current fix status in
[bugs/bug-c-audio-axis.md](bugs/bug-c-audio-axis.md).

Short: observer side-stream primed K/V content on σ_a axis while audio x_prev sits on σ_v
post-Fix-A. Plant-axis fix (PR #32 original) FALSIFIED for fade-audio (GPU 2026-09-01:
single-frame clean / fade hiss persists). Content-axis fix SHIPPED (PR #32 revised, commits
4644fcf+e4a9940): `_audio_observer_ratio` maps content via `shift⁻¹(m·σ_a)` on σ_v (Möbius
inverse). GPU PENDING. See [euler-ancestral-per-row-fix.md](euler-ancestral-per-row-fix.md) for
full design + status.

## Bug C-remaining — H2 REJECTED; see Bug E for the primary open bug

See [bugs/bug-c-audio-axis.md](bugs/bug-c-audio-axis.md) for the H2 rejection and σ̃ status.
The primary open bug is Bug E below; see [audio-carry-identity.md](audio-carry-identity.md) for the
Consequence-2 ρ mechanism (REAL, ancestral-amplified, GPU discriminator `0/0/49/73` 2026-08-29).

## Bug D

**`H3InjectSampler` required `inject_list` → "Missing Connection" on zero injects (FIXED, pending merge).**

**Symptom:** the node declared `inject_list` as a required input, so wiring zero injects — or
bypassing all of them — raised a ComfyUI "Missing Connection" error, blocking a plain
passthrough/no-inject run.

**Fix** (branch `fix-optional-inject-list`): make `inject_list` optional so zero injects ==
passthrough (no-op inject). **Status: FIXED, pending merge.**

## Bug E — OPEN: Long-fade video-latent interference (moiré/streamers), sampler-independent

**Symptom:** moiré / streamers / electric patterns when a substantial frozen held block coexists
with a long fade ramp; sampler-independent; audio tracks via joint attention. DECOUPLED 2026-08-28:
**ERROR ⟺ held ≥ ~28 AND ramp ≥ 51** (both terms independently necessary; M-B unique survivor;
M-A/M-C/M-D/M-E refuted). Full data, decoupling matrix, and mechanism theory in
[long-fade-grid-beat.md](long-fade-grid-beat.md).

## Bug F — OPEN: euler_ancestral not wired to clean-K/V single-forward → fractional video keyframes ghost

**Symptom** (user, GPU 2026-09-01, PR #31): under `euler_ancestral`, a single-frame fractional
inject at **0.3** denoise showed a distorted ghost/noise frame; the same at **0.6** was clean —
low-fractional-worst, matching the core `0 < min_denoise < 1` keyframe-ghost signature.

**Root cause (source-confirmed):** the clean-K/V observer-splice `_single_forward_denoised` (the
fade-ghost fix from PR #28, [clean-kv-split.md](c2-rho-fix-paths/observed-level-plant/clean-kv-split.md))
is wired ONLY into `_euler_step` (routes at sampler.py:391). `_euler_ancestral_rf_step`
(sampler.py:412) calls `ctx.model(...)` directly at the carrier sigma (sampler.py:467), so
fractional rows under `euler_ancestral` bypass the ghost fix and ghost.

**Not introduced by PR #31** — the gap exists on `main`; PR #31 is a no-op on video. Stochastic
ancestral noise can make it vary run-to-run.

**Proposed fix:** route `_euler_ancestral_rf_step`'s per-row denoised through
`_single_forward_denoised` as `_euler_step` does. This is half 1 of the combined fix design in
[euler-ancestral-per-row-fix.md](euler-ancestral-per-row-fix.md) (half 2 = the σ_v axis for Bug C).
Distinct from Bug B's stochastic renoise (which also hits euler_ancestral); both vanish under `euler`.

**Post-#32 attribution (2026-09-01, retrodiction):** post-PR #32 both the video ghost AND the
co-located audio noise are gone. The single-frame drop-mode audio noise is ATTRIBUTED to Bug F
(H3's shared A/V attention imprinted the ghost-contaminated VIDEO denoised on co-located audio —
consistent with Bug E audio-tracks-visual GPU precedent in
[audio-axis-verdict.md](audio-axis-verdict.md)). NOT isolated by a dedicated A/B; retrodiction
only. Clean-K/V wiring fix may be included in PR #32 or resolved indirectly via the plant-axis fix.

Source footnotes for Bug A (`^plin`/`^pconds`/`^ec`/`^fwd`/`^sli`) live with its full record in
[bugs/bug-a-audio-scale.md](bugs/bug-a-audio-scale.md).
