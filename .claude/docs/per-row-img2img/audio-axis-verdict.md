<!-- provenance: bug+theory (single-cause axis verdict + "euler clean" FALSIFIED by GPU 2026-08-28; Fix A a correct LOCAL improvement, NOT the #76 cure; σ_a-load-bearing-for-LABEL proof still valid; true cause under investigation) -->
<!-- verified: 2026-08-28 · Fix-A GPU A/B (user, branch fix-audio-ancestral-axis-mismatch) FALSIFIED the verdict;
     _euler_step byte-identical main↔branch (deduction below); σ_a-label proof from source comfy-ref @b78cec87;
     commit 41c488d; prior branch fix-audio-carrier-recovery @2483914 -->
# Audio ancestral axis mismatch — verdict FALSIFIED; Fix A a correct local improvement

Consequence 3 of the [audio carry identity](audio-carry-identity.md).
Read when debugging `_euler_ancestral_rf_step`, Fix A, or the audio noise floor ([bugs.md](bugs.md)).

## FALSIFIED by GPU 2026-08-28 — read this first

The user GPU-tested Fix A (move only the ancestral integration to σ_v) on branch
`fix-audio-ancestral-axis-mismatch`. Result:

1. Fractional-frame "muffled squeaking" (FACT 1) **PERSISTS** under euler_ancestral — Fix A did
   NOT resolve it.
2. A persistent **low noise floor** runs through the NON-INJECTED (m=1, fully free) timeline.
3. That floor is present under euler_ancestral AND ALSO (quieter) under deterministic **euler** —
   the sampler previously believed clean.

**Decisive deduction (verified from code):** the Fix A diff touches ONLY
`_euler_ancestral_rf_step` (plus threading `sig_row_v` through `_StepContext`). The deterministic
`_euler_step` (sampler.py:278, registered `"sample_euler": _euler_step` at :404) is UNCHANGED —
byte-for-byte identical between `main` and the fix branch. So the floor the user now hears under
euler was ALREADY PRESENT on main; the earlier "euler is clean" A/B only ever established the
absence of the loud ancestral RINGING/ECHO, never a noise-free floor — nobody had listened for the
floor under euler before.

Consequences (all falsify the sections below):
- **(a)** The floor is SAMPLER-INDEPENDENT and pre-existing. euler runs no ancestral renoise yet
  shows it → the ancestral step is NOT its root cause. The "single-cause axis mismatch" verdict
  and the "euler-clean discriminator" are **FALSIFIED**.
- **(b)** Fix A cannot fix the floor (doesn't touch euler) and did NOT fix the fractional squeak
  under euler_ancestral → the axis mismatch was not the squeak's sole cause either.
- **(c)** The floor appears in m=1 FREE audio, where Consequence 2's error ratio ρ = 1 exactly →
  it is NOT Consequence 2 either. The cause lives in the shared every-step/every-row audio-carry
  machinery that corrupts even fully-free audio. Candidates, all UNVERIFIED: the
  `process_latent_in` ×S audio scale, the `forward` σ_a/σ_v carry, or packed-trajectory handling.
  **Root cause is UNDER INVESTIGATION — do not assert one.** See [bugs.md](bugs.md) noise-floor bug.

## Fix A — a correct LOCAL improvement, but does NOT resolve #76

Fix A moves ONLY `_euler_ancestral_rf_step` `denoised_r` (:380) and the ancestral terms
si/sigma_down/ratio/renoise_coeff (:384-393) onto the σ_v-axis per-row sigma. It is a **correct
local improvement**: audio at m=1 becomes bit-exact vs stock `sample_euler_ancestral_RF`, and
video is byte-identical. But it does NOT resolve #76's symptoms (fractional squeak persists; the
noise floor is sampler-independent and untouched by it). Keep it as a correctness fix on the
ancestral path; it is not the #76 cure. Whether to merge is a separate call.

## ~~Single-cause verdict (Consequence 3)~~ — FALSIFIED

~~BOTH GPU facts share ONE root cause: the axis mismatch in `_euler_ancestral_rf_step`.~~
**FALSIFIED (2026-08-28):** the noise floor is sampler-independent and present under euler, which
runs none of these lines; and Fix A left FACT 1 intact. The two symptoms do not share this single
cause. (Historical detail retained: the stock `sample_euler_ancestral_RF`,
comfy-ref/comfy/k_diffusion/sampling.py:247-265, computes sigma_down/ratio/renoise_coeff from pure
σ_v with zero AV branching, and our sampler puts audio's ancestral terms on σ_a at :380,384-393 —
so Fix A's local correctness claim stands, but it is not the #76 root cause.)

## ~~Euler clean — live A/B~~ — FALSIFIED as a discriminator

~~Swapping euler_ancestral→euler makes ringing VANISH, ruling out all sampler-independent
causes.~~ **FALSIFIED (2026-08-28):** the A/B only showed euler lacks the loud ancestral RINGING;
it never established a noise-free floor, and the Fix-A test now reveals a low floor present under
euler too. euler is NOT clean; it merely lacks the ancestral echo. The discriminator cannot rule
out sampler-independent causes — the floor IS one.

## Renoise is (only) the forward-echo engine — narrowed

The ancestral renoise injection (sampler.py:395-397, eta>0) still explains the LOUD
ringing/echo that euler lacks (euler has no renoise). That much survives: renoise is the echo
engine for the loud component. It does NOT explain the sampler-independent floor.

## σ_a is load-bearing for the LABEL — proof STILL VALID (independent of the falsification)

This model-contract proof is independent of the axis verdict and remains valid.

**VERDICT: σ_a is removable for the ancestral INTEGRATION but LOAD-BEARING for the LABEL.**
Fix B (full-unification: σ_v for BOTH label AND integration) remains **REJECTED — by
model-contract proof, not assumption.**

**PROOF (independently verified from source + model, not circular):** our sampler passes the model
a fraction `w = sig_row/sig_g` (sampler.py:755,758); the model computes `t_row = 1 − w·σ_g`
(comment sampler.py:754; model `_forward` comfy model.py:604-605 `rows_t = 1 − m·σ_a`). For
audio, `sig_g = sig_a[i]` which EQUALS the model's own internally-derived σ_a (both from
time_shift_sigma(σ_v)). So current `w = sig_row_a/sig_a[i]` → model yields `1 − sig_row_a`
(truthful label). If the σ_v fraction were passed instead (`w = sig_row_v/sig_v[i]`), the model
still multiplies by ITS σ_a → `1 − sig_row_v·(σ_a/σ_v)`. Since time_shift_sigma is nonlinear
(σ_a/σ_v ≈ 0.27→1.0 across the schedule,
[dit-forward.md](native-h3-mechanism/dit-forward.md)), this MISLABELS audio, worst at
early/high-σ steps. The σ_a denominator is required for a truthful label.

**σ_a IS LOAD-BEARING IN THREE SITES** (only the ancestral integration axis is questioned):

1. Per-row label denominator `w = sig_row/sig_g` (sampler.py:557,755) — proven above.
2. Observer-label K/V split — observer labels `t_obs = 1−m·σ` use shifted σ_a for audio
   ([label-ratio-and-observer-split.md](schedule-tail-late-delta/label-ratio-and-observer-split.md):95).
3. Deterministic r-scaling `r = (sig_row−sig_row_next)/(sig_g−sig_g_next)` in `_euler_step`
   (:305) and `_fallback_step` (:272) — unifying to σ_v would perturb a path with zero benefit.

**PROVENANCE — deliberate design:** σ_a axis was intentional, introduced in commit 41c488d
("Ship schedule-tail remap + observer-label K/V split": "complete the audio port: audio rows run
the remap on the sigma-shifted audio schedule via time_shift_sigma"). The schedule-tail remap idx
(`k_d`, `_stream_row_sigma` sampler.py:496-499) is axis-INDEPENDENT; the 17n+5 A/V tail join
controls layout, not sigma axis — so σ_a affects only sigma VALUES, not tail alignment.
