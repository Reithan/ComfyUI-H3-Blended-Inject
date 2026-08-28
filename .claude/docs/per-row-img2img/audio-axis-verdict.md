<!-- provenance: confirmed (axis mismatch DEFINITE; Fix A confirmed; Fix B REJECTED by model-contract proof 2026-08-28; σ_a load-bearing for label; code change itself still UNVERIFIED on GPU) -->
<!-- verified: 2026-08-28 · stock euler_a comparison (k_diffusion/sampling.py:247-265) + live euler A/B GPU (user)
     + model-contract σ_a-load-bearing proof; source comfy-ref @b78cec87; commit 41c488d;
     prior branch fix-audio-carrier-recovery @2483914 -->
# Audio ancestral axis mismatch — verdict and fix design

Consequence 3 of the [audio carry identity](audio-carry-identity.md).
Read when debugging `_euler_ancestral_rf_step` or reasoning about Fix A vs Fix B.

## Single-cause verdict (Consequence 3)

BOTH GPU facts — FACT 1 (fractional ringing, euler_ancestral-only) and FACT 2 (/sig_g loud noise,
GPU 2026-08-27) — share ONE root cause: the axis mismatch in `_euler_ancestral_rf_step`
(sampler.py:364,379-397). Model eval + velocity `v` are computed on the σ_v carrier (lines
364,379), but `denoised_r` and the full ancestral update (si/sigma_down/ratio/renoise_coeff) run
on the σ_a-axis `sig_row` (lines 380,384-393).

## DEFINITE via stock source comparison (2026-08-28)

`sample_euler_ancestral_RF` (comfy-ref/comfy/k_diffusion/sampling.py:247-265) computes
sigma_down/ratio/renoise_coeff from `sigmas[i]/sigmas[i+1]` = pure σ_v, applied uniformly to
audio AND video with zero AV branching. Audio is pre-scaled onto the σ_v frame before sampling
(model_base.py:2158-2159, model_sampling.py:344-347 audio_scale); the σ_a shift is applied
ENTIRELY inside `forward()` (entry carry model.py:534-538, per-row label 601-609, exit re-encode
547-551). Stock treats audio's outer integration identically to video; σ_a is purely an internal
model detail. Our sampler.py wrongly puts audio's ancestral terms on σ_a (:380,384-393).
Direct source-to-source comparison upgrades the diagnosis from "well-supported" to **DEFINITE**.

## Euler clean — live A/B (GPU, 2026-08-28)

Same graph, same seed, only sampler swapped euler_ancestral→euler: ringing VANISHES. Controlled
A/B on the actual ringing workflow (stronger than the 2026-08-23 standalone euler run). Rules out
ALL sampler-independent causes (ρ/Consequence 2, forward carry, process_latent_in) and localizes
the bug to exactly the velocity-recovery + ancestral-renoise lines euler does not execute.
Consequence 2's ρ-error runs every step — if ρ caused audible ringing, euler would show it too.
Euler is clean; ρ is present-but-imperceptible. **SUPERSEDES the 2026-08-27 reinterpretation that
named ρ as the hiss cause**; Consequence 2 is documented in audio-carry-identity.md as a
real-but-imperceptible second-order effect only.

## Renoise is the forward-echo engine

The ancestral renoise injection (sampler.py:395-397, eta>0 default 1.0) adds σ_a-scaled noise
into the σ_v packed trajectory every step, carried forward via the persistent seeded noise sampler
— producing ringing/echo feedback that propagates forward into FACT 1. `_euler_step`
(sampler.py:278-308) has NO renoise → no forward echo → euler clean.

## m-dependence

Residual `denoised_r = C + (σ_v−σ_a)(ε−C)` scales with axis gap `(σ_v − sig_row)`. Since
`sig_row = m·σ_a`, gap grows as m→0 ⇒ residual and ringing grow as m decreases = FACT 1's
free-vs-fractional asymmetry. The "mismatch is uniform across rows" refutation is WRONG.

## Earlier /sig_g GPU result still explained

Removing the residual via /sig_g strips the σ_v-axis correction from a σ_a-axis update; schedules
don't share shape (shift 12 vs 3), mismatch accumulates step-over-step → loud fluctuating noise
(FACT 2). `/carrier` is load-bearing; do NOT revert.

## σ_a is load-bearing for the label — Fix B REJECTED by proof (2026-08-28)

**VERDICT: σ_a is removable for the ancestral INTEGRATION but LOAD-BEARING for the LABEL.**
Fix B (full-unification: σ_v for BOTH label AND integration) is **REJECTED — by model-contract
proof, not assumption.**

**PROOF (independently verified from source + model, not circular):** our sampler passes the model
a fraction `w = sig_row/sig_g` (sampler.py:755,758); the model computes `t_row = 1 − w·σ_g`
(comment sampler.py:754; model `_forward` comfy model.py:604-605 `rows_t = 1 − m·σ_a`). For
audio, `sig_g = sig_a[i]` which EQUALS the model's own internally-derived σ_a (both derived from
time_shift_sigma(σ_v)). So current `w = sig_row_a/sig_a[i]` → model yields `1 − sig_row_a`
(truthful label). If the σ_v fraction were passed instead (`w = sig_row_v/sig_v[i]`), the model
still multiplies by ITS σ_a → `1 − sig_row_v·(σ_a/σ_v)`. Since time_shift_sigma is nonlinear
(σ_a/σ_v ≈ 0.27→1.0 across the schedule,
[dit-forward.md](native-h3-mechanism/dit-forward.md)), this MISLABELS audio, worst at
early/high-σ steps. The σ_a denominator is required for a truthful label.

**σ_a IS LOAD-BEARING IN THREE SITES** (only the ancestral integration is wrong):

1. Per-row label denominator `w = sig_row/sig_g` (sampler.py:557,755) — proven above; GPU-clean
   via euler A/B.
2. Observer-label K/V split — observer labels `t_obs = 1−m·σ` use shifted σ_a for audio
   ([label-ratio-and-observer-split.md](schedule-tail-late-delta/label-ratio-and-observer-split.md):95).
3. Deterministic r-scaling `r = (sig_row−sig_row_next)/(sig_g−sig_g_next)` in `_euler_step`
   (:305) and `_fallback_step` (:272) — GPU-validated clean on σ_a; unifying to σ_v would
   perturb a validated path (extra risk, zero benefit).

**PROVENANCE — deliberate design:** σ_a axis was intentional, introduced in commit 41c488d
("Ship schedule-tail remap + observer-label K/V split", msg: "complete the audio port: audio rows
run the remap on the sigma-shifted audio schedule via time_shift_sigma"). The schedule-tail remap
idx (`k_d`, `_stream_row_sigma` sampler.py:496-499) is axis-INDEPENDENT (same integer indexes for
dense_v and dense_a); the 17n+5 A/V tail join controls layout, not sigma axis — so σ_a affects
only sigma VALUES (w, r, ancestral terms), not tail alignment.

## Fix A — confirmed design (code change UNVERIFIED on GPU)

**Move ONLY `_euler_ancestral_rf_step` `denoised_r` (:380) and ancestral terms
si/sigma_down/ratio/renoise_coeff (:384-393) to the σ_v-axis per-row sigma.** `sv =
_stream_row_sigma(i, dense_v, sig_v)` at sampler.py:509 already exists but is DISCARDED for
audio by the `where` at :513. Feed `sv` (and its next) into those lines; leave label (:557) and
deterministic r-scaling (_euler_step :305, _fallback :272) on σ_a (euler-validated — do NOT
perturb). Fix B is now **REJECTED with proof** — not merely "unnecessary" — recomputing the label
on σ_v would mislabel audio worst at early/high-σ steps. **GUARANTEE GAINED:** audio at m=1
becomes bit-exact stock `sample_euler_ancestral_RF` (currently the docstring guarantee holds only
for video). Durable code → needs fail-then-pass regression test.

**Test order:** axis-split fix FIRST (alone). Predicted to resolve both FACT 1 and FACT 2 with no
ρ compensation. Pursue 1/ρ Consequence-2 compensation ONLY IF artifacts survive the axis fix;
validate on pure euler (where the axis fix is inert).
