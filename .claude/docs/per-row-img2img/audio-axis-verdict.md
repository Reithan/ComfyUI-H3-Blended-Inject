<!-- provenance: bug+proof (free-audio ancestral axis fix VALIDATED, controlled GPU 2026-08-28; earlier falsification retracted; σ_a-load-bearing-for-LABEL proof valid; one separate fractional-audio Consequence-2 bug OPEN) -->
<!-- verified: 2026-08-28 · controlled GPU A/B (user, branch fix-audio-ancestral-axis-mismatch, no fractional injects, minimal graph);
     supersedes the 94b1597 falsification (that run used fractional injects → conflated two phenomena);
     σ_a-label proof from source comfy-ref @b78cec87; commit 41c488d; prior branch fix-audio-carrier-recovery @2483914 -->
# Audio ancestral axis mismatch — Fix A VALIDATED for free audio; one separate fractional bug remains

Consequence 3 of the [audio carry identity](audio-carry-identity.md).
Read when debugging `_euler_ancestral_rf_step`, Fix A, or fractional-region audio ([bugs.md](bugs.md)).

## VALIDATED for free audio (controlled GPU 2026-08-28) — read this first

A prior wiki commit (94b1597) marked this verdict FALSIFIED and called Fix A "not the cure." That
run was done WITH fractional injects, which muddied the signal by mixing two separate phenomena.
**The falsification is RETRACTED as premature.** New controlled GPU tests (user, 2026-08-28: same
prompt, NO fractional injects, minimal graph) isolate the variables and REINSTATE Fix A for the
free-audio case:

| setup | euler | euler_ancestral |
|---|---|---|
| STOCK KSampler (our node OUT), no injects | CLEAN | CLEAN |
| OUR node, `main`, free audio (1 guide, no injects → all rows m=1) | CLEAN | **TINNY/REVERB/NOISY** |
| OUR node, Fix A branch, same free-audio setup | CLEAN | **CLEAN** |
| OUR node, Fix A branch, WITH a video fade-in inject | CLEAN* | audio distortion + loud noise IN THE FADE REGION only |

\*euler behaviour in the fade-region case is TBD (A/B pending); the loud noise was observed under
the fade inject, localized to the fractional region — a SEPARATE bug (see below).

## Fix A — VALIDATED for the free-audio (m=1) ancestral case

On `main`, our `euler_ancestral` diverged from stock on FREE audio: tinny/reverb/noisy, while stock
KSampler ancestral was clean. **This is an OUR-NODE bug** (stock is clean). Fix A moves the
ancestral integration terms — `denoised_r` (:380) and si/sigma_down/ratio/renoise_coeff (:384-393)
— onto the σ_v-axis per-row sigma. With Fix A, free-audio `euler_ancestral` is **CLEAN and matches
stock**; audio at m=1 becomes bit-exact vs stock `sample_euler_ancestral_RF`; video byte-identical.

**Root cause (free-audio bug):** on main, audio rows computed the ancestral RENOISE terms on the
σ_a schedule (`sig_row`) while the packed audio actually lives on the σ_v trajectory → a mis-scaled
renoise was injected every step → accumulating tinny/reverb noise. `euler` runs NO renoise, so it
was clean both ways (main and branch). Fix A puts the renoise integration back on the σ_v axis the
packed audio really follows. **FIXED by Fix A.**

## The earlier "euler-clean discriminator" is VALID for free audio

Controlled result: our `euler` on free (m=1) audio is CLEAN, matching stock, on BOTH main and the
fix branch. So swapping euler_ancestral→euler DOES discriminate: it removes the loud ancestral
renoise component, and for free audio the remainder is clean. The 94b1597 "Bug C:
sampler-independent noise floor" framing is **WRONG and retracted** — there is no free-audio floor
under euler. The floor/noise the user heard under euler in the earlier session was in the
WITH-fractional-injects setup: a different, fractional-region phenomenon (next section).

## Remaining OPEN bug — fractional-region audio distortion (Consequence 2 shaped)

With a VIDEO fade-in inject, audio distorts + emits loud noise IN THE FADE / fractional region.
Mechanism (from `schedule.py:200-215` `RowSchedule.audio_denoise`): in `audio_mode="fade"` the
audio rows follow the video denoise envelope, so a video fade compresses AUDIO rows to fractional m
and they blend toward the packed clean S·A under the **[Consequence 2](audio-carry-identity.md) ρ
mis-scale** (ρ≠1 for 0<m<1). This is the long-known fractional-audio ρ mis-scale, now reproduced —
SEPARATE from the free-audio ancestral bug Fix A fixes.

**Still TBD (characterization in progress):** sampler-dependence (does deterministic euler also
distort in the fade region?) and the exact `audio_mode` used. A euler A/B + an audio_mode
confirmation are pending. Do NOT yet assert it is ancestral-specific OR fully sampler-independent.
Tracked as the open item in [bugs.md](bugs.md).

## σ_a is load-bearing for the LABEL — proof STILL VALID (independent of everything above)

This model-contract proof is independent of the axis fix and remains valid.

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

**σ_a IS LOAD-BEARING IN THREE SITES** (only the ancestral integration axis was ever questioned;
Fix A moves ONLY that integration, not the label):

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
