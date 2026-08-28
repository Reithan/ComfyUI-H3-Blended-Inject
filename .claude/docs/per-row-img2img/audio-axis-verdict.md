<!-- provenance: bug+proof (Fix A VALIDATED free audio; H2 carry-contract fractional-audio renoise ROOT CAUSE CONFIRMED by discriminator matrix 2026-08-28; fix designed, awaiting GPU) -->
<!-- verified: 2026-08-28 · controlled GPU A/B (user, branch fix-audio-ancestral-axis-mismatch);
     Fix A: no-fractional-injects run VALIDATED; H2 matrix: euler+fade=CLEAN isolates renoise as fault;
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
| OUR node, Fix A branch, WITH a video fade-in inject | CLEAN (confirmed) | audio distortion + loud noise IN THE FADE REGION only (ancestral only — see H2 section below) |

euler+fade is now CONFIRMED CLEAN (discriminator matrix, 2026-08-28). The loud noise under the
fade inject is ancestral-specific; root cause and fix design in the H2 section below.

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

## Fractional-region audio distortion — H2 carry-contract: ROOT CAUSE CONFIRMED, FIX DESIGNED

**Discriminator matrix** (controlled GPU, user, 2026-08-28, branch fix-audio-ancestral-axis-mismatch):

| setup | result |
|---|---|
| euler + fade (audio fractional 0<m<1) | CLEAN |
| euler_ancestral + drop (audio m=1 free) | CLEAN |
| euler_ancestral + keep (audio m=0 frozen) | CLEAN |
| euler_ancestral + fade (audio fractional 0<m<1) | AUDIO NOISY |
| video fractional under euler_ancestral | CLEAN |

**What the matrix proves:** euler+fade CLEAN rules out [Consequence 2](audio-carry-identity.md) (ρ
mis-scale) as the audible cause — euler exercises the identical σ_a init lerp and fractional clean
blend, and is clean. It also rules out an init-axis mismatch (euler uses the same init). The ONLY
difference between clean euler and noisy euler_ancestral is the ancestral RENOISE block
(sampler.py:395-408). The fault is renoise-specific.

**Root cause (H2, confirmed):** the H3 model applies a GLOBAL scalar carry = σ_a/σ_v to the audio
slice every step (comfy-ref/comfy/ldm/minimax/model.py:528-538; output :548-550). The sampler's
packed audio lives at scaled amplitude σ̃ = (σ_v/σ_a)·sig_row = w·σ_v. Fix A renoises on
sig_row_v = σ̃ correctly ONLY when carry=1: video (carry=1, exact) and m=1 (w→1, validated).
For fractional audio, sig_row = time_shift_sigma(sig_row_v) is nonlinear while carry ≠ 1, so
σ̃ ≠ sig_row_v; the ancestral renoise (sigma_down/ratio/renoise_coeff on sig_row_v at :395-404)
injects a mis-scaled noise magnitude every step → accumulates → audible noise. Masked at m=1
(σ̃≡sig_row_v) and at m=0 (renoise_coeff→0). This is the audio-vs-video renoise divergence the
user predicted: video renoise composes correctly (carry=1), audio's does not (carry≠1).

**Fix design (H2 fix, on top of Fix A):** in `_euler_ancestral_rf_step`, swap sig_row_v/
sig_row_v_next → carry-consistent σ̃/σ̃_next for ALL ancestral terms (denoised_r projection,
si, sip1, sigma_down, alpha, renoise_coeff, ratio). Per row: construct `sig_row_c = sig_row_v`
(video/m=1) or `w*sig_v[i]` (fractional audio, w=sig_row/sig_g.clamp(min=1e-8)).
Guarantees: video byte-identical, m=1 bit-exact (w=1→σ̃=sig_row_v), m=0/terminal unchanged.
The +1 term uses next-step globals (carry re-evaluated each model call). Leave global
noise_sampler args (sigmas[i], sigmas[i+1]) unchanged.

**Residual caveat:** σ̃ corrects renoise noise magnitude but NOT the packed clean-coefficient
mismatch (Consequence 2 ρ). ρ is NOT the audible fault (euler+fade is clean despite identical ρ).
If a faint residual survives the H2 fix on GPU, ρ-compensation is the next suspect.

**Status:** fix designed + being implemented on branch fix-audio-ancestral-axis-mismatch;
awaiting user GPU confirmation (task #77). Falsifiable prediction: euler_ancestral+fade audio
becomes CLEAN; drop(m=1)/keep(m=0)/video stay clean and m=1 stays bit-identical.
Tracked in [bugs.md](bugs.md).

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
