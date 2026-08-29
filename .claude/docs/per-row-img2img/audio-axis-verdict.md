<!-- provenance: bug+proof (Fix A VALIDATED free audio; H2 FALSIFIED as fade-length confound 2026-08-28; primary open bug = long-fade video interference, RCA in progress; σ_a-LABEL proof valid) -->
<!-- verified: 2026-08-28 · Fix A: no-fractional-injects GPU A/B VALIDATED; H2 falsified by late fade-length GPU data (same date);
     euler-discriminator collapsed (tests differed in fade length, not sampler); σ_a-label proof from comfy-ref @b78cec87 -->
# Audio ancestral axis mismatch — Fix A VALIDATED for free audio; H2 FALSIFIED; primary long-fade video bug open

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

⚠ euler+fade=CLEAN and euler_ancestral+fade=NOISY are NO LONGER a valid sampler discriminator —
those tests used different fade lengths (~30f vs ~60f). See the H2 section below for the
falsification and the primary open bug (long-fade video interference, sampler-independent).

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

**Shipped form (2026-08-29):** Fix A is extracted standalone on branch
`fix-audio-ancestral-sigma-v-axis` (σ_v-axis ancestral integration only, via a new `row_sigma_v(i)`
helper + `sig_row_v` on `_StepContext`). The σ̃/`sig_row_c` carry-consistent layer (H2) was
DELIBERATELY DROPPED — its justifying discriminator is falsified (see H2 section) — so the
mergeable branch is Fix-A-only. 620 passed, ruff clean.

## The earlier "euler-clean discriminator" is VALID for free audio

Controlled result: our `euler` on free (m=1) audio is CLEAN, matching stock, on BOTH main and the
fix branch. So swapping euler_ancestral→euler DOES discriminate: it removes the loud ancestral
renoise component, and for free audio the remainder is clean. The 94b1597 "Bug C:
sampler-independent noise floor" framing is **WRONG and retracted** — there is no free-audio floor
under euler. The floor/noise the user heard under euler in the earlier session was in the
WITH-fractional-injects setup: a different, fractional-region phenomenon (next section).

## Fractional-region audio distortion — H2 FALSIFIED (fade-length confound); primary bug is video-primary, open

**The original discriminator** (earlier this branch) showed euler+fade=CLEAN vs euler_ancestral+fade=NOISY.
⚠ **This discriminator has COLLAPSED.** New controlled data (user, 2026-08-28 late) reveals those
tests used different fade lengths (~30f clean vs ~60f noisy). Fade length, not sampler, was the
real independent variable.

**What the new controlled data shows:**
- Short fade (~30f): CLEAN under both euler and euler_ancestral.
- Long fade (~60f): VIDEO-latent INTERFERENCE — moiré, jagged streamers/ribbons/electric patterns —
  under BOTH euler AND euler_ancestral (sampler-independent).
- Audio tracks the visual interference via A/V joint-attention coupling (SFX matches visual texture).
  The artifact is VIDEO-PRIMARY; audio is a secondary symptom, not an independent audio bug.
- Present on main BEFORE either fix: rollback to Fix A only (d70d1767) AND to pre-both-fixes
  both still show the long-fade artifact. Our σ̃ fix and Fix A neither cause nor fix it.

**H2 carry-contract renoise hypothesis — REJECTED as cause of the primary artifact.**
H2 targeted the ancestral-renoise mis-scale (σ̃ ≠ sig_row_v for fractional audio). The evidence
that isolated it conflated short-fade with euler vs long-fade with euler_ancestral. The σ̃
implementation is video-byte-identical and m=1 bit-exact (harmless), but its justifying
discriminator is falsified. Do NOT present it as a validated fix for the audible artifact.

**Minor open thread (AMBIGUOUS):** on the σ̃ branch, short euler_ancestral+fade was heard as
"clean-ish, maybe a whistle" — possibly the Consequence-2 ρ residual, possibly prompt variation.
Status: AMBIGUOUS, not validated. Separate minor thread; do not conflate with the primary bug.

**Primary open bug (see [bugs.md](bugs.md) Bug E and [long-fade-grid-beat.md](long-fade-grid-beat.md)):**
long-fade VIDEO-latent interference (moiré/streamers/electric), sampler-independent, present on
main, audio tracks via joint attention. Decoupling factorial DONE 2026-08-28 (single-variable
matrix on 0/0/39/90 + L=55 shift series): held-alone (M-E), pure-ramp (M-A), midpoint (M-D), and
trailing-free-alone (M-C) all REFUTED; M-B (held ≥ ~28 AND ramp ≥ 51) is the unique surviving model,
both terms independently necessary. See
[long-fade-grid-beat/ramp-length-decouple.md](long-fade-grid-beat/ramp-length-decouple.md).

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
