<!-- provenance: bug+proof (Fix A VALIDATED free audio; H2 FALSIFIED as fade-length confound 2026-08-28;
     short-fade whistle RESOLVED = C2 ρ ancestral-amplified, GPU 0/0/49/73 2026-08-29 — canonical in audio-carry-identity.md;
     primary open bug = long-fade video interference, RCA in progress; σ_a-LABEL proof valid;
     PR #31 σ_v RE-EXTRACT UNMERGED (being closed) — designed + unit-tested on branches #26/#31; NOT shipped to main;
     σ_v axis is NECESSARY BUT INSUFFICIENT ALONE — superseded by the combined fix on branch fix-euler-ancestral-per-row-renoise) -->
<!-- verified: 2026-08-28 · Fix A: no-fractional-injects GPU A/B VALIDATED; H2 falsified by late fade-length GPU data (same date);
     euler-discriminator collapsed (tests differed in fade length, not sampler); σ_a-label proof from comfy-ref @b78cec87 -->
# Audio ancestral axis mismatch — Fix A VALIDATED for free audio; H2 FALSIFIED; primary long-fade video bug open

Consequence 3 of the [audio carry identity](audio-carry-identity.md).
Read when debugging `_euler_ancestral_rf_step`, Fix A, or fractional-region audio ([bugs.md](bugs.md)).

## PR #31 σ_v RE-EXTRACT — UNMERGED; NECESSARY BUT INSUFFICIENT ALONE (GPU 2026-09-01) — read first

⚠ **Merge-state correction (2026-09-01):** PR #31 was NEVER shipped to `main` and is being closed.
The σ_v-axis re-extraction was **designed + unit-tested on unmerged branches (#26, #31)** only;
`main` carries no σ_v-axis change. It is **superseded by the combined fix** on branch
`fix-euler-ancestral-per-row-renoise` — see [euler-ancestral-per-row-fix.md](euler-ancestral-per-row-fix.md).

PR #31 (`reextract-audio-ancestral-sigma-v-axis`) re-extracts the ancestral integration onto the
σ_v axis in `_euler_ancestral_rf_step` via new `sig_row_v`/`sig_row_v_next` (per-row x0 from the
global-carrier velocity projected onto σ_v). Real, unit-verified change — σ_a vs σ_v per-row
schedules genuinely differ (max diff ~0.32) — but a **no-op for every VIDEO row**
(`sig_row_v == sig_row`); it perturbs ONLY audio rows.

GPU (user, 2026-09-01; single-frame fractional injects): on the σ_v branch alone the audible audio
noise **PERSISTS** under `euler_ancestral`. So the σ_v axis is **NECESSARY BUT INSUFFICIENT ALONE** —
NOT the wrong cause. The earlier "chased the wrong cause / FALSIFIED as the audio cause" framing is
**corrected**: PR #31 failed on FRACTIONAL audio only because the clean-K/V splice bypass (Bug F)
still fed the ancestral step a contaminated denoised. Both halves are required together.

**Decisive discriminator (same branch/prompt, only sampler changed):** `euler` produces NEITHER the
audio noise NOR the video ghost; `euler_ancestral` produces BOTH. The two symptoms share the
`euler_ancestral` renoise path but need distinct fixes, combined in one change:

- Audio noise = σ_v-axis integration (Bug C, half 2) fed a CLEAN denoised (Bug F clean-K/V wiring,
  half 1). Neither half alone clears it — that is why the σ_v-only PR #31 branch still had noise.
  The stochastic-renoise reading ([bugs.md](bugs.md) Bug B) alone was too narrow; the axis IS
  load-bearing once the denoised is clean.
- Video ghost = the same clean-K/V wiring gap (clean-K/V not routed under euler_ancestral) —
  see [bugs.md](bugs.md) Bug F.
- Deterministic `euler` already routes through the clean-K/V splice and runs no renoise → clean.

Fix A below is unaffected (it fixed the m=1 free-audio renoise mis-scale, GPU-validated
2026-08-28); this note concerns the fractional-inject audio artifact only.

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

**Branch form (2026-08-29; UNMERGED):** Fix A is extracted standalone on branch
`fix-audio-ancestral-sigma-v-axis` (σ_v-axis ancestral integration only, via a new `row_sigma_v(i)`
helper + `sig_row_v` on `_StepContext`). The σ̃/`sig_row_c` carry-consistent layer (H2) was
DELIBERATELY DROPPED — its justifying discriminator is falsified (see H2 section) — so the branch is
Fix-A-only. 620 passed, ruff clean. NOT on `main`; folded into the combined fix
([euler-ancestral-per-row-fix.md](euler-ancestral-per-row-fix.md)).

**Scope: the σ_v axis applies to ALL per-row integration, not just `_euler_ancestral_rf_step`.** The
per-row multistep steps (PR3 `add-per-row-multistep-steps`) and the future DPM++ SDE spine (PR4)
carry the same dependency — their x0 recovery and integration must run on σ_v for audio too. See
[sampler-class-support.md](sampler-class-support.md) PR3 σ_v-axis coherence note.

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

**Short-fade ancestral whistle — RESOLVED (GPU 2026-08-29).** On the σ̃ branch, short
euler_ancestral+fade was heard as "clean-ish, maybe a whistle." A controlled discriminator (config
`0/0/49/73`, fade-out ramp=24 < Bug E's 51: `euler` CLEAN, `euler_ancestral` soft buzz for ~2.5s on
the fade-OUT ramp) resolves it: this is the **Consequence-2 ρ error, real and ancestral-amplified**
— audible on euler_ancestral fade ramps, silent on deterministic euler. This SUPERSEDES the earlier
"possibly prompt variation" framing. Canonical mechanism + fix paths:
[audio-carry-identity.md](audio-carry-identity.md). Separate from the primary video-primary bug.

**Primary open bug (see [bugs.md](bugs.md) Bug E and [long-fade-grid-beat.md](long-fade-grid-beat.md)):**
long-fade VIDEO-latent interference (moiré/streamers/electric), sampler-independent, present on
main, audio tracks via joint attention. Decoupling factorial DONE 2026-08-28 (single-variable
matrix on 0/0/39/90 + L=55 shift series): held-alone (M-E), pure-ramp (M-A), midpoint (M-D), and
trailing-free-alone (M-C) all REFUTED; M-B (held ≥ ~28 AND ramp ≥ 51) is the unique surviving model,
both terms independently necessary. See
[long-fade-grid-beat/ramp-length-decouple.md](long-fade-grid-beat/ramp-length-decouple.md).

## σ_a is load-bearing for the LABEL — proof STILL VALID (independent of everything above)

Carved out to [audio-axis-verdict/sigma-a-label-proof.md](audio-axis-verdict/sigma-a-label-proof.md)
(char/line budget). Short: **σ_a is removable for the ancestral INTEGRATION but LOAD-BEARING for the
model LABEL** — Fix B (σ_v for BOTH label and integration) stays REJECTED by model-contract proof.
σ_a is load-bearing in three sites (per-row label denominator, observer-label K/V split,
deterministic r-scaling). See the child for the full source proof and provenance.
