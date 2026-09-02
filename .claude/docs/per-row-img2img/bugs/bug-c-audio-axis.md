<!-- provenance: bug (OPEN — plant-axis FALSIFIED for fade-audio GPU 2026-09-01; content-axis fix SHIPPED PR #32 revised commits 4644fcf+e4a9940, GPU PARTIAL 2026-09-01; C2 durable port DECIDED, GPU pending) -->
<!-- verified: 2026-09-01 GPU falsification of plant-axis; content-axis GPU PARTIAL; Fix A m=1 free audio GPU-CONFIRMED 2026-08-28 -->
# Bug C — Audio observer axis defect (euler_ancestral, fade-region hiss)

Full record for Bug C (audio axis defect) and Bug C-remaining (H2 rejection). This is the
fade-audio hiss under `euler_ancestral` fractional ramps. Read with
[../euler-ancestral-per-row-fix/content-axis.md](../euler-ancestral-per-row-fix/content-axis.md)
for the current fix design.

## Bug C — current state (2026-09-01)

**Free-audio euler_ancestral distortion — our-node observer content-axis bug.**
Content-axis fix SHIPPED PR #32 revised commits 4644fcf+e4a9940 (GPU PARTIAL 2026-09-01 —
ring narrowed mid-m; both ends clean; hiss shorter; euler UNCHANGED).
Plant-axis fix FALSIFIED for fade-audio (GPU 2026-09-01, single-frame clean / fade hiss persists).
C2 durable port GPU CONFIRMED (user 2026-09-01, PR #32; residual localized: low-m band ≈0.75–1.0 s;
anchor fix implemented fe0343a+91078cc, GPU pending) — see
[../euler-ancestral-per-row-fix/audio-anchor-scale.md](../euler-ancestral-per-row-fix/audio-anchor-scale.md).

**Co-location note (2026-09-01):** Bug C is TIMELINE-WIDE — all audio rows are m=1 in drop mode
(`audio_denoise=1.0`); `sig_a≠carrier` applies uniformly, not just to inject-local rows. So
"single-frame drop-mode audio noise = Bug C" was a mis-attribution. The inject-local co-located
noise was Bug F (joint-attention coupling). Bug C stays REAL — it is just timeline-wide.

⚠ **Merge-state correction (2026-09-01):** NO σ_v-axis change is on `main`.
`_euler_ancestral_rf_step` (sampler.py:412) is untouched since PR #16 and calls `ctx.model(...)`
directly at sampler.py:467; `sig_row_v` does not exist in main's sampler.py. Fix A / the σ_v axis
was designed and GPU-validated for free audio on unmerged branches (`fix-audio-ancestral-sigma-v-axis`,
PR #26, PR #31) but never landed. Bug C is therefore **OPEN on main**.

A prior wiki commit (94b1597) called this a persistent, sampler-INDEPENDENT noise floor present
even under deterministic euler on free (m=1) audio, and marked the axis verdict FALSIFIED. That run
used FRACTIONAL injects, conflating two phenomena. **Retracted.** Controlled GPU A/B (user,
2026-08-28: same prompt, NO fractional injects, minimal graph) shows:

- STOCK KSampler (our node OUT): euler CLEAN, euler_ancestral CLEAN.
- OUR node, `main`, free audio (m=1): euler CLEAN, euler_ancestral TINNY/REVERB/NOISY.
- OUR node, Fix A branch, free audio: euler CLEAN, euler_ancestral **CLEAN**.

So free-audio `euler` is CLEAN (no noise floor); the distortion was an OUR-NODE `euler_ancestral`
bug.

**Root cause (chain, 2026-09-01):** on main, audio rows computed ancestral RENOISE terms on the σ_a
schedule while the packed audio lives on the σ_v trajectory → mis-scaled renoise injected every
step → accumulating tinny/reverb noise.

Post-plant-fix (GPU falsified 2026-09-01), the residual hiss reveals a deeper PER-STEP defect: the
observer side-stream K/V CONTENT was primed on the σ_a axis (ratio = m·σ_a on σ_a grid) while the
audio content post-Fix-A sits on the σ_v axis. **Fix: `_audio_observer_ratio` maps content via
`shift⁻¹(m·σ_a)` on σ_v (Möbius inverse).** σ_a stays load-bearing for the LABEL only.

Full design: [../euler-ancestral-per-row-fix/content-axis.md](../euler-ancestral-per-row-fix/content-axis.md).
Full axis verdict: [../audio-axis-verdict.md](../audio-axis-verdict.md).

## Bug C-remaining — H2 REJECTED (fade-length confound)

**What H2 was:** euler+fade=CLEAN vs euler_ancestral+fade=NOISY was read as a sampler discriminator
pointing to the ancestral-renoise mis-scale (σ̃ ≠ sig_row_v for fractional audio) as root cause.

**Why H2 is REJECTED:** new controlled data (user, 2026-08-28 late) shows the original test pair
used different fade lengths (~30f clean vs ~60f noisy). Fade length, not sampler, was the
independent variable. euler+LONG fade also artifacts. The discriminator is spurious; H2 is not
the root cause.

**σ̃ implementation status:** video-byte-identical and m=1 bit-exact (harmless), but UNVALIDATED
as a fix for the audible artifact. Do not present it as the fix. The mergeable branch
`fix-audio-ancestral-sigma-v-axis` is Fix-A-only — σ̃/`sig_row_c` is dropped, not shipped.

**The primary open bug is Bug E** (see [../bugs.md](../bugs.md)). Consequence-2 ρ status: REAL and
ancestral-amplified — audible on `euler_ancestral` fade ramps, silent on deterministic `euler`
(GPU discriminator `0/0/49/73`, 2026-08-29). Canonical mechanism + fix paths:
[../audio-carry-identity.md](../audio-carry-identity.md).
