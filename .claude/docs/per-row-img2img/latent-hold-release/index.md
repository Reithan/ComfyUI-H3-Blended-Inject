<!-- provenance: status (SUPERSEDED — route-1 latent hold-and-release, branch proto-latent-hold-release; replaced by 3-lever arch) -->
<!-- verified: 2026-08-27 · HOLD-27 GPU run; route superseded by 3-lever arch GPU-CONFIRMED 2026-08-23 -->
# Route-1 latent hold-and-release — historical record (index)

Prototype branch `proto-latent-hold-release` tested whether a latent-resident anchor-and-release
could give a mid-timeline keyframe intuitive img2img denoise with neighbor blend (no ghost, no pop).
**SUPERSEDED** by the 3-lever architecture (init-noise lerp + fractional denoise_mask conditioning +
REQUIRED denoised correction + noise_mask=None), GPU-CONFIRMED 2026-08-23. This cluster is a
historical record of GPU experiments and failure modes.

Design goal: ONE user-facing knob `d` (via `hold`+`m`), resolution-invariant or res-aware internally.
Parent design: [crux-and-mechanism-2](../highres-underdenoise-model/crux-and-mechanism-2.md).
Open paths: [status-and-open-paths](../status-and-open-paths.md) path 1.

## Key GPU findings (chronological)

**Findings 1–3:** `latent_hold_frac` (step-fraction) is the wrong unit — H3's back-loaded schedule makes
60% of step-space only ~7% of sigma-space. Hold-residency confirmed. Two early reasoning errors retracted:
composition is NOT late-only; latent CAN attract.
See [mechanism-and-early-findings](mechanism-and-early-findings.md).

**Findings 4–6:** Attraction is baseline to co-evolving per-row injects; hold not required.
Frozen m=0 (denoise=0.0) also attracts. The Fable +0.5 envelope trap applies only to linear/eased
interpolation, not the user's `none` setting.
See [attraction-and-envelope](attraction-and-envelope.md).

**Findings 7–11 (central bug): Provenance-blind confound.**
`anchor_mask=(m>0)&(m<1)` froze the opening video inject's fade-out ramp instead of the r40/r60
keyframes. The frozen wrong rows propagated via H3 global attention and corrupted r40's blend.
Fix: hold only single-frame injects (`source_length==1`) at fractional denoise.
GPU-CONFIRMED: armed count dropped ~1.08M to 97,920; r40 attracts/blends correctly.
See [hold-mechanism-and-confounds](hold-mechanism-and-confounds.md).

**Finding 12:** After the provenance fix, r40 under-denoises. `m·denoised+(1−m)·inp` caps redraw at m;
the tail is ordinary per-row img2img. Direct lever: raise `min_denoise`.
See [anchor-denoise-after-clean-fix](anchor-denoise-after-clean-fix.md).

**Findings 13–14:** m=0.99 probe shows neighbor temporal attention strongly shapes anchor's result
(contagion AFFECTS amount, does not set it). Higher m gave better-looking r40. Leading explanation:
res-compression. Still-repeat-with-fade: fixes seam but anchors artifacts and reads as freeze-frame; not viable.
See [anchor-denoise-m-vs-res](anchor-denoise-m-vs-res.md).

**HOLD-15:** High-m generalizes to r60; not position-specific.

**HOLD-16/17:** Sweep: `frac` tracks re-noise level monotonically; m in {0.8,0.9,0.99} all clean at early
release. "Release-step = quality gate, NOT amount knob" RETRACTED (user, 2026-08-25): hold=0.75 was
non-diagnostic. Reversal: keep release EARLY, leave m variable.
See [held-keyframe-m-vs-sdedit](held-keyframe-m-vs-sdedit.md).

**HOLD-18 (decisive res test): Route VIABLE.** hold=0.25/m=0.99 at 1MP: frac 0.551→0.553 (delta 0.002).
Hold reopens the 1MP window closed for m-only. Res-invariance at high-m is the supported band.
See [resolution-invariance-goal](resolution-invariance-goal.md).

**HOLD-19:** m NOT res-invariant below m=0.99 — m=0.8@1MP frac 0.362 (vs 0.388 @0.5MP), keyframe smears.
Amount-floor CONFIRMED at ~frac 0.39 @1MP. Trilemma: clean keyframe + good blend jointly reachable only
at near-full amount (m≈1). See [amount-floor-and-step0-redesign](amount-floor-and-step0-redesign.md).

**HOLD-20:** Prenoise toggle built. frac +0.09–0.14 vs clean-hold; direct evidence neighbors shape anchor's
realized redraw via contagion. Good high-m configs stay clean.

**HOLD-21:** Step[0] redesign FALSIFIED as smear-fix. frac delta +0.01 at 1MP m=0.8 (essentially flat). The
smear is the anchor's own low-m partial denoise, not neighbor-target mismatch.

**HOLD-22/23 (0.2MP):** hold=0.5/m≈0.99 reads ~0.5 denoise at a third resolution. d=0.5 perceptual point
res-robust across 0.2/0.5/1.0MP. `frac` DECOUPLED from perceived-d: 0.273→0.608 all read ~0.5.
No-hold ruler frac 0.273 ≈ hold=0.01 proxy frac 0.272 (dead heat; prior divergence prediction FALSIFIED).
See [knob-design-open-questions](knob-design-open-questions.md).

**HOLD-24:** Per-frame scheduled release with schedule-sigma pin — output well-blended locally but
STRUCTURALLY INCOHERENT (camera moves, morphing). Cause: inverted influence ordering; strongest anchors
hidden during early structure-setting window. FALSIFIED on structural coherence.

**HOLD-25 (GPU-CONFIRMED):** "Neighbors see release" fix: pin each held row at its own release sigma
`L = sigmas[k_row]` from step 0. Structural incoherence resolved; blend and denoise solid.
See [per-frame-scheduled-release](per-frame-scheduled-release.md).

**HOLD-26 design:** Min-free-steps floor decouples release LEVEL from TIMING. Implemented 074e443.

**HOLD-27 (2026-08-27): GPU-FALSIFIED.** All 5 runs over-denoise, including previously-accurate Option-1.
Two failures: (1) per_frame path has no denoised correction — free-step count governs redraw, not level pin;
(2) floor is provenance-blind (hits opening fade-out ramp; same bug class as Findings 7–11).
See [min-free-steps-floor](min-free-steps-floor.md).

## Knob design — final state before supersession

`d` is perceptual: "how completely is the inject redrawn." `m=d` internally. Release EARLY via
sigma-threshold (sigma_sw >= ~0.95). Design B (decoupled level L) dissolved. Clean res-invariant output
available only in the high band (m≈0.9–0.99, frac ~0.4–0.55). Low-d partial redraws smear at high res.

## Child docs

- [mechanism-and-early-findings](mechanism-and-early-findings.md) — Findings 1–3: build + early bugs
- [attraction-and-envelope](attraction-and-envelope.md) — Findings 4–6: attraction A/B + envelope decode
- [hold-mechanism-and-confounds](hold-mechanism-and-confounds.md) — Findings 7–11: provenance-blind confound + GPU fix
- [anchor-denoise-after-clean-fix](anchor-denoise-after-clean-fix.md) — Finding 12: under-denoise post-fix
- [anchor-denoise-m-vs-res](anchor-denoise-m-vs-res.md) — Findings 13–14: m probe + res/attention mechanism dispute
- [resolution-invariance-goal](resolution-invariance-goal.md) — HOLD-18: res-invariance at high-m; decisive test result
- [held-keyframe-m-vs-sdedit](held-keyframe-m-vs-sdedit.md) — HOLD-15/16/17: sweep data + retraction trail
- [knob-design-open-questions](knob-design-open-questions.md) — Design decisions: yardstick, best anchor, open frontier
- [amount-floor-and-step0-redesign](amount-floor-and-step0-redesign.md) — HOLD-19/20/21: floor confirmed + prenoise FALSIFIED
- [per-frame-scheduled-release](per-frame-scheduled-release.md) — HOLD-24 FALSIFIED / HOLD-25 CONFIRMED: per-frame mechanism
- [min-free-steps-floor](min-free-steps-floor.md) — HOLD-26 design + HOLD-27 GPU-FALSIFIED: floor failure analysis
