<!-- provenance: status (SUPERSEDED — route-1; knob-design synthesis + retraction + yardstick/lens/anchor; 2026-08-25) -->
<!-- verified: 2026-08-25 · HOLD-16–23 + user retraction + yardstick/lens/anchor decisions @proto-latent-hold-release -->
# The single-knob design: retraction + yardstick + anchor + open problem (SUPERSEDED)

Continues [resolution-invariance-goal](resolution-invariance-goal.md). HOLD-18 proved the high-m hold recipe
is res-invariant; this doc records how the single-d knob was designed.
Index: [index](index.md).

## RETRACTION: "hold/release-step = quality gate, NOT amount lever" (user, 2026-08-25)

The HOLD-16 conclusion that "release-step = quality gate, NOT an amount knob" is RETRACTED as unproven.
The hold=0.75 bad run is non-diagnostic: the inject frame was authored expecting ~0.4–0.6 denoise, so any
low-d approximation looks bad regardless of mechanism. Meanwhile hold=0.25 (frac 0.553) and hold=0.5
(frac 0.499) are BOTH clean at different frac — hold moves realized redraw; clean is a CANDIDATE amount lever.
Any downstream conclusion leaning on "hold = quality gate" is REOPENED. The hold=0.75 confound
(BOTH a later step AND lower frac) was not separable: "bad because tail-starved" vs "bad because sub-floor
amount" are not discriminated.

## Mechanism model (HYPOTHESIS)

- Blend quality is governed by sigma_sw at release (neighbor maturity). Back-loaded schedule keeps neighbors
  at sigma≈0.975 through step ~10, so hold≤~0.5 releases while neighbors are still ≈full-noise; they
  co-denoise together. hold≥0.75 releases after sigma has dropped (neighbors committing) — abrupt blend.
- Amount is governed by the renoise LEVEL (= m·sigma_sw today). At a fixed early release m sets amount.
- Res-stability comes from m≈1 = native denoise, NOT from hold. m<1 compresses the schedule via label+correction.

## Knob design conclusion (after Fable review): Design B dissolved

**There is no clean third lever; `m` at a fixed early release IS the amount knob.** Fable's algebra shows
decoupled level L is either redundant at early release or forces the bad late-release regime.

- **User knob:** `keyframe_denoise` d in (0,1]; m=d internally.
- **Internal:** release EARLY via sigma-threshold (latest step with sigma_sw>=~0.95; schedule-robust).
- **Ship landmines:** (1) quantize `ceil(m·256)/256` collapses d>~0.996 to 1.0, disarming hold — cap m<=0.99;
  (2) d=1.0 semantics = hold + full renoise; keep arming provenance-aware, not the non-fractional path.

## YARDSTICK for `d` (user decision, 2026-08-25)

`d` is PERCEPTUAL/SEMANTIC: "how completely is the inject frame blended and/or redrawn into the timeline?"
It is NOT a numeric property and is NOT required to match SDXL's numeric denoise (H3 has timeline
temporal-contagion; SDXL single-image denoise does not). `frac` (realized-redraw magnitude) stays an INTERNAL
secondary readout — a magnitude, not a quality and not `d`.

## READING LENS: failure-mode split (user's guess, NOT data, 2026-08-25)

Two failure modes:
- **"poor/low denoise"** = structure PRESERVED, just under-redrawn. A clean-but-under-redrawn result is the
  wrong `d`; the mechanism is not broken.
- **"smudgey"** = structure CHANGED but INCORRECTLY. This is a mechanism failure, not a tuning problem.

## CURRENT BEST ANCHOR + the open frontier (2026-08-25)

**Best anchor:** hold=0.5, m=0.99 (~1), plus or minus `hold_prenoise_step0` — reads ~0.5 denoise.
CONFIRMED at a THIRD resolution: 0.2MP (HOLD-23) joins 0.5MP and 1.0MP. frac OFF 0.379 / ON 0.608; both
read ~0.5. **The d=0.5 perceptual point is res-robust across all 3 tested resolutions.**

**`frac` is DECOUPLED from perceived-d.** Three 0.2MP runs spanning frac 0.273→0.379→0.608 all read ~0.5
(user placed them inside a 0.4–0.6 visual band). Frac magnitude is not the yardstick; perceived-d is.
Prenoise grows the frac-kick as res drops (+0.23 @0.2MP vs +0.09 @0.5MP) but leaves perceived-d UNCHANGED.

## No-hold ruler + no-hold ≈ hold=0.01 (HOLD-22)

At 0.2MP no hold machinery is needed — **m IS the knob (d≈m)**. TRUE no-hold m=0.5: frac 0.273 (reads ~0.5).
The earlier hold=0.01 proxy gave 0.272 — a dead heat. Mechanistic reason: both put the anchor into the tail at
the same magnitude (init-lerp |x0|=0.5362 ≈ m·sigma_sw=0.4988). A prior prediction that the two would DIVERGE
is FALSIFIED.

**Open frontier (low-d at high res):** The supported band is only d=0.5 at 1MP. Routes tried produced smudgey
output (hold=0.75 confounded; m=0.8@1MP HOLD-21). Leading untested lever: SPLIT m_release from the tail m so
the tail keeps m≈1 (on-manifold correction) while lowering only the release level L. Distinct from the
dissolved Design-B "decoupled L" (that lowered L at a fixed early release with a single m); here the split
preserves the m≈1 correction.

## Caveats

- **frac ≠ d:** full-strength redraw tops out at frac≈0.55 (conditioning keeps regen correlated with clean).
- **d_content floor is UNPROVEN:** only sub-floor point (frac 0.237) is triple-confounded; no early-release
  low-frac run exists. If real, document as img2img semantics; do NOT silently over-redraw.
