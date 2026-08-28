<!-- provenance: status (SUPERSEDED — route-1; design goal + decisive cross-resolution test; intra-res sweeps do NOT settle this) -->
<!-- verified: 2026-08-25 · user reframing + prior 1MP window-closed (the-real-bug) + all-hold-runs-0.5MP -->

> **North star (this prototype drive):** ONE user-facing knob that maps visually to a normal img2img denoise `d`,
> produced by modulating `hold` and `m` — **resolution-invariant in general (IDEAL)**, or via a
> **resolution-aware internal mapping `lever = f(d, res)` (ACCEPTABLE fallback)**. The user sets ONE value and
> must NOT re-tune per resolution. Every input, mechanism, formula, and output in this prototype drive exists to
> serve that mapping; anything that does not is out of scope for the drive.

# The real axis: ONE resolution-invariant setting (not a per-res retune)

Continues [held-keyframe-m-vs-sdedit](held-keyframe-m-vs-sdedit.md). The user (2026-08-25) re-centered the
goal, and it exposes a confound in that doc's sweep conclusions. Index: [index](index.md).

## The goal (governs the design)
See north-star above. Clarification: internal `lever = f(d, res)` compensation is allowed (runtime resolution
is known, so res-awareness is free); what must be single + intuitive is the USER knob.

## The confound in the sweep conclusions (concede)
`held-keyframe-m-vs-sdedit`'s "re-noise LEVEL = amount knob, release-STEP = quality gate" is an
**intra-resolution** finding — every run was **0.5MP**. It does NOT show any setting is res-invariant. The
known counter-fact: **m=0.5/no-hold is bad at 0.5MP and WORSE at 1MP** (res-dependent). So the two-axis split
holds *within* a resolution but is silent on the axis that matters.

## What's settled vs open (for res-invariance)
- **m-only: DEAD across res.** Nominal m res-compresses (realized ≪ nominal @1MP); does not converge to a
  working value in 0.5–1.0MP. And [the-real-bug](../highres-underdenoise-model/the-real-bug.md) found the
  **1MP window CLOSED without hold** — no single d hits content AND blend. (This is *why* hold/route-1 exists.)
- **hold-only: res-behavior UNTESTED.** Every HOLD-* run is 0.5MP. We do not know if a hold recipe survives
  to 1MP. Also, an intra-res limitation already shows hold-only is a poor *amount* knob: reducing amount via
  hold means releasing LATER, which starves the tail → quality drops (HOLD-16 hold=0.75 = BAD). So "hold-only
  single knob" has a viability question even before resolution.
- **combined `m`+`hold` (res-aware): the fallback.** More complex; only justified if neither single lever is
  viable. Cannot rank the three until we have ONE cross-resolution data point.

## DECISIVE test — run the good hold recipe at 1MP — RAN 2026-08-25, PASSED
Ran the good recipe at 1.0MP, proxy at both resolutions (HOLD-18):
- **hold=0.25, m=0.99:** frac **0.551 @0.5MP → 0.553 @1MP** (Δ 0.002). Clean + well-blended at both.
- **hold=0.5, m=0.99:** ~0.50 visual @0.5MP → frac **0.499 @1MP**. Clean + well-blended at both.

⇒ The high-m hold recipe's realized redraw is **resolution-invariant across 0.5↔1.0MP** (a 2× area jump moved
frac by 0.002). **Hold REOPENS the 1MP window** that was closed for m-only. The route is VIABLE.

**Hypothesis for WHY (UNVERIFIED):** the res-invariance is a property of **m≈1 = NATIVE denoise**, not of hold
per se. At m≈0.99 the release re-noises the anchor to near-full (level 0.97–0.98) and the tail runs at
near-native labels (t_row≈σ) + near-native correction ⇒ standard SDEdit from ~full noise = the model's own
denoise = res-stable. The m-COMPRESSION (m<1 shrinking the schedule via label+correction) is the res-unstable
part (the m=0.5 case). Hold's job is the BLEND (neighbors compose around clean), NOT the res-stability.

**What this predicts + the next test:** if res-invariance = m≈1/native, the LOWER amounts are the hard part —
they need either m<1 (res-unstable) or late release (quality-killer). So the open single-knob question is
whether the AMOUNT can move res-invariantly. **DISCRIMINATOR (unrun): re-run the m-sweep (m=0.8, 0.9 at
hold=0.5) at 1MP**, compare frac to 0.5MP (0.388 / 0.442):
- frac@1MP ≈ frac@0.5MP ⇒ the high-m band [0.8–0.99] at fixed early hold is a **res-invariant amount knob**
  over frac≈[0.39–0.55] → a single m-knob (fixed early hold) is viable for that range.
- frac@1MP < frac@0.5MP ⇒ m res-compresses even in the high band → the res-invariance was the near-full/native
  case only → a single m-knob won't cover a range; combined/other needed.
