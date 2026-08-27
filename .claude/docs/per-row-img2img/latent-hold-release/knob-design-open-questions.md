<!-- provenance: status (knob-design synthesis + retraction + yardstick/lens/anchor; 2026-08-25) -->
<!-- verified: 2026-08-25 · HOLD-16–23 + user retraction + yardstick/lens/anchor decisions @proto-latent-hold-release -->
# The single-knob design: retraction + yardstick + anchor + open problem

Continues [resolution-invariance-goal](resolution-invariance-goal.md). HOLD-18 proved the high-m hold recipe is
res-invariant (1MP window reopened); now: how do we turn it into ONE intuitive img2img-`d` knob? Index: [index](index.md).

## RETRACTION — "hold/release-step = quality gate, NOT amount lever" (user, 2026-08-25)
The conclusion from HOLD-16 that "release-step = quality gate, NOT an amount knob" is RETRACTED as unproven.
The hold=0.75 bad run is NON-DIAGNOSTIC: the inject frame was authored expecting ~0.4–0.6 denoise, so any
low-`d` approximation looks bad even if the mechanism worked perfectly. Meanwhile hold=0.25 (frac 0.553) and
hold=0.5 (frac 0.499) are BOTH clean at different frac → **hold moves realized redraw while clean = a CANDIDATE
clean amount lever** (opposite of the retracted claim). Any downstream conclusion leaning on "hold = quality
gate" (e.g. old Q2 "release early, set amount via m") is REOPENED.

The confound still stands: hold=0.75 has BOTH a later step AND lower frac (0.237). "Bad because tail-starved"
is not separable from "bad because sub-floor amount on a rough source frame." Neither is confirmed.
Discriminator (queued): m=0.7 at early release (hold=0.5), BOTH res — see queued discriminator below.

## Mechanism model that frames the answers (HYPOTHESIS)
- **Blend quality is governed by σ_sw at release** (= neighbor maturity). Back-loaded schedule keeps neighbors
  at σ≈0.975 through step ~10, so hold≤~0.5 releases the anchor while neighbors are still ≈full-noise → they
  co-denoise together → good blend. hold≥0.75 releases after σ has dropped (neighbors committing) → abrupt.
- **Amount is governed by the renoise LEVEL** (= m·σ_sw today). At a fixed early release (σ_sw≈0.97) m sets it.
- **Res-stability comes from m≈1 = native denoise**, NOT from hold. m<1 compresses the schedule (label m·σ +
  correction keeps (1−m)·inp) = off-manifold + res-unstable.

## The 3 open questions — working answer (HYPOTHESIS; 1MP m-sweep + Fable pending)
**Q1 — vary step[0] or step[hold] noise differently?** step[0]: keep anchor CLEAN (the hold's whole purpose is
neighbors composing around a clean anchor). step[hold]: candidate change = **decouple the renoise level from
m·σ_sw into a first-class knob L**, released at the schedule step where σ≈L so the label matches the noise
(on-manifold). Only needed if m can't be the res-invariant amount knob (see Q3).
**Q2 — release@hold vs m / res / const?** Keep release **EARLY and ~constant** (σ_sw high, neighbors immature)
— it's the BLEND control, not the amount knob. Independent of m (coupling them is the confound above). Possible
WEAK res-dependence (is neighbor-maturity-vs-σ res-invariant? untested) → allow a small res-aware nudge if HOLD
runs at 1MP show the blend needs it; default = const.
**Q3 — mask m vs res / hold?** Keep m INDEPENDENT of hold. Res: the 1MP m-sweep decides — if m={0.8,0.9} stay
res-invariant, **m IS the single knob** (or a light m_eff=f(d,res) map), hold fixed early = the dream. If m
res-compresses, m won't cover a range → pin m=1 (native) and move amount to the decoupled level-L knob (Q1),
accepting the combined/res-aware fallback.

## Knob design conclusion (after Fable review) — Design B dissolved
**There is no clean third lever; `m` at a fixed early release IS the amount knob.** (Design B conceded:
Fable's algebra shows decoupled level L is either redundant at early release or forces the bad late-release
regime.)
- **User knob:** `keyframe_denoise` d∈(0,1]; m=d internally.
- **Internal:** release EARLY via σ-threshold "latest step with σ_sw≥~0.95" (schedule-robust).
- **Ship landmines:** (1) quantize `ceil(m·256)/256` collapses d>~0.996 → DISARMS hold → cap m≤0.99;
  (2) d=1.0 semantics = hold+full renoise; keep arming provenance-aware, not the non-fractional path.

## YARDSTICK for `d` — perceptual/semantic (user decision, 2026-08-25)
`d` is PERCEPTUAL/SEMANTIC: "how completely is the inject frame blended and/or redrawn into the timeline
where it was injected?" It is NOT a numeric property and is NOT required to match SDXL's numeric denoise
(H3 has timeline temporal-contagion; SDXL single-image denoise does not). Instrumenting an SDXL-denoise
reference (KSampler-clone stats) was CONSIDERED and REJECTED. `frac` (realized-redraw magnitude) stays an
INTERNAL secondary readout — a magnitude, not a quality and not `d` — never the target/yardstick.

## READING LENS — failure-mode split (user's guess, NOT data, 2026-08-25)
Distinguish two failure modes when judging a run:
- (a) **"poor/low denoise"** = structure PRESERVED, just under-redrawn = working-but-wrong-`d`.
  A clean-but-under-redrawn result is the wrong `d`; the mechanism is not broken.
- (b) **"smudgey"** = structure CHANGED but INCORRECTLY = BROKEN/contaminated denoise = mechanism failure.
Apply this split to every run. A "smudgey" result is a mechanism failure; "clean-but-low" is a tuning problem.

## CURRENT BEST ANCHOR + open problem (2026-08-25)
**Best anchor:** hold=0.5, m=0.99 (~1), ±`hold_prenoise_step0` → "looks like ~0.5 denoise". Now CONFIRMED at a
THIRD resolution: 0.2MP (HOLD-23) joins 0.5MP + 1.0MP. At 0.2MP prenoise OFF frac **0.379**, ON frac **0.608**
(level 0.9674 = m·σ_sw ⇒ m≈0.992; the "0.5m" raw-log label was a user typo — `m_release=m_packed` at nodes.py:282
pins m). BOTH read ~0.5 denoise. **The d=0.5 perceptual point is res-robust across all 3 tested resolutions.**

**Reinforced — `frac` is DECOUPLED from perceived-`d`.** Three 0.2MP runs spanning frac 0.273→0.379→0.608 ALL
read ~0.5 (user placed them inside a 0.4–0.6 visual band). Frac magnitude is not the yardstick; perceived-`d` is.
Corollary: the prenoise (step[0]) contagion frac-kick GROWS as res drops (+0.23 @0.2MP vs +0.09 @0.5MP) yet leaves
perceived-`d` UNCHANGED — prenoise moves frac magnitude, not the perceived denoise.

## The 0.2MP no-hold ruler + no-hold ≈ hold=0.01 (HOLD-22)
At 0.2MP NO machinery is needed — **m IS the knob (d≈m)**. TRUE no-hold m=0.5 gave realized redraw frac **0.273**
(|x0|=0.5362, |clean|=0.7319); user reads it as ~0.5. This is the perceptual "ruler" against which high-res hold
configs are compared. The earlier **hold=0.01 proxy gave 0.272 — a dead heat**, so the proxy was empirically valid
despite taking a DIFFERENT branch (no-hold = init-lerp x0=m·x+(1−m)·clean + per-step correction; hold=0.01 = freeze
1 step then renoise to level m·σ_sw=0.4988 then tail). Mechanistic reason: both put the anchor into the tail at the
SAME magnitude (no-hold |x0|=0.5362 ≈ hold=0.01 renoise |x_mid|=0.5361, i.e. init-lerp level ≈ m·σ_sw) and run ~the
same step count, so they converge. A prior chat prediction that the two would DIVERGE is **FALSIFIED**.

**Open problem:** tune this ONE clean config to ALSO approximate "looks-like-0.25" / "looks-like-0.75" across ALL
resolutions, staying clean (not smudgey).
- MORE redraw (→0.75): likely EASY — earlier release / less hold = healthy many-tail-step regime.
- LESS redraw (→0.25): the HARD end. At 0.2MP any d≈m trivially; at 1MP only d=0.5 is nailed. Routes tried produced
  smudgey output (hold=0.75 confounded; m=0.8@1MP HOLD-21). Low-`d`-at-high-res is the open frontier.

**Leading UNTESTED lever for low-`d` at high res (candidate/plan, NOT a result):** SPLIT `m_release` from the tail
`m` (both currently = m_packed at nodes.py:282) so the tail can keep `m≈1` (clean/on-manifold correction) while
lowering ONLY the release level `L` to redraw less. This is the axis HOLD-19/21 could not reach (there, lowering m
also degraded the tail correction → smear). Distinct from the dissolved Design-B "decoupled L" (that lowered L at a
fixed early release with a single m); here the split preserves the m≈1 correction that keeps output on-manifold.

## Open alternative — is HOLD-18 invariance native-m or CONTAGION? (Fable — important)
HOLD-18 was attributed to "m≈1 native denoise," but Finding 13 has neighbor attention AFFECTING the anchor. If
the neighbor field is res-stable (whole-video native gen is) and the anchor is partly contagion-SET, it could
inherit res-invariance THROUGH contagion at ANY m — predicting the in-flight 0.8/0.9 run ALSO returns invariant,
for a DIFFERENT reason. The two stories AGREE on the band (0.8/0.9) and DIVERGE at LOW m. ⇒ an invariant
in-flight result does NOT confirm native-m; it only settles the band. Live dispute:
[anchor-denoise-m-vs-res](anchor-denoise-m-vs-res.md).

## Queued discriminator: m=0.7 at BOTH res (pre-Block-D)
m=0.7 (+ m=0.6 if cheap), hold=0.5 (early release), 0.5MP AND 1MP; frac + visual. Three answers:
(i) floor vs late-release — clean at frac≈0.3 ⇒ no floor in good regime; artifact-y ⇒ content floor (FEATURE);
(ii) res-invariance at LOW end (native-m predicts compression; contagion predicts invariance — diverge here);
(iii) bounds clean range below frac 0.39.

## Caveats to carry
- **frac ≠ d:** full-strength redraw tops out at frac≈0.55 (conditioning keeps full regen correlated with
  clean). If mapping d→target-frac, normalize by this ceiling.
- **d_content floor is UNPROVEN:** only sub-floor point (frac 0.237) is triple-confounded; no early-release
  low-frac run exists. If real, document as img2img semantics ("low d preserves source flaws — pre-clean the
  keyframe"), do NOT silently over-redraw.
