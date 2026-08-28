<!-- provenance: status (SUPERSEDED — route-1; HOLD-19 result + amount-floor confirmed + trilemma + step[0] redesign; trilemma/smear-cause = HYPOTHESIS) -->
<!-- verified: 2026-08-25 · 1MP m-sweep GPU run + fixed-release isolation @proto-latent-hold-release -->
# The amount-floor, trilemma, and step[0] redesign (HOLD-19–21, SUPERSEDED)

Continues [knob-design-open-questions](knob-design-open-questions.md). Index: [index](index.md).

## HOLD-19 data: 1MP, hold=0.5 (release step 10, sigma_sw=0.975, 10 tail steps), only `m` varies

| m | frac @0.5MP | frac @1MP | @1MP visual |
|---|---|---|---|
| 0.8 | 0.388 | **0.362** | blend OK, **keyframe pops as a smeared frame** |
| 0.9 | 0.442 | **0.392** | blend very good, minimal pop (minor smudge) |
| 0.99 | ~0.50 | 0.499 | clean + well-blended |

## Correction 1: `m` is NOT res-invariant (hold did not fix it)

Same nominal m gives LESS realized redraw at 1MP (0.388→0.362, 0.442→0.392); compression → ~0 only as m→1
(0.99: 0.50→0.499). The res-invariant-AND-clean point is NARROW — near-full redraw (m≈0.99) only.
Hold never touched m's res-behavior. Tagging m to res (m_eff=f(d,res)) was the original, already-doubted path.

## Correction 2: the amount-floor is REAL and SEPARATE

The m-sweep holds release FIXED (hold=0.5) — only amount changes. At 1MP m=0.8 (frac 0.362) **smears the
keyframe but blends fine**; m=0.9 (0.392) is clean. A low-amount failure at a GOOD release with identical
tail/neighbors — this is amount alone, below a floor (~frac 0.39 @1MP), smearing the keyframe.

Two DISTINCT symptoms:
- **abrupt/noisy blend** = release too late (low sigma_sw, mature neighbors) — only hold=0.75 shows it.
- **smeared/popped keyframe, blend fine** = amount below floor — m=0.8@1MP shows it at a good release.

hold=0.75 was CONFOUNDED (hit BOTH: low amount + late release). m=0.8@1MP ISOLATES the floor. Clean,
res-stable band: roughly **m in [0.9,0.99], frac [0.39,0.55]**.

## The trilemma (HYPOTHESIS — why a clean PARTIAL redraw is hard)

- A **clean** keyframe wants native denoise (m=1 correction, on-manifold).
- Native denoise from a **partial** level wants releasing at that level's schedule step = few tail steps +
  mature neighbors = **bad blend**.
- An **early** release (good blend) at a partial amount uses m<1 correction = off-manifold partial-denoise = **smear**.

Clean keyframe + good blend jointly reachable only at ~FULL amount (m≈1). Partial amount forces smear-or-bad-blend.

## The step[0] redesign (the user's proposed lever)

Today the hold presents a 0-noise (clean) anchor to neighbors across steps 0 to release, so neighbors converge
toward the **clean INPUT** — but the final keyframe is the **redrawn** version. At release the keyframe is torn:
neighbors (locked onto clean) pull it back while the tail redraws it. This is a plausible source of the smear,
worse at high res (targets differ in more high-freq detail).

**Fix (user):** at step[0], pre-noise the anchor to its anticipated release state (≈ level m·sigma_sw) and present
THAT through the hold, so neighbors converge toward the keyframe's release/redraw trajectory.
**Prediction:** less keyframe tear at partial amounts, possibly lowering the floor.
**Caveat:** a noised anchor is a weaker content signal; blend may degrade.

## BUILT: the `hold_prenoise_step0` toggle (HOLD-20)

Implemented as a boolean node widget (default OFF = current clean-hold). On: `renoised = level·eps + (1−level)·clean`
computed up front; `x0 = where(anchor, renoised, x0)` before the hold. The release-step re-assert becomes a no-op.

## HOLD-20: prenoise ON, 0.5MP, m=0.99, hold sweep

| hold | frac OFF (prior) | frac ON | ON visual |
|---|---|---|---|
| 0.25 | 0.55 | **0.673** | blend + denoise fine |
| 0.50 | ~0.50 | **0.589** | blend + denoise fine |
| 0.75 | 0.24 | **0.376** | neighbor smudges + poorly denoised (bad) |

**Two reads:** (1) no regression — good high-m configs (0.25/0.5) stay clean. (2) Prenoise is a real amount
lever acting PURELY through contagion: anchor enters the tail at `renoised` in BOTH modes (the release-step
`where` forces it), so ONLY what NEIGHBORS saw during the hold differs. Yet frac rose +0.09–0.14 — direct
evidence neighbors shape anchor's realized redraw. hold=0.75 stays bad (release timing unchanged).

## HOLD-21: prenoise ON, 1MP, m=0.8, hold=0.5 — redesign FALSIFIED as smear-fix

Exact config where clean-hold smeared (HOLD-19 frac 0.362). Prenoise ON: frac **0.372** (delta +0.010, essentially
FLAT) — still smudgey/bad. The step[0] redesign does NOT fix the smear. The premise was wrong: if the smear were
the "converge-to-clean-then-jump" tear, removing it would clean it. It didn't, and barely moved the amount.

**Refined model:** the anchor's OWN correction `m·denoised+(1−m)·inp` at low m clamps realized redraw near a low
value (the 0.2·inp pull dominates); neighbors can't overcome it. The floor is a property of the anchor's own
partial denoise, not the neighbor target.

**Leading hypothesis (ties to RES-1):** a genuinely partial single-frame redraw at 1MP smears in the base model
with NO hold at all — exactly RES-1 from the highres-underdenoise thread. m=0.99 is clean at both res because it
is a NEAR-FULL redraw (model-safe); m=0.8 partial + 1MP hits the base deficiency. Hold-release levers act on
neighbors and init; they cannot repair a deficiency in the ANCHOR'S OWN denoise.

## Where this leaves the knob

Clean, res-invariant output lives only in the HIGH-amount band (m≈0.9–0.99, frac ~0.4–0.67).
Genuinely low-d partial redraws smear at high res; prenoise does not rescue them. Two paths:
- **A (ship narrow):** map d onto the supported high-amount band; document low-d as unsupported at high res.
- **B (attack the root):** res-corrected effective-m (highres-underdenoise-model gamma≈1.6 up-map) — orthogonal
  to hold-release (hold gives the blend; res-correction gives the clean partial redraw).
