<!-- provenance: status (HOLD-19 result + amount-floor confirmed + the trilemma + the user's step[0] redesign; my trilemma/smear-cause = HYPOTHESIS) -->
<!-- verified: 2026-08-25 · 1MP m-sweep GPU run (user) + fixed-release isolation @proto-latent-hold-release -->
# The amount-floor, the trilemma, and the step[0] redesign (post-HOLD-19)

Continues [knob-design-open-questions](knob-design-open-questions.md). The 1MP m-sweep (HOLD-19) forces two
corrections and surfaces a design trilemma. Index: [index](index.md).

## HOLD-19 data — 1MP, hold=0.5 (release step 10, σ_sw=0.975, 10 tail steps), only `m` varies
| m | frac @0.5MP | frac @1MP | @1MP visual |
|---|---|---|---|
| 0.8 | 0.388 | **0.362** | blend OK, **keyframe pops as a smeared frame** |
| 0.9 | 0.442 | **0.392** | blend very good, minimal pop (minor smudge) |
| 0.99 | ~0.50 | 0.499 | clean + well-blended |

## Correction 1 — `m` is NOT res-invariant (hold did not fix it; original finding re-confirmed)
Same nominal m gives LESS realized redraw at 1MP (0.388→0.362, 0.442→0.392); the compression → ~0 only as m→1
(0.99: 0.50→0.499). So the res-invariant-AND-clean point is NARROW — near-full redraw (m≈0.99). The hold never
touched m's res-behavior. Tagging m to res (m_eff=f(d,res)) is the original, already-doubted path — the user is
right to distrust a "constant hold makes m res-invariant" claim; the data kills it.

## Correction 2 — the amount-floor is REAL and SEPARATE (the point-5 confound, resolved by SYMPTOM)
The m-sweep holds the release FIXED (hold=0.5) — only amount changes. At 1MP m=0.8 (frac 0.362) **smears the
keyframe but blends fine**; m=0.9 (0.392) is clean. A low-amount failure at a GOOD release with identical
tail/neighbors ⇒ **amount alone, below a floor (~frac 0.39 @1MP), smears the keyframe.** Two DISTINCT symptoms:
- **abrupt/noisy blend** = release too late (low σ_sw, mature neighbors) — ONLY hold=0.75 shows it.
- **smeared/popped keyframe, blend fine** = amount below floor — m=0.8@1MP shows it at a good release.
hold=0.75 was CONFOUNDED (hit BOTH: low amount + late release); m=0.8@1MP ISOLATES the floor. Content-floor
CONFIRMED (I wrongly folded it into the late-release story on Fable's framing — conceded). The clean,
res-stable band is roughly **m∈[0.9,0.99], frac [0.39,0.55]**; high res crosses the floor below that.

## The trilemma (HYPOTHESIS — why a clean PARTIAL redraw is hard)
- A **clean** keyframe wants a NATIVE denoise (m=1 correction, on-manifold).
- Native denoise from a **partial** level wants releasing at that level's schedule step = few tail steps +
  mature neighbors = **bad blend**.
- An **early** release (good blend) at a partial amount uses the m<1 correction = the off-manifold partial-denoise
  regime (same family as the old m=0.5 artifacts) = **smear**.
⇒ (clean keyframe + good blend) has only been jointly reachable at ~FULL amount (m≈1). Partial amount forces
smear-or-bad-blend. This is why the res-invariant-clean point is narrow.

## Point 1 — the user's step[0] redesign (the lever that attacks the trilemma)
Today the hold presents a **0-noise (clean)** anchor to neighbors across steps 0→release, so neighbors converge
toward the **clean INPUT** — but the final keyframe is the **redrawn** version, a DIFFERENT target. At release
the keyframe is torn: neighbors (locked onto clean) pull it back toward clean while the tail redraws it → a
plausible source of the smear, worse at high res (targets differ in more high-freq detail — matches HOLD-19).
**Fix (user):** at step[0], pre-noise the anchor to its anticipated release state (≈ level m·σ_sw) and present
THAT through the hold, so neighbors converge toward the keyframe's release/redraw trajectory — killing the
"converge-to-clean-then-jump" discontinuity. **Prediction:** less keyframe tear at partial amounts, possibly
LOWERING the floor. **Caveat:** a noised anchor is a weaker content signal → the blend MAY degrade; that is the
empirical tradeoff to measure. (Mechanism-agnostic per user point 6 — we care whether it cleans the result.)

## BUILT — the `hold_prenoise_step0` toggle (A/B ready)
Implemented as a BOOLEAN node widget `hold_prenoise_step0` (default OFF = current clean-hold), threaded
`nodes.py`→`hold_release["prenoise_step0"]`→`sampler.py`. Mechanism: `renoised = level·eps + (1−level)·clean`
(`level = m_rel·σ_sw`) is computed UP FRONT (deterministic in x); when ON, `x0 = where(anchor, renoised, x0)`
BEFORE the hold, so the existing `m_hold=0` freeze holds the anchor at its release-state (not clean) through
steps 0→release. The release-step `torch.where(anchor, renoised, x_mid)` then becomes a no-op re-assert (anchor
already ≈ renoised) — no converge-then-jump. Log line tags `prenoise-step0` vs `clean-hold` and prints whether
x0 should ≈ renoised or clean. **Two defaulted sub-decisions (flag if wrong):** (1) hold at the EXACT release
level `m_rel·σ_sw`, not bare σ_sw; (2) keep the trusted `t_row` label during the hold (relabeling the held
anchor's timestep is a SEPARATE untried lever). All 576 tests pass; ruff clean. Not pushed (user pushes).

## HOLD-20 — prenoise ON, 0.5MP, m=0.99, hold sweep (vs prior clean-hold OFF)
| hold | frac OFF (prior) | frac ON | ON visual |
|---|---|---|---|
| 0.25 | 0.55 | **0.673** | blend + denoise fine |
| 0.50 | ~0.50 | **0.589** | blend + denoise fine |
| 0.75 | 0.24 | **0.376** | neighbor smudges + keyframe poorly denoised (bad) |

**Two clean reads:**
1. **No regression** — the good high-m configs (0.25/0.5) stay clean; prenoise doesn't break them.
2. **Prenoise is a real amount lever, and it acts PURELY through contagion.** By construction the anchor enters
   the tail at `renoised` in BOTH modes (the release-step `where(anchor, renoised, x_mid)` forces it; the hold
   only freezes, it doesn't evolve the anchor). So the anchor's own IC + tail schedule are IDENTICAL OFF vs ON —
   the ONLY difference is what NEIGHBORS saw during the hold (clean vs the noised release-state). Yet frac rose
   **+0.09 to +0.14** at every hold. That delta can only travel through global attention ⇒ **direct evidence
   neighbors SHAPE the anchor's realized redraw** (the user's contagion read), and prenoise rescales the amount
   knob UPWARD (seed the anchor noisy → it departs from clean more).

**hold=0.75 still bad (frac 0.376):** both symptoms (neighbor smudge + poor keyframe). Prenoise raised frac
(0.24→0.376) but did NOT rescue it — release timing is unchanged, so the late-release blend failure persists, and
0.376 is still below the ~0.39 floor. NOT decisive on the floor (0.75 is the confounded double-failure).

## HOLD-21 — prenoise ON, 1MP, m=0.8, hold=0.5 (the decisive test) → redesign FALSIFIED as a smear-fix
The exact config where clean-hold smeared (HOLD-19: frac 0.362). Prenoise ON: frac **0.372** (Δ+0.010, essentially
FLAT) — still smudgey/bad; blend in/out OK. **The step[0] redesign does NOT fix the smear**, and it falsifies the
premise: if the smear were the converge-to-clean-then-jump TEAR, removing it (prenoise) would clean it. It didn't,
and barely moved the amount ⇒ the 1MP low-m smear is NOT the neighbor-target mismatch.

Contrast HOLD-20 (0.5MP m=0.99, frac +0.09–0.14): prenoise's contagion lever has leverage in the HEALTHY high-m
band but almost NONE at 1MP m=0.8. Refined model: the anchor's OWN correction `m·denoised+(1−m)·inp` at low m
clamps realized redraw near a low value (the 0.2·inp pull dominates); neighbors can't overcome it. **The floor is
a property of the anchor's own partial denoise, not the neighbor target.**

## Leading hypothesis — the low-amount smear IS the base-model high-res deficiency (ties to RES-1)
A genuinely PARTIAL single-frame redraw at 1MP smears in the base model with NO hold at all — that is exactly
RES-1 (1MP single frame d=0.5 → source-identical pop) from the highres-underdenoise thread. m=0.99 is clean at
both res because it is a NEAR-FULL redraw (model-safe); m=0.8 partial + 1MP hits the base deficiency. Our
hold-release levers act on NEIGHBORS and INIT — they cannot repair a deficiency in the ANCHOR'S OWN denoise ⇒
prenoise can't help (HOLD-21). **Discriminator (largely already answered):** isolated single 1MP frame, SDEdit
partial strength, no hold/neighbors — clean or smeared? RES-1 says smeared → base model. If so, the low-d band is
fundamentally hard at high res via ANY neighbor-side route.

## Where this leaves the knob (USER decides)
Clean, res-invariant output lives only in the HIGH-amount band (m≈0.9–0.99 ≈ near-full redraw, frac ~0.4–0.67).
Genuinely low-d partial redraws smear at high res; prenoise does NOT rescue them. Two honest paths:
- **A (ship narrow):** map the single `d` knob onto the supported high-amount band; document low-d as unsupported
  at high res.
- **B (attack the root):** the anchor's OWN res-corrected effective-m (highres-underdenoise-model γ≈1.6 up-map) —
  ORTHOGONAL to hold-release (hold gives the blend, res-correction gives the clean partial redraw).
