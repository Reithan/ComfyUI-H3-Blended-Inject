<!-- provenance: status (design fork + next-test plan; NEW inference from RUN m-probes + the r60 run; hypotheses marked UNVERIFIED) -->
<!-- verified: 2026-08-25 · r60 m=0.99/hold=0.5 GPU run + sampler.py level=m·σ_sw trace @proto-latent-hold-release -->
# Held single-frame keyframe — is `m` the lever, or the release σ? (design fork + next tests)

Continues [anchor-denoise-m-vs-res](anchor-denoise-m-vs-res.md). The runs now let us attack the user's two
prototype questions directly: **Q1** — under hold, pin `m@1` or leave it variable? **Q2** — what noise level
at step 0 and at release, as a function of `m` and `hold`? Index: [index](index.md).

## New result — high-m GENERALIZES to r60 (last open single-frame point closed)
r60 **m=0.99, hold=0.5** (r40=m=0.0), 0.5MP: r60 ALSO well-blended and "properly denoised to ~0.5 visual
accuracy" — same as r40 at these settings. The m=0.99 recipe is NOT position-specific. Logged HOLD-15;
removed from experiments-run's DEFERRED list.

## Key re-reading: at hold=0.5, "m=0.99" IS already pin-m=1 (schedule is back-loaded)
Re-noise level = `m·σ_sw` (sampler.py). At hold=0.5 the release step sits at **σ_sw≈0.975** — linear_quadratic
is back-loaded (steps 0→10 drop σ only ~2.5%; 10→20 drop ~97.5%; Finding 12). So m=0.99 → level
`0.99·0.975≈0.965` ≈ FULL noise. **The result we like is already a near-full, on-schedule redraw — effectively
`m=1` SDEdit at high strength**, NOT a variable-m<1 partial. The BAD result (m=0.5 → artifacts) is the
variable-m<1 regime (level 0.49; correction keeps 50% of the un-redrawn frame → off-manifold). ⇒ **The
existing m=0.5-vs-0.99 A/B is ALREADY a crude "variable-m vs pin-m=1" comparison, and pin-m=1 won.**

## Two DIFFERENT ways to make a "partial denoise" — the real fork
- **(a) variable `m<1`** (today's knob): `corrected = m·denoised+(1−m)·inp` always retains `(1−m)` of the
  un-redrawn frame → an OFF-manifold blend (the m=0.5 artifacts). Amount and manifold-cleanliness are COUPLED.
- **(b) pin `m=1`, set amount via the release σ** (proper SDEdit strength): full on-manifold redraw, but
  STARTED from a chosen noise level σ_L. Less noise → smaller but still-clean departure from source. Amount
  DECOUPLED from cleanliness.
We have only ever seen (b) at ~FULL strength (hold=0.5 ⇒ σ_sw≈0.975). We have NOT tested whether (b) yields a
clean, controllable LOWER amount. That gap is exactly what Q1/Q2 hinge on.

## Q2 answer (HYPOTHESIS — ⚠ RETRACTED by the sweep results below; kept as the reasoning trail)
`level = m·σ_sw` conflates two things: WHICH step we release at (σ_sw) and HOW FAR BELOW the schedule we land
(the `m` factor). Proposal: release AT the schedule (`level = σ_sw`, m=1, on-manifold) and set strength by
CHOOSING σ_sw — release later ⇒ smaller redraw. Step-0 = clean (held). This makes `m` inert for held
keyframes and the **release σ** the amount knob. Because the schedule is back-loaded, parametrize by a
**target σ_L** (release at the step where σ≈L) so the knob is schedule-independent — NOT by raw `hold_frac`
(hold=0.5 is already ~full strength; you'd need ~0.75–0.9 to move σ_sw meaningfully).

## Next tests (yes/no; grepped experiments-run.md — NEITHER is run: prior hold sweeps were m=0.5 attraction)
**Test A — release-σ sweep at pinned m=1 (answers Q1 + controllability).** r40 (or r60), m=1, vary the
release so σ_sw spans e.g. {~0.9, ~0.6, ~0.3} (via a target-σ, or holds ~0.5 / 0.75 / 0.9). Judge realized
redraw of the anchor.
- Redraw decreases monotonically AND stays CLEAN as σ_sw shrinks ⇒ **(b) works → PIN m@1, expose release-σ as
  the min_denoise knob.** (Q1 = pin; Q2 = release level = σ_sw.)
- Redraw stuck ~0.5 regardless of σ_sw (neighbor-coupling attractor), OR low-σ_sw goes artifact-y ⇒ (b) is
  not a clean knob → keep `m` variable, or seek another amount lever. Reopens.

**Test B — hold vs no-hold at m=1, matched σ (answers Q2 step-0 / is the hold even needed).** m=1, same
release σ: (i) clean hold to the release step; (ii) NO hold — init the anchor directly at σ_sw, neighbors
co-evolve from step 0.
- Equivalent ⇒ the step-0 hold adds nothing; use plain SDEdit-strength init (simpler prototype).
- Hold blends better (neighbors compose around clean first) ⇒ keep the hold; step-0 = clean is correct.

Run both with the `‖x_final[anchor]−clean‖` proxy logged per keyframe — an amount SWEEP judged by eye is
exactly where the proxy turns "looks ~0.5" into a monotonic curve.

**Proxy WIRED (sampler.py, hold-release path):** after the release tail it prints
`realized anchor redraw: |x_final−clean|=… (frac of |clean|=…); |x_final|=…`. Read the `frac` value —
≈0 = untouched source, ↑ = more regeneration. Read-only, no behavior change. Grep the run log for
`realized anchor redraw`.

## Sweep RESULTS (2026-08-25) — `frac` tracks re-noise LEVEL; release-step sets QUALITY

r40, 0.5MP, euler/20-step. `frac` = realized-redraw proxy. Sorted by re-noise level = m·σ_sw:

| run | hold | σ_sw | m | level | frac | visual |
|---|---|---|---|---|---|---|
| hold=0.75 | 0.75 | 0.725 | 0.99 | 0.719 | **0.237** | BAD — under-denoised, abrupt/noisy blend |
| m=0.8 | 0.5 | 0.975 | 0.8 | 0.781 | 0.388 | good |
| m=0.9 | 0.5 | 0.975 | 0.9 | 0.880 | 0.442 | good |
| m=0.99 | 0.5 | 0.975 | 0.99 | 0.965 | ~0.50 | good (prior) |
| hold=0.25 | 0.25 | 0.9875 | 0.99 | 0.980 | 0.551 | good |

**CORRECTION (user, 2026-08-25):** "release-step = quality gate, NOT amount knob" below is RETRACTED as
unproven. The hold=0.75 bad run is non-diagnostic — the inject frame expected ~0.4–0.6 denoise, so a
lower-`d` approximation looks bad regardless of mechanism. Meanwhile hold=0.25 (frac 0.553) and hold=0.5
(frac 0.499) are BOTH clean at different frac → hold moves realized redraw while clean = a CANDIDATE clean
amount lever. Full retraction + new yardstick/anchor: [knob-design-open-questions](knob-design-open-questions.md).

**Two axes, now separated by the data:**
1. **Re-noise LEVEL = the redraw-amount knob.** `frac` rises monotonically with level (0.72→0.24 … 0.98→0.55).
   At a FIXED early release, **`m` cleanly sets the amount** — m∈{0.8,0.9,0.99} all look good.
2. **Release STEP = a quality gate, NOT an amount knob** (⚠ this conclusion is RETRACTED; see above). hold=0.75 (release step 15, σ_sw=0.725, only 5 tail
   steps, neighbors ~75% done) looks BAD even though its level (0.719) ≈ m=0.8's (0.781): under-denoised +
   abrupt blend. Releasing LATE to lower the amount starves the tail and commits the neighbors. hold≤0.5
   (≥10 tail steps, neighbors mid-denoise) is the good regime.

**⇒ REVERSAL — the "pin m=1, control amount via the release σ" lean above is RETRACTED.** σ_sw/release-step is
the quality gate; lowering it (proper-SDEdit-late-release) is exactly the bad regime. The data says the
opposite: **keep the release EARLY (hold≈0.25–0.5), leave `m` VARIABLE, set the redraw amount via `m`.**

**Q1 answered — do NOT pin m@1; leave m variable.** m∈[~0.8,1.0] is a clean monotonic amount knob at an early
release; pinning m=1 discards the one working lever. Also WEAKENS the "any m<1 is off-manifold" hypothesis —
m=0.8/0.9 are clean partials, not artifact-y.

**Q2 answered (useful range).** Step 0 = clean (hold). Release EARLY (σ_sw near-full, ~½ steps as tail).
Re-noise level = m·σ_sw = the amount; pick m for the target redraw (frac≈0.39 @m=0.8 → ~0.55 @m=0.99). Do NOT
reduce amount by releasing later.

**Still OPEN:** (a) is LOW redraw (frac<0.4) achievable CLEANLY, or is there a **d_content floor** (~0.4 redraw
needed to clean the anchor's own artifacts — the m=0.5 frac~0.15 "artifact" run hints at a floor)? Cheap next:
m=0.6/0.7 at hold=0.5. (b) late-release badness confounds tail-length vs neighbor-maturity. (c) Test B (hold
vs no-hold at matched level) still unrun.

**Prototype gotcha (user, 2026-08-25):** m=0.999 did NOT arm the hold (read as non-fractional). Likely cause:
`quantize_denoise` = `ceil(m·256)/256`, and `ceil(0.999·256)=256 → 1.0`, so any m>255/256≈0.9961 quantizes to
full-generation. Test ceiling = m≤0.99. VERIFY in mask.py/quantize path before treating as a real fix.
