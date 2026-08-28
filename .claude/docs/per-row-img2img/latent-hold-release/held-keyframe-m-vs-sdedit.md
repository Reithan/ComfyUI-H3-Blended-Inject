<!-- provenance: status (SUPERSEDED — route-1; design fork + sweep results + retraction; 2026-08-25) -->
<!-- verified: 2026-08-25 · r60 m=0.99/hold=0.5 GPU run + sampler.py level=m·sigma_sw trace @proto-latent-hold-release -->
# Held single-frame keyframe: `m` vs release sigma (SUPERSEDED)

Continues [anchor-denoise-m-vs-res](anchor-denoise-m-vs-res.md). Index: [index](index.md).

## New result: high-m GENERALIZES to r60 (HOLD-15)

r60 **m=0.99, hold=0.5** (r40=m=0.0), 0.5MP: r60 also well-blended and "properly denoised to ~0.5 visual
accuracy" — same as r40. The m=0.99 recipe is not position-specific.

## Key re-reading: at hold=0.5, "m=0.99" IS already pin-m=1 (schedule is back-loaded)

Re-noise level = `m·sigma_sw`. At hold=0.5 the release step sits at **sigma_sw≈0.975** — linear_quadratic
is back-loaded (steps 0→10 drop sigma only ~2.5%; 10→20 drop ~97.5%). So m=0.99 → level `0.99·0.975≈0.965`
≈ FULL noise. The result we like is already a near-full, on-schedule redraw — effectively `m=1` SDEdit at
high strength, NOT a variable-m<1 partial. The bad result (m=0.5) is the variable-m<1 regime (level 0.49;
correction keeps 50% of the un-redrawn frame). The existing m=0.5-vs-0.99 A/B is already a crude
"variable-m vs pin-m=1" comparison, and pin-m=1 won.

## Two ways to make a "partial denoise" — the real fork

- **(a) variable m<1** (today's knob): `corrected = m·denoised+(1−m)·inp` always retains (1−m) of the
  un-redrawn frame — an off-manifold blend. Amount and manifold-cleanliness are coupled.
- **(b) pin m=1, set amount via the release sigma** (proper SDEdit strength): full on-manifold redraw started
  from chosen noise level sigma_L. Amount decoupled from cleanliness.

Path (b) was only tested at ~FULL strength (hold=0.5 → sigma_sw≈0.975). Not tested whether (b) yields a
clean, controllable LOWER amount. This gap is what Q1/Q2 hinged on.

## Q2 hypothesis (RETRACTED — kept as reasoning trail)

Proposal: release AT the schedule (`level = sigma_sw`, m=1, on-manifold) and set strength by CHOOSING sigma_sw.
Step-0 = clean (held). This would make `m` inert for held keyframes and the **release sigma** the amount knob.
Because the schedule is back-loaded, parametrize by a **target sigma_L** (schedule-independent).
**Retracted below by the sweep data.**

## Sweep results (HOLD-16/17, 2026-08-25): `frac` tracks re-noise LEVEL; release-step sets quality

r40, 0.5MP, euler/20-step. `frac` = realized-redraw proxy. Sorted by re-noise level = m·sigma_sw:

| run | hold | sigma_sw | m | level | frac | visual |
|---|---|---|---|---|---|---|
| hold=0.75 | 0.75 | 0.725 | 0.99 | 0.719 | **0.237** | BAD — under-denoised, abrupt/noisy blend |
| m=0.8 | 0.5 | 0.975 | 0.8 | 0.781 | 0.388 | good |
| m=0.9 | 0.5 | 0.975 | 0.9 | 0.880 | 0.442 | good |
| m=0.99 | 0.5 | 0.975 | 0.99 | 0.965 | ~0.50 | good (prior) |
| hold=0.25 | 0.25 | 0.9875 | 0.99 | 0.980 | 0.551 | good |

**CORRECTION (user, 2026-08-25):** "release-step = quality gate, NOT amount knob" is RETRACTED as unproven.
The hold=0.75 bad run is non-diagnostic — inject frame expected ~0.4–0.6 denoise, so any low-d approximation
looks bad regardless of mechanism. Meanwhile hold=0.25 (frac 0.553) and hold=0.5 (frac 0.499) are BOTH clean
at different frac — hold moves realized redraw; clean is a CANDIDATE amount lever.
Full retraction + yardstick: [knob-design-open-questions](knob-design-open-questions.md).

**Two axes, now separated by the data:**
1. **Re-noise LEVEL = the redraw-amount knob.** `frac` rises monotonically with level (0.72→0.24 ... 0.98→0.55).
   At a FIXED early release, `m` cleanly sets the amount; m in {0.8,0.9,0.99} all look good.
2. **Release STEP = a quality gate, NOT an amount knob (RETRACTED).** hold=0.75 (5 tail steps, neighbors ~75%
   done) looks BAD even at level 0.719 ≈ m=0.8's 0.781: under-denoised + abrupt blend.

**REVERSAL:** the "pin m=1, control amount via release sigma" lean is RETRACTED. Lowering sigma_sw is exactly
the bad regime. The data says: **keep release EARLY (hold≈0.25–0.5), leave `m` VARIABLE, set redraw amount via `m`.**

**Q1 answered:** do NOT pin m@1; leave m variable. m in [~0.8,1.0] is a clean monotonic amount knob at early
release. Pinning m=1 discards the working lever. Also weakens "any m<1 is off-manifold" — m=0.8/0.9 are clean
partials.

**Q2 answered (useful range):** step 0 = clean (hold). Release EARLY (sigma_sw near-full, ~half steps as tail).
Re-noise level = m·sigma_sw = the amount; pick m for target redraw (frac≈0.39 @m=0.8 → ~0.55 @m=0.99).

**Still open (at time of writing):** is LOW redraw (frac<0.4) achievable cleanly, or is there a d_content floor?
Answered by HOLD-19: YES, floor confirmed at ~frac 0.39 @1MP. See [amount-floor-and-step0-redesign](amount-floor-and-step0-redesign.md).

**Prototype gotcha (user, 2026-08-25):** m=0.999 did NOT arm the hold. Likely cause: `quantize_denoise =
ceil(m·256)/256`, and `ceil(0.999·256)=256 → 1.0` — any m>255/256≈0.9961 quantizes to full generation.
Test ceiling: m<=0.99. Verify in mask.py/quantize path before treating as a real fix.
