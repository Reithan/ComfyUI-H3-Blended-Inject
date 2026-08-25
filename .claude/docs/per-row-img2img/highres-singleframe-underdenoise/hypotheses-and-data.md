<!-- provenance: theory (H1 CONFIRMED by elimination; bimodality GPU-confirmed 2026-08-24 — lock+chaos walls bracketed @1MP; exonerated alternatives, kill-shot data) -->
<!-- verified: 2026-08-24 · repo @debug-single-frame-underdenoise · 1MP run (row40@0.83=chaos, row60@0.45=lock) -->
# Hypotheses, exonerations, and kill-shot data

Index: [highres-singleframe-underdenoise](../highres-singleframe-underdenoise.md). Resolution ladder (FINDING 1/2/3): [resolution-ladder](resolution-ladder.md). Temporal effects: [temporal-and-contagion](temporal-and-contagion.md).

## Exonerated (with evidence — NOT the cause)

Our three levers' arithmetic (endpoint `clean − m·∫v dσ`, provably step/res-independent), mask
construction/pack/pool (uniform per-row `m`, idempotent amax), `post_composite_preserve` (`m==0`
only — never touches the 0.5 row), grid chunking (purely temporal, f136→row40 in both res),
external/native cond-row conditioning (user: none on hot path), sigma schedule σ_max resolution
dependence (shift is a fixed *user* value, not res-coupled; area-normalized RoPE). **Key nuance:**
the x-space invariance proofs are correct — they just don't imply *perceptual-corruption*
invariance (see H1).

## Leading hypotheses (ranked; Fable brainstorm + source)

### Measured so far (1MP, 40-step euler, debug branch)
- **Build:** single-frame inject latent is well-formed (std≈0.97–0.99 ≈ video rows) → H4/VAE-special-case dead.
- **Per-step trajectory (row 40, m=0.5, calls 0→37, t 1.00→0.27):** `|inp|` **0.5595→0.6972**
  (rising toward clean-norm ~0.77), `|den−inp|` **1.095→0.262** (converging). The row denoises
  FULLY and is NOT frozen — velocity is integrated every step. `t=1.00` on early calls = the
  `shift_video` schedule compressing early sigmas near 1.0 (expected, not stuck).
- **Init-lerp verified correct.** Metric calibration: pure-noise (GEN) reads `|inp|≈0.79`, clean
  (std 0.97) reads ~0.77, so a half-noise lerp reads `0.5·√(0.79²+0.77²)≈0.55` = row 40's 0.5595 at
  step 0. Lever 1 + per-token m land exactly on the keyframe → freeze / m-misalignment REFUTED.
- **Lands on SOURCE, not a generic gen.** At call 37 row 40 `|inp|`=0.697 sits ABOVE the GEN rows
  (0.66) and tracks its generated neighbor row 39 (0.687): GEN rows started at noise (0.79) and
  settled DOWN to a generic x0 (~0.66); row 40 started at the lerp (0.56) and rose UP toward the
  source's higher-structure norm. The predicted x0 ≈ `clean` = source ⇒ denoising returns the
  source ⇒ source-identical output. **This IS refined H1.**

### KILL-SHOT RESULT (α=√5 map, 1MP: row40 d=0.70, row60 d=0.66) — UNDERSHOOTS, quantified
Final `|out−clean|` normalized by the **d=1.00 companion rows** (39=0.956, 59=0.938) as full-gen:
- **row 40 d=0.70:** |out−clean|=0.278, |out|=0.801 → **R=0.29**
- **row 60 d=0.66:** |out−clean|=0.249, |out|=0.816 → **R=0.27**
Vs the d=0.5 run (row40 0.182, row60 0.166): displacement rose +53% (row40 0.182→0.278) — **raising
effective-m works directionally.** But still undershoots three ways in agreement: R≈0.28 for nominal
d≈0.7; `|out|`≈0.80 barely above clean-norm ~0.77 (frame hardly moved); and the `|inp|` rebound to
0.706 (self-reconstruction recapture). **R is CONVEX in d @1MP** — R(0.5)≈0.21, R(0.7)≈0.29, but
R(1.0)≡1.0 by definition: flat through the mid-range then steep near the top. That flat region IS
the self-reconstruction basin; proportional movement only starts once d clears it (est. ~0.85+). A
√5 map lands inside the flat zone → **α=√5 confirmed insufficient; the true correction is steeper
and nonlinear (basin escape), not a fixed power law.** **0.95 run (user, live): neighbors FINALLY
follow f40 and blend with it** — basin escape restores the temporal blend from the neighbor side
too (once the frame leaves source it becomes something neighbors resolve WITH, not around). 0.95
looks slightly too high → sweet spot in [0.8, 0.9]. NEXT: (a) run the 0.2MP baseline WITH debug
(same frames, d=0.5/0.45) to log its `|out−clean|`+full-gen → gives the TARGET R to match; then
(b) sweep 1MP d∈{0.8,0.85,0.9} to the d where R hits target AND the `|inp|` rebound vanishes
(monotonic settle to a generic norm, no climb back to ~0.70). Calibrate α from that d, not √5.
