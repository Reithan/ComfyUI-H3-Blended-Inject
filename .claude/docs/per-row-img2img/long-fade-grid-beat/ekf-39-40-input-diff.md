<!-- provenance: confirmed (EXECUTED per-row/per-step diff of real repo code; two pivots independently re-run by orchestrator) -->
<!-- verified: 2026-08-28 · evaluate_envelope + quantize_denoise + k_d executed on repo code (files byte-identical to origin/main); rows 11 & 13 pivots re-confirmed by orchestrator -->
# Bug E — the exact ekf=39→40 input delta (RETRACTS "inputs identical")

Parent: [long-fade-grid-beat.md](../long-fade-grid-beat.md).

This is the EXHAUSTIVE, executed mechanistic diff between the two canonical pivots
0/0/39/90 (GPU ERROR) and 0/0/40/90 (GPU CLEAN), holding sfi=0/skf=0/efo=90,
clip=90 / 27 rows / 20 steps / linear schedule / min_denoise=0. Every quantity was
diffed across all 27 rows × 20 steps.

**Correction:** the CPU audit ([cpu-audit.md](cpu-audit.md)) previously concluded the two
configs were CPU-identical / "no discrete feature separates". That is RETRACTED. The delta
is real and collapses to TWO discrete loci (rows 11 & 13) plus a sub-0.016 continuous label
cloud. It sits at the HOLD→FADE SEAM, not the fade midpoint.

## Discrete GATE ① — row 11 observer K/V membership flip

The ghost-observer defect ([ghost-observer-defect.md](ghost-observer-defect.md)) fires exactly
at this boundary:

- ekf=39: row 11 raw m=0.002404 → quantized q=1/256=0.003906 (>1e-6) → `_fractional_rows`
  (observer_split.py:69) INCLUDES it → DiT block K/V patched, advertised to attention at t_obs≈0.996.
- ekf=40: row 11 m=0.0 → EXCLUDED from the observer set.
- BUT the sampler freezes row 11 to clean in BOTH: k_d=round(20*(1−q))=20 → never=True
  (sampler.py:496–497), so row 11's own output is identical. The only downstream difference is
  the ERROR config exposes ONE extra barely-noisy observer row to attention.

Root of gate ①: an internal inconsistency between `classify_row_region` (uses `cc < ekf`,
reports row 11 'preserve' for both) and `evaluate_envelope` (uses `t <= ekf-1`, gives row 11
nonzero m at ekf=39). Row 11 center times = (35.5,36.5,37.5,38.5); the 38.5 center tips into
fade-out only for ekf=39 (hold_end=38), not ekf=40 (hold_end=39).
(This classify-vs-envelope mismatch was reported by the diff agent — REPORTED, not yet
independently re-run.)

## Discrete GATE ② — row 13 single k_d integer step (17→18)

The ONLY row whose sampler trajectory changes:

- q_A[13]=0.136719 → round(20*0.863281)=round(17.27)=17
- q_B[13]=0.121094 → round(20*0.878906)=round(17.58)=18
- Crosses the round() half-integer boundary.

Propagates to:
- dense-sigma indices at all 20 steps (idx_A=340+3i vs idx_B=360+2i, sampler.py:521),
- init composite at i=0 (0.15·x+0.85·clean vs 0.10·x+0.90·clean, sampler.py:600),
- per-row w constant 0.15 vs 0.10 across all steps (linear-schedule property),
- row 13's per-step DiT label t_row=1−w·sig_g.

## Continuous cloud — rows 12, 14–26

Slightly lower m under ekf=40 (holds one more frame); t_obs / pooled-label diffs ≤0.016.
Row 25 collapses to an identical quantized value (234/256). The `never` set is IDENTICAL in
both ([0..11]). Fade-midpoint row 19 has k_d=9 in BOTH — it does NOT change.

## Key reconciliation — seam, not midpoint

The input delta sits at the HOLD→FADE SEAM (rows 11–13, frames ~35–45, front of the ramp),
NOT the fade midpoint (rows 17–18, frames ~56–67) where the artifact visually originates.
The earlier "midpoint" framing (visual read + Fable's r19 feature-match, see
[first-frac-row.md](first-frac-row.md)) inspected the WRONG rows. Perturbation ENTERS at the
seam and MANIFESTS downstream at the visual midpoint via attention spreading.

## Status

- This is the exact mechanistic diff. The prior "inputs identical" claim is RETRACTED.
- The gate is one or both of ① (row-11 observer membership) / ② (row-13 k_d step);
  discrimination between them is the open question.
- Both loci are binary/discrete — consistent with the binary-artifact observation.
- No new GPU test proposed yet (user directive: find the diff first — done).
