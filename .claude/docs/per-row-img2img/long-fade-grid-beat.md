<!-- provenance: theory (UNVERIFIED — M-A & M-C falsified; survivors M-B/M-D/M-E confounded; short-ramp decoupler + code-quant dig pending) -->
<!-- verified: 2026-08-28 · L=55 shift series GPU-falsified length model; 3 survivors remain confounded at fixed clip=90 -->
# Bug E: long-fade video interference — shift-series falsifies length model; 3 survivors confounded

Bug E symptom: moiré / streamers / ribbons / electric patterns in video latent with long fades;
sampler-independent; audio tracks via joint attention.
Established in [bugs.md](bugs.md) and [audio-axis-verdict.md](audio-axis-verdict.md).
Note: σ̃/H2 and Fix A do NOT address Bug E — present on main before either fix, sampler-independent.

## GPU-FALSIFIED models (tombstones)

**M-A (ramp-length band) — FALSIFIED 2026-08-28:** L=55 shift series gives two CLEAN (0/0/15/70,
0/0/25/80) and two ERROR (0/0/30/85, 0/0/35/90); ramp length alone does not determine outcome.
Do not revisit.

**M-C (trailing-free-heals) — FALSIFIED 2026-08-28:** 0/0/30/85 has freetail=5 (same as CLEAN
0/0/0/85) yet is ERROR; free-tail size does not gate the artifact. Do not revisit.

**S2 (cell-alignment) — FALSIFIED 2026-08-28:** Fit 13/13 efo=90 configs as a spurious coincidence;
broke 3/4 on out-of-sample GPU tests (0/0/34/90 → ERROR vs CLEAN predicted; 0/0/47/90 → CLEAN vs
ERROR predicted; 0/0/39/90 → ERROR vs CLEAN predicted). Do not revisit.

## Full data table — 20 GPU-tested configs (clip=90 for all shift-series runs)

Notation: a/b/c/d = fade_in/skf/ekf/fade_out. held = c, ramp L = d−c, mid = (c+d)/2,
freetail = clip−d (clip=90 throughout; ★ = new shift-series run 2026-08-28).

| Config | held | ramp L | midpoint | freetail | result |
|--------|------|--------|----------|----------|--------|
| 0/0/0/17 | 0 | 17 | 8.5 | 73 | CLEAN |
| 0/0/0/69 | 0 | 69 | 34.5 | 21 | CLEAN |
| 0/0/0/73 | 0 | 73 | 36.5 | 17 | CLEAN |
| 0/0/0/75 | 0 | 75 | 37.5 | 15 | CLEAN |
| 0/0/0/85 | 0 | 85 | 42.5 | 5 | CLEAN |
| 0/0/7/75 | 7 | 68 | 41.0 | 15 | CLEAN |
| 0/0/15/70 ★ | 15 | 55 | 42.5 | 20 | CLEAN |
| 0/0/25/80 ★ | 25 | 55 | 52.5 | 10 | CLEAN |
| 0/0/30/85 ★ | 30 | 55 | 57.5 | 5 | ERROR |
| 0/0/30/90 | 30 | 60 | 60.0 | 0 | ERROR |
| 0/0/34/90 | 34 | 56 | 62.0 | 0 | ERROR |
| 0/0/35/90 ★ | 35 | 55 | 62.5 | 0 | ERROR |
| 0/0/36/90 | 36 | 54 | 63.0 | 0 | ERROR |
| 0/0/37/90 | 37 | 53 | 63.5 | 0 | ERROR |
| 0/0/38/90 | 38 | 52 | 64.0 | 0 | ERROR |
| 0/0/39/90 | 39 | 51 | 64.5 | 0 | ERROR |
| 0/0/40/90 | 40 | 50 | 65.0 | 0 | CLEAN |
| 0/0/44/90 | 44 | 46 | 67.0 | 0 | CLEAN |
| 0/0/47/90 | 47 | 43 | 68.5 | 0 | CLEAN |
| 0/0/60/90 | 60 | 30 | 75.0 | 0 | CLEAN |

0/0/35/90 is the shift-series anchor (previously known ERROR). 0/0/30/85 is the decisive new
falsifier: same ramp L=55 as the clean configs but ERROR — killing M-A and M-C simultaneously.

## Surviving models — all fit all 20 configs; confounded at fixed clip=90

Sliding a fixed-L=55 window at clip=90 moves held, midpoint, and freetail together — they remain
mutually confounded. A short-ramp test (decoupler, see below) is required to separate them.

**M-B held+ramp:** error iff held ≥ ~28 AND ramp ≥ 51. At fixed clip=90, held = 90 − L so
held≥28 ⟺ L≤62 — consistent with all 20 cases. Long ramps without a long held block (ramp≥51,
held<28) would be safe; long held blocks alone (held≥40, ramp<51) already confirmed safe.

**M-D fade-midpoint absolute position:** error iff (ekf+efo)/2 ∈ ~[55, 64.5]. The shift series
extends the band downward (0/0/30/85 mid=57.5 → ERROR; 0/0/25/80 mid=52.5 → CLEAN). Algebraic
caution: within efo=90, mid = 90 − L/2, so M-D and M-A were algebraically identical — but M-A's
falsification does NOT automatically rule out M-D. M-D survives as an independent hypothesis.

**M-E held-band (NEW, 2026-08-28):** error iff held (= ekf, the frozen block length) ∈ ~[28, 39].
Evidence: held 0/7/15/25 → clean; held 30–39 → error; held 40/44/47/60 → clean. Clean on BOTH
sides; no ramp term needed. Most parsimonious of the three survivors.

## Noise-localization observation (2026-08-28)

Dissecting 0/0/34/90: noise is STRONGEST at output frames ~60–64; fade spans [34, 90], midpoint=62.
Consistent with degeneracy forming mid-ramp where m≈0.5 and rows release at mid-schedule (~step 10).
Noise at the midpoint is expected under all three surviving models — it does not decide the causal
variable, only narrows timing-within-schedule.

## Short-ramp decoupler — decisive experiment (UNRUN)

No tested config has ramp < 43. A short ramp (ramp < 51) separates all three survivors in two runs:

| Run (clip=90) | held | ramp | midpoint | M-B | M-D | M-E |
|---------------|------|------|----------|-----|-----|-----|
| 0/0/50/70 | 50 | 20 | 60.0 | CLEAN (ramp<51) | ERROR (mid∈[55,64.5]) | CLEAN (held>39) |
| 0/0/35/65 | 35 | 30 | 50.0 | CLEAN (ramp<51) | CLEAN (mid<55) | ERROR (held∈[28,39]) |

Read-off: 0/0/50/70 ERROR → M-D confirmed; 0/0/35/65 ERROR → M-E confirmed;
both CLEAN → M-B (held AND ramp both required). At most one model survives per run.

## Code-quantization investigation (IN PROGRESS — Fable)

A Fable code-side dig is tracing whether quantization/rounding/snapping produces a degenerate
mid-fade transition row. Bracket: 0/0/25/80 CLEAN vs 0/0/30/85 and 0/0/35/90 ERROR vs 0/0/40/90
CLEAN. Placeholder — update with findings.

## Epistemic note

Four falsified mono-causal rules committed from single-family covariable sweeps:
1. Grid-cycle-count → falsified (early).
2. Held+ramp → retired as confound when S2 appeared, un-retired after S2 fell.
3. S2 cell-alignment → GPU-falsified 2026-08-28.
4. Ramp-length band (M-A) → GPU-falsified 2026-08-28 by L=55 shift series.

**Lesson: do not declare a root cause from a sweep where candidate variables covary. Require a
decoupling factorial before committing any model.** Three models still fit all 20 data points.

## Fix direction (hold until short-ramp decoupler resolves)

- M-B confirmed: snap either held or ramp below the respective threshold (~28 or ~51).
- M-D confirmed: snap fade midpoint (ekf+efo)/2 outside ~[55, 64.5].
- M-E confirmed: snap held (=ekf) outside ~[28, 39] — simplest constraint, no ramp term.

Do not implement any fix until the decoupler data arrives.
