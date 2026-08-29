<!-- provenance: confirmed (GPU: M-B held≥~28 AND ramp≥51 unique survivor; M-A/M-C/M-D/M-E refuted; REFINED two-stage FORMATION∧NOT-HEALED, ramp upper-edge monotone) + theory (seam-attention SNR-trough UNVERIFIED) -->
<!-- verified: 2026-08-28 · decoupling matrix on 0/0/39/90 + 3 two-stage-refinement configs (0/0/39/107, 0/0/38/89, 0/0/29/80), branch fix-audio-ancestral-axis-mismatch; prior: exhaustive CPU audit -->
# Bug E: long-fade video interference — decoupled; M-B (held ≥ ~28 AND ramp ≥ 51) is the unique surviving model

Bug E symptom: moiré / streamers / ribbons / electric patterns in video latent with long fades;
sampler-independent; audio tracks via joint attention.
Established in [bugs.md](bugs.md) and [audio-axis-verdict.md](audio-axis-verdict.md).
Note: σ̃/H2 and Fix A do NOT address Bug E — present on main before either fix, sampler-independent.

## Child docs (drill-down)

- [long-fade-grid-beat/ramp-length-decouple.md](long-fade-grid-beat/ramp-length-decouple.md):
  **CURRENT HEADLINE** — GPU single-variable decoupling matrix on `0/0/39/90`. **M-B
  (held ≥ ~28 AND ramp ≥ 51) is the unique surviving model; both terms independently necessary.**
  Pure-ramp (M-A), held-alone (M-E), midpoint (M-D), and trailing-free-alone (M-C) all REFUTED.
- [long-fade-grid-beat/two-stage-heal.md](long-fade-grid-beat/two-stage-heal.md): **REFINES M-B**
  (numbers unchanged) into TWO stages — FORMATION (ramp ≥ 51, MONOTONE, no upper edge through 68) ∧
  NOT-HEALED (held ≥ ~28, SOFT boundary; held≈29 → MIXED). 3 new GPU configs; retires standing-wave.
- [long-fade-grid-beat/kv-observer-mismatch.md](long-fade-grid-beat/kv-observer-mismatch.md):
  **CONFIRMED (numeric):** the KV/observer curvature mismatch — `σ_row > m·σ_g` every fade row (up to
  ~7.7×), LIE worst HELD-SIDE (m≈0.18); Δ×m DAMAGE peaks MIDPOINT (m≈0.5). The FORMATION seed.
- [long-fade-grid-beat/survival-model.md](long-fade-grid-beat/survival-model.md): **THEORY (attention
  CONFIRMED global; mechanism UNVERIFIED):** healing = attention-share DILUTION / softmax outvote, not
  out-of-range; held-size suspected POSITION-confound (pending #82); only SETTLED m=1 heals; dilution
  SPATIAL not temporal.
- [long-fade-grid-beat/cpu-audit.md](long-fade-grid-beat/cpu-audit.md): exhaustive CPU audit —
  every discrete/threshold/segmentation surface opened and FALSIFIED as the gate (two independent
  passes, both negative). Supersedes the snap/rounding + observer-split hypotheses.
- [long-fade-grid-beat/first-frac-row.md](long-fade-grid-beat/first-frac-row.md): `first_frac_row
  ∈ [8,11]` ⇔ ERROR — perfect 14/14 CPU proxy (a reparametrization of ekf), + decisive next GPU
  runs with sharp predictions.
- [long-fade-grid-beat/ghost-observer-defect.md](long-fade-grid-beat/ghost-observer-defect.md):
  genuine non-gate correctness bug (observer fractional-row low threshold vs sampler `never`
  boundary) — open, fix deferred. Now has a CONFIRMED boundary instance: row 11 flips observer
  membership exactly at ekf 39/40.
- [long-fade-grid-beat/ekf-39-40-input-diff.md](long-fade-grid-beat/ekf-39-40-input-diff.md):
  the EXECUTED ekf=39→40 input diff. RETRACTS the "inputs identical" claim — the delta is TWO
  discrete loci (row-11 observer flip, row-13 k_d 17→18) at the HOLD→FADE SEAM, not the midpoint.

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
| 0/0/29/80 ✦ | 29 | 51 | 54.5 | 10 | MIXED |
| 0/0/38/89 ✦ | 38 | 51 | 63.5 | 1 | ERROR |
| 0/0/39/107 ✦‡ | 39 | 68 | 73.0 | 0 | ERROR |

0/0/35/90 is the shift-series anchor (previously known ERROR). 0/0/30/85 is the decisive new
falsifier: same ramp L=55 as the clean configs but ERROR — killing M-A and M-C simultaneously.

✦ = 2026-08-28 two-stage run. ‡ = clip=107 (NOT 90); breaks the held=90−ramp weld (tail vs clip=107).
**MIXED** = FORMS mid-fade (preview, ~step 5) then HEALS. `0/0/39/107` (ramp 68) resolves "ramp upper
edge" as a clip=90-weld artifact — formation MONOTONE, no edge through 68. See
[long-fade-grid-beat/two-stage-heal.md](long-fade-grid-beat/two-stage-heal.md).

## Surviving models — DECOUPLED 2026-08-28 (single-variable matrix)

NOTE (2026-08-28): the confound is now BROKEN by a single-variable perturbation matrix on
`0/0/39/90` — see [long-fade-grid-beat/ramp-length-decouple.md](long-fade-grid-beat/ramp-length-decouple.md).
Verdict: **M-B (held ≥ ~28 AND ramp ≥ 51) is the WINNER / unique survivor; both terms
independently necessary. Pure-ramp (M-A), M-D, and M-E all REFUTED; M-C reconfirmed dead.** The
matrix adds the OFF-90 point `0/0/39/85` (held39, ramp46, CLEAN) that makes the ramp term provably
necessary independent of held, and — together with the L=55 shift series held-necessity pair — kills
pure-midpoint and pure-held. The three entries below are retained for history; read the decouple doc
for current status.

**M-B held+ramp — WINNER:** error iff held ≥ ~28 AND ramp ≥ 51. Fits all 22 GPU configs. To KILL
the error, break EITHER term — held<28 OR ramp<51. Projection insight: at fixed clip=90 held = 90 −
ramp, so "ramp≥51" and "held≥28 (⟺ ramp≤62)" collapse onto one interval — the apparent clip=90
"ramp band L∈[51,62]" was just M-B's two thresholds projected under held+ramp=90; decoupling needed
an off-90 point (`0/0/39/85`) to separate them.

**M-D fade-midpoint absolute position — REFUTED:** was error iff (ekf+efo)/2 ∈ ~[57.5, 64.5].
Killed by the off-90 point `0/0/39/85`: midpoint (39+85)/2 = 62 ∈ the band yet CLEAN. Retained for
history only.

**M-E held-band — REFUTED:** was error iff held (= ekf) ∈ ~[28, 39]. Killed by `0/0/39/85`: held
39 ∈ [28,39] yet CLEAN. Held alone does not gate; it is one NECESSARY TERM in M-B, not a standalone
band. Retained for history only.

## Noise-localization observation (2026-08-28)

0/0/34/90: noise STRONGEST at frames ~60–64; fade [34,90], midpoint=62 — mid-ramp m≈0.5 releasing at
mid-schedule (~step 10). Consistent with M-B (and the Δ×m damage peak); narrows timing, does not
decide the causal variable.

## CPU discrete surface — EXHAUSTIVELY FALSIFIED (2026-08-28)

No code-side discrete gate exists. Two independent passes (Fable numeric replication + a Sonnet
mechanism auditor) both came back negative: snap/rounding family, observer split, denoise-mask
quantization, dense sigma grid, and frame_to_row/evaluate_envelope arithmetic are all opened and
falsified. Structural reason: a strictly-monotone linear ramp cannot produce a degenerate
segmentation or dedup merge, so the observer-split and cell-alignment hypotheses are conclusively
dead. Full detail + killer pairs in [long-fade-grid-beat/cpu-audit.md](long-fade-grid-beat/cpu-audit.md).

**Verdict:** the binary artifact is NOT explained by CPU-side quantization/segmentation. This
CONFIRMS the binary character ("happens or doesn't", no gradient) while locating the gate DOWNSTREAM
in GPU model/attention dynamics — the DiT's response to where the held prefix ends in the row grid.
The perfect CPU PROXY is `first_frac_row ∈ [8,11]` ⇔ ERROR (14/14; a reparametrization of ekf),
which pins the m≈0.5 row to latent rows 17–18 / frames 56–67 where the artifact originates — see
[long-fade-grid-beat/first-frac-row.md](long-fade-grid-beat/first-frac-row.md). A genuine non-gate
"ghost-observer" correctness defect also surfaced: [long-fade-grid-beat/ghost-observer-defect.md](long-fade-grid-beat/ghost-observer-defect.md).

## Epistemic note

Six falsified mono-causal rules committed from single-family covariable sweeps:
1. Grid-cycle-count → falsified (early).
2. Held+ramp → retired as confound when S2 appeared, un-retired after S2 fell.
3. S2 cell-alignment → GPU-falsified 2026-08-28.
4. Ramp-length band (M-A) → GPU-falsified 2026-08-28 by L=55 shift series.
5. CPU snap/rounding family → CPU-falsified 2026-08-28 (Fable 13-config sweep, two-sided).
6. Observer-split / dedup-collapse → CPU-falsified 2026-08-28 with a STRUCTURAL reason: a
   strictly-monotone linear ramp cannot produce a degenerate segmentation or dedup merge, so no
   observer-side discrete gate can exist (see cpu-audit.md).

**Lesson: do not declare a root cause from a sweep where candidate variables covary. Require a
decoupling factorial before committing any model.** That factorial is now DONE (single-variable
matrix on 0/0/39/90 + the L=55 shift series): M-A/M-D/M-E fell, M-C stayed dead, and M-B (held ≥ ~28
AND ramp ≥ 51) survived as the unique two-term model — see
[long-fade-grid-beat/ramp-length-decouple.md](long-fade-grid-beat/ramp-length-decouple.md).

## Fix direction (decoupler resolved — M-B is the model)

- Model: ERROR ⟺ held ≥ ~28 AND ramp ≥ 51. To kill the error, break EITHER term — held < 28 OR
  ramp < 51 (edge sits between 50 and 51 for ramp). Both terms are independently necessary.
- M-A (pure-ramp band), M-D (midpoint), and M-E (held-band) are REFUTED — do not build fixes on
  them. In particular the ramp lever alone is NOT sufficient (the L=55 series proves held matters).
- Projection caution: at clip=90 the two thresholds collapse (held = 90 − ramp), so a clip=90-only
  sweep looks like a single "ramp band L∈[51,62]"; that is an artifact — off-90 fades decouple.
- Ramp UPPER EDGE — RESOLVED 2026-08-28 (supersedes the old "untested / do not assume monotone"
  caveat): `0/0/39/107` (ramp 68, held 39) still ERRORs, so ramp FORMATION is MONOTONE with no upper
  edge through 68. The apparent ~64 upper edge was an artifact of the clip=90 weld (held=90−ramp).
- Refinement: Bug E is now a TWO-STAGE process — FORMATION (ramp, monotone) ∧ NOT-HEALED (held, a
  soft healing boundary ~28–30; held≈29 → MIXED). Numbers match M-B; a very short held or short ramp
  still kills the error, but the held lever now works by HEALING the formed pattern, not preventing
  formation. See [long-fade-grid-beat/two-stage-heal.md](long-fade-grid-beat/two-stage-heal.md).
