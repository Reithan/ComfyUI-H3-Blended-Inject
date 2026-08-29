<!-- provenance: confirmed (GPU single-variable decoupling matrix on 0/0/39/90) — the final FORMATION-mechanism section is theory (UNVERIFIED; standing-wave RETIRED → seam-attention / mid-ramp SNR-trough) -->
<!-- verified: 2026-08-28 · GPU single-variable perturbation matrix, branch fix-audio-ancestral-axis-mismatch -->
# Bug E — ramp-length decoupling matrix: M-B (held ≥ ~28 AND ramp ≥ 51) is the unique surviving model

Parent: [long-fade-grid-beat.md](../long-fade-grid-beat.md).
REFINED 2026-08-28 → [two-stage-heal.md](two-stage-heal.md): M-B's single rule decomposes into
FORMATION (ramp, monotone — no upper edge through ramp 68) ∧ NOT-HEALED (held, a soft healing
boundary; held≈29 → MIXED). M-B's numeric predictions are unchanged; read that doc for the
mechanism update below.

A single-variable perturbation matrix on the canonical ERROR config `0/0/39/90` (sfi/skf/ekf/efo,
clip=90, steps=20, shift_v=12, schedule linear_quadratic, res 0.2MP, audio mode=drop) DECOUPLES
the long-standing three-model confound (held-block-size vs ramp-length vs trailing-free-heals).
Each row below changes exactly ONE variable off the baseline — the changes are separate, not
additive.

Define: **held = ekf − skf**, **ramp = efo − ekf**, **free tail = clip − efo**.

## The matrix

| Perturbation (off 0/0/39/90) | held | ramp | tail | result |
|------------------------------|------|------|------|--------|
| baseline 0/0/39/90 | 39 | 51 | 0 | ERROR |
| interp ease_in_out→linear | 39 | 51 | 0 | ERROR |
| min_denoise 0→0.1 | 39 | 51 | 0 | ERROR |
| inject_at 0→17 | 39 | 51 | 0 | ERROR |
| steps 20→30 | 39 | 51 | 0 | ERROR |
| shift_video 12→10 | 39 | 51 | 0 | ERROR |
| schedule linear_quadratic→simple | 39 | 51 | 0 | ERROR |
| clip 90→107 (efo still 90) | 39 | 51 | 17 | ERROR |
| efo 90→85 (0/0/39/85) | 39 | 46 | 5 | CLEAN |
| ekf 39→40 (0/0/40/90) | 40 | 50 | 0 | CLEAN |
| resolution 0.2→0.3MP | 39 | 51 | — | clean* |

`*` resolution 0.3MP is **DILUTION-SUSPECT** — do NOT count it as a clean data point; the user
flagged it may only be hiding the signal by spreading it over more tokens, not fixing it.

## Conclusions (CONFIRMED by this GPU matrix)

1. **`ramp = efo − ekf` is a NECESSARY term — but NOT the sole discriminator.** Within THIS matrix
   held was pinned ≈39 (≥28 in every row), so ramp was the only FREE lever; that is a slice
   artifact, not global sufficiency. The pre-existing L=55 shift series (two CLEAN, two ERROR at
   identical ramp=55) directly refutes any pure-ramp band. What this matrix DOES prove is that the
   ramp term is necessary INDEPENDENT of held: `0/0/39/85` (held39, ramp46) is CLEAN vs
   `0/0/39/90` (held39, ramp51) ERROR — held identical, only ramp moved across the 50/51 edge.
   Combined with the held-necessity from the L=55 series (`0/0/25/80` held25 CLEAN vs `0/0/30/85`
   held30 ERROR, ramp=55 fixed in both), BOTH terms are independently necessary → **M-B is the
   unique survivor** (see model status below).
2. **Held-block-size ALONE is REFUTED.** `efo 90→85` holds held=39 (same as baseline) yet goes
   CLEAN → held=39 yields BOTH ERROR and CLEAN. Kills M-E (held-band ∈[28,39]→error).
3. **Trailing-free-heals ALONE is REFUTED (twice).** `ekf 39→40` has tail=0 like baseline yet
   CLEAN; and `clip 90→107` adds a 17-frame free tail yet stays ERROR while `efo→85`'s 5-frame
   tail goes CLEAN. Bigger tail errors, smaller tail heals = anti-monotone. Reconfirms M-C dead.
4. **The 39→40 flip is just the ramp crossing 51→50** — NOT a discrete input-quantization feature.
   No row-11 observer-membership flip and no row-13 k_d 17→18 step is the cause; those were
   quantized shadows of a continuous ramp-length crossing. This RESOLVES the earlier
   "seam-vs-midpoint gap" (the exhaustive 39/40 input diff put all discrete deltas at the seam rows
   11–13, while the artifact appears at midpoint rows 17–18 — see
   [ekf-39-40-input-diff.md](ekf-39-40-input-diff.md) and [first-frac-row.md](first-frac-row.md)).
5. **Sampler/steps/schedule/shift/interp/min_denoise/inject_at are all invariant** — none of the
   seven non-geometry perturbations moves the outcome (all ramp=51, all ERROR). In particular
   `steps 20→30` KILLS every steps-dependent / k_d-quantization framing (incl. first_frac_row / the
   H4 angle): the gate is geometric (fade shape), not schedule-quantized.

## Model status after decoupling

- **M-B (held+ramp) → WINNER / unique survivor**: ERROR ⟺ held ≥ ~28 AND ramp ≥ 51, where
  held = ekf−skf and ramp = efo−ekf. It fits ALL 22 GPU configs (the 20 in the index full-data
  table + the two new single-variable CLEAN points `0/0/39/85` and `0/0/40/90` + the clip=107
  point). BOTH terms are now INDEPENDENTLY NECESSARY: ramp-necessity from `0/0/39/85` (CLEAN) vs
  `0/0/39/90` (ERROR) at fixed held; held-necessity from `0/0/25/80` (CLEAN) vs `0/0/30/85`
  (ERROR) at fixed ramp=55.
- **Pure-ramp band (M-A) → REFUTED, stays dead**: the L=55 shift series (two CLEAN, two ERROR at
  identical ramp=55) directly refutes it. This matrix does NOT revive pure ramp — it establishes
  ramp as ONE necessary term inside M-B, not a standalone discriminator.
- **M-D (midpoint absolute position) → REFUTED**: `efo→85` has midpoint (39+85)/2=62 ∈ the claimed
  [57.5,64.5] band yet is CLEAN.
- **M-E (held-band) → REFUTED**: `0/0/39/85` has held 39 ∈ [28,39] yet is CLEAN (conclusion 2
  applies — held alone does not gate).
- **M-C (trailing-free-heals) → REFUTED, reconfirmed** (conclusion 3).
- Reconciliation with the M-A tombstone: M-A was falsified as a GLOBAL 1-D ramp band by the L=55
  shift series where held COVARIED. This matrix isolates ramp with held fixed (~39) and shows the
  ramp term is necessary — but pure ramp-band remains correctly dead. The matrix does NOT revive
  it; it completes M-B's two-term necessity.
- Projection insight: on the efo=90 line, held = 90 − ramp, so "ramp≥51" and "held≥28
  (⟺ ramp≤62)" collapse onto one interval — the old apparent "ramp band L∈[51,62]" at clip=90 was
  just M-B's two thresholds PROJECTED onto that clip. Decoupling therefore required an OFF-90 point;
  `0/0/39/85` supplies it and is what makes ramp provably necessary independent of held.

## FORMATION mechanism — standing-wave RETIRED → seam-attention / mid-ramp SNR-trough (theory, UNVERIFIED)

**Standing-wave / resonant-band — TOMBSTONE (retired 2026-08-28).** WHY: a resonance needs a
tuned/banded ramp length (it would DETUNE → an upper edge). Formation is MONOTONE with NO upper
edge through ramp 68 (`0/0/39/107`: ramp 68, held 39 → ERROR), so no resonant band exists. The old
picture — a standing interference with its antinode at the ramp midpoint, ramp length setting the
wavelength, forming only above ~51 frames — is kept only as history. Do not revisit.

**Leading mechanism now (theory — UNVERIFIED): seam-attention reach vs a mid-ramp SNR trough.** The
ramp is bounded by two high-confidence seams (hold→fade near m=0, clean-anchored; fade→end near
m=1, generation-anchored) whose attention attenuates inward. The ramp MIDPOINT is both lowest-SNR
(m≈0.5) and furthest in row/token distance from both anchors; a longer ramp pushes it beyond
seam-attention reach → the model cannot resolve it → moiré. Gives monotone-in-length, a
midpoint-localized artifact, and a binary threshold WITHOUT a resonance. Full statement + the
survival stage + the interp-density nuance: [two-stage-heal.md](two-stage-heal.md).

The seam-attention argument is the PROPAGATION half only. The FORMATION SEED is now quantified: the
KV/observer curvature mismatch (`σ_row > m·σ_g`, worst held-side at m≈0.18, up to ~7.7×) — CONFIRMED
numerically in [kv-observer-mismatch.md](kv-observer-mismatch.md). Curvature gives the seed + sign;
seam-attention gives the WHERE (mid-ramp) and the length dependence.

Any mechanism must explain BOTH terms: ramp gates FORMATION (monotone), held gates SURVIVAL/HEALING
(soft boundary ~28–30). A very short held heals the formed pattern even at ramp≥51 (`0/0/25/80`
held25, ramp55 CLEAN).
