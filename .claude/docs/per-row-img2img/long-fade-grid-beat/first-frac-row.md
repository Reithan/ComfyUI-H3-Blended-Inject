<!-- provenance: theory (UNVERIFIED — perfect 14/14 CPU proxy for Bug E; a reparametrization of ekf, not a newly-found code gate; predicts GPU outcomes) -->
<!-- verified: 2026-08-28 · Fable round-2 — first_frac_row∈[8,11] ⇔ ERROR on all 14 configs; GPU predictions still UNRUN -->
# Bug E — the `first_frac_row` coordinate: a perfect CPU proxy

Parent: [long-fade-grid-beat.md](../long-fade-grid-beat.md). CPU audit:
[cpu-audit.md](cpu-audit.md).

## Perfect CPU separator (14/14)

`first_frac_row ∈ [8, 11]` ⇔ ERROR.

Definition: the smallest latent-row index whose quantized per-row denoise is strictly fractional
(1e-6 < m_q < 0.999), computed from `evaluate_envelope(0,0,ekf,efo,0.0,'linear',90,27,0)` then
`ceil(d*256)/256`. CLEAN when first_frac_row ≤ 7 or ≥ 12.

For clip=90 / 27-row this is a bijection with ekf∈[30,39] — it is a REPARAMETRIZATION of ekf, NOT a
newly-found code gate. But it is the mechanistically meaningful coordinate: a latent row index tied
to the 17-frame chunk grid.
- row 8 = chunk1 local3
- row 9 = chunk1 local4
- row 10 = chunk2 boundary (1-frame)
- row 11 = chunk2 local1

## Why it reconciles the user's midpoint observation

When first_frac_row∈[8,11], the ramp's m≈0.5 row lands on latent rows 17–18 (frames 56–67) —
exactly where the artifact visually originates. Early/late placement moves that mid-fade row out of
the frames-56–67 zone → clean. This subsumes survivors M-B/M-D/M-E (held+ramp, midpoint-position,
held-band): all three are now predicted by first_frac_row, but remain confounded until decoupled on
GPU.

**Update (2026-08-28) — the INPUT perturbation is at the SEAM, not the midpoint.** An executed
ekf=39→40 diff found the actual input delta at rows 11–13 (frames ~35–45, front of ramp): a row-11
observer K/V membership flip and a row-13 k_d 17→18 step. The midpoint (rows 17–18) is where it
MANIFESTS via attention spreading, not where it enters. The r19 midpoint feature-match above
inspected the wrong rows. See [ekf-39-40-input-diff.md](ekf-39-40-input-diff.md).

## Falsifiable predictions (sharpen the lower boundary at rows 7/8)

- ekf=28 → first_frac_row=7 → predict CLEAN.
- ekf=32 → first_frac_row=8 → predict ERROR.

## Decisive next GPU data (both clip=90-valid)

- **0/0/35/65** (held35 / ramp30 / mid50) → first_frac_row=10 → PREDICT ERROR. If ERROR despite its
  SHORT ramp and EARLY midpoint, this kills the ramp-length (M-A) and midpoint-position (M-D)
  models and pins cause to first_frac_row (where the held block ends).
- **0/0/50/70** (held50 / ramp20 / mid60) → first_frac_row ≥ 12 → PREDICT CLEAN.
- **Boundary-sharpening:** ekf=28 (row7 → CLEAN), ekf=32 (row8 → ERROR) to nail the lower edge at
  rows 7/8.
