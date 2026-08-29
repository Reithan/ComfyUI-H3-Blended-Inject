<!-- provenance: confirmed (CPU discrete/threshold/segmentation surface exhaustively audited; no code-side gate exists — two independent passes both negative) -->
<!-- verified: 2026-08-28 · Fable numeric replication (round-1 + round-2) + Sonnet mechanism auditor; both negative -->
# Bug E — CPU discrete surface: exhaustively audited, no code-side gate

Parent: [long-fade-grid-beat.md](../long-fade-grid-beat.md).

Two independent passes — a Fable numeric replication and a Sonnet mechanism auditor — both came
back negative. Every discrete surface is now opened and FALSIFIED as the binary gate. The CPU-side
discrete/threshold/segmentation surface is FULLY EXHAUSTED.

## Round-1: code-quantization sweep — FALSIFIED (Fable)

A Fable agent computed the full per-row schedule for 13 clip=90 configs using the REAL
`evaluate_envelope` + `k_d = round(20*(1-m)).clamp(0,20)`. It swept ~20 discrete quantities:
SNAP20 (ramp row rounding to k_d=20 → preserve/never=True → hard clean-restore at sampler.py:638),
SNAP0 (ramp row k_d=0, w saturates at 1.0), k_d collisions, idx collisions (sampler.py:521),
w saturation (:592), round-half ties, etc.

- **Error configs (8):** 0/0/30/85, 0/0/30/90, 0/0/34/90, 0/0/35/90, 0/0/36/90, 0/0/37/90,
  0/0/38/90, 0/0/39/90.
- **Clean configs (5):** 0/0/15/70, 0/0/25/80, 0/0/40/90, 0/0/44/90, 0/0/47/90.

Verdict: no CPU-side rounding/snapping/collision quantity separates error from clean. Endpoint-snap
conjunction (SNAP20 AND NOT SNAP0) and its XOR refinement are BOTH falsified two-sided:
- 0/0/36/90 is ERROR with NO snap of any kind (no k_d=20, no k_d=0, zero collisions).
- 0/0/47/90 is CLEAN despite preserve-snap (row 13, m=0.0028 → k_d=20 → hard restore at
  sampler.py:638) that every snap-bearing error config carries.

k_d collisions fail independently: errors 36/37/39 have zero; clean 15/70 has one. The only features
separating the 13 are monotone in ekf — trivial restatements of "held-end frame ~26–39", not
mechanisms; the contiguous error bracket means any ekf-monotone quantity manufactures a window.

## Round-2 + auditor: every discrete surface opened & falsified

- **Observer split (observer_split.py):** single contiguous observer segment; n_unique(t_obs) ==
  n_frac (ZERO dedup collapse) in all 14 configs. A strictly-monotone linear ramp structurally
  CANNOT produce a degenerate segmentation or dedup merge; the 1e-6/1e-3 tolerances
  (observer_split.py:69) exclude nothing. No layout mismatch ((b−a)==stream["n"]==27 always).
- **_denoise_mask_values / make_pooled (nodes.py:253, comfy model_base.py):** quantizes to 1/256;
  idempotent for pre-quantized per-row m, uniform per-step for w. Cannot discriminate on ekf.
- **Dense sigma grid (sampler.py:519–522):** exact integer indexing, no structural boundary at
  midpoint indices.
- **frame_to_row / evaluate_envelope (grid.py, envelope.py):** correct integer arithmetic, no
  boundary bug.
- **Fable round-2 feature families:** midpoint-row features, second-order (never-rows-removed)
  features, and k_d-ladder-shape features ALL overlap both labels. The midpoint row r19 is
  identical in the killer pair 0/0/39/90 (ERROR) vs 0/0/40/90 (CLEAN) (same k_d=9).

**RETRACTED (2026-08-28):** the earlier claim that this killer pair is "CPU-identical on every
discrete feature" is WRONG. An executed per-row/per-step diff found the delta is rows 11 & 13 at
the HOLD→FADE SEAM (row-11 observer K/V membership flip + row-13 k_d 17→18), invisible to
midpoint-centric features. See [ekf-39-40-input-diff.md](ekf-39-40-input-diff.md).

## Conclusion

The binary artifact is NOT explained by CPU-side quantization or segmentation. This CONFIRMS the
binary character ("happens or doesn't", no gradient) while locating the gate DOWNSTREAM in GPU
model/attention dynamics — the DiT's discrete response to where the held prefix ends in the row
grid. See [first-frac-row.md](first-frac-row.md) for the perfect CPU proxy and the GPU next-steps.
