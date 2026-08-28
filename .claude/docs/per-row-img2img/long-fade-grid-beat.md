<!-- provenance: theory (CPU-EXACT separator S2, GPU out-of-sample PENDING — code-exact across 13 tested configs; no GPU out-of-sample confirmation yet) -->
<!-- verified: 2026-08-28 · CPU numerical analysis of frame_to_row/evaluate_envelope/sampler; source repo @ branch fix-audio-ancestral-axis-mismatch -->
# Bug E: long-fade video interference — chunk-seam cell-alignment (S2)

Bug E symptom: moiré / streamers / ribbons / electric-like patterns in video latent with long
fades; sampler-independent; audio tracks via joint attention.
Established in [bugs.md](bugs.md) and [audio-axis-verdict.md](audio-axis-verdict.md).

Note: σ̃/H2 and Fix A do NOT address Bug E — present on main before either fix, sampler-independent.

## Separator S2 — code-exact predictor (13/13)

The artifact occurs **iff** the fade-out ramp-start row — the latent row containing `ekf`
(end_keyframe), where the frozen held block meets the ramp — is **adjacent to a chunk reset row**.

H3 grid: (1,4,4,4,4) per 17-frame chunk. Each chunk has 5 rows: `local_row = global_row % 5`.

| local_row | position in chunk | S2 |
|-----------|-------------------|----|
| 0 | chunk reset / anchor (1-frame cell) | SAFE |
| 1 | immediately AFTER reset | ERROR |
| 2 | interior (4-frame cell) | SAFE |
| 3 | interior (4-frame cell) | SAFE |
| 4 | immediately BEFORE next reset | ERROR |

**S2 = True iff `local_row ∈ {1, 4}`, i.e. `global_row % 5 ∈ {1, 4}`.**

## Evidence — 13/13 perfect separation

Notation: a/b/c/d = fade_in/skf/ekf/fade_out. Ramp-start `global_row = frame_to_row(ekf)`.

| Config | ramp-start row | cell | S2 | GPU result |
|--------|---------------|------|----|------------|
| 0/0/0/85,17,69,73,75 (c=0) | row 0 | 0 (reset) | F | CLEAN |
| 0/0/7/75 (c=7) | row 2 | 2 | F | CLEAN |
| 0/0/40/90 (c=40) | row 12 | 2 | F | CLEAN |
| 0/0/44/90 (c=44) | row 13 | 3 | F | CLEAN |
| 0/0/60/90 (c=60) | row 18 | 3 | F | CLEAN |
| 0/0/35/90 (c=35) | row 11 | 1 | T | ERROR (streamers) |
| 0/0/36/90 (c=36) | row 11 | 1 | T | ERROR (geometric) |
| 0/0/37/90 (c=37) | row 11 | 1 | T | ERROR (green electric) |
| 0/0/30/90 (c=30) | row 9 | 4 | T | ERROR (streamers; original error case) |

S2 predicts all 13 cases correctly. GPU out-of-sample confirmation is PENDING.

## Prior theory retired — "held+long-ramp mid-schedule disparity" was a confound

The previous revision hypothesized that BOTH a substantial held block AND a long ramp were
required. S2 shows this was a confound.

The original pair that suggested it:
- 0/0/30/90 (held 30f, ramp 60f → NOISY): ramp-start row 9, cell 4 → S2=T
- 0/0/60/90 (held 60f, ramp 30f → CLEAN): ramp-start row 18, cell 3 → S2=F

The outcome differs by WHERE `ekf` lands in the chunk — NOT by held size or ramp length.
**Held+ramp framing is retired as a confound explained by S2. Do not revisit it.**

## Structural, not a denoise/k_d-level effect

Config c=37: ramp-start row 11, `k_d = 20` (same as a fully-frozen row; denoise ≈ 1.0) → still
ERRORS. The artifact is not driven by the denoise level at the ramp-start row. It is the geometric
coincidence of the frozen→ramping boundary landing on a reset-adjacent cell. Interior cells 2–3
keep the boundary inside the chunk, with frozen context buffering both sides of every seam.

## Mechanism — chunk-seam decode-overlap (same confirmed H3 failure mode)

H3 VAE encodes each 17-frame chunk independently and DECODES with overlapping blended windows.
This chunk-boundary decode blend is the **confirmed root of the task #25 "pop"** in this codebase
— see [highres-singleframe-underdenoise/resolution-ladder.md](highres-singleframe-underdenoise/resolution-ladder.md)
(memory key `h3-vae-decode-overlap`).

Placing the frozen→ramping discontinuity on a reset-adjacent cell (1 or 4) lands the abrupt signal
jump exactly at a chunk seam. The overlapping decode pass blends that jump into adjacent decoded
frames → structured visual noise. Cells 2–3 (interior) keep the discontinuity ≥2 rows inside the
chunk; frozen context buffers both sides of every seam and the decode blend sees a smooth signal.

## Pending GPU out-of-sample predictions (UNRUN)

| Config | ramp-start row | cell | Prediction |
|--------|---------------|------|------------|
| 0/0/47/90 (c=47) | row 14 | 4 | ERROR — confirms cell4 independent of c=30 |
| 0/0/34/90 (c=34) | row 10 | 0 (reset) | CLEAN |
| 0/0/38/90 (c=38) | row 11 | 1 | ERROR |
| 0/0/39/90 (c=39) | row 12 | 2 | CLEAN — pins error cliff at the cell1→cell2 boundary |

The c=38/c=39 pair pins the exact transition: last frame of cell1 → first frame of cell2.

## Fix direction (speculative — confirm GPU out-of-sample first)

Snap `ekf`/`efo` (and by symmetry `sfi`/`skf`) OFF the reset-adjacent cells (1 and 4), to the
reset cell (0) or interior cells (2–3). This is analogous to the existing `inject_at` 17-frame
floor-snap in `sanitize.py` (`snap_inject_at`, lines 26–68), which ensures `inject_at` lands on
a valid chunk boundary. A fade-endpoint snap would keep the frozen→ramp boundary away from
every chunk seam.

**Do not implement until GPU out-of-sample predictions are confirmed.**
