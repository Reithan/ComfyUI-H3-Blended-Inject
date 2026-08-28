<!-- provenance: theory (UNVERIFIED — analytical/CPU-side, no GPU confirmation) -->
<!-- verified: 2026-08-28 · CPU numerical characterization of the schedule-tail remap; GPU verification pending -->
# Long-fade video interference — 17-frame grid-beat theory (Bug E RCA)

Bug E symptom: moiré / streamers / ribbons / electric-like patterns in video latent with long
fades (~60f); absent at short fades (~30f); sampler-independent; audio tracks via joint attention.
Established in [bugs.md](bugs.md) and [audio-axis-verdict.md](audio-axis-verdict.md).

This doc records the leading mechanistic hypothesis from CPU numerical characterization.
It is a THEORY: the arithmetic is consistent; GPU confirmation is required before treating this
as the definitive cause.

Note: the σ̃/H2 and Fix A work does NOT address Bug E — Bug E is present on main, before either
fix, and is sampler-independent.

## Leading mechanism — 17-frame grid-beat (UNVERIFIED)

The proposed cause is a BEAT between the per-row fade denoise ramp and H3's 17-frame latent
grid structure.

The grid maps frames to rows as (1,4,4,4,4) per 17-frame chunk (grid.py:138-168).
Every 5th row is a 1-frame "boundary" row that samples the m-ramp at a SINGLE point.
The four surrounding rows each average over 4 frames (envelope.py:125-243 averages
row_center_times).
This imposes a periodic (period = 5 rows = 17 frames) non-uniform sampling of the fade ramp
onto k_d and the per-row sigma-labels.

k_d = round(steps*(1-m)) (sampler.py:496) and the dense-grid idx (sampler.py:521) embed that
periodic modulation into per-row conditioning.
H3's full joint attention (no mask, [dit-forward.md](native-h3-mechanism/dit-forward.md))
renders it as a periodically-structured latent → visible moiré, which couples into audio.

## Why fade length is the variable (the crux)

A 60f fade spans ~4 complete grid cycles (20 rows = 4×5).
A 30f fade spans only ~2.2 cycles (11 rows).
~4 repetitions is enough to resolve a spatial frequency as a visible moiré; ~2 is not.

This is the classic two-near-commensurate-periods (ramp vs 17-frame grid) moiré origin.
The SAME per-period structure is sub-perceptual at 30f and perceptible at 60f.

## Corroborating detail — stuck pairs at 60f (secondary, not the whole story)

At 60f the k_d steps are fine (avg Δk_d≈1.2/row), producing two "stuck pairs": adjacent rows
straddling a chunk boundary get IDENTICAL k_d despite different content.
Rows 4&5 (frames 16/17 boundary) and rows 15&16 (frames 51/52) are the stuck pairs.
At 30f (coarse steps, avg Δk_d≈3) NO stuck pairs occur.

Secondary note: H3 VAE encodes each 17-frame chunk INDEPENDENTLY (grid.py:3).
A long partial-denoise ramp crossing 3-4 independent chunks could add inter-chunk phase mismatch
on the same 17-frame period.
This variant shares the period and is hard to separate from the sampling-periodicity story
without GPU.

## Alternatives contradicted by the arithmetic

**k_d release-band quantization ("more bands = more moiré"): CONTRADICTED.**
30f has COARSER k_d steps (Δk_d≈3) than 60f (Δk_d≈1.2).
If coarse banding caused it, 30f would be WORSE. It is clean. Banding per se is not the cause.

**Stretched-tail row-crossing / non-monotonic sigma: CONTRADICTED by algebra.**
idx(A,i)−idx(B,i) = (k_A−k_B)(steps−i) ≥ 0 for all i, so per-row sigma order is strictly
monotone in k_d; no crossings exist.

## Falsifiable predictions — discriminating GPU experiments

1. **Fade-length sweep** (30/40/50/60/70f, all else fixed): artifact onset should track
   GRID-CYCLE COUNT (rows_in_fade / 5), NOT k_d-band count. Predict onset around ~3 cycles.

2. **Spatial/temporal period check**: the visible interference should have a ~17-frame
   (≈5-latent-row) period. If the pattern repeats on that period, this strongly confirms grid-beat.

3. **Fade-endpoint snap / phase shift**: snapping fade endpoints to 17-frame chunk boundaries,
   or phase-shifting a fixed-length fade, should CHANGE or reduce the pattern if grid-beat is
   the cause; should NOT change it if the cause were pure k_d banding.

## Fix directions (SPECULATIVE — confirm the mechanism first)

Do not implement until the mechanism is GPU-confirmed. Each targets a different part of the theory:

(a) Break the periodic sampling bias — evaluate the ramp so boundary (1-frame) and delta
    (4-frame) rows are sampled consistently.

(b) Continuous per-row release — replace round() with dither/interpolation on k_d to eliminate
    stuck pairs.

(c) Constrain or snap fade geometry to whole grid cycles.

Lead with confirming the mechanism via the GPU experiments above, NOT with a code fix.
