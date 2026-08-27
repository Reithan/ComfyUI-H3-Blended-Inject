<!-- provenance: reference (implemented surface — commit 1fea318 branch proto-schedule-tail-release; replaces the schedule_tail_release boolean) -->
<!-- verified: 2026-08-27 · coordinator relay of commit 1fea318; not independently source-read -->
# `prototype_mode` ablation combo (1fea318)

Child of [schedule-tail-composite-release](../schedule-tail-composite-release.md).

Commit `1fea318` REPLACES the earlier `schedule_tail_release` boolean with a `prototype_mode`
combo on H3InjectSampler (takes precedence over `per_frame_release` and `latent_hold_frac`) —
a 2×2-ish ablation isolating schedule remap vs composite-drop, built to diagnose the STR-2
underbake (see [gpu-results](gpu-results.md)).

## Modes

- `both` (default; = prior toggle-ON behavior): schedule remap + per-step clean composite until
  release step k_d, then dropped; weight = label = σ_row/σ_glob.
- `rescheduled`: remap only — ONE init composite at step 0 (weight w₀ places the row on its
  noise-line at σ_row(0)), no per-step re-inject; labels + per-row step lerp as in `both`.
  Per-region SDEdit on the stretched tail. **Working candidate after STR-3..7.**
- `mask-drop`: official mask mechanism (raw label m + per-step clean composite toward clean)
  but dropped at k_d (label → 1 after); NO schedule remap, no r-lerp.
- `official`: in-loop emulation of the official mask mechanism — label m + per-step clean
  composite every step, never dropped; no remap. Baseline.
- `default`: stock per-row img2img lever path (init lerp + fractional labels + denoised
  correction), unchanged.

## Notes

- Logging: the anchor-provenance logging from the old proto branch doesn't exist on main, so the
  runtime banner/redraw logging reports over ALL fractional rows (0<d<1), not just
  keyframe-anchor rows — mechanism unchanged.
- Ablation legs `mask-drop` and `official` not yet run (as of STR-7).
