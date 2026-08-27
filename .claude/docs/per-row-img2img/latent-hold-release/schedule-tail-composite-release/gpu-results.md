<!-- provenance: confirmed (GPU runs STR-1..8, 2026-08-27) + theory (hypotheses/implications marked UNVERIFIED) -->
<!-- verified: 2026-08-27 · GPU runs STR-1..8 (user test deployment; modes 'both' @c7afc85, 'rescheduled'/'mask-drop' @1fea318) -->
# Schedule-tail GPU results — STR series log

Child of [schedule-tail-composite-release](../schedule-tail-composite-release.md). Pointer rows
live in [experiments-run/hold-continued](../../experiments-run/hold-continued.md).

## Mode `both` (c7afc85, 0.2MP, 2026-08-27)

- **STR-1** — 40 steps, both injects d=0.5: solid blend, denoise level looks correct. Minor blur
  on part of the second inject — suspected prompt/inject issue, not mechanistic.
- **STR-2** — 20 steps, inject r40 d=0.4 + r60 d=0.2: BOTH under-denoised well below dial value
  (d=0.4 visually reads as ~0.2; d=0.2 reads as ~0.1). Step-count-dependent.
- **Initial hypothesis (UNVERIFIED, now WEAKENED by STR-3/4):** phase 2 gives a row only steps·d
  free steps to traverse its ENTIRE stretched tail (release at k_d = steps·(1−d), so d=0.2 @ 20
  steps = 4 free steps) — an under-discretized tail reads as underbake; 40 steps @ d=0.5 = 20
  free steps looked fine.

## Mode `rescheduled` (1fea318, 2026-08-27)

- **STR-3** — 0.2MP, injects d=0.4/0.2 (steps not restated by user; PRESUMED 20 as in STR-2):
  both injects somewhat TOO denoised (overbake vs dial) but smooth and well blended — no errors,
  no seams, no blending issues. First fully-clean blend result.
- **STR-4** — 0.2MP, d=0.2/0.1: d=0.2 still slightly too denoised; d=0.1 slightly too LITTLE.
  No errors or blending issues. User notes their own denoise calibration may be a factor.
- **STR-5** — 0.2MP, injects d={0.2, 0.15}: solid result, matches the user's intent "spot-on" —
  the first dial-calibrated fully-satisfying result.
- **STR-6** — same settings at 0.5MP (explicit resolution-invariance test): very solid; the
  d=0.2 inject possibly slightly under-denoised but very close. 0.5MP output notably higher
  quality overall, for the ordinary higher-res reasons. Resolution invariance of the mechanism
  effectively holds; only a mild resolution sensitivity of perceived denoise (slightly under at
  higher res at the same dial).
- **STR-7** — 0.5MP, d={0.3, 0.2}: "almost perfect."

## Mode `mask-drop` (1fea318, 2026-08-27)

- **STR-8** — settings otherwise as the recent STR runs: FAILURE — audio freak-out, color
  flashing in the video fade-in region, bad inject blending.
- **Interpretation (theory, UNVERIFIED but design-predicted):** `mask-drop` releases WITHOUT the
  schedule remap, so at each row's release step the label jumps discontinuously from m to 1
  while the row's content sits at effective level ~m·σ — a level/label lie plus a per-row label
  discontinuity. Fade regions have per-row d, so their release steps cascade across rows → the
  observed per-row color flashing in the fade-in. Audio rides the same labels through the
  σ-shift (12→3) nonlinearity, amplifying the jump → the audio freak-out. This is the
  discriminator confirming the schedule remap (truthful σ_row labels + per-row step lerp) is
  precisely what makes release seamless in `rescheduled`/`both`: there, the row is ON its
  labeled noise-line at release so nothing jumps. Strengthens `rescheduled` as the mechanism;
  naive DD-style drop on raw-percentage labels is FALSIFIED for H3.

## Implications

- **Anchoring dominates the `both` underbake (theory, UNVERIFIED):** STR-3/4 WEAKEN the STR-2
  under-discretization hypothesis as the primary cause. `rescheduled` and `both` place a row at
  the SAME level σ_row(k_d) by the release step and share an identical post-release tail
  discretization; they differ only in whether the first k_d steps are genuine model evolution
  (`rescheduled` — redraw accrues) or per-step clean-composite anchoring (`both` — content stays
  source-pinned). `both` underbakes at the same dial where `rescheduled` overbakes ⇒ the
  held-phase composite anchoring, NOT tail discretization, is the dominant driver.
- **Dial calibration:** `rescheduled` effective img2img strength is sigmas[k_d] (shift-12
  top-heavy: d=0.4→σ≈0.89, d=0.2→≈0.75, d=0.1→≈0.57), which reads stronger than the dial
  suggests at mid d — yet d=0.1 read as too WEAK, so perceived-redraw vs σ_eff is nonlinear.
  After STR-5..7 the dial is top-heavy but usable BY EYE, and mildly resolution-dependent
  (slightly under-reads at higher res at the same dial).
- **Status after STR-1..8:** `rescheduled` (per-region SDEdit on the stretched schedule tail —
  init-only composite at σ_row(0), remapped labels + per-row step lerp, no per-step re-inject)
  is the WORKING candidate mechanism: clean blends across 0.2MP and 0.5MP, calibratable dial, no
  seams/errors in any run. `both` remains underbake-prone (held-phase composite anchoring,
  STR-2 vs STR-3). `mask-drop` FAILED (STR-8: naive drop on raw-percentage labels falsified,
  remap confirmed as the seamless-release ingredient). `official` ablation leg not yet run.
