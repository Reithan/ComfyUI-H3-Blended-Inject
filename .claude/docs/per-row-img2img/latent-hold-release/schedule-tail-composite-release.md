<!-- provenance: status (index for the schedule-tail composite-release thread; children carry design/theory/results tags) -->
<!-- verified: 2026-08-27 · GPU runs STR-1..8; design + comfy-ref source read (comfy/ldm/minimax/model.py ~587–609) -->
# Schedule-tail composite release — DD-style unification of the official mask (index)

Design: the official H3 clean-composite and its `1−m·σ` label are ONE self-consistent mechanism
(the composite makes the label true; the ghost is its terminal alpha-blend). This thread rebuilds
Differential Diffusion's timing insight on that convention: per-row stretched schedule-tail
sigmas `σ_row(i) = sigmas[k_d + i·(steps−k_d)/steps]` with `k_d = round(steps·(1−d))`, one
weight `w = σ_row/σ_glob` serving as BOTH label and composite, release = drop the composite.
Implemented `c7afc85` on branch `proto-schedule-tail-release` (recreated FRESH from `main`
7877d4d at the user's request; earlier sha 155c911 no longer exists); ablation combo `1fea318`.

**Status (2026-08-27, after STR-1..8):** mode `rescheduled` (per-region SDEdit on the stretched
tail — init-only composite at σ_row(0), remapped labels + per-row step lerp, no per-step
re-inject) is the WORKING mechanism, and by **user decision (2026-08-27) is expected to become
the CANONICAL implementation** barring new bugs or quality problems — i.e. plan for owned,
tested, first-class code on this path rather than treating it as a prototype branch. It gives
clean blends across 0.2MP and 0.5MP, a calibratable dial, and no seams or errors in any run so
far. Dial calibration remains top-heavy (σ_eff = sigmas[k_d]) and
mildly resolution-dependent, but usable by eye. Mode `both` remains underbake-prone (held-phase
composite anchoring — STR-2 vs STR-3). Mode `mask-drop` FAILED (STR-8: naive DD-style drop on
raw-percentage labels falsified for H3; the remap is the seamless-release ingredient).
`official` ablation leg not yet run.

## Child docs

- [design-and-mechanism](schedule-tail-composite-release/design-and-mechanism.md) — the
  self-consistency insight, the schedule-tail refinement, the full mechanism (w, phases,
  release, r-lerp, degeneracies, audio, universal application), and how it differs from the
  falsified HOLD-26/27 static-pin approach.
- [ablation-modes](schedule-tail-composite-release/ablation-modes.md) — the `prototype_mode`
  combo (`1fea318`, replaces the `schedule_tail_release` boolean): `both` / `rescheduled` /
  `mask-drop` / `official` / `default`, plus the all-fractional-rows logging note.
- [consistency-audit](schedule-tail-composite-release/consistency-audit.md) — source-verified
  three-channel (label/content/step) audit of `rescheduled`: exact for video under Euler; the
  audio stream's carried coordinate breaks three ways at FRACTIONAL audio ticks (already exercised
  by the test video's opening fade — real but mild, fix deferred until video settles), plus five
  negligible/known divergences.
- [multistep-stochastic-support](schedule-tail-composite-release/multistep-stochastic-support.md)
  — design sketch (unimplemented): why `dpmpp_2m` silently runs Euler under the loop today, and
  how recovering per-element `denoised` from the Euler slice lets the loop run a per-row update
  rule for `dpmpp_2m`, `res_multistep`, and `euler_ancestral_RF` (the stochastic gap).
- [gpu-results](schedule-tail-composite-release/gpu-results.md) — the STR-1..7 run log
  (`both` underbake, `rescheduled` clean-blend series incl. the 0.5MP res-invariance test),
  the anchoring-dominates implication, and dial-calibration notes.

Predecessors (falsified): [min-free-steps-floor](min-free-steps-floor.md),
[per-frame-scheduled-release](per-frame-scheduled-release.md). Ghost/DD background:
[differential-diffusion](../differential-diffusion.md).
