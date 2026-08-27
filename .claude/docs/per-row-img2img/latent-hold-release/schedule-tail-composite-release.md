<!-- provenance: theory (design — IMPLEMENTED c7afc85 + ablation combo 1fea318, branch proto-schedule-tail-release; first GPU data MIXED, NOT fully verified) -->
<!-- verified: 2026-08-27 · design session + comfy-ref source read (comfy/ldm/minimax/model.py ~587–609); GPU runs STR-1/2 (mode 'both') + STR-3/4 (mode 'rescheduled'), 0.2MP -->
# Schedule-tail composite release — DD-style unification of the official mask

## Motivating insight (source-derived)

The official H3 denoise-mask is exactly self-consistent (comfy-ref `comfy/ldm/minimax/model.py`
~587–609): the per-step clean composite `m·x + (1−m)·x₀` places a row at `(1−m·σ)x₀ + m·σ·ε` —
precisely the RF interpolation at effective sigma `m·σ` — and the label `1 − m·σ` states exactly
that. The re-inject is what MAKES the label true; composite and label are one mechanism, not two
knobs. The ghost is its designed fixed point: the clean component stays frozen at the ORIGINAL x₀
every step (the model's own predictions are discarded at rate 1−m), so the terminal step yields
the literal alpha-blend `m·x_gen + (1−m)·x₀` — a double exposure. Binary masks (m∈{0,1}) and
transient fade ramps are correct/benign; SUSTAINED fractional m used as an img2img amount knob is
off-label. The field's canonical fix is Differential Diffusion's TIMING insight: stop compositing
at the region's threshold and let it evolve (per-region SDEdit). This design is DD's timing
insight rebuilt on H3's own self-consistent clean-composite convention — which H3's
`scale_latent_inpaint` override natively supports. See [differential-diffusion](../differential-diffusion.md).

**User refinement (2026-08-27):** the official label is a RAW percentage of the current σ
(`m·σ_t` — a compressed COPY of the schedule), which is schedule-position-misaligned for
non-linear schedules (e.g. linear_quadratic at step 1: 0.5·0.987 = 0.494, a level the schedule
only reaches late). Instead, the row's label should follow the SCHEDULE's own tail: run the last
d-fraction of the schedule's σ-values, stretched over all steps.

## Mechanism

Code: `sampler.py`, `schedule_tail` branch in `build_per_row_sampler_function`; label branch in
`build_conditioning_wrapper`.

- Per row: `k_d = round(steps·(1−d))` where d = the row's envelope denoise m. Release step = k_d
  — "run the last d-fraction of the denoise schedule".
- Per step i: stretched-tail sigma `σ_row(i) = sigmas[k_d + i·(steps−k_d)/steps]` (float index,
  lerp'd on the sigma grid). `σ_row ≤ σ_glob` always.
- ONE number does both jobs: `w_i = σ_row(i)/σ_glob(i)` is BOTH the label mask (model computes
  `t_row = 1 − w·σ_glob = 1 − σ_row`) AND the composite weight — content and label are truthful
  in both phases.
- Phase 1 (i < k_d, held): pre-step official-style clean composite `w·x + (1−w)·x₀` puts the
  row's content exactly at level σ_row, source-anchored; neighbors attend to a resolving anchor.
- Release (i = k_d): the composite is simply dropped. No seam: the row already sits on a valid
  noise-line at its own tail-entry level — a legit SDEdit entry whose noise component is the
  co-evolved stream content, not fresh ε.
- Phase 2 (i ≥ k_d, free): the clean component evolves (no ghost); per-row step lerp
  `x ← x_prev + r·(x_step − x_prev)` with `r = Δσ_row/Δσ_glob` confines each row's integral to
  its flatter stretched tail. The r-lerp is applied to ALL rows every step (r=1 for full-denoise
  rows; held rows get re-projected next step anyway, but scaling them makes the handoff seamless).
- Degeneracies: d=1 → σ_row=σ_glob, w≡1, r≡1 → untouched full gen. d=0 → w≡0 → composite =
  clean every step = exact preserve (plus belt-and-braces final `torch.where(never, clean, x)`).
- Audio: the same w ratio is applied to audio rows/labels; the model computes the audio label
  `1 − w·σ_a`, composite uses the same weight → the audio analog is self-consistent per stream
  (video-ratio approximation, as in prior prototypes).
- UNIVERSAL application: every fractional row, NO provenance filter — a deliberate simplification
  for this prototype that deliberately ignores the HOLD-27 provenance lesson, to test the
  mechanism universally first. Fade ramps now also ride stretched tails instead of official
  behavior — watch for fade regressions in GPU runs.
- Surface (commit `1fea318` REPLACES the earlier `schedule_tail_release` boolean): a
  `prototype_mode` combo on H3InjectSampler (takes precedence over `per_frame_release` and
  `latent_hold_frac`) — a 2×2-ish ablation isolating remap vs composite-drop to diagnose the
  STR-2 underbake:
  - `both` (default; = prior toggle-ON behavior): schedule remap + per-step clean composite
    until release step k_d, then dropped; weight = label = σ_row/σ_glob.
  - `rescheduled`: remap only — ONE init composite at step 0 (weight w₀ places the row on its
    noise-line at σ_row(0)), no per-step re-inject; labels + per-row step lerp as in `both`.
    Per-region SDEdit on the stretched tail.
  - `mask-drop`: official mask mechanism (raw label m + per-step clean composite toward clean)
    but dropped at k_d (label → 1 after); NO schedule remap, no r-lerp.
  - `official`: in-loop emulation of the official mask mechanism — label m + per-step clean
    composite every step, never dropped; no remap. Baseline.
  - `default`: stock per-row img2img lever path (init lerp + fractional labels + denoised
    correction), unchanged.
- The per-step loop re-enters base_fn one interval at a time (Euler-appropriate, deterministic
  samplers only). No tests (prototype; user pushes with --no-validate).

## Status

Implemented `c7afc85` on branch `proto-schedule-tail-release` (recreated FRESH from `main`
7877d4d at the user's request; the earlier sha 155c911 no longer exists on any branch). The
anchor-provenance logging from the old proto branch doesn't exist on main, so the runtime
banner/redraw logging reports over ALL fractional rows (0<d<1), not just keyframe-anchor rows —
mechanism unchanged. Confirmed unrun in experiments-run before build — no prior experiment holds
a descending, co-evolving composite; all HOLD-* pins were static states. NOT fully verified —
mode `both` has one positive + one negative point (STR-1/2); the `prototype_mode` ablation combo
(`1fea318`) built to diagnose the negative delivered STR-3/4: mode `rescheduled` gives the first
fully-clean blend results, with dial CALIBRATION as the open question.

## First GPU results (2026-08-27, 0.2MP; STR-1/2 mode `both` @c7afc85, STR-3/4 mode `rescheduled` @1fea318)

Pointer rows STR-1/STR-2 in [experiments-run/hold-continued](../experiments-run/hold-continued.md).

- **STR-1** — 40 steps, both injects d=0.5: solid blend, denoise level looks correct. Minor blur
  on part of the second inject — suspected prompt/inject issue, not mechanistic.
- **STR-2** — 20 steps, inject r40 d=0.4 + r60 d=0.2: BOTH under-denoised well below dial value
  (d=0.4 visually reads as ~0.2; d=0.2 reads as ~0.1). Step-count-dependent.
- **Working hypothesis (UNVERIFIED):** phase 2 gives a row only steps·d free steps to traverse
  its ENTIRE stretched tail (release at k_d = steps·(1−d), so d=0.2 @ 20 steps = 4 free steps) —
  an under-discretized tail reads as underbake; 40 steps @ d=0.5 = 20 free steps looked fine.
- **STR-3** — mode `rescheduled`, injects d=0.4/0.2 (steps not restated by user; PRESUMED 20 as
  in STR-2): both injects somewhat TOO denoised (overbake vs dial) but smooth and well blended —
  no errors, no seams, no blending issues.
- **STR-4** — mode `rescheduled`, d=0.2/0.1: d=0.2 still slightly too denoised; d=0.1 slightly
  too LITTLE. No errors or blending issues. User notes their own denoise calibration may be a
  factor.
- **Implication (theory, UNVERIFIED):** STR-3/4 WEAKEN the STR-2 under-discretization hypothesis
  as the primary underbake cause. `rescheduled` and `both` place a row at the SAME level
  σ_row(k_d) by the release step and share an identical post-release tail discretization; they
  differ only in whether the first k_d steps are genuine model evolution (`rescheduled` — redraw
  accrues) or per-step clean-composite anchoring (`both` — content stays source-pinned). `both`
  underbakes at the same dial where `rescheduled` overbakes ⇒ the held-phase composite
  anchoring, NOT tail discretization, is the dominant driver of the `both` underbake. The open
  question is now DIAL CALIBRATION: `rescheduled` effective img2img strength is sigmas[k_d]
  (shift-12 top-heavy: d=0.4→σ≈0.89, d=0.2→≈0.75, d=0.1→≈0.57), which reads stronger than the
  dial suggests at mid d — yet d=0.1 read as too WEAK, so perceived-redraw vs σ_eff is nonlinear.

## How it differs from prior attempts

HOLD-26/27 pinned a STATIC state (fixed renoise at level L) and moved release timing — HOLD-27
showed timing levers can't buy level fidelity without a displacement clamp, plus a
provenance-blind floor bug. This design instead keeps the row ON a truthful noise-line
continuously via the descending co-evolving composite (level always right by construction), and
timing is not a separate knob at all: level and timing both derive from d through the stretched
tail. Falsified predecessors: [min-free-steps-floor](min-free-steps-floor.md) and
[per-frame-scheduled-release](per-frame-scheduled-release.md).
