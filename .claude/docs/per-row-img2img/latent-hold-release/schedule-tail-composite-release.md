<!-- provenance: theory (design — IMPLEMENTED commit 155c911 branch proto-schedule-tail-release, NOT GPU-verified) -->
<!-- verified: 2026-08-27 · design session + comfy-ref source read (comfy/ldm/minimax/model.py ~587–609); no GPU run yet -->
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
- Surface: `schedule_tail_release` BOOLEAN toggle on H3InjectSampler (default OFF; takes
  precedence over `per_frame_release` and `latent_hold_frac`). The per-step loop re-enters
  base_fn one interval at a time (Euler-appropriate, deterministic samplers only). No tests
  (prototype; user pushes with --no-validate).

## Status

Implemented `155c911` on branch `proto-schedule-tail-release` (branched from
proto-latent-hold-release for the loop plumbing). NOT GPU-verified. Confirmed unrun in
experiments-run before build — no prior experiment holds a descending, co-evolving composite; all
HOLD-* pins were static states.

## How it differs from prior attempts

HOLD-26/27 pinned a STATIC state (fixed renoise at level L) and moved release timing — HOLD-27
showed timing levers can't buy level fidelity without a displacement clamp, plus a
provenance-blind floor bug. This design instead keeps the row ON a truthful noise-line
continuously via the descending co-evolving composite (level always right by construction), and
timing is not a separate knob at all: level and timing both derive from d through the stretched
tail. Falsified predecessors: [min-free-steps-floor](min-free-steps-floor.md) and
[per-frame-scheduled-release](per-frame-scheduled-release.md).
