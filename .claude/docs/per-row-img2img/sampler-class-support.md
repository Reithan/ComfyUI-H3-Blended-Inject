<!-- provenance: theory (design + one SOURCE-CONFIRMED finding; per-row steps UNVERIFIED, no GPU) -->
<!-- verified: 2026-08-27 · comfy-ref k_diffusion/sampling.py @b78cec87 lines 240-266 (euler_ancestral_RF), 796-818 (dpmpp_2m), 1394+ (res_multistep); repo sampler.py build_per_row_sampler_function -->
# Sampler-class support under the schedule-tail remap (stochastic + multi-step)

**Status: discussion-stage design, nothing built, direction not user-confirmed.**
See also: [bugs.md Bug B](bugs.md#bug-b) · [stochastic-recovery-theory.md](stochastic-recovery-theory.md)
· [audio-carry-identity.md](audio-carry-identity.md) · [PER_ROW_IMG2IMG_NOTES.md](../PER_ROW_IMG2IMG_NOTES.md)

## Finding 1 (SOURCE-CONFIRMED, 2026-08-27): multistep samplers silently degrade to first order

The shipped remap loop (`sampler.py::build_per_row_sampler_function`) calls
`base_fn(model, x, sigmas[i:i+2], ...)` one global interval at a time.
History-based samplers re-initialize `old_denoised = None` at function entry —
`dpmpp_2m` (sampling.py:796-818) and `res_multistep` (1394+) both follow this pattern —
so with a 2-sigma slice they take their first-order fallback branch on **every step**.

Under the remap, dpmpp_2m and res_multistep therefore run as first-order (euler/DDIM-like)
**everywhere**, including free m=1 rows. Not a correctness bug — outputs are valid first-order
trajectories, and this was true for all GPU validation runs of the shipped mechanism. But
"supported: dpmpp_2m/res_multistep" currently means "runs as first-order," an accuracy
regression vs the retired three-lever path where `base_fn` ran the full schedule once.

## Finding 2: Bug B persists in remap form

For stochastic samplers the remap's per-row r-scaling `x_cur = x_prev + r·(x_cur − x_prev)`
linearly rescales a displacement that contains the renoise term.
`sample_euler_ancestral_RF` (sampling.py:240-266) has affine alpha terms:
`alpha_ip1/alpha_down = (1−σ_{i+1})/(1−σ_down)` as the x-multiplier, and
`renoise_coeff = (σ_{i+1}² − σ_down²·α_{i+1}²/α_down²)^½` — neither scales linearly under
`σ → σ_row`, so fractional/preserved rows are still corrupted.
Same root cause as [Bug B](bugs.md#bug-b); the mechanism changed but the crack is unchanged.

## Design (UNVERIFIED): per-row step functions replace the black-box base_fn

The remap loop already owns exact per-row sigma tensors (`sig_row`, `sig_row_next` from the
dense steps²+1 grid; audio rows pre-shifted via `_shift_schedule`). Both sampler classes become
supportable by replacing the per-interval `base_fn` call with in-house step functions written
elementwise over those tensors; the r-scaling then applies only to a generic-fallback path.

This **simplifies** [stochastic-recovery-theory.md](stochastic-recovery-theory.md): no
"corrected denoised" identity needed — the truthful `w` labels already make the model return
per-row velocity `v_r`. Recover it as `v = (x − denoised)/σ_g` (comfy wrapper divides by global
σ) and form `denoised_r = x − σ_row·v`.

### Per-row RF-ancestral (kills Bug B)

One model eval per interval. Then elementwise with tensor σ_i = sig_row, σ_ip1 = sig_row_next:

- `downstep_ratio = 1 + (σ_ip1/σ_i − 1)·eta`; `σ_down = σ_ip1·downstep_ratio`
- Euler sub-step with `denoised_r`; renoise with per-row alpha terms and coeff.
- m=0 rows: clamp σ_i to ε ⇒ ratio→0, σ_down→0, coeff→0 ⇒ row frozen; no NaN reaches the model.
- Replicate `s_noise·noise_scale` (model_sampling attr) and
  `default_noise_sampler(x, seed=extra_args["seed"])`.

### Per-row multistep (restores 2nd order for dpmpp_2m / res_multistep, including free rows)

Same formulas with tensor sigmas (`t = −log σ`, phi functions broadcast elementwise). Carry
per-row `old_denoised` / `old_sigma_down` across our loop iterations. Clamp σ ≥ ε before log
for m=0 rows (final `where(never, clean, ·)` still guards the output).

### Dispatch

Whitelist by sampler name → native per-row step. Unknown deterministic → current wrap+r-scale
fallback (first-order-correct). Unknown stochastic → keep `sampler_is_stochastic` warning.

### Deferred: within-interval multi-eval samplers

Heun, dpm_2, dpmpp_2s_ancestral, dpmpp_sde additionally need pooled-label refresh before each
inner model eval (`w_mid = σ_row_mid/σ_carrier`). Doable since we own the loop, but extra
plumbing; deferred.

## Risks and verification notes

- **Leak risk** unchanged from stochastic-recovery-theory: internal model σ-dependence beyond the
  label channel would break the identity; a GPU spike (euler_ancestral, 39f fade case) is the test.
- **Audio:** ancestral alpha terms on the SHIFTED audio σ need a pass through
  [audio-carry-identity.md](audio-carry-identity.md) before trusting fractional-audio + stochastic.
- **Maintenance:** each native step is a small reimpl that can drift from comfy upstream.
- **Status:** discussion-stage design; nothing built; direction not yet user-confirmed.

## Delivery plan (3 PRs, tasks #66–#70)

Ordering rationale: stochastic first — it is the user-visible failure and validates the leak risk
before investing in multistep work.

**PR1 `refactor-per-row-step-dispatch` (task #66)** — behavior-identical refactor only.
Restructures `build_per_row_sampler_function` to a per-row step-function protocol + dispatch by
sampler name. Native per-row euler step is numerically equivalent to the current wrap+r-scale for
euler; all other samplers keep the wrap+r-scale fallback unchanged. CPU equivalence tests:
m=0 / fractional / m=1, audio-shifted rows. No GPU gate — zero behavior change.

**PR2 `add-per-row-rf-ancestral-step` (task #67)** — per-row RF-ancestral step.
Elementwise `sample_euler_ancestral_RF` formulas over tensor σ_row; σ clamped ≥ε so m=0 rows
freeze; seeded `default_noise_sampler`; `s_noise·noise_scale` replicated. Whitelisted
euler_ancestral stops the stochastic warning; non-whitelisted stochastic still warns.
CPU tests vs scalar reference; analytical audio-carry-identity check for shifted-σ audio rows.
**Merge gated on USER GPU spike (task #68):** 39f fade Bug-B repro + min_denoise 0.2–0.3
checklist — this is the leak-risk test (hidden σ-dependence beyond the label channel).

**PR3 `add-per-row-multistep-steps` (task #69)** — per-row dpmpp_2m + res_multistep.
Deterministic (eta=0) form; per-row `old_denoised` history carried across the outer loop — fixes
Finding 1's first-order degradation. Key test: all-m=1 rows over a full schedule must reproduce
the stock sampler bit-for-bit on a toy model. Gated on USER GPU quality check (task #70) at the
d=0.2/0.15 sweet-spot config.
