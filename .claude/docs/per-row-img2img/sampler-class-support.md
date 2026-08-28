<!-- provenance: theory + confirmed (Finding 2 GPU-CONFIRMED for euler_ancestral 2026-08-27; Finding 1 SOURCE-CONFIRMED; PR3/PR4 design UNVERIFIED) -->
<!-- verified: 2026-08-27 · PR2 GPU pass task #68 @ede2d8c -->
<!-- source: comfy-ref k_diffusion/sampling.py @b78cec87: 240-266 euler_ancestral_RF, 738-792 dpmpp_sde,
     796-818 dpmpp_2m, 822-873 dpmpp_2m_sde, 68-176 helpers, 1394+ res_multistep; repo sampler.py, observer_split.py -->
# Sampler-class support under the schedule-tail remap (stochastic + multi-step)

**Status: PR2 shipped + GPU-CONFIRMED (task #68, 2026-08-27); PR1 refactor pending; PR3/PR4 design stage.**
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

## Finding 2 (GPU-CONFIRMED 2026-08-27): Bug B killed for euler_ancestral

**CONFIRMED:** PR2 (`add-per-row-rf-ancestral-step`, shipped ede2d8c) GPU-validated in task #68.
Config: 0.2MP, two injects at fractional min_denoise d=0.20 and d=0.15, fade-in at start, 20
steps; both `euler` (deterministic baseline) and `euler_ancestral` (new RF-ancestral path) tested.
Result: no grey static / no corruption on fractional or preserved rows, no ghosting;
euler_ancestral quality matches euler. The per-row velocity-recovery identity
`v = (x − denoised)/σ_carrier`, `denoised_r = x − σ_row·v` holds on the real H3 model with
NO hidden σ-dependence beyond the label channel. Bug B is killed for euler_ancestral.

*Historical context (why the remap form cracked before PR2):* the remap's per-row r-scaling
`x_cur = x_prev + r·(x_cur − x_prev)` linearly rescales a displacement that contains the
renoise term. `sample_euler_ancestral_RF` (sampling.py:240-266) has affine alpha terms:
`alpha_ip1/alpha_down = (1−σ_{i+1})/(1−σ_down)` as the x-multiplier, and
`renoise_coeff = (σ_{i+1}² − σ_down²·α_{i+1}²/α_down²)^½` — neither scales linearly under
`σ → σ_row`. PR2 resolves this for the single-eval RF-ancestral case. Same root cause as
[Bug B](bugs.md#bug-b); PR2 closes it for euler_ancestral; multistep / DPM++ not yet verified.

→ **AUDIO: OPEN (root cause CONFIRMED 2026-08-27, task #76).** GPU spike also revealed euler_a
audio hiss. Root cause: implementation divides by `ctx.sigmas[i]` (σ_v) instead of `ctx.sig_g`
(per-modality global: σ_v video, σ_a audio). Effective lerp weight `w/S ≈ w/4`: audio under-denoises
~4×; at m=1 ~75% terminal noise survives → the hiss. Fix: `v = (x−denoised)/ctx.sig_g`. Video
is a provable no-op (sig_g == sigmas[i] for video; #68 GPU pass cannot regress). Bug also present
in PR3's recovery reuse; PR3 held. GPU audio confirm pending (task #76).

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

Heun, dpm_2, dpmpp_2s_ancestral additionally need pooled-label refresh before each inner model
eval (`w_mid = σ_row_mid/σ_mid`). `dpmpp_sde` graduated to PR4 below; the rest stay deferred
until PR4's label-refresh plumbing proves out.

## Risks and verification notes

- **Leak risk (GPU-CONFIRMED clear for euler_ancestral, task #68):** the spike ran and passed; no
  hidden σ-dependence detected beyond the label channel for the single-eval RF-ancestral path.
  PR3 (multistep) and PR4 (DPM++ SDE) reuse the same recovery identity but add multistep
  history / a second in-step eval — their leak surface is larger and they remain UNVERIFIED.
- **Audio (CONFIRMED OPEN, task #76):** root cause identified: recovery divides by `ctx.sigmas[i]`
  (σ_v) not `ctx.sig_g` (per-modality global). Fix: one-line sigma swap; regression test =
  audio-slice denoised_r == denoised at m=1 terminal. See
  [audio-carry-identity.md](audio-carry-identity.md) (Consequence 3). Bug in PR2 + PR3; PR3 held.
- **Maintenance:** each native step is a small reimpl that can drift from comfy upstream.
- **Status:** discussion-stage design; nothing built; direction not yet user-confirmed.

## Delivery plan (4 PRs, tasks #66–#73)

Ordering rationale: stochastic first — it is the user-visible failure and validates the leak risk
before investing in multistep work.

**PR1 `refactor-per-row-step-dispatch` (task #66)** — behavior-identical refactor only.
Restructures `build_per_row_sampler_function` to a per-row step-function protocol + dispatch by
sampler name. Native per-row euler step is numerically equivalent to the current wrap+r-scale for
euler; all other samplers keep the wrap+r-scale fallback unchanged. CPU equivalence tests:
m=0 / fractional / m=1, audio-shifted rows. No GPU gate — zero behavior change.

**PR2 `add-per-row-rf-ancestral-step` (task #67) — SHIPPED (ede2d8c) + GPU-VALIDATED (task #68).**
Elementwise `sample_euler_ancestral_RF` formulas over tensor σ_row; σ clamped ≥ε so m=0 rows
freeze; seeded `default_noise_sampler`; `s_noise·noise_scale` replicated. Whitelisted
euler_ancestral stops the stochastic warning; non-whitelisted stochastic still warns.
CPU tests vs scalar reference; analytical audio-carry-identity check for shifted-σ audio rows.
GPU spike (task #68) passed: 0.2MP, d=0.20/0.15 injects, fade-in, 20 steps, both euler +
euler_ancestral — no corruption, no ghosting; leak-risk test clear.

**PR3 `add-per-row-multistep-steps` (task #69)** — per-row dpmpp_2m + res_multistep.
Deterministic (eta=0) form; per-row `old_denoised` history carried across the outer loop — fixes
Finding 1's first-order degradation. Key test: all-m=1 rows over a full schedule must reproduce
the stock sampler bit-for-bit on a toy model. Gated on USER GPU quality check (task #70) at the
d=0.2/0.15 sweet-spot config.
Introduces shared DPM++ spine helpers (reused by PR4 SDE family): `_recover_row_denoised(ctx)`
(velocity-recovery + model-eval block), `_dpmpp_time_coeffs(σ_a, σ_b)` (t = −log σ, h, sigma_ratio),
`_dpmpp_2m_second_order(denoised_r, old_denoised_r, r)` ((1+1/2r)·d − (1/2r)·d_old multistep combiner).

**PR4 `add-per-row-dpmpp-sde-steps` (task #72)** — per-row `dpmpp_sde` + `dpmpp_2m_sde` + `dpmpp_3m_sde`. Sequenced
AFTER PR3: 2M SDE is literally PR2's stochastic renoise ∩ PR3's history carry. Source facts
(sampling.py @b78cec87): `sample_dpmpp_sde` (738-792) = TWO model evals/step (denoised @σ_i, then
denoised_2 @midpoint σ_s_1, r=1/2 default), NO history, noise injected after EACH sub-step (781,
791); `sample_dpmpp_2m_sde` (822-873) = ONE eval/step, carries `old_denoised` AND `h_last`, SDE
noise `σ_{i+1}·(−2hη).expm1().neg().sqrt()·s_noise` (869), solver_type midpoint (default) / heun
(877). Both: BrownianTreeNoiseSampler, `s_noise·noise_scale`, `offset_first_sigma_for_snr`.
Corrections vs the draft plan: no `_RF` variants, but RF is IMPLICIT — `sigma_to_half_log_snr`
(152) branches on CONST → λ = −logit(σ), so α = 1−σ lives INSIDE the helpers; per-row steps run
the logit form elementwise with σ_row clamped to [ε, 1−ε] (the first-σ offset at 168 fixes only
the global scalar). `get_ancestral_step` operates in exp(−λ) = σ/α space (776, 785), python
`min()` (73) → per-row reimpl needs `torch.minimum`. Label refresh is pooled-only: the observer self-refreshes
per model call from the CALL's σ (`t_obs = 1 − m·σ`, observer_split.py:38-59), but the pooled `w`
stash publishes once per outer step (sampler.py:558) → `_StepContext` gains a `publish_labels(w)`
closure so the step sets `w_mid = σ_row_mid/σ_s_1` before eval 2 (elementwise midpoint:
`λ_row_s1 = λ_row + r·h_row`, `σ_row_mid = sigma_fn(λ_row_s1)`). BrownianTreeNoiseSampler is
scalar-interval-only (`sort` uses `a < b`, 116) → query at global carrier σ; output is
unit-variance (149), so all per-row scaling stays in the elementwise coefficients (noise
correlation structure follows the global schedule — accepted approximation).
**Merge gated on NEW USER GPU spike (task #73):** leak surface LARGER than PR2's — these lean on
the SNR mapping rather than the clean RF alpha identity, so rerun the label→timestep leak test
(39f fade, min_denoise 0.2–0.3) with both samplers before merge.
Spine composition (PR3 helpers): 2M SDE = recover → time_coeffs → 2nd-order combiner + stochastic
renoise (1 eval/step, 2M history). 3M SDE = recover → time_coeffs → 3-term 2-deep combiner +
stochastic renoise (1 eval/step). `dpmpp_sde` = recover → time_coeffs → mid-eval label refresh +
BrownianTree (2 evals/step, no history). **All three SDE variants blocked on task #76** (audio-carry
renoise miscalibration — same hiss that currently appears under euler_a).
