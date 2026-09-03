<!-- provenance: theory + confirmed (RF-ancestral + multistep steps BUILT; SDE family (PR4) BUILDING — source-confirmed @b78cec87; other within-interval multi-eval samplers deferred) -->
<!-- verified: 2026-09-02 · RF-ancestral (PR2) GPU-confirmed task #68; per-row multistep BUILT + GPU-CONFIRMED (@6a5e786, user local both good, CPU tests); SDE family (PR4) BUILDING (add-per-row-dpmpp-sde-steps @b78cec87) -->
# Design (RF-ancestral + multistep BUILT): per-row step functions replace the black-box base_fn

Child of [sampler-class-support.md](../sampler-class-support.md) — the native-step design that
the [delivery-plan.md](delivery-plan.md) PRs implement.

The remap loop already owns exact per-row sigma tensors (`sig_row`, `sig_row_next` from the
dense steps²+1 grid). Both sampler classes become supportable by replacing the per-interval
`base_fn` call with in-house step functions written elementwise over those tensors; the r-scaling
then applies only to a generic-fallback path.

**Axis-blind (post-#33):** `sampler.py::row_sigma`/`global_sigma` return the VIDEO σ_v for EVERY
row — there is NO σ_a projection here. Audio integrates on σ_v exactly as stock, and audio's fade
rides the official KSamplerX0Inpaint noise_mask composite ([../audio-native-composite.md](../audio-native-composite.md)).
The step functions below need no per-axis special-casing.

This **simplifies** [stochastic-recovery-theory.md](../stochastic-recovery-theory.md): no
"corrected denoised" identity needed — the truthful `w` labels already make the model return
per-row velocity `v_r`. Recover it as `v = (x − denoised)/σ_g` (comfy wrapper divides by global
σ) and form `denoised_r = x − σ_row·v`.

### Per-row RF-ancestral (kills Bug B) — IMPLEMENTED (PR2 @ede2d8c, `_euler_ancestral_rf_step`)

One model eval per interval. Then elementwise with tensor σ_i = sig_row, σ_ip1 = sig_row_next:

- `downstep_ratio = 1 + (σ_ip1/σ_i − 1)·eta`; `σ_down = σ_ip1·downstep_ratio`
- Euler sub-step with `denoised_r`; renoise with per-row alpha terms and coeff.
- m=0 rows: clamp σ_i to ε ⇒ ratio→0, σ_down→0, coeff→0 ⇒ row frozen; no NaN reaches the model.
- Replicate `s_noise·noise_scale` (model_sampling attr) and
  `default_noise_sampler(x, seed=extra_args["seed"])`.

### Per-row multistep — IMPLEMENTED (PR3 @6a5e786, `_dpmpp_2m_step` / `_res_multistep_step`)

Restores 2nd order for dpmpp_2m / res_multistep, including free rows. Same formulas with tensor
sigmas (plain log-σ `t = −log σ`, phi functions broadcast elementwise). Per-row `old_denoised`
(res_multistep also `old_sigma_down`/`old_sig_row`) carried across the outer loop via `ctx.state`.
Clamp σ ≥ ε before log for m=0 rows (final `where(never, clean, ·)` still guards the output);
terminal rows take the first-order branch. Shared spine `_recover_row_denoised(ctx, carrier)`:
one model eval at the global carrier σ, `v = (x_prev − denoised)/carrier`,
`denoised_r = x_prev − sig_row·v`. CPU tests reproduce stock BIT-FOR-BIT for all-m=1 (atol=1e-6);
GPU-CONFIRMED (user local 2026-09-02, dpmpp_2m + res_multistep both good). The plain samplers use
log-σ, NOT the logit/half-log-SNR form — that
form is confined to the SDE family (PR4).

### Dispatch

Whitelist by sampler name → native per-row step. Unknown deterministic → current wrap+r-scale
fallback (first-order-correct). Unknown stochastic → keep `sampler_is_stochastic` warning.

### Per-row SDE (PR4, BUILDING — `add-per-row-dpmpp-sde-steps`, source-confirmed @b78cec87)

`dpmpp_2m_sde` / `dpmpp_3m_sde` / `dpmpp_sde`. **RF is IMPLICIT:** H3 is CONST, so
`sigma_to_half_log_snr(σ) = log((1−σ)/σ)` and `alpha_t = (1−σ)` live INSIDE the SNR helpers — no
`_RF` variants. Per-row ports run the logit form elementwise over `sig_row`, clamped to `[ε, 1−ε]`
at BOTH ends (comfy's first-σ offset clamps only the global scalar).

- `_dpmpp_2m_sde_step`: 1 eval/step, 1-deep history `(denoised_r, h)`; midpoint solver; SDE renoise
  `sip1·(−2·h·eta).expm1().neg().sqrt()·s_noise`. PR3's `_dpmpp_2m_step` + logit λ + renoise.
- `_dpmpp_3m_sde_step`: 1 eval/step, 2-deep history `(denoised_1, denoised_2, h_1, h_2)`; 3M
  combiner, 2M one-history fallback, Euler no-history fallback.
- `_dpmpp_sde_step`: 2 evals/step, NO history. Eval1 → `denoised_r` (recover spine). Per-row
  midpoint `λ_s_1 = λ_s + r·h` (r=½), `σ_row_mid = sigmoid(−λ_s_1)`; step-1 ancestral
  `get_ancestral_step` reimplemented with `torch.minimum` (comfy uses python `min()`). Eval2 at a
  GLOBAL midpoint carrier `σ_mid_g = sigmoid(−(λ_s_g + r·h_g))` after publishing
  `w_mid = σ_row_mid/σ_mid_g`; recover `denoised_2_r` via the same velocity identity.

**Label refresh plumbing:** a `publish_labels(w)` closure (wraps the loop's
`make_pooled`→`schedule_tail["pooled_current"]`), stashed in `step_state`, lets `_dpmpp_sde_step`
refresh pooled labels between its two evals. **Noise:** `BrownianTreeNoiseSampler` is
scalar-interval-only → queried once at the global carrier; unit-variance output, so all per-row
scaling stays in the elementwise coefficients. **Fractional rows:** `2m_sde`/`3m_sde` reuse
`_recover_row_denoised` for free; `dpmpp_sde`'s 2nd eval needs its own side-stream priming at the
midpoint σ (the one new fractional path, main GPU risk). Axis-blind: σ_v every row, no σ_a.

### Deferred: within-interval multi-eval samplers

Heun, dpm_2, dpmpp_2s_ancestral additionally need pooled-label refresh before each inner model
eval (`w_mid = σ_row_mid/σ_mid`) — the same `publish_labels` plumbing PR4 builds for `dpmpp_sde`.
They stay deferred until PR4's label-refresh proves out on GPU.
