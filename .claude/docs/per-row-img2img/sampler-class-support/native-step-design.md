<!-- provenance: theory + confirmed (RF-ancestral + multistep steps IMPLEMENTED; only within-interval multi-eval / SDE (PR4) remain deferred) -->
<!-- verified: 2026-09-02 · RF-ancestral (PR2) GPU-confirmed task #68; per-row multistep BUILT + GPU-CONFIRMED (add-per-row-multistep-steps @6a5e786, user local both good, CPU bit-for-bit tests); SDE (PR4) UNVERIFIED -->
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

### Deferred: within-interval multi-eval samplers

Heun, dpm_2, dpmpp_2s_ancestral additionally need pooled-label refresh before each inner model
eval (`w_mid = σ_row_mid/σ_mid`). `dpmpp_sde` graduated to PR4 (see [delivery-plan.md](delivery-plan.md));
the rest stay deferred until PR4's label-refresh plumbing proves out.
