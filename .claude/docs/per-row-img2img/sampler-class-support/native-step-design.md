<!-- provenance: theory (UNVERIFIED — analytical design; no experimental confirmation of PR3/PR4 paths) -->
<!-- verified: 2026-08-28 · design stage; PR2 (euler_ancestral) form GPU-confirmed task #68, the rest UNVERIFIED -->
# Design (UNVERIFIED): per-row step functions replace the black-box base_fn

Child of [sampler-class-support.md](../sampler-class-support.md) — the native-step design that
the [delivery-plan.md](delivery-plan.md) PRs implement.

The remap loop already owns exact per-row sigma tensors (`sig_row`, `sig_row_next` from the
dense steps²+1 grid; audio rows pre-shifted via `_shift_schedule`). Both sampler classes become
supportable by replacing the per-interval `base_fn` call with in-house step functions written
elementwise over those tensors; the r-scaling then applies only to a generic-fallback path.

This **simplifies** [stochastic-recovery-theory.md](../stochastic-recovery-theory.md): no
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
eval (`w_mid = σ_row_mid/σ_mid`). `dpmpp_sde` graduated to PR4 (see [delivery-plan.md](delivery-plan.md));
the rest stay deferred until PR4's label-refresh plumbing proves out.
