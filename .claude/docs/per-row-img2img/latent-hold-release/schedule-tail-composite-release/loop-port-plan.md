<!-- provenance: status (implementation plan — approved direction, NOT yet implemented) -->
<!-- verified: 2026-08-28 · plan only, nothing built or GPU-run; code @34a5925 -->
# Loop-port implementation plan (multistep + stochastic under `rescheduled`)

## Chosen approach & why

Port each sampler's ~10-line update formula into the schedule-tail loop, tensorized per-row with
the exact dense-grid `(σ_row(i), σ_row(i+1))`. Rationale in full lives in
[multistep-stochastic-support](multistep-stochastic-support.md); the alternatives and why they
lost:

- **(a) Wrap / doctored `denoised`** (`denoised_d = α·x + β·denoised` in the model wrapper, with
  `base_fn` given the full schedule) — needs the same per-sampler knowledge as the port, its
  second-order history mixes doctored values with global-`h` coefficients (approximate on
  fractional rows), and it requires fragile sigma→step-index inference in the wrapper.
- **(b) Magnitude-only `noise_sampler` shim** — falsified as Bug B under the old compressed path,
  and even under `rescheduled` it would only fix noise *magnitude*, never the per-row
  deterministic `sigma_down`.
- **(c) Stock sampler plus a per-step composite** — that is the official H3 mechanism, i.e. the
  ghosting this repo exists to fix.

The canonical-direction decision (user, 2026-08-27) settles it: `rescheduled` is headed for
first-class owned integrator code with tests, not corrections threaded through a wrapper.

## Steps (ordered)

1. **Refactor the step core to a direct model call.** Replace the per-step
   `base_fn(model, x, sigmas[i:i+2], ...)` probe with
   `denoised = model(x_cur, sigmas[i] * s_in, **extra_args)` — the same call every k-diffusion
   sampler makes. `model` (the `KSamplerX0Inpaint`-wrapped callable) is already in scope in
   `sampler_function`. This decouples the loop from `base_fn` slicing quirks (the `heun` /
   `dpmpp_2s` two-call impurity) and makes `denoised` first-class. The per-row Euler update then
   becomes explicit: `v = (x − denoised)/σ_glob`, `x ← x + Δσ_row·v`. That is algebraically
   identical to today's probe-plus-r-lerp, so **step 1 must reproduce current GPU-verified
   behavior**; for a non-Euler selected sampler the behavior goes from *silently* Euler to
   *explicitly* Euler, same output. The callback is invoked manually with the remapped index,
   since the loop now owns it.
2. **Per-row `dpmpp_2m`** (integrator chosen when `sampler_name == dpmpp_2m`): keep
   `old_denoised`; per-element `t_row = −log σ_row`, `h = t_next − t`, `h_last`; extrapolate
   `denoised_d`; update `x ← (σ_row_next/σ_row)·x − expm1(−h)·denoised_d`. Guard `σ_row = 0`
   (frozen `d = 0` rows) with `torch.where` safe denominators — those rows are restored by
   `never` regardless. Whole-frame quality win, since full rows are Euler-ized today.
3. **Per-row `euler_ancestral` (RF form)** — port `sample_euler_ancestral_RF` elementwise:
   `downstep_ratio = 1 + (σ_row_next/σ_row − 1)·eta`; `sigma_down = σ_row_next·downstep_ratio`;
   the alpha ratios; `renoise_coeff = (σ_row_next² − sigma_down²·α_next²/α_down²)^0.5`. The
   deterministic part is `x ← (σ_down/σ_row)·x + (1 − σ_down/σ_row)·denoised`, then renoise with
   ONE seeded ε per step (comfy `default_noise_sampler`) scaled per-row by `renoise_coeff`.
   Rows at `σ_row = 0` self-protect (coeff 0) on top of the where-guards. **GPU spike required:**
   this is the live test of the stochastic-recovery hypothesis, since `rescheduled` has no
   x-space compression and therefore lacks Bug B's falsifier.
4. **Fallback:** any other `sampler_name` falls back to the Euler port, plus the existing
   stochastic warning where it applies. Document `heun` and `dpmpp_2s` as Euler-fallback.
   `res_multistep` is optional later; SDE/Brownian stay out of scope (they would need per-row
   Brownian trees).

## Test plan

Canonicalization makes this a tested feature rather than a prototype, so:

- **Full-denoise-rows equivalence:** per-row 2M with an all-ones mask ≡ stock `sample_dpmpp_2m`
  on a synthetic schedule (fake model, CPU).
- **`d = 0` exact preserve** under every integrator, ancestral included (renoise coeff 0).
- **Scale-invariance:** a fractional row under per-row Euler ≡ stock Euler run on that row's
  stretched schedule alone.
- **Ancestral determinism** under a fixed seed, and a label-stream (`w`) regression against the
  current build.

## Open items carried

- Audio per-stream `σ_a_row` ([consistency-audit](consistency-audit.md) finding A) applies to
  every integrator, and ancestral audio noise inherits it. Fold into the audio extension rather
  than solving it per-integrator.
- The `both` / `mask-drop` / `official` ablation modes stay on the Euler path; all of this is
  scoped to `rescheduled`.
