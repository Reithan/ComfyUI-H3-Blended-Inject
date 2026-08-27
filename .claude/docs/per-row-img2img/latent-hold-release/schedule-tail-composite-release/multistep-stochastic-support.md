<!-- provenance: theory (design sketch — analytical, NOT implemented, NOT GPU-verified) -->
<!-- verified: 2026-08-27 · comfy-ref source-verified claims noted inline; code @34a5925; nothing GPU-run -->
# Supporting multistep & stochastic samplers under `rescheduled`

What the schedule-tail loop currently does to non-Euler samplers, and a single design that would
let it drive most of them correctly.

## Current behavior (source-verified)

The loop calls `base_fn` once per step with the two-element slice `sigmas[i:i+2]`, so a
history-based multistep sampler never accumulates history: every step takes its first-order
branch.

- **`dpmpp_2m`, `res_multistep`:** first-order branch only. For `dpmpp_2m` that branch is
  algebraically identical to Euler in sigma-space —
  `(σ_next/σ)·x − expm1(−h)·denoised = (σ_next/σ)·x + (1 − σ_next/σ)·denoised = euler`
  (comfy-ref `comfy/k_diffusion/sampling.py`, `sample_dpmpp_2m`). So selecting `dpmpp_2m` under
  any prototype mode today **silently runs Euler**: output is correct and the r-lerp is exact,
  but the user gets none of the second-order benefit they asked for.
- **Two-call single-step samplers (`heun`, `dpmpp_2s`):** both model calls happen inside the
  slice, so the sampler itself is intact; the r-lerp then rescales a second-order displacement
  linearly. That is a first-order approximation of the right answer. Roughly works, not exact.
- **Stochastic samplers:** unsupported, and the node warns. On H3 `euler_ancestral` routes to
  `sample_euler_ancestral_RF` (the CONST check at the top of `sample_euler_ancestral`). Its
  `downstep_ratio`, `sigma_down`, and the alpha-ratio `renoise_coeff` are all computed from the
  **global** sigma pair, and our r-lerp then rescales the whole displacement — injected noise
  included — by `r`. Wrong twice over.

## Unifying design: own the per-row integrator

From a one-step Euler slice the model's per-element prediction is exactly recoverable, because
the global sigma cancels. With `x_prev` in, `x_step` out, and `Δσ_glob = σ_next − σ_glob`:

```
d        = (x_step − x_prev) / Δσ_glob          # per-element, sign per implementation
denoised = x_prev − σ_glob · d
```

Once per-element `denoised` is in hand, **every k-diffusion update rule is pure elementwise
tensor algebra**, and the global pair `(σ_i, σ_{i+1})` can simply be replaced by the row's exact
`(σ_row(i), σ_row(i+1))` from the dense grid. The r-lerp then goes away, replaced by a per-row
update rule mirroring whichever sampler the user picked. Zero comfy surface change; the existing
callback remap keeps working.

- **(a) per-row Euler** — identical result to the existing r-lerp. Already done.
- **(b) per-row `dpmpp_2m` (~40 lines)** — keep `old_denoised`; per-element `t = −log σ_row`, so
  `h`, `h_last`, and `r` all become per-element tensors. Guard `σ_row = 0` rows (frozen `d = 0`
  rows) with where-masks. Deterministic, so no stochastic gate is needed. This is real multistep
  support, not an Euler stand-in.
- **(c) per-row `res_multistep`** — same pattern, more formula. Medium effort.
- **(d) per-row `euler_ancestral_RF` (~50 lines)** — port the RF renoise with per-row sigmas:
  `downstep_ratio`, `sigma_down`, the alpha ratios, and `renoise_coeff` all become per-element
  tensors of `σ_row`. One fresh ε draw per step (comfy's `default_noise_sampler`, seeded) scaled
  per-row by `renoise_coeff_row`. Rows at `σ_row ≡ 0` self-protect, since their coeff is 0.
  **Key point:** Bug B falsified the magnitude shim *under the stock lever path's x-space
  compression*. Under `rescheduled` rows sit at their TRUE `σ_row` scale with truthful labels, so
  there is no compression — the precondition that killed the shim is absent. This is the
  [stochastic-recovery-theory](../../stochastic-recovery-theory.md) "per-row ancestral step",
  finally on a mechanism where its preconditions hold. See also Bug B in [bugs](../../bugs.md).
- **(e) SDE / Brownian samplers (`dpmpp_sde` etc.)** — `BrownianTreeNoiseSampler` is keyed to
  global sigma pairs; per-row correctness needs per-row trees. Large effort, low payoff.
  **Out of scope.**

## Caveats

- Audio inherits finding A from [consistency-audit](consistency-audit.md): per-row σ for audio
  ticks should eventually be `σ_a_row`. Per-row noise injection on the audio slice adds a **new**
  instance of the same carried-coordinate issue (binary audio stays exact; fades are presumed
  mild). Fold this into the planned audio extension rather than solving it twice.
- Scope all of this to `rescheduled`, the working candidate. In `both` mode the held-phase
  composite would fight freshly injected noise on every step.
- Recommended order: **(d)** first if the goal is to close the repo's known stochastic gap, or
  **(b)** first for a quality win on deterministic sampling. Both are cheap.

## Wrap/shim alternative considered (2026-08-27) — loop-port preferred

A **wrap path genuinely exists**, and it is worth recording that it was evaluated rather than
overlooked. Hand `base_fn` the FULL schedule (which restores multistep history) and redirect the
per-row integration from inside the model wrapper, using a **doctored `denoised`**:

- For Euler, `denoised_d = r·denoised + (1 − r)·x` — that is *exactly* the existing
  `m·denoised + (1 − m)·x` wrapper correction with `m = r` for that step.
- It generalizes to any update rule affine in `(x, denoised)` via `denoised_d = α·x + β·denoised`
  with per-row `(α, β)`.
- Ancestral noise magnitude is reachable too: pre-scale ε per-row through the existing
  `noise_sampler` hook.

**Rejected in favor of the loop-port**, for three reasons:

1. **No generality win.** The per-sampler `(α, β)` still has to be derived by hand — the same
   knowledge the port needs, just expressed indirectly.
2. **Second-order terms stay approximate.** History terms would mix doctored values with
   coefficients computed from the *global* `h`, so fractional rows get an approximation. The
   loop-port instead runs each rule with the row's own `h` on the true `denoised`, which is exact
   by construction.
3. **Fragile plumbing.** The wrapper would have to infer the step index back from sigma, which is
   brittle in general and worse for two-call samplers like `heun`, where two wrapper invocations
   share one step.

**Also note:** the current slicing euler-izes **full-denoise rows too**, not just held ones. Under
any prototype mode the entire generation runs Euler regardless of which sampler the user selected.
Every GPU result to date used Euler anyway, so none of the recorded runs are affected — but the
claim "sampler X was used" is not true of any prototype run until this is fixed.

The user decision that `rescheduled` is expected to become the canonical implementation (see the
[index](../schedule-tail-composite-release.md) status line) strengthens the port choice further:
a canonical mechanism deserves first-class owned integrator code with tests, not corrections
threaded through a wrapper.
