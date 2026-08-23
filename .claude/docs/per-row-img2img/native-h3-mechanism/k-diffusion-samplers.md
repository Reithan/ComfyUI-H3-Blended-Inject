<!-- provenance: reference (comfy-ref file:line map) -->
<!-- verified: 2026-08-23 · comfy-ref @b78cec87 -->
# k_diffusion samplers — deterministic vs stochastic

Part of [native-h3-mechanism](../native-h3-mechanism.md). Read when deciding which samplers are
supported or debugging per-row scale-invariance. Sibling internals:
[model-and-scaling](model-and-scaling.md) · [sampler-loop](sampler-loop.md) ·
[dit-forward](dit-forward.md).

The real axis is **DETERMINISTIC (works free) vs genuinely-STOCHASTIC (ancestral/SDE, eta>0)** —
NOT sampler order/family. `sampler_accepts_noise_sampler` (checks for `noise_sampler` in the
signature) OVER-detects: res_multistep has the param but is deterministic.

- `sample_euler`[^euler]: plain, deterministic, NO RF reroute. `x += (x−denoised)·(σ_next−σ)/σ`
  → ratio → **scale-invariant under σ→m·σ**. Works free (given the correction).
- `sample_euler_ancestral`[^eulera]: **delegates to `sample_euler_ancestral_RF` for CONST
  models**. So euler_ancestral on H3 = RF variant.
- `sample_euler_ancestral_RF`[^euleraRF]: renoise uses `alpha_ip1 = 1−sigma_{i+1}`,
  `alpha_down = 1−sigma_down`, `renoise_coeff = sqrt(sigma_{i+1}² − sigma_down²·alpha_ip1²/
  alpha_down²)`, `x = (alpha_ip1/alpha_down)·x + noise·s_noise·renoise_coeff`. The deterministic
  part is scale-invariant, but the `alpha = 1−sigma` terms are **affine (degree-0 offset), NOT
  homogeneous** ⇒ do NOT survive σ→m·σ. A noise-scale shim can only scale injected noise, not the
  `alpha_ip1/alpha_down` x-rescale ⇒ **per-row compression impossible for RF ancestral via a
  shim** (see [bugs · Bug B](../bugs.md#bug-b)).
- `sample_res_multistep`[^resmulti]: public wrapper passes `eta=0.` ⇒ deterministic. Coeffs are
  log-sigma diffs/ratios → invariant. Works free.
- `sample_dpmpp_2m`[^dpmpp2m]: no noise_sampler, ratios/log-diffs only → invariant. Works free.

This deterministic-only reality plus DD's stochastic-only reality is the
[duality](../differential-diffusion.md#the-duality-the-real-reason-this-isnt-simple) that drives
the whole design.

[^euler]: `comfy/k_diffusion/sampling.py` ~190 (`git show HEAD:`).
[^eulera]: `comfy/k_diffusion/sampling.py` ~216-218 (`git show HEAD:`).
[^euleraRF]: `comfy/k_diffusion/sampling.py` ~240 (`git show HEAD:`).
[^resmulti]: `comfy/k_diffusion/sampling.py` ~1459 (`git show HEAD:`).
[^dpmpp2m]: `comfy/k_diffusion/sampling.py` ~796 (`git show HEAD:`).
