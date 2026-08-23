<!-- provenance: reference (comfy-ref file:line map) -->
<!-- verified: 2026-08-23 · comfy-ref @b78cec87 -->
# Native inpaint sampler loop

Part of [native-h3-mechanism](../native-h3-mechanism.md). Read when tracing how comfy wraps the
model per step (PRE/POST composite, guider, wrapper payload). Sibling internals:
[model-and-scaling](model-and-scaling.md) · [dit-forward](dit-forward.md) ·
[k-diffusion-samplers](k-diffusion-samplers.md).

`KSamplerX0Inpaint.__call__`[^ksx0]:
```
latent_mask = 1 - denoise_mask
x   = x*denoise_mask + scale_latent_inpaint(x, sigma, noise, latent_image, denoise_mask)*latent_mask   # PRE
out = inner_model(x, sigma, ...)                                                                        # denoised
out = out*denoise_mask + self.latent_image*latent_mask                                                  # POST  ← GHOST
```
POST forces `out = m·D + (1−m)·clean` = convex blend of two coherent latents ⇒ double-exposure
ghost for fractional rows — the
[ghost math](../differential-diffusion.md#why-native-inpaint-ghosts-the-math). This is exactly why our
path passes `noise_mask=None` (see [our-architecture](../our-architecture.md#plus)).

- `KSAMPLER.sample`[^ksample]: `sampler_function(model_k, noise, sigmas, extra_args, ...)`. No
  noise_sampler passed by default.
- `ksampler`[^ksamplerfn]: returns the RAW k_diffusion fn (which DOES declare `noise_sampler`
  for stochastic samplers — see [k-diffusion-samplers](k-diffusion-samplers.md)).
- `CFGGuider`[^cfgguider]: `process_latent_in` then returns `process_latent_out(samples)` ⇒
  **`sample_custom` output is RAW space** (audio already ÷4); our post-composite (raw clean) is
  consistent. ✓
- `calc_cond_batch`[^calccond]: `model_function_wrapper` gets `{"input": input_x, "timestep",
  "c", "cond_or_uncond"}` where `input_x` is the sampler's packed x. This is the hook our
  correction wrapper rides on ([our-architecture](../our-architecture.md#the-three-levers)).

[^ksx0]: `comfy/samplers.py` 630-643 (PRE 639, POST 642).
[^ksample]: `comfy/samplers.py` 983-1007.
[^ksamplerfn]: `comfy/samplers.py` 1010-1035.
[^cfgguider]: `comfy/samplers.py` — `process_latent_in` 1224, `process_latent_out` return 1238.
[^calccond]: `comfy/samplers.py` 208 (`calc_cond_batch` public entry); private impl `_calc_cond_batch` at 221; `model_function_wrapper` call site at 332-333 inside `_calc_cond_batch`.
