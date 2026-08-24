<!-- provenance: reference (file:line map; verify before trusting line numbers) -->
<!-- verified: 2026-08-23 · repo @c2b3bc6 · comfy-ref @b78cec87 -->
# File:line quick index

Read this when you just need a source location without the surrounding narrative. comfy-ref
paths relative to `/home/reithan/projects/comfy-ref/`; our-code paths relative to repo root.
Files marked `(git show)` are tracked but not on disk — read via `git show HEAD:comfy/<path>`.

For the *why* behind any location: [our-architecture](our-architecture.md) ·
[native-h3-mechanism](native-h3-mechanism.md) · [differential-diffusion](differential-diffusion.md) ·
[bugs](bugs.md).

## comfy-ref

| What | Where |
|---|---|
| audio_scale = 4.0 | `comfy/model_sampling.py` `ModelSamplingAV.audio_scale` ~344 |
| video scale_factor = 1.0 | `comfy/latent_formats.py` `MiniMaxH3Video` |
| calculate_denoised uses outer σ | `comfy/model_sampling.py` `CONST` ~86-101 |
| process_latent_in (audio ×4) / out | `comfy/model_base.py` 2158-2159 / 2161-2162 |
| _denoise_mask_values (pooled conds) | `comfy/model_base.py` 2234-2243 |
| process_timestep (embedding only) | `comfy/model_base.py` ~1228-1237 |
| scale_latent_inpaint (fixed cond t=0.999) | `comfy/model_base.py` 2248-2272 (aug 2255-2256, comment 2249) |
| KSamplerX0Inpaint PRE/POST (ghost = 642) | `comfy/samplers.py` 630-643 |
| denoise_mask_function hook (no POST-only hook) | `comfy/samplers.py` 636-637 |
| KSAMPLER.sample / ksampler | `comfy/samplers.py` 983-1007 / 1010-1035 |
| CFGGuider process_latent_in/out | `comfy/samplers.py` 1224 / 1238 |
| Differential Diffusion | `comfy_extras/nodes_differential_diffusion.py` |
| euler_ancestral → RF for CONST | `comfy/k_diffusion/sampling.py` ~216-218 (git show) |
| RF renoise (affine alpha) | `comfy/k_diffusion/sampling.py` ~240 (git show) |
| sample_euler / res_multistep / dpmpp_2m | `comfy/k_diffusion/sampling.py` ~190 / ~1459 / ~796 (git show) |
| time_shift_sigma | `comfy/ldm/minimax/model.py` ~36-38 |
| DiT forward audio carry | `comfy/ldm/minimax/model.py` 527-551 |
| DiT per-row t_v / t_a + cond pin | `comfy/ldm/minimax/model.py` `_forward` 553-626 (593-609) |

## our code

| What | Where |
|---|---|
| per-row init-lerp | `sampler.py::per_row_init_lerp` |
| correction + conditioning wrapper | `sampler.py::build_conditioning_wrapper` |
| per-row sampler function | `sampler.py::build_per_row_sampler_function` |
| audio-scale fix | `sampler.py::scale_packed_audio` + `nodes.py::_run_sampler` |
| clean reference / post-composite | `composite.py::build_clean_reference` / `post_composite_preserve` |
| GPU pipeline | `nodes.py::_run_sampler` (pragma no cover) |
| stochastic shim (dead/deferred) | `sampler.py::make_per_row_noise_sampler`, `sampler_accepts_noise_sampler` |
