<!-- provenance: reference (comfy-ref file:line map) -->
<!-- verified: 2026-08-23 · comfy-ref @b78cec87 -->
# Native H3 mechanism — comfy-ref reference map (index)

Read this when debugging how comfy/H3 actually behaves under the hood. Content is split into
bite-sized pages under [`native-h3-mechanism/`](native-h3-mechanism/) — open only the one you
need. Bare source locations: [file-line-index](file-line-index.md). Related siblings:
[our-architecture](our-architecture.md) · [differential-diffusion](differential-diffusion.md) ·
[bugs](bugs.md).

Reading comfy-ref: it's a SPARSE checkout. `k_diffusion/sampling.py` and `latent_formats.py`
are tracked but NOT on disk — read via `git show HEAD:comfy/<path>`. Add files with
`cd /home/reithan/projects/comfy-ref && git sparse-checkout add /path/to/file`.

## Pages

- [model-and-scaling](native-h3-mechanism/model-and-scaling.md) — model wiring, latent formats,
  audio_scale=4.0, `process_latent_in/out`, `_denoise_mask_values`, `scale_latent_inpaint` (the
  native inject setup). *Read for audio scaling / latent packing / native inject.*
- [sampler-loop](native-h3-mechanism/sampler-loop.md) — `KSamplerX0Inpaint` PRE/POST composite
  (the ghost), `KSAMPLER.sample`, `CFGGuider` (raw-space output), `calc_cond_batch` wrapper
  payload. *Read for how comfy wraps the model per step.*
- [dit-forward](native-h3-mechanism/dit-forward.md) — `time_shift_sigma`, audio carry, per-row
  `t_v`/`t_a`, cond-timestep pin, `process_timestep` (embedding-only compression). *Read for
  per-row timestep / audio sigma-shift.*
- [k-diffusion-samplers](native-h3-mechanism/k-diffusion-samplers.md) — deterministic (scale-
  invariant, work free) vs stochastic (RF ancestral not compressible). *Read for sampler
  support / scale-invariance.*
