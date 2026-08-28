<!-- provenance: reference (comfy-ref file:line map) -->
<!-- verified: 2026-08-23 · comfy-ref @b78cec87 -->
# Model wiring & latent/audio scaling

Part of [native-h3-mechanism](../native-h3-mechanism.md). Read when debugging audio scale, latent
packing, or the native inject setup. Sibling internals:
[sampler-loop](sampler-loop.md) · [dit-forward](dit-forward.md) ·
[k-diffusion-samplers](k-diffusion-samplers.md).

## Model wiring

- `MiniMaxH3`[^supported]: `sampling_settings = {"shift": 12.0, "audio_shift": 3.0}`,
  `latent_format = MiniMaxH3AV`.
- Latent formats[^latentfmt]: `MiniMaxH3Video` → `latent_channels=24`, **`scale_factor=1.0`**
  (⇒ video `process_in` is identity), spatial downscale 16, temporal 4. `MiniMaxH3AV` →
  `latent_channels=32`; `fix_empty_latent` builds `NestedTensor((video[1,24,T,Hl,Wl],
  audio[1,32,2,audio_t]))`; `audio_frame_rescale = 5/3`.
- `ModelSamplingAV.audio_scale`[^audioscaleprop]: `shift/audio_shift`, user-configurable via
  the "ModelSamplingMiniMaxH3 / MiniMax H3 Sigma Shift" node (official guidance: `shift_video`
  ~10–14+ tuned per movement/detail, `shift_audio` ≈ 3±1 leave at default). = 4.0 only at defaults
  12/3; never hardcode ×4. Does NOT depend on latent_shapes; safe to read anytime via
  `model.model_sampling.audio_scale`.
- `CONST`[^const]: `calculate_input = noise`; **`calculate_denoised = model_input −
  model_output·sigma`** (uses the OUTER sigma, root of the denoised-correction bug, lever 3 in
  [our-architecture](../our-architecture.md)); `noise_scaling = sigma·noise + (1−sigma)·latent`;
  `inverse_noise_scaling = latent/(1−sigma)`.

## `MiniMaxH3` model_base

- `audio_scale()`[^mbaudioscale]: returns `model_sampling.audio_scale` when sampling the packed
  latent (`latent_shapes` present, len≥2), else 1.0.
- `_scale_audio_slice`[^scaleslice]: nested → scales stream[1]; packed flat →
  `n = prod(latent_shapes[0][1:]); latent[..., n:] *= scale`.
- `process_latent_in`[^plin] / `process_latent_out`[^plout]: **video ×1.0, audio ×S** (and
  inverse), S = audio_scale. Applied automatically by CFGGuider.
- `extra_conds`[^extraconds]: builds `minimax_payload` (CONDConstant); sets
  `payload["audio_scale"] = audio_scale()`; injects `denoise_mask`/`audio_denoise_mask` conds.
- `_token_grid_masks`[^tokengrid]: pool to token grid (amax over 2×2 DiT patch / audio frame), then `ceil(mask*256)/256` quantize. (Pool-first, quantize-second; the doc previously had this reversed.)
- `_denoise_mask_values`[^dmvalues]: returns `{'denoise_mask', 'audio_denoise_mask'}` only for
  streams whose min < 1−1e-3. **This is what we call to build `pooled` conds.**
- `scale_latent_inpaint`[^sli]: native per-row fractional inject setup:
  - video: `cleans[0] = aug·clean + (1−aug)·noise`, `aug = VISUAL_COND_TIMESTEP ≈ 0.999`
    (fixed cond-timestep, INDEPENDENT of σ, the reason DD cracks on H3; see
    [differential-diffusion](../differential-diffusion.md#why-dd-cracks-on-h3-source-level)).
  - audio carry: `factor = (sigma_v/sigma_a)/audio_scale`,
    `sigma_a = time_shift_sigma(sigma_v, shift, audio_shift)`, `cleans[1] *= factor` (the native
    per-step compensation for the `forward` carry); unused in our path (`noise_mask=None`). Its
    per-row analogue is the candidate fix in
    [audio-carry-identity](../audio-carry-identity.md).
  - `injected = pack(cleans)`; if no `x`/`denoise_mask` → return injected; else
    `x_blend_weight = clamp((token_grid_mask − denoise_mask)/(1−denoise_mask))`,
    `return injected + x_blend_weight·(x − injected)` (handles sub-token-grid granularity).

[^supported]: `comfy/supported_models.py` → `class MiniMaxH3`.
[^latentfmt]: `comfy/latent_formats.py` → `MiniMaxH3Video`, `MiniMaxH3AV` (via `git show HEAD:`).
[^audioscaleprop]: `comfy/model_sampling.py` → `ModelSamplingAV.audio_scale` property, ~line 343.
[^const]: `comfy/model_sampling.py` → `CONST` class, ~lines 86-101.
[^mbaudioscale]: `comfy/model_base.py` 2140-2144.
[^scaleslice]: `comfy/model_base.py` 2146-2156.
[^plin]: `comfy/model_base.py` 2158-2159.
[^plout]: `comfy/model_base.py` 2161-2162.
[^extraconds]: `comfy/model_base.py` 2164-2213 (payload audio_scale at 2199, mask conds 2201-2203).
[^tokengrid]: `comfy/model_base.py` 2230-2232.
[^dmvalues]: `comfy/model_base.py` 2234-2243.
[^sli]: `comfy/model_base.py` 2248-2272 (aug 2255-2256; audio carry 2257-2265; blend 2269-2272).
