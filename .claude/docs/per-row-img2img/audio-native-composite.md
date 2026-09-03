<!-- provenance: status + theory (design decision, UNVERIFIED — CPU/GPU pending; supersedes the C2 audio line) -->
<!-- verified: 2026-09-02 (branch replace-c2-with-native-audio-composite) · comfy-ref source-grounded -->
# Audio fades → the official ComfyUI mask composite (drop C2)

## Core decision

Stop routing fractional AUDIO rows through the custom C2 packed-axis ancestral correction. Route
them through ComfyUI's OFFICIAL mask composite instead, and DELETE the entire C2 apparatus for
audio.

## Why — the reframe

- **Audio has no in-frame (spatial) axis.** It is 2D, purely temporal across the timeline. The
  per-row img2img engine exists to fix a VIDEO problem: perceived in-frame denoising mismatching
  the mask variable, which broke frame-to-frame blending (the keyframe partial-denoise ghost at
  `0 < min_denoise < 1`). That class of problem simply cannot exist for audio.
- **Audio fades are `min_denoise = 0` TEMPORAL fades** — exactly the class Motion Context / the
  official composite handles cleanly, INCLUDING under stochastic (ancestral). GPU control: the
  incumbent Motion Context run (stock KSampler, injected latent + fade mask) produced clean audio
  under `euler_ancestral` (spectrogram4), while our C2 path produced accumulating static.
- **The C2 static is self-inflicted.** The official sampler is axis-blind:
  `sample_euler_ancestral_RF` (comfy-ref `comfy/k_diffusion/sampling.py:256-265`) runs ONE
  ancestral update on the flat packed latent at the single video sigma. Audio's separate schedule
  is handled INSIDE the model forward (`comfy/ldm/minimax/model.py:527-551`) via a velocity
  chain-rule correction, so the velocity the sampler sees is already in the σ_v frame. C2
  re-derives that axis transform by hand on a reconstructed σ_c axis and commits the ancestral
  fresh-noise draw against a slightly-off clean estimate each step → accumulating broadband static.

## Why the old `noise_mask=None` objection is VIDEO-only

- We set `noise_mask=None` (nodes.py) because the official mask composite forms a DUALITY with our
  engine on H3 (see [differential-diffusion.md](differential-diffusion.md)): the official mask path
  is correct on stochastic/ancestral but CRACKS on deterministic euler; our per-row engine is
  correct on deterministic, breaks on stochastic (Bug B).
- The crack/ghost is a VIDEO phenomenon. In H3's `scale_latent_inpaint` override
  (`comfy/model_base.py:2248-2272`): the VIDEO stream `cleans[0]` is pinned to
  `VISUAL_COND_TIMESTEP=0.999` (sigma-independent cond injection = the crack/ghost source). The
  AUDIO stream `cleans[1]` gets ONLY a rescale `factor = (σ_v/σ_a)/scale` — NO cond-timestep pin —
  then a token-grid-aware blend (lines 2269-2272) purpose-built for the packed-A/V fractional-mask
  case. So the audio branch is clean; the objection never applied to audio.

## Mechanism — how to implement

- Pass `noise_mask` with VIDEO rows = 1.0 (the composite is a no-op at mask=1:
  `x*1 + injected*0 = x`, so our engine still owns video unchanged) and AUDIO rows = the fade
  fractions.
- `clean_nested` is already passed as `latent_image` to `sample_custom` (nodes.py:402), so
  `KSamplerX0Inpaint` already holds the clean latent; only `noise_mask=None` (nodes.py:403)
  currently suppresses the composite.
- `KSamplerX0Inpaint` wraps the model handed to our `sampler_function` UPSTREAM of our per-row
  loop, so every `ctx.model(...)` call automatically gets the official PRE/POST composite for
  free — nothing is re-derived.
- Keep audio fractional in the CONDITIONING mask (`model._denoise_mask_values`) — official H3
  applies the mask to BOTH conditioning and composite; they are complementary, not double-counting.
- Exclude AUDIO rows from per-row sigma-compression: audio rides the global sigma schedule (m=1 in
  our engine); the official composite realizes the fade.
- Disable the custom observer clean-K/V AUDIO splice (the `H3BI_SPLICE_AUDIO` prototype toggle
  becomes permanently off): audio goes through the official composite, not our custom splice. The
  observer splice remains for fractional VIDEO rows only.
- DELETE the C2 apparatus: `_c2_audio_ancestral_update`, `_pooled_leak_lambda`, the pool toggles,
  `eps_carry` threading, the C2 debug logger.
- RE-APPLY the video clean-K/V splice routing in `_euler_ancestral_rf_step` (the `obs`/`frac_mask`
  → `_single_forward_denoised` guard). This is the `euler_ancestral` VIDEO ghost fix (Bug F, was
  commit 06c7029) — it is NOT on main and must be carried forward.

## Caveat (UNVERIFIED, for GPU)

The official composite CRACKS on DETERMINISTIC euler (documented video signature in
[differential-diffusion.md](differential-diffusion.md)). It is clean on ancestral (our case; MC's).
If audio fades are ever run under plain euler, that path needs watching — or keep our engine for
audio under euler specifically. Whether audio even exhibits the crack on deterministic (vs it being
hidden like the video ghost is hidden by motion) is untested.
