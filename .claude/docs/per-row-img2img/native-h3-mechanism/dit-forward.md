<!-- provenance: reference (comfy-ref file:line map) -->
<!-- verified: 2026-08-27 · comfy-ref @b78cec87 -->
# DiT forward: per-row timestep & audio carry

Part of [native-h3-mechanism](../native-h3-mechanism.md). Read when reasoning about per-row
timestep compression, the audio sigma-shift, or the cond-timestep pin. Sibling internals:
[model-and-scaling](model-and-scaling.md) · [sampler-loop](sampler-loop.md) ·
[k-diffusion-samplers](k-diffusion-samplers.md).

- `time_shift_sigma(sigma, from, to)`[^timeshift]: `base = sigma/(from + sigma·(1−from))`; then
  `to·base/(1 + (to−1)·base)`. (12→3): 1→1; 0.5→0.2; 0.1→0.027 ⇒ `sigma_a/sigma_v` ~0.27–1
  (audio denoises "ahead" of video).
- `forward`[^forward]: `scale = payload["audio_scale"]`. If `scale != 1.0`, carries audio:
  `carry = sigma_a/sigma_v`, `x = [x[0], audio_src*carry]` on input; out[1] adjusted on output.
  **The carry RUNS in our path** (corrected 2026-08-23): `sample_custom → process_conds` calls
  `model.extra_conds`, which unconditionally sets `payload["audio_scale"]` whenever the latent has
  an audio stream (`model_base.py` ~2199, 2140-2144); a prior claim that our wrapper bypasses
  this was false. Consequences: [audio-carry-identity](../audio-carry-identity.md).
- `_forward`[^_forward]: per-row timestep labels; video: `rows_t = (1 − m·sigma_v).clamp(max=
  t_pin_v)` ⇒ effective video sigma = **m·sigma_v**; audio: `rows_t = (1 − m·sigma_a).clamp(max=
  t_pin_a)` ⇒ effective audio sigma = **m·sigma_a**. `VISUAL_COND_TIMESTEP`/`AUDIO_COND_TIMESTEP`
  pin fully-preserved rows near 1; this pin is why
  [DD cracks on H3](../differential-diffusion.md#why-dd-cracks-on-h3-source-level).
- Where compression happens (corrected 2026-08-23): `MiniMaxH3(BaseModel)`[^h3class] has **no
  `process_timestep` override** (the `~1228` one belongs to MiniMaxAV, a different class);
  BaseModel's is identity, so `forward` receives the GLOBAL sigma. All per-row compression happens
  in `_forward`'s row-label computation above. **It compresses only the network's timestep
  EMBEDDING, not the outer sigma used by `calculate_denoised`.** This mismatch is the root of the
  REQUIRED denoised correction (lever 3 in
  [our-architecture](../our-architecture.md#the-three-levers)).

## Attention and label mechanics (source-confirmed 2026-08-27)

- **No cross-frame attention weighting:** `Attention.forward` calls
  `optimized_attention(q, k, v, heads, mask=None)`[^attn]: one full bidirectional self-attention
  over the entire packed sequence (text + cond keyframe rows + refs + audio + video). No attention
  mask, bias, or timestep-dependent gating exists; RoPE supplies relative position only.
- **adaLN is the sole per-row timestep channel:** per-row `t_emb` → scale/shift on the row's own
  tokens before QKV and gates on residual updates (`_mod_scale_shift`/`_mod_gate`[^adaln]). Neighbor
  attention to a row is fully EMERGENT: Q·K content similarity where the row's K/V come from its
  t-modulated features. The t label tells the network how to interpret the row's features; all
  anchoring is learned, not architectural.
- **THE PIN (exact):** `VISUAL_COND_TIMESTEP = 0.999`, `AUDIO_COND_TIMESTEP = 1.0`[^pin].
  An `m=0` row gets label exactly 0.999, the identical label native cond/reference keyframe rows
  carry (`seg_t["cond"] = max(t_v, vis_aug)`, aug default 0.999). Any `m>0` row gets
  `1−m·σ < 0.999` (until `m·σ < 0.001`); a generic partially-noised peer, not the cond-frame
  category. See [schedule-tail-late-delta: label-category cliff](../schedule-tail-late-delta/data-and-hypotheses.md#falsified-label-category-cliff-as-dominant-cause)
  for the anchoring implications.
- **adaLN t_emb is smooth in t** (`use_adaln_curves` lerps a table[^adalncurves]) but the learned
  downstream response to the label need not be; nothing in the architecture makes anchoring
  linear in `m`.

[^timeshift]: `comfy/ldm/minimax/model.py` ~36-38.
[^forward]: `comfy/ldm/minimax/model.py` 527-551 (input carry 538, output 549-550).
[^_forward]: `comfy/ldm/minimax/model.py` 553-626 (video 593-600, audio 601-609).
[^h3class]: `comfy/model_base.py` 2136 (`class MiniMaxH3(BaseModel)`); BaseModel identity
`process_timestep` at 255, call site 240.
[^attn]: `comfy/ldm/minimax/model.py` 195 (`Attention.forward`).
[^adaln]: `comfy/ldm/minimax/model.py` 231-242 (`_mod_scale_shift`/`_mod_gate`), 617-641
(`mod_segments` assembly).
[^pin]: `comfy/ldm/minimax/model.py` 32-33 (constants), 581-585 (`seg_t["cond"]` assignment).
[^adalncurves]: `comfy/ldm/minimax/model.py` 684-689 (`use_adaln_curves` table lerp).
