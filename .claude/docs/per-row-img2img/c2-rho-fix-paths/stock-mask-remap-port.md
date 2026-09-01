<!-- provenance: status + confirmed (the least-squares math is CPU/source-confirmed; the node is prototype/unrun on GPU) -->
<!-- verified: 2026-08-31 · branch proto-observed-level-inject-noise · math = CPU numerics + comfy source read (ldm/minimax/model.py:587-605); node unrun on GPU -->
# Stock-mask remap port: rescale a noise_mask to reproduce σ_row

Parent: [index.md](index.md).
Related: [residual-accounting.md](residual-accounting.md) (adds this as discriminator (d)),
[observed-level-plant.md](observed-level-plant.md) (sibling tool — content-side Δ closure),
[../long-fade-grid-beat/kv-observer-mismatch.md](../long-fade-grid-beat/kv-observer-mismatch.md)
(the σ_row/(m·σ_g) ratio table this fit reproduces).

> **PROTO-ONLY.** This node (`H3RescaleNoiseMask` + `comfyui_h3_blended_inject/mask_rescale.py`) lives
> ONLY on `proto-observed-level-inject-noise`. It was NOT shipped to the durable branch
> `clean-kv-observer-splice` (which ships only the clean-K/V euler splice). History, not durable code.

## The node
Branch `proto-observed-level-inject-noise` adds:
- `comfyui_h3_blended_inject/mask_rescale.py` — pure-torch helper `rescale_mask_to_remap(mask, sigmas)`.
- `comfyui_h3_blended_inject/nodes.py` — node `H3RescaleNoiseMask` (display "H3 Rescale Noise
  Mask"), registered in NODE_CLASS_MAPPINGS/NODE_DISPLAY_NAME_MAPPINGS. Inputs: model (MODEL),
  mask (MASK), scheduler (combo), steps (INT default 20), modality (["video","audio"]). Output:
  MASK. Method `rescale` is `# pragma: no cover` (comfy-dependent); the math lives in the pure helper.

## Stock vs our per-row placement (confirmed)
The STOCK H3 masked denoise puts a row/token with denoise-mask value `m` at effective sigma
`m·σ_g` — LINEAR (comfy source `ldm/minimax/model.py:587-596`, label `t = 1 − m·σ`; mask polarity
1=denoise/generate, 0=preserve; audio mask multiplies the audio-SHIFTED sigma per `model.py:601-605`).
Our per-row schedule-tail remap instead puts the row at the CURVED per-row sigma
`σ_row(τ) = S(1 − m(1−τ))`, τ=i/steps, S=sigma schedule as fn of normalized position (S(0)=σ_max, S(1)=0).

## The least-squares scalar
No single scalar `m_new` makes `m_new·σ_g(τ)` equal the curved `σ_row(τ)` at every step (linear vs
curved). The node returns the least-squares-optimal scalar PER mask element:

    m_new = argmin_a Σ_τ (a·σ_g(τ) − σ_row(τ))²  =  Σ_τ σ_g·σ_row / Σ_τ σ_g²

a σ_g²-weighted mean of the per-step ratio σ_row/σ_g (emphasises structure-bearing high-sigma early
steps). Endpoints are exact FIXED POINTS: m=0→0 (σ_row≡0), m=1→1 (σ_row≡σ_g); a LINEAR schedule maps
to the identity (only the interior bends, ∝ schedule curvature). Because σ_row ≥ m·σ_g everywhere
(the confirmed Δ>0 gap, see kv-observer-mismatch.md), the fit gives m ≤ m_new ≤ 1 — the rescaled
mask is always LARGER (the stock linear path needs a bigger mask to reach our higher curved σ_row).

## CPU-validated numbers
Shift-12 concave schedule s(u)=12u/(1+11u), steps=20:
m=0.05→0.246, m=0.2→0.621, m=0.5→0.863, m=0.8→0.960, m=1→1.0; linear schedule → identity (err ~6e-8).
The light-mask upscaling is consistent with kv-observer-mismatch.md's σ_row/(m·σ_g) ratio table
(7.74× at m=0.05, 3.75× at m=0.2).

## Discriminator role
It ports our remap's per-row σ_row trajectory into the STOCK sampler via a rescaled `noise_mask`,
WITHOUT our custom per-row step function or observer-split. A/B: if stock+m_new ≈ our node's output,
the remap VALUE (curved σ_row) is the whole story and our custom machinery is unnecessary; if they
differ, the machinery (per-row step, observer split) carries independent weight.

NOT a decoupling test. The stock path applies the LINEAR observer label natively, so this reproduces
the σ_row MAGNITUDE, not the curved-self/linear-obs split. It cannot separate the decoupling-residue
from the r-lerp arithmetic — for that see the (b)/(c) discriminators in residual-accounting.md.
