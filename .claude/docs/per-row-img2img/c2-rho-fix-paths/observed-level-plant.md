<!-- provenance: theory + index (the idea/targets; GPU results + DC-debias plan live in child docs) -->
<!-- verified: 2026-08-31 · branch proto-observed-level-inject-noise · idea/analysis analytical; GPU stamps live in the child docs -->
# Observed-level plant: close Δ on the CONTENT side

Parent: [index.md](index.md).
Related: [residual-accounting.md](residual-accounting.md) (the decoupling-residue vs r-lerp fork this
discriminates), [../long-fade-grid-beat/kv-observer-mismatch.md](../long-fade-grid-beat/kv-observer-mismatch.md)
(the Δ closed form + #81, the OBSERVATION-side dual of this test),
[stock-mask-remap-port.md](stock-mask-remap-port.md) (sibling tool — ports σ_row into the STOCK
sampler via a rescaled noise_mask instead of touching content noise).

Child docs (this page split when it passed the char ceiling):
- [observed-level-plant/gpu-result.md](observed-level-plant/gpu-result.md) — GPU RESULT: Δ closes on
  the content side, a colour tint appears; mechanism + the two-sided-Δ core consequence + forward options.
- [observed-level-plant/dc-debias.md](observed-level-plant/dc-debias.md) — the chosen follow-up:
  per-step self-side DC de-bias PLAN + the GPU partial-win result + the low-pass frequency-split extension.
- [observed-level-plant/second-stream.md](observed-level-plant/second-stream.md) — Option I second-stream
  (OVERTURNED 2026-08-31): two-pass architecture with split OFF caused self-reception regression +
  weak blend. Retained as history; superseded by clean-kv-split.md below.
- [observed-level-plant/clean-kv-split.md](observed-level-plant/clean-kv-split.md) — the CURRENT
  replacement (GPU-CONFIRMED 2026-08-31, DURABLE on `clean-kv-observer-splice`): keep the split
  architecture (self-reception + broadcast), change only the K/V content source to
  `x_obs = clean + m·(x_prev − clean)` re-noised to `m·σ_g` (STATIC clean anchor = exact x0;
  prev_denoised regressed). SOLE always-on mechanism.

> **Durable-branch status:** the CONTENT-side ANCESTRAL renoise experiment below (target #2) is
> PROTO-ONLY — it stays on `proto-observed-level-inject-noise`. The durable branch
> `clean-kv-observer-splice` reverted `_euler_ancestral_rf_step` to stock and ships only the clean-K/V
> euler splice ([observed-level-plant/clean-kv-split.md](observed-level-plant/clean-kv-split.md)).

## The idea

A fractional inject row carries two sigma-labels: its true self-attention/content noise `σ_row`
(the curved schedule-tail remap) and the observer/KV broadcast `t_obs = 1 − m·σ_g` (observed level
`m·σ_g`). The confirmed gap is `Δ = σ_row − m·σ_g > 0` for EVERY fade row — the content is NOISIER
than neighbours are told (kv-observer-mismatch.md).

This prototype tests closing Δ on the **CONTENT side**: reduce the ACTUAL noise put into the
injected latent from the self-attention level `σ_row` DOWN to the observed level `m·σ_g`, while
KEEPING the self-attention self-evolution at the higher `σ_row` — the Q/gate/MLP path, the pooled
self-label `t_row = 1 − σ_row`, and the deterministic per-row `r`/ancestral integration all stay on
`σ_row`. Only the injected NOISE MAGNITUDE drops to the observed level.

This is the **DUAL of #81**. #81 closes Δ on the OBSERVATION side (broadcast the curved `σ_row` to
K/V, moving observation UP to meet content); this moves content DOWN to meet observation.

## Two code targets (both change injected noise magnitude only, not the self-label)

1. **Step-0 init composite** (`sampler.py` ~line 580, `x_cur = w·x_cur + (1−w)·clean`): replace the
   plant coefficient `w = sig_row/sig_g` with the observed fraction `w_obs = m`, so the row is
   planted at `m·σ_g` instead of `σ_row`. Shared by ALL samplers (it lives in the loop, not the step
   fn).
2. **`_euler_ancestral_rf_step` renoise** (`sampler.py` ~line 396–411): compute the ancestral
   renoise operation (`sigma_down`, `alpha_ip1/alpha_down` prefactor, `renoise_coeff`) on OBSERVED
   sigmas `m·σ_g` / `m·σ_g_next` instead of `sig_row` / `sig_row_next`. The deterministic denoised
   recovery (`denoised_r` from the global-carrier velocity) and the deterministic ancestral write
   (`ratio·x_prev + (1−ratio)·denoised_r`) STAY on `sig_row` — only the fresh-noise magnitude drops
   to observed level.

## Why BOTH (not init-only)

wiki v5 ([residual-accounting.md](residual-accounting.md), "NOT init-state-driven") found a one-shot
`i==0` init-state change produced NO GPU effect: the DiT self-corrects one-shot state errors early;
only PERSISTENT re-applied-every-step errors sustain the residual. Deterministic euler adds noise
ONLY at `i==0` (no per-step renoise), so on the deterministic path the init change is the only lever
and may be self-corrected away. The ancestral path DOES re-add noise every step (`renoise_coeff`,
keyed to `sig_row`), so target #2 is the PERSISTENT lever that survives v5.

## Predictions (now answered — see [gpu-result.md](observed-level-plant/gpu-result.md))

Pre-GPU this doc predicted fade noise would drop if content-side closure was effective, flagged a
GHOST-DIRECTION under-denoise risk from planting LESS noise, and cautioned that target #2 renoises
at observed level while keeping the deterministic write on `σ_row` (ancestral marginal deliberately
not maintained). Outcome: the fade noise DID drop (Δ closed on the observer side), but a self-side
colour tint appeared instead of clean output — see [gpu-result.md](observed-level-plant/gpu-result.md).

## Relation / ranking

This is a new discriminator on the "PRIMARY candidate = decoupling-residue vs raw `_euler_step`
r-lerp arithmetic" fork in [residual-accounting.md](residual-accounting.md). It attacks the
decoupling-residue side directly by removing the content-side half of Δ: if the fade noise is the
decoupling residue, dropping content to the observed level should quiet it; if the noise is the
r-lerp arithmetic (attention-label-innocent), it should persist.
