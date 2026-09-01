<!-- provenance: theory (UNVERIFIED design — exactness proven analytically from comfy source; no GPU run yet) -->
<!-- verified: analytical only (2026-08-31) — DiT patch-embed linearity + Q/KV separability confirmed from comfy/ldm/minimax/model.py; whole path is GPU-only, unverified -->
# Option II — exact single-forward fractional side-stream (performance)

Parent: [../observed-level-plant.md](../observed-level-plant.md).
Refines: [clean-kv-split.md](clean-kv-split.md) (which said "No Option II needed" — that was
CORRECTNESS-scoped; Option II is a PERFORMANCE path, see below).

## Purpose

The GPU-confirmed clean-K/V mechanism runs TWO full forwards per euler step (capture on `x_obs`
at observer labels `m`; splice on `x_prev` at truthful `σ_row`, K/V overwritten with the capture
cache). H3 has no uncond, so that is literally 2× the model cost per step.

Option II collapses it to ~1 forward by carrying a small side-hidden `h_m` for ONLY the fractional
band, reusing the main stream's already-computed neighbour K/V. It is EXACT — it reproduces the
two-forward result bit-for-bit — not an approximation.

## Why it is exact (three source-confirmed facts)

1. **Neighbour-identity.** Non-fractional rows carry IDENTICAL content (`x_prev`) AND identical
   labels in BOTH forwards — held rows maskval 0, free rows maskval 1 in capture (`m_dev`) and
   splice (`w = σ_row/σ_g`) alike. Only the fractional band's own hidden genuinely differs
   (capture: `x_obs` at `m`; splice: `x_prev` at `σ_row`). So neighbour hidden evolves identically
   block-for-block and can be computed ONCE.

2. **Patch-embed linearity.** `video_patch_proj`/`audio_patch_proj` are affine Linear with a shared
   bias and NO additive positional embed on the token hidden (RoPE is applied inside attention,
   content-independent). So `embed(a·x + b·y) = a·embed(x) + b·embed(y)` exactly when `a + b == 1`,
   and embed is row-wise. → block-0 init of the band is exact.

3. **Q/KV separability.** The fused `qkv_proj` split yields disjoint Q|K|V; K/V depend only on the
   modulated hidden; per-row labels enter only each row's own slice. → one combined K/V set can be
   read by two different queries.

Source map: comfy/ldm/minimax/model.py — patch proj ~464-466 & embed assembly ~646-681;
DiTBlock.forward 286-291; adaLN `_mod_scale_shift`/`_mod_gate` 226-242; Attention 169-196;
block loop 696-711.

## Algorithm (per euler step, ONE forward)

- **Block-0 init:** `h_m[band] = ratio·h_main0[band] + (1−ratio)·h_clean[band]`, where
  `ratio = clamp(σ_obs/σ_row, max=1)`, `σ_obs = m·σ_g`, `h_main0` = the main stream's block-0 input
  (embed of `x_prev`), and `h_clean` = embed of the static `clean` inject at the band (computed
  ONCE per generation — `clean` is static).

- **Each block:** build ONE combined K/V set = the main stream's K/V with the band rows overwritten
  by the side-stream's `k_m/v_m` (computed from `h_m` at observer modulation). Read it with TWO
  queries:
  - (a) main Q at `σ_row` over all tokens → main attention → main hidden evolves → its final
    `denoised` feeds euler (this reproduces the splice forward exactly);
  - (b) band `Q_m` at observer modulation over all tokens → side attention → `h_m` evolves (this
    reproduces the capture forward's fractional hidden, which is what produces next block's band
    K/V).

- `h_m` threads block-to-block in state; the interleaving is causally sound (capture block `i`
  depended only on capture block `i−1` frac + capture neighbours = main hidden).

## The observer modulation (`t_emb_m`)

The side stream must be modulated at the observer label `m`, but a splice-labelled forward's adaLN
table only contains `σ_row`-derived timesteps. So the single forward carries a SECOND
time-embedding `t_emb_m` for just the band's `m` levels, computed ONCE per step by REUSING the
model's own time-embedder submodule on our observer timesteps (`t_row_m = 1 − m·σ`).

Band mod-index = `searchsorted(level) × 3 + modality tag` (mirrors `rows_to_mod_index`,
model.py ~617-622). Per-block side params = `block.adaln_proj(t_emb_m)` gathered for the band.

## Why NOT an approximation

Any single-forward cheaper than this side-stream (e.g. relabel the band's own noisy hidden without
a separate stream) reduces to the OLD label-only relabel that already FAILED — it re-modulates the
still-noisy hidden and leaks residual fade-noise (see observer_split.py module docstring and
second-stream.md). The side-stream is the minimal exact construction; there is no correct cheaper
approximation.

## Cost

~1 full forward + O(band) per block (side qkv + band×seq attention + band MLP), vs 2 full forwards.
Savings scale with `(1 − band/total_tokens)`.

## Verifiability caveat

The entire path is GPU-coupled (comfy model internals) and cannot be CPU-verified for exactness.
CPU tests cover only the pure math: block-0 init linear combo, combined-K/V construction, and
pos/label bookkeeping. GPU verification is a USER task.

## Status

Design recorded 2026-08-31 on branch `single-forward-clean-kv-splice` (durable). Implementation in
progress in observer_split.py + sampler.py.
