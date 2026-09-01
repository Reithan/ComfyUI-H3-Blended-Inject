<!-- provenance: theory (OVERTURNED 2026-08-31 — GPU regression; clean-kv-split.md is the replacement) -->
<!-- verified: 2026-08-31 · GPU test of Option I build; two-cause decomposition confirmed; architecture overturned -->
# Second-stream fix (Option I) — OVERTURNED

Parent: [../observed-level-plant.md](../observed-level-plant.md).
Superseded by: [clean-kv-split.md](clean-kv-split.md) (clean-sourced K/V inside the split).
Formerly superseded: [dc-debias.md](dc-debias.md) (DC/low-pass patch — now also superseded by clean-kv-split).

## What Option I was (historical record)

A fractional fade row carries ONE physical content-noise level but needs TWO:
SELF needs `σ_row` (curved schedule-tail) so its Q/gate/MLP denoises correctly (no keyframe ghost).
OBSERVER needs `m·σ_g` (linear) so neighbours read it at its clean denoise-destination level.
Gap `Δ = σ_row − m·σ_g > 0` every fade row. Option I absorbed Δ by two model forwards per step:

- Pass B (SELF, `w = σ_row/σ_g`, `x_prev` as-is, observer-split OFF): fractional rows' denoised.
- Pass A (OBSERVER, `w = m`, content `x_obs`): free rows' denoised.
- Composite: `denoised[free] = A`, `denoised[fractional] = B`, then euler step.

Observer content: `x_obs = denoised_self + σ_obs·û`, `σ_obs = m·σ_g` — the frame's denoised
appearance re-expressed at observer noise level, preserving x0 evolution. Env gates:
`H3_SECOND_STREAM` (enable); `H3_SS_CLEAN_ANCHOR` (older clean-anchored fallback). BOTH DELETED.

## GPU result + overturn (2026-08-31)

Two simultaneous regressions vs the known-good observer-label split baseline:
(1) in-frame OVER-CHANGE — "mask 0.2 changes like 0.5";
(2) weak timeline blend.

Present on SINGLE-FRAME injects (one fractional row → zero fade→fade attention) — confirming
the cause is NOT the multi-row fade→fade approximation; it is the split-off architecture itself.

**Cause 1 — over-change (split-off self-reception):** Option I disabled `observer_split`. With
the split off, the fractional row's K/V reverted to `σ_row` → the frame perceived itself as a
`σ_row`-noisy token and ran full-strength denoise. KEY INSIGHT: the observer-label split governs
SELF-RECEPTION (what a frame reads of its own K/V) as well as broadcast (what neighbours read).
Split OFF → self-anchor lost → over-change. Full statement: [clean-kv-split.md §Finding 1](clean-kv-split.md).

**Cause 2 — weak blend (content fidelity regression):** With split off and observer content
anchored on the static clean inject (`x_obs = clean + ratio·(x_prev − clean)`), neighbours
read a flatter, less-evolved inject → weaker blend than Request A's m·σ_g-planted evolved inject.

**Why Option II does NOT fix this:** Option II (true per-block K/V splice) would also remove
the split from the fractional frame, reproducing cause 1. The correct fix is to keep the split
ARCHITECTURE and change only the K/V CONTENT SOURCE. See [clean-kv-split.md](clean-kv-split.md).
