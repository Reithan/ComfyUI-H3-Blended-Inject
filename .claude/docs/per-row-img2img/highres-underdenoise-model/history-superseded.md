<!-- provenance: theory (historical analytical models, ALL SUPERSEDED by T_N(d) — kept for provenance only) -->
<!-- verified: 2026-08-24 · 1MP GPU runs (0.45/0.60/0.68 lock, 0.75/0.78 smear=window closed, 0.83 chaos) + Fable adjudication · data in highres-singleframe-underdenoise.md -->
# Historical superseded models — γ, α=ρ, √ρ, bimodality

Index: [highres-underdenoise-model](../highres-underdenoise-model.md). Current live model: see the T_N(d) section in the index. Metrics that replaced these: [metrics-detectors](metrics-detectors.md).

## GPU-REFIT (2026-08-24) — single exponent γ≈1.6 [HISTORICAL: STRUCTURALLY WRONG, superseded by T_N]

⚠ **SUPERSEDED 2026-08-24 (Fable).** The single-γ model is wrong in STRUCTURE, not just value: one
exponent for both edges predicts the window's log-odds width is N-invariant, but **closure @1MP proves the
lock-edge and chaos-edge have DIFFERENT N-scalings (γ_chaos < γ_lock) that CROSS at N≈0.3-0.7MP** — the
window shrinks to zero, it doesn't merely shift. γ≈1.6 did its job (placed the 0.75 probe correctly,
falsified openness) — keep it as a historical bracket, but **d\* is dead as a target** and the core object
is now T_N(d) (see CURRENT MODEL). Also retracted: "window open-but-narrow by default" (closed is data);
the four-régime ladder now reads "coherent band EXISTS at low N, VANISHED by 1MP"; the régime-4 read of
d=0.95 as "inject lost" is SOFTENED to UNVERIFIED (T-suppression means realized displacement at 0.95 was
likely < nominal, so the inject may have been less lost than believed).

Original refit (kept for provenance): The α=ρ law below is **FALSIFIED** (0.83=chaos, not coherent).
Refit `odds_edge ∝ N^{γ/2}` against ALL constraints — 0.2MP smooth @0.45-0.50, 0.1MP chaos @0.50, 1MP lock ≤0.70, 1MP chaos @0.83:
lock-edge ⇒ γ≥1.31, chaos-edge ⇒ γ≤1.99, all points consistent with a single **γ∈[1.31,1.99],
center ≈1.6** → multiplier ρ^0.8≈3.6 → mapping the 0.45-0.50 anchor gives **d\*≈0.75-0.78 @1MP**.
Neither ρ (γ=2→0.83=chaos wall) nor √ρ (γ=1→0.64-0.69, straddles the 0.70 lock) fits; truth is between.

**FOUR régimes, monotone in d (not three):** `lock → coherent → chaos → generic-gen`. Chaos is the
anchor-CONFLICT band (partial anchor fights the front, loses messily = "smeared"). As d→1 the anchor
becomes irrelevant and the row degenerates to an ordinary generated frame — trivially seamless but the
**inject is LOST**. This reconciles the old "d=0.95 blends, slightly high" (that was régime 4,
inject-lost, NOT a coherent blend). Window is **open-but-narrow** by default; declare CLOSED only if
the 0.75 region also fails with the seam gate high.

## (SUPERSEDED by the refit above) The calibration law — noise-ODDS linear in token count (α = ρ)

Let N = spatial tokens/frame (H/16 · W/16): ~416 @0.1MP, ~836 @0.2MP, ~4128 @1.0MP. Anchor at the
validated-smooth point (0.2MP, d≈0.45-0.50). Map denoise across resolution by holding **noise-odds ×
token-count** constant:

  **d′/(1−d′) = (N′/N_ref) · d/(1−d)**   ⇒  α = ρ = N′/N_ref  (token ratio, not its root)

- 0.2MP→1.0MP (ρ=4.94): d=0.50 → **d′=0.83**, d=0.45 → **d′=0.80**. ← dead-center in the empirically
  found 0.8-0.9 window.
- 0.2MP→0.1MP (ρ=0.498): d=0.50 → **d′≈0.33**, d=0.45 → **d′≈0.29**. ← **KEY TEST: inject at d≈0.30
  @0.1MP; the odds-linear law predicts chaos→smooth.** Run this one experiment; it cleanly
  discriminates the model (caveat below).

## Why √ρ HAD to undershoot (and why linear)

Per-band SNR of a red-spectrum (power ∝ f^−β) latent sampled at N tokens, noise fraction m
(odds m/(1−m)): **SNR_band = N·((1−m)/m)²·|c(f)|²/σ² — LINEAR in N** at fixed physical frequency.
Holding it constant = noise-odds × √ρ = exactly SD3/Flux's shift rule = our α=√5 trial. But the data
refutes single-factor SNR: with Λ₁ = N·((1−m)/m)², the 1MP d=0.70 point (Λ₁≈**758**, still POPS) sits
BELOW the 0.2MP d=0.45 smooth point (Λ₁≈**1249**) — no threshold on Λ₁ separates pop from smooth, so a
second N-dependence exists. Fitting the exponent to the sweet spot pins **γ≈2 ⇒ noise-odds ∝ N
(linear)**. Two candidate mechanisms for the extra √ρ (both testable):
- **(a) commitment-timing** — generated content crosses its own SNR-visibility threshold earlier at
  high res, shifting WHEN the row must commit by another √ρ in odds space (SNR sets *whether* it can
  self-reconstruct; timing sets *when* it commits vs the front). TEST: at 1MP d=0.5, raise
  `shift_video` (12→20+) instead of d — (a) predicts partial fix (delays everyone's commitment,
  restores synchrony without touching SNR).
- **(b) generative-prior gain** — the DiT does super-resolution-style completion from a low-freq
  skeleton, beating the linear matched-filter bound by ~√N. TEST: same shift_video bump — (b)
  predicts little effect.

## Bimodality = a SLIDING WINDOW between two rising thresholds (not a peaked scalar)

With Λ = N²·((1−m)/m)² (γ=2 units): **Λ>Λ_lock** → posterior collapses on source at t=0 → straight-
line ODE → POP (under-denoise). **Λ<Λ_id** → surviving signal below identity threshold → no anchor →
hallucinated/CHAOS. **Λ_id<Λ<Λ_lock** → coherent blend. Both bounds rise with N, so at fixed m=0.5 the
ladder walks across the window: 0.1MP (below-noise→chaos), 0.2MP (inside), 1MP (locked). Window spans
only ~5-7× in Λ → genuinely narrow (why goldilocks felt sharp).
- **HONEST CAVEAT:** 1MP d=0.95 was "slightly high" but NOT chaotic, yet its Λ is below 0.1MP-d=0.5's
  Λ — so the chaos boundary likely has a DIFFERENT mechanism than the lock boundary. Candidates: 0.1MP
  (~416 tok) is at/below H3's training-distribution floor (off-manifold, intrinsically unstable), OR
  the temporal front is too weak to steer at low token counts. The d≈0.30 @0.1MP test resolves it:
  smooth ⇒ law covers both modes; still chaotic ⇒ chaos is a resolution FLOOR and the law governs only
  the lock side (the side we care about in practice).
