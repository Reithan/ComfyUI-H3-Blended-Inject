<!-- provenance: theory + confirmed (two-layer stack model; Layer 1 decoupling residue is
     theory/UNVERIFIED; Layer 2 C2 correctable; GPU test (c) confirmed Layer 1 two-sided;
     round-10 δ from delta-reinjection.md). -->
<!-- verified: GPU test (c) 2026-08-31 (content-side closure reveals tint = same Δ from
     opposite side); discriminating-test plan (a)/(b)/(d) analytical + pending;
     naturalization caveat GPU 2026-08-31; round-10 δ = analytical. -->
# Reconciliation: two-layer stack of the fade noise

Parent: [../residual-accounting.md](../residual-accounting.md)

Reconciles the PRIOR observer-split / label-vs-attention decoupling residue with the CURRENT
audio-fade (C2) findings. The Fable endgame re-derivation owns final mechanism attribution — this
extends its ranking, does not replace it.

THROUGH-LINE: fade noise = accepted DECOUPLING residue (both modalities, FLOOR) +
C2 carry-compression (audio-only, CORRECTABLE) + ancestral ρ-drift (AMPLIFIER).

## Layer 1 — decoupling Δ residue (FLOOR, both modalities) — theory (UNVERIFIED)

The linear-vs-curved observer-split gap `Δ = σ_row − m·σ_g`
([../../long-fade-grid-beat/kv-observer-mismatch.md](../../long-fade-grid-beat/kv-observer-mismatch.md),
REFRAME section) is the ONLY already-identified wiki mechanism that explains the VIDEO-side fade
noise — video rows have S=1, so C2 ≡ 0 for video. Per the reframe it is analytically IRREDUCIBLE
(not zeroable by any single label choice without reviving the keyframe ghost) — an accepted design
trade-off. The Δ math is CPU/source-confirmed; the "this residue IS the artifact / irreducible"
claim is analytical only. Discriminators (#81 kill-switch, stock-linear long-fade) are PENDING.

## Layer 2 — C2 carry-compression (AUDIO-ONLY, CORRECTABLE)

Exact math + fixes v3/v4/v6, applied ONLY in `_euler_ancestral_rf_step`, ABSENT from
`_euler_step` (sampler.py:305-335). Porting corrects a correctable layer, NOT the floor.

**Layer 3 — ancestral ρ-drift (v7): AMPLIFIER**, as ranked in candidates.md.

## PRIMARY candidate: two rivals, not yet distinguishable (both theory, UNVERIFIED)

- Δ / observer-split decoupling residue (Layer 1) — leading IDENTIFIED mechanism;
  attention-label driven.
- Raw `_euler_step` r-lerp arithmetic — legacy r-lerp on the carrier axis, NO C2/σ_c corrections;
  a distinct, attention-label-INNOCENT candidate. The wiki cannot yet tell these apart.

## Discriminating tests

**SUPERSEDE the plain H3_FORCE_ETA=0 plan; eta=0 DEMOTED to a cheap tie-breaker.**

- **(a)** Port C2 v3/v4/v6 into `_euler_step`, 5-step euler: audio drops + video persists ⇒
  two-layer model confirmed.
- **(b)** #81 kill-switch (broadcast curved `σ_row` to K/V), 5-step euler: fade noise vanishes
  (ghost returns) ⇒ noise IS the decoupling residue; persists ⇒ r-lerp arithmetic is the source,
  decoupling innocent.
- **(c)** content-side Δ closure — plant injected noise at observed `m·σ_g`, keep self-evolution
  at `σ_row` (DUAL of #81, attacks decoupling-residue from the content side).
  **RUN (GPU 2026-08-31):** content-side closure REMOVES the observer-side fade noise + Bug-E
  break and REVEALS a self-side colour tint — the noise and the tint are the SAME Δ seen from
  opposite sides (confirms Layer 1 is real + two-sided).
  See [../observed-level-plant.md](../observed-level-plant.md).
- **(d)** stock-mask remap port — `H3RescaleNoiseMask` rescales a `noise_mask` (least-squares
  scalar `m_new = Σσ_g·σ_row / Σσ_g²`) so the STOCK sampler reproduces our curved `σ_row`
  MAGNITUDE without our per-row step fn. A/B stock+m_new vs our node isolates whether the σ_row
  value is the whole story. NOT a decoupling test (stock applies the linear observer label
  natively — reproduces magnitude, not the curved-self/linear-obs split).
  See [../stock-mask-remap-port.md](../stock-mask-remap-port.md).

## Gate-mismatch tension

Config `0/0/49/73` ramp=24 ≪ Bug E's ~51 threshold; its own gate predicts CLEAN yet 5-step
deterministic noise is present. "Sub-threshold residue everywhere, runaway only in the gate" is
plausible but UNPROVEN. Bug E stays RULED OUT as direct cause; tension noted.

## Naturalization caveat

High steps naturalize the deterministic injection error into diegetic sound (2026-08-31 finding);
not prior doctrine.

## Round-10 — δ re-injection (C2-internal, audio-only, UNVERIFIED)

Network clean-estimate error δ leaks m-graded static at low m; outranked by the mode-independent
injection error above (audio-only — cannot explain video or euler noise).
Detail: [../../euler-ancestral-per-row-fix/delta-reinjection.md](../../euler-ancestral-per-row-fix/delta-reinjection.md).
