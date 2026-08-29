<!-- provenance: theory (mixed — attention is CONFIRMED global/source-derived; the dilution/outvote survival mechanism itself is UNVERIFIED) -->
<!-- verified: 2026-08-28 · source (observer_split.py, model.py:169-196) + GPU config set (ramp 50/51/46/68, held25/29/30, inject_at=17, 0.3MP, steps 20→30), branch fix-audio-ancestral-axis-mismatch -->
# Bug E — unified survival model: healing is attention-share dilution

Parent: [../long-fade-grid-beat.md](../long-fade-grid-beat.md).
Sibling: [two-stage-heal.md](two-stage-heal.md) (the FORMATION∧NOT-HEALED decomposition this refines);
[kv-observer-mismatch.md](kv-observer-mismatch.md) (the Δ×mutability formation seed).

Unifies the Stage-2 survival observations into one mechanism and flags the open confounds.

## Healing is attention-share DILUTION, not out-of-range

H3 attention is GLOBAL all-to-all: `optimized_attention` runs with `mask=None`
(observer_split.py `_attention_with_observer_kv`; model.py:169-196). So the clean anchors — the held
block (m=0, Δ≈0) and settled free generation (m=1, Δ≈0) — are NEVER out of the midpoint's attention
range. Distance cannot be the gate the way a windowed-attention model would impose it.

Ramp length gates healing through SOFTMAX MASS instead. A longer ramp packs more high-Δ mid-band
peers into the sequence. The midpoint's attention budget is fixed (softmax sums to 1), so more poison
peers DILUTE the share that reaches the clean anchors — the anchors get OUTVOTED. RoPE adds only a
mild distance taper on top; it does not create a hard window. This predicts monotone-in-ramp (longer
= worse) with no upper edge, matching the GPU set: ramp 50 CLEAN / 51 ERROR / 46 CLEAN / 68 ERROR.

CONFIRMED: attention is global (source). UNVERIFIED: the outvote/dilution mechanism itself is theory.

## Held-size is a SUSPECTED POSITION-CONFOUND (hypothesis, pending #82)

Both the outvote model and a naive anchor-mass model predict MORE held → MORE clean m=0 anchor mass →
MORE healing. The data is OPPOSITE: `0/0/25/80` (held25) is CLEAN, `0/0/30/85` (held30) is ERROR.
More held healed LESS.

Resolution: with `skf=0` in every tested config, `held = ekf` = the ramp seam's absolute distance
from frame 0. Held-size and ramp-POSITION are WELDED. Since position is now shown to matter
independently (`0/0/29/80` MIXED→ERROR when shifted to inject_at=17), the "held ≥ ~28" survival term
is plausibly POSITION in disguise — collapsing held and position into ONE survival term.

Discriminator: test #82 runs `0/0/25/80` and `0/0/39/90` at inject_at ∈ {0, 17, 34} to break the
weld. UNVERIFIED until #82.

## "m=1 heals" needs a qualifier: only SETTLED m=1

Only SETTLED, low-Δ m=1 regions heal. A fresh UPSTREAM free head is m=1 but actively generating —
noisy early in sampling, high transient Δ — which is why the inject_at=17 free head HURT rather than
healed. A "healing neighbor" is a CLEAN ANCHOR (Δ≈0), not any frame that merely happens to sit at
m∈{0,1}. Mutability alone does not qualify a frame as a healer; low deviation-from-clean does.

## Three-config table (regime-aware) — GHOST and MOIRÉ live in OPPOSITE regimes

| config | self label | obs label | composite | short-fade / single-keyframe | long fade (ramp ≥51) |
|--------|-----------|-----------|-----------|------------------------------|----------------------|
| stock / MC | linear 1−m·σ | linear | per-step recomposite | GHOST (both sources) | CLEAN (predicted — #83) |
| our remap, no decouple | curved σ_row | curved | init-only | (ghost gone) | BREAKS (obs miscalibrated) |
| our node (current) | curved σ_row | linear | init-only | CLEAN (ghost fixed) | MOIRÉ (Bug E) |

The fade SHAPE (per-frame m-gradient) is common to ALL rows, so it is NOT the differentiator. The
GHOST is regime-specific — short-fade / single-keyframe only; on normal/long fades stock never
ghosted. The MOIRÉ is the residual of the self≠obs LABEL decoupling, appearing in the opposite
(long-fade) regime. The `long fade` stock=CLEAN cell is a PREDICTION pending #83. This grounds the
DISFAVORED OOD-shape verdict and the stock-linear discriminator test (ROW 1) in
[kv-observer-mismatch.md](kv-observer-mismatch.md).

## Two ghost sources (both removed by our node)

The short-fade ghost had TWO separate origins; our node removes each with a distinct lever:

- **Self under-denoise:** velocity integrated over the full global interval → fixed by per-row
  `r`-scaling onto the compressed tail (`sampler.py` `_fallback_step`/`_euler_step`).
- **Per-step recomposite / re-pin of the original inject:** `out·m + latent·(1−m)` compounds every
  step (retained fraction → ~100% for m=0.4 over 20 steps) → fixed by the init-only composite
  (`sampler.py:563-564`, never re-injected) + `noise_mask=None`.

These are two SEPARATE removals: the label fix (curved self) and the composite fix (init-only) are
orthogonal levers.

## Interaction hypothesis (UNVERIFIED) — the ghost fixes may ENABLE the moiré

The init-only composite means fade rows are never re-anchored to clean after step 0. A per-step
recomposite would drag rows back toward clean each step, partially SUPPRESSING the moiré (at the
cost of reintroducing the ghost). So the recomposite removal may bear on Bug E, not just the ghost:
the two ghost fixes TOGETHER (curved self + never-re-anchor) may be what LETS the moiré develop.
Testable adjacent to #83 — re-enable a per-step recomposite on a long fade and check whether the
moiré suppresses. UNVERIFIED.

## Dilution is SPATIAL, not temporal

The 0.3MP "CLEAN" result is cosmetic, not real extra healing: a fixed noise pattern spread over more
spatial tokens lowers its per-token amplitude (the existing DILUTION-SUSPECT flag). Steps 20→30
stayed ERROR — MORE sampling steps do NOT dilute the artifact. So the dilution that matters is
SPATIAL token density only; temporal/step budget does not buy healing. This keeps the resolution
result from being misread as evidence that more compute heals Bug E.
