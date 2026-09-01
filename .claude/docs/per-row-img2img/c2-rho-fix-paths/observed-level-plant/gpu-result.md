<!-- provenance: GPU-RESULT + mechanism analysis (Δ closes on content side; a colour tint appears) -->
<!-- verified: 2026-08-31 · branch proto-observed-level-inject-noise · user GPU (euler + euler_a, 0/0/49/73 and 0/0/39/90, ours vs official); mechanism/analysis remain analytical -->
# GPU RESULT — Δ closes on content side; a colour tint appears

Parent: [../observed-level-plant.md](../observed-level-plant.md).

Request A was implemented: init composite planted at `w_obs = m` (sampler.py:608-610) and the
ancestral path renoised at `m·σ_g` (`_euler_ancestral_rf_step`). Results:

- **euler (deterministic), 0/0/49/73:** audio fade noise GONE. But the fade region is now BRIGHTER
  and BLUER — a colour/brightness tint (DC/hue offset, not structured noise).
- **euler, prior-"broken" 0/0/39/90 (Bug E window):** no longer breaks; no audio artifacts; only
  the bright/blue tint.
- **euler_ancestral, 0/0/39/90:** same blue tint PLUS extra audio noise and video flickering.
- **Official/stock sampler, 0/0/39/90, euler AND euler_ancestral:** clean (no noise, no tint).
- **Official, ancestral, single-frame inject denoise=0.3:** the known failure — inadequate blend
  into/out of the inject, inject frame not noticeably denoised vs source (the ghost this repo
  exists to fix). Expected, unchanged.

## Mechanism (analysis)

A fractional row has ONE physical content-noise level but TWO adaLN labels: SELF label `σ_row`
(curved; drives Q/gate/MLP/velocity) and OBSERVER label `t_obs = 1 − m·σ_g` (linear; what
neighbours' K/V read — observer_split.py re-modulates the row's hidden content under the observer
adaLN). Request A dropped the PHYSICAL content σ_row → m·σ_g.

- **Observer side becomes CONSISTENT** (content m·σ_g ↔ label m·σ_g) → fade noise + Bug-E break
  vanish. This CONFIRMS the fade noise WAS the observer-side decoupling residue Δ (Layer 1 of
  residual-accounting.md / kv-observer-mismatch.md REFRAME), corroborated from a NEW direction:
  content-side closure = discriminator (c).
- **Self side becomes MISMATCHED** (content m·σ_g ↔ label σ_row): the model is told the row is
  noisier than it is, over-travels the RF velocity field, and lands with a deterministic
  LOW-FREQUENCY bias = the brighter/bluer tint (DC/hue offset, NOT structured noise).
- **euler_ancestral** re-injects the low level every step while the deterministic write stays on
  σ_row → re-exposes the mismatch stochastically → tint + extra audio noise + flicker. Deterministic
  euler is the clean isolation.
- **Official @0/0/39/90** has content = self = observer = m·σ_g everywhere (fully linear) → no noise
  AND no tint, but GHOSTS / under-denoises (the single-frame result).

## CORE CONSEQUENCE — the noise and the tint are the SAME Δ, seen from opposite sides

The fade noise and the colour tint are the SAME irreducible Δ observed from the two sides. The
constraints jointly over-determine the label:

- {no noise ⇒ content = observer} + {no tint ⇒ content = self} + {good denoise ⇒ self = σ_row}
  jointly FORCE observer = σ_row — which is exactly #81, predicted to revive the keyframe ghost
  (kv-observer-mismatch.md "#81 prediction").

So a single PHYSICAL content level merely TRADES the observer residue for the self residue; it
cannot remove both. This two-sided confirmation is the main result.

## REFINES residual-accounting v5 (a boundary condition, not a contradiction)

v5 held that a one-shot `i==0` init change self-corrects. Nuance: init changes self-correct ONLY
when the new start stays ON the labeled trajectory. Request A moved the start OFF the σ_row
trajectory (planted at m·σ_g, integrated on the σ_row tail), so the deterministic euler ODE lands
at a BIASED fixed point — the tint persists. Not a contradiction; a boundary condition on v5.

## FORWARD OPTIONS (candidates — do NOT pick here)

1. **Post-hoc colour/exposure match** of fractional rows: the tint is DC/low-freq, so a cheap
   per-channel moment-match to the free-region colour statistics could give no-noise + no-tint +
   good-denoise IF the bias is truly DC. Fast prototype test.
2. **Observer-ONLY content clean:** keep physical content at σ_row (self correct) and downscale only
   the CONTENT the observer's K/V see to m·σ_g (not just relabel). Principled but HARD — observer_split
   operates in DiT hidden space (hn); a raw-latent reblend-toward-clean isn't available there without
   clean's hidden activations or a second pass.
3. **Revert Request A**, pursue a different lever.

The chosen follow-up (kill the self-side tint without reviving the ghost) is in
[dc-debias.md](dc-debias.md).
