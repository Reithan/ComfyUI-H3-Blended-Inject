<!-- provenance: theory (Fable analytical model + 1MP GPU validation 2026-08-24; THE CRUX = t_row coupling proven; mechanism space + switched-mode analysis) -->
<!-- verified: 2026-08-24 · 1MP GPU runs (0.45/0.60/0.68 lock, 0.75/0.78 smear=window closed, 0.83 chaos) + Fable adjudication · data in highres-singleframe-underdenoise.md -->
# THE CRUX, mechanism space, and switched-mode

Index: [highres-underdenoise-model](../highres-underdenoise-model.md).
Fable verdict and step-starvation: [crux-and-mechanism-2](crux-and-mechanism-2.md).
Experiments flowing from this: [experiments](experiments.md) / [fix-strategies](fix-strategies.md).

## THE CRUX — t_row couples denoise-rate AND attention-trust (2026-08-24)

The whole difficulty in one sentence: **`denoise_mask` = `t_row` is a single per-row scalar that H3's
architecture forces to do two jobs** — it feeds adaLN (→ the row's own velocity / denoise rate) AND it
shapes that row's hidden state, which becomes the attention key/value neighbors read (→ how much they
trust/attend it). One number, two couplings you cannot set independently:
- **low t_row** → trusted context + barely denoised = **LOCK** (source-identical; neighbors follow but the
  frame itself doesn't move).
- **high t_row** → denoised at full rate + UNtrusted (neighbors don't anchor to it) = **CHAOS** (frame
  smears, no coherent blend).
- The **coherent middle** is only where the two effects happen to balance → that's *why* it's a narrow
  window, and why one scalar can't widen it. **Bimodality is the direct signature of this coupling.**

"Have your cake and eat it" (user) = **decouple denoise-rate from attention-trust.** For the joint knob
model (A latent-content / B mask / C cond-aug / D composite) and how the user's "neighbors see clean, keyframe
denoises full-time" ideal maps onto the routes, see
[keyframe-two-views-and-knobs](../keyframe-two-views-and-knobs.md). There are exactly
three routes, differing in WHERE they break the coupling:
1. **In TIME — anchor-then-release.** low-t early (trust while composition locks) → high-t late (denoise).
   Cost: step-starvation at extreme d / low K (see below).
2. **By DUPLICATION — two forward passes/step.** Pass A: anchor locked (low t) → gives neighbors a trusted
   field. Pass B: anchor unlocked (high t) → denoises the anchor. Combine (lerp/slerp neighbors from A,
   anchor from B). ~2× cost, and the two fields are mutually inconsistent (B's neighbors never saw a
   trusted anchor) — but as a **CORRECTNESS ORACLE it's gold**: if two-pass yields a coherent 1MP single
   frame, it PROVES a good answer exists and gives a cheap-method target to match. Ship-cost aside.
3. **In ARCHITECTURE — attention-logit boost.** Keep the anchor at its TRUE high t (denoise at the real
   rate) AND separately add +log(b) to neighbor→anchor attention logits (trust). The ONLY single-pass
   route that truly decouples — this IS the user's "denoise 40%, attend at 60%." Cost: needs a DiT
   attention hook (invasive), and the mask→b mapping is unknown. = Fable's route ③, now promoted from
   "diagnostic" to "the elegant single-pass answer."

**MC re-read in this light:** MC's alpha-blend clamps `x_row` toward source in LATENT space every step —
it never touches attention, so neighbors still see a high-t (untrusted) anchor. That buys STABILITY (the
clamp keeps the row on the source manifold → never chaos) at the price of GHOST (clamp persists into
render) and NO true attention-anchoring. It's route-nothing: full denoise + latent leash, which is why it
"mostly works" but breaks for single-frame / no-fade / non-zero d. Our reverted hold-and-release was a
crude route-1. **The target is route 3 (or route 1 done right); route 2 is the oracle that tells us the
ceiling.**

> **CHALLENGED BY GPU (2026-08-25):** A/B with only `latent_hold_frac` toggled showed the baseline
> co-evolving per-row inject (NO hold) already blends neighbors; hold-release hard-cut. "Attraction
> needs a hold" looks wrong. `denoise=0.0` (permanent freeze, `none`-interpolation, code-verified)
> BLENDS. The hold-ON cut is explained: `anchor_mask` froze the OPENING video inject's fade-out
> (wrong row) and propagated via global attention to corrupt r40 — not a release artifact.
> This challenges both "a co-evolving op is INCAPABLE of attraction" and "attraction needs a
> low-noise-EARLY anchor" below. Treat those claims as challenged/unverified, not refuted.
> See [latent-hold-release/hold-mechanism-and-confounds](../latent-hold-release/hold-mechanism-and-confounds.md)
> Findings 7–10.

## Mechanism space — why re-injection can't make attraction (2026-08-24)

Framing the per-step update as a design space of {denoise, re-inject-source, re-noise, decay-schedule}:
- MC = constant re-inject (denoise THEN blend x←m·x+(1−m)·noised_clean) → stable/never-chaos but GHOST.
- Ours = no re-inject (correction toward EVOLVING x) → no ghost but bimodal.
- **User's order-inversion (blend-THEN-denoise):** marginal — both orders re-expose source every step;
  the only real difference is the LAST step (blend-then-denoise leaves a clean final denoise
  un-re-frozen, slightly less terminal ghost). Not the lever.
- **KEY THEORY (CHALLENGED — see GPU block above):** any re-inject/blend op acts on the ANCHOR ROW
  ONLY; it cannot make NEIGHBORS attract toward it. Neighbor attraction is an ATTENTION effect: a
  LOW-NOISE row reads as trusted context that neighbors compose around. ⇒ a purely anchor-row-local
  latent op is likely INCAPABLE of producing attraction; the only latent-space route is **low-noise
  EARLY** (auto attention-attractor). **Candidate:** hold row resolved for k<k_sw → RE-NOISE to
  m'·σ_sw → release into per-row img2img + denoised-correction. Open: k_sw, m', fresh-vs-correlated
  re-noise. Test = does it WIDEN the basin vs the ≤0.15 denoise-only window?

### User's "switched-mode" (2026-08-24) — = route-1 + a cond token; convergent, cond redundant/risky

**Proposal:** inject into BOTH latent & cond rows; for X steps hold latent row locked + cond at some aug;
at release, unlock latent for Y steps + drop cond. **This IS route-1 anchor-then-release**, re-derived
independently. The cond token is the only addition; assessment:
- **Cond during HOLD is REDUNDANT.** A locked clean latent row (low `t_row`) is already the trusted low-noise
  field neighbors compose around. If added anyway, it MUST be clean (`aug≈0.999`) — fractional-aug is a
  contagion source ([conditioning-row-inject](../conditioning-row-inject.md), 0.2MP aug=0.4 test).
- **"Cond @ aug 0" ≠ off.** `aug=0` = pure-noise reference (max contagion), NOT removal. Remove the token to
  drop the cond.
- **Real handoff is on the LATENT, not the cond.** Releasing a clean row without RE-NOISING = clean-content-
  at-noisy-label mismatch (test-2 contagion). Route-1 requires **re-noise to `m'·σ_sw` + denoised correction**
  at release; the cond cannot supply this.
- **`aug=(1−d)` arithmetic: right spirit, wrong variables.** `k_sw = measured k_comp` (not `(1−d)` of steps);
  re-noise level = `m'·σ_sw` (m'≈0.3–0.5), not directly `(1−d)`. The proposal conflates denoise fraction,
  switch point, and re-noise level — three separate quantities.

**Net:** switched-mode ⇒ build **pure route-1 (no cond)** as MVP. **Clean-aug cond during HOLD only** is a
legitimate second variant (possible neighbor-anchor boost), lower-priority than shipping route-1. The distinct
cond idea, **live-cond-mirror (variant #4)** — cond carries the LIVE anchor at release — is principled but
carries positive-feedback lock risk ([conditioning-row-inject](../conditioning-row-inject.md) exp order §2).
