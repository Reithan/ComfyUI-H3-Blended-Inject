<!-- provenance: theory (Fable analytical model + 1MP GPU validation 2026-08-24; THE CRUX = t_row coupling proven; mechanism space + switched-mode analysis) -->
<!-- verified: 2026-08-24 · 1MP GPU runs (0.45/0.60/0.68 lock, 0.75/0.78 smear=window closed, 0.83 chaos) + Fable adjudication · data in highres-singleframe-underdenoise.md -->
# THE CRUX, mechanism space, and switched-mode

Index: [highres-underdenoise-model](../highres-underdenoise-model.md). Fable verdict and step-starvation: [crux-and-mechanism-2](crux-and-mechanism-2.md). Experiments flowing from this: [experiments](experiments.md) / [fix-strategies](fix-strategies.md).

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

> **CHALLENGED BY GPU (2026-08-25) — key discriminator still pending:** the section below argues a
> low-noise EARLY anchor is "THE latent route to attraction" and that a co-evolving anchor-row-local
> op is "likely INCAPABLE of producing attraction." A single-variable A/B (only `latent_hold_frac`
> toggled) showed the OPPOSITE direction: the baseline co-evolving per-row inject (NO hold) already
> blends neighbors toward the keyframe, and the hold-release run hard-cut. So "attraction needs a
> hold" looks wrong. The clean discriminator has RUN: `denoise=0.0` (a permanent freeze, no release)
> with the user's `none`-interpolation config = a TRUE frozen clean m=0 anchor (envelope
> code-verified), and it **BLENDS**. So the freeze is fine. And the hold-ON cut is now EXPLAINED
> (2026-08-25): the provenance-blind `anchor_mask` froze the OPENING video inject's fade-out (not the
> keyframes), and that wrong-row freeze PROPAGATED via H3's global attention to corrupt r40's blend —
> NOT a release artifact, NOT evidence the freeze/attraction machinery fails. This still contradicts
> "a co-evolving anchor-row-local op is INCAPABLE of attraction" below AND "attraction needs a
> low-noise-EARLY anchor": a frozen clean anchor blends and so does a co-evolving fractional one.
> Treat the claims below as challenged/unverified, not refuted. See
> [latent-hold-release/hold-mechanism-and-confounds](../latent-hold-release/hold-mechanism-and-confounds.md)
> Findings 7–10.

## Mechanism space — why re-injection can't make attraction (2026-08-24)

Framing the per-step update as a design space of {denoise, re-inject-source, re-noise, decay-schedule}:
- MC = constant re-inject (denoise THEN blend x←m·x+(1−m)·noised_clean) → stable/never-chaos but GHOST.
- Ours = no re-inject (correction toward EVOLVING x) → no ghost but bimodal.
- **User's order-inversion (blend-THEN-denoise):** marginal — both orders re-expose source every step;
  the only real difference is the LAST step (blend-then-denoise leaves a clean final denoise
  un-re-frozen, slightly less terminal ghost). Not the lever.
- **KEY THEORY (to confirm w/ Fable):** any re-inject/blend op acts on the ANCHOR ROW ONLY — it tunes
  the anchor's immutability but CANNOT make NEIGHBORS attract toward it. Neighbor→anchor following
  (FL2VA; the d=0.95 "neighbors followed" obs) is an ATTENTION effect: a LOW-NOISE row reads as trusted
  context that neighbors compose around. ⇒ a purely anchor-row-local latent op is likely INCAPABLE of
  producing attraction; the only latent-space route is to make the anchor **low-noise EARLY** (auto
  attention-attractor), i.e. anchor-then-release is not "another blend schedule," it's THE latent route
  to attraction, working through attention. **Concrete candidate:** hold row resolved (x=clean, low t)
  for k<k_sw (attraction while composition locks) → RE-NOISE to m'·σ_sw at k_sw → release into per-row
  img2img + denoised-correction. Composes {anchor + re-noise-at-release + denoise + correction}. Open:
  k_sw (σ where composition locks, ~shift_video-tied), m', fresh-vs-correlated re-noise. Test = does it
  WIDEN the basin vs the ≤0.15 denoise-only window, or just move the same needle?

### User's "switched-mode" (2026-08-24) — = route-1 + a cond token; convergent, cond redundant/risky

**Proposal:** inject into BOTH latent & cond rows; for X steps hold the latent row locked + run the cond at
some aug; at a release point, unlock the latent to denoise for Y steps + drop the cond. **This IS route-1
anchor-then-release** (hold→release the latent), re-derived independently — good, it's our top build. The cond
token is the only addition; assessment:
- **Cond during HOLD is largely REDUNDANT.** A locked clean latent row (low `t_row`) is ALREADY the trusted
  low-noise field neighbors compose around (the whole point of the hold). Adding a cond token duplicates that.
  If added anyway, it MUST be clean (`aug≈0.999`) — a fractional-aug cond is a contagion source
  ([conditioning-row-inject](../conditioning-row-inject.md), 0.2MP aug=0.4 test).
- **"Cond @ aug 0" at release ≠ off.** `aug=0` = pure-noise reference (max contagion), NOT removal. To drop
  the cond, remove the token. Semantic correction to the proposal.
- **The real handoff is on the LATENT, and the proposal omits it:** releasing a clean row into a σ_sw stream
  labeled at low t without RE-NOISING = clean-content-at-noisy-label mismatch = the test-2 contagion again.
  Route-1 already requires **re-noise to `m'·σ_sw` + denoised correction** at release — that, not the cond, is
  what makes "cond & latent agree at release." The cond can't supply it.
- **The `aug=(1−d)` arithmetic — right spirit, wrong variables.** The handoff-consistency instinct is exactly
  route-1's re-noise constraint. But (a) the split point `k_sw` is set by where COMPOSITION LOCKS
  (`k_sw = measured k_comp`, ~shift_video-tied), NOT by `(1−d)` of the steps; (b) the release re-noise level
  is `m'·σ_sw` (remaining-schedule + m'≈0.3–0.5), NOT directly `(1−d)`. The proposal conflates the final
  denoise fraction `d`, the switch point, and the re-noise level — three separate quantities.

**Net:** switched-mode ⇒ build **pure route-1 (no cond)** as the MVP (already the top build). A **clean-aug
cond during HOLD only** is a legitimate *second* variant to A/B against the MVP (possible neighbor-anchor
boost), but today's contagion data makes it lower-priority than shipping route-1 itself. The genuinely
distinct cond idea remains **live-cond-mirror (variant #4)** — cond carries the LIVE (evolving) anchor at
release, not the stale clean still — which is the principled "keep cond at release," but it has the
positive-feedback lock risk ([conditioning-row-inject](../conditioning-row-inject.md) exp order §2).
