<!-- provenance: theory (analysis + source-grounded; blended fade design analysis + SHARPENED cond-token decoupler GPU-confirmed @1MP 2026-08-24) -->
<!-- verified: 2026-08-24 · mc @d299ea5 · comfy-ref @b78cec87 (+ GPU: MC keyframes blend r40/r60 @1MP, anchors m=0-wrong) -->
# Fade on the conditioning path and the cond-token crux-decoupler

Index: [conditioning-row-inject](../conditioning-row-inject.md). What the cond path is: [what-and-verdict](what-and-verdict.md). Aug falsifications and contagion: [aug-mechanism](aug-mechanism.md).

## Can we build a blended fade in/out on the conditioning path? (user's core question)

**Yes — but only by CONSTRUCTING it ourselves; there is no native fade knob.** Two source-allowed
levers (comfy-ref @b78cec87):

1. **Per-keyframe strength via latent pre-blend.** ⚠ **FALSIFIED 2026-08-24 — see the ⚠ FALSIFIED
   section below.** A latent-only pre-blend leaves the cond timestep label at "clean" (global,
   un-relabelable per-row) → the DiT reads noised content as clean signal → static. `aug` is a COUPLED
   (noise+timestep) pair; noising the latent without relabeling the row's timestep is broken, and the
   timestep is not per-row-settable for cond rows. Kept here for provenance; do not use.
2. **Per-timeline-position fade via graduated placement.** `cond_t = cursor + FRAME_RESCALE·
   resolved_frame_index` is an unquantized float RoPE coord (`model.py:343-352`); nothing stops
   emitting the SAME base image at several `resolved_frame_index` values, each pre-blended (lever 1)
   to a decreasing strength. The full-attention copy mechanism then yields a graduated influence
   across the timeline — a fade in/out, no core-model change.

**The catch that reframes the whole question:** every graduated placement is another full-attention
cond segment, so a smooth fade **multiplies the attention cost** — worsening the exact downside
(finite attention budget) that makes the user default to latent injects. A latent fade costs zero
extra attention. AND a cond inject **doesn't produce a preserve-seam pop** in the first place (the
output row is *generated* to match, not frozen — no hard boundary), so it doesn't NEED a fade to
hide a seam the way our latent path does. A single cond placement already gives a soft *implicit*
falloff (holistic influence attenuates with RoPE distance). So on the conditioning path a "fade" is
about **dialing how far/strong the keyframe's reach extends**, not about blending away a pop.

**Design read:** a conditioning-path fade is *buildable* (levers 1+2) but is a poor trade for the
latent-fade use case — it pays attention for a seam that isn't there. Where it could genuinely win
is the **holistic-guidance** use case the user flagged (steer the *whole* clip toward a keyframe),
when the attention budget is free. Verdict unchanged: not a substitute for the latent path, and
building a dedicated fade node isn't worth it now. UNVERIFIED on GPU.

## Open hypothesis (UNVERIFIED — interop probe, not a build)

**Combining** both on the SAME keyframe may improve blend quality: our per-row img2img latent anchor
(controllable `min_denoise`) PLUS the same still as one native cond row (single placement, cheap on
attention) for holistic steer. The cond row may tighten motion/identity on the *generated blend
rows* and reduce VAE-interior softness, at ~zero ghost risk. Test as **pass-through interop first**
(wire MC's `H3 Custom Keyframes` upstream of our sampler — it already flows through, see below),
measure, and only then consider a convenience toggle.

## SHARPENED (2026-08-24) — the cond token as a NATIVE crux-decoupler (route 3 by other means)

**New GPU data (user, 2026-08-24):** MC "H3 Custom Keyframes" (= this native `minimax_keyframes`
path) blends r40 & r60 **very well at 1MP**, but the anchors look "wrong" — the output rows COPY the
near-clean reference (m=0-equivalent), so they don't match the gen. This is a clean **high-res GPU
confirmation of the attention-anchoring half** (neighbors compose around a single anchor at 1MP —
what FL2VA only suggested; see [highres-underdenoise-model.md](../highres-underdenoise-model.md) crux).

**The decoupling insight:** the cond token carries its OWN timestep label (`aug≈0.999`,
`model.py:583-585`), living at the same `cond_t` as the anchor's output row but INDEPENDENT of that
row's `denoise_mask`/`t_row`. That is precisely the crux's coupled scalar split onto two tokens =
the user's "denoise the anchor 45%, neighbors attend at ~100%" — achieved NATIVELY (no attention-logit
patch). **Hybrid to test:** same keyframe on BOTH paths — latent row r40 at `m=0.45` (our fractional
regen of the anchor itself) + one cond token at `cond_t=r40` (the high-res-confirmed neighbor anchor);
trust knob (`aug`) and denoise knob (`t_row`) now on separate tokens.

**Honest tension (why it's a lead, NOT a slam dunk — do not oversell, cf. the FL2VA overreach):**
1. The cond token's strongest copy-pull lands on the SAME `cond_t` as the anchor's own output row →
   it may FIGHT the 45% denoise and re-freeze the anchor (worse, not better). Tuning `aug` DOWN
   trades anchor-freeze against neighbor-anchor strength — a 2-knob search (`m`, `aug`).
2. Neighbors attend to the CLEAN keyframe (cond token), while the anchor row is a 45%-regen → a
   possible new mini-seam between a keyframe-blended neighborhood and a partially-regenerated anchor
   (mitigated if 45%-regen stays keyframe-close, unproven).
3. Does NOT by itself fix mechanism-A basin-collapse in the latent row (anchor still barely denoises
   @1MP); it fixes the NEIGHBOR-blend, and the cond source-pull may deepen the freeze. So the hybrid
   plausibly buys clean neighbor blending but NOT the anchor's own fractional regen — the harder half.
**Cheap to probe** (pure pass-through interop, no build): wire MC keyframes upstream of our sampler,
run r40 at latent `m=0.45` + cond token, sweep `aug`, read seam z + anchor T_realized. Higher-value
than pure-latent AtR ONLY if probe #1 shows the anchor keeps regen at some `aug`; else AtR/route-1
stays primary.
