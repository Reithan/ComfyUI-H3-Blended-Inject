<!-- provenance: theory (analysis synthesizing confirmed source facts into a joint mental model; the route-mapping + re-freeze risk are UNVERIFIED) -->
<!-- verified: 2026-08-24 · comfy-ref @b78cec87 model.py/model_base.py/samplers.py (knob facts source-confirmed; interplay claims analytical) -->
# The keyframe: two questions, four knobs (joint mental model, 2026-08-24)

Consolidates the session's pieces (user request 2026-08-24: "align how the neighbors SEE the keyframe
vs how the keyframe itself is RESOLVED"). Sibling: [highres-underdenoise-model](highres-underdenoise-model.md)
(the crux) · [conditioning-row-inject](conditioning-row-inject.md) (aug/cond facts).

## Split the problem into TWO independent questions

1. **Neighbor-view** — how do the OTHER rows perceive/attend to the keyframe? (governs blend/attraction)
2. **Anchor-resolution** — how does the keyframe's own OUTPUT row denoise? (governs the anchor's final look)

The whole bug is that H3's native machinery **ties these two to one scalar** (`t_row` = the latent row's mask
label does BOTH — [the crux](highres-underdenoise-model/crux-and-mechanism.md)). "Decoupling" = answering the two
questions with two independent controls.

## The four knobs (all source-confirmed)

| # | Knob | Lives on | Controls | Notes |
|---|------|----------|----------|-------|
| A | **Latent row CONTENT** | video/output row | what actually gets denoised | clean still / noise / blend |
| B | **Latent row MASK `m`** | video/output row | anchor DENOISE-RATE (`t_row=1−m·σ`) AND colors what neighbors read off the row's hidden state | the crux: ONE knob, BOTH jobs. `m=0`=preserve, `m=1`=full gen |
| C | **Cond token + `aug`** | separate cond row (never read out) | a PURE neighbor-guidance channel | `aug`≈0.999=clean reference; fractional `aug`=noisy reference that SPREADS noise ([contagion](conditioning-row-inject.md)) |
| D | **Sampler composite** (`noise_mask`) | output row, per step | re-blends clean latent into output each step | MC=on → GHOST; ours=`noise_mask=None` → ghost-free |

Neighbor-view is fed by **B (the row's label/hidden state) + C (the cond token)**. Anchor-resolution is fed by
**A (content) + B (rate) + D (composite)**. Knob **B is the only one on both axes** — that's why it can't serve
both cleanly, and why every "route" adds a SECOND channel to relieve it.

## The user's ideal (2026-08-24) restated in these terms

> Every step: neighbors treat the keyframe like any clean guide (**neighbor-view = CLEAN, always**), while the
> keyframe's own row denoises `d → 0` over the FULL step budget (**anchor-resolution = faithful img2img**).

This is exactly **decoupling the two views** — neighbor-view pinned clean, anchor-resolution running fractional.
It maps onto the routes as follows (and NOT cleanly onto route-1):

- **Always-on HYBRID (cheap, single-pass, UNTESTED in clean form):** clean cond token (**C**, `aug≈0.999`) for
  neighbor-view + our ghost-free fractional latent row (**A**=clean, **B**=`m=d`, **D**=off) for
  anchor-resolution, BOTH every step. Directly implements the ideal. **RISK = anchor RE-FREEZE:** the anchor's
  output row also attends the clean cond token (`mask=None`, no self-exclusion) and may COPY it → stop
  denoising → back to the pop. Untested with CLEAN aug (prior hybrid tests used MC-composite latent [ghost] or
  fractional aug [contagion]); this specific combo is the missing experiment.
- **Route-2 two-pass (guaranteed, 2× cost):** Pass A anchor=clean → take NEIGHBOR outputs (they saw a clean
  anchor). Pass B anchor=`m=d` denoising → take ANCHOR output. Both full-budget, every step. This is the LITERAL
  implementation of "lie to neighbors + let the keyframe denoise," with **no truncation and no re-freeze** (the
  denoising anchor in B never sees a clean copy of itself). Cost aside, it's the clean answer / oracle.
- **Route-3 attention-logit boost (single-pass, invasive):** keep the anchor at its TRUE denoising `t_row` and
  bias neighbor→anchor logits. Caveat: neighbors then attend MORE to the anchor's ACTUAL (partially-noised)
  hidden state — it boosts attention to a noisy representation, so it does NOT by itself deliver "see it as
  clean." Weaker fit to the ideal than the hybrid/route-2.
  **Briefly PROMOTED under H1, then REJECTED 2026-08-27 (latent-side mandate + SDPA kernel-fallback perf); see [schedule-tail-late-delta](schedule-tail-late-delta.md).**
- **Route-1 anchor-then-release (time-split):** does NOT match the ideal — the keyframe is frozen during hold
  (no denoise) then denoises in a TRUNCATED tail, and neighbors react to the release transition. The user's two
  worries about hold-and-release are BOTH valid; route-1 trades the ideal's simultaneity for one cheap scalar.
  The cond-channel variant (timed-removal, knob **C**) escapes this objection — see
  [isolated-frame-attention-support](isolated-frame-attention-support.md).
  **Empirical support (HOLD-24/25, GPU ~2026-08-25):** schedule-sigma per-frame pin blends cleanly yet breaks
  GLOBAL structure (anchors hidden as noise in the early structure-setting window) — concrete evidence for
  "neighbors see clean." The "neighbors see release" variant (HOLD-25) GENERALIZES "neighbors see clean" and
  is GPU-CONFIRMED: structural coherence restored (solid blend/denoise; fade tuning open). See
  [per-frame-scheduled-release](latent-hold-release/per-frame-scheduled-release.md).

## Takeaway

The user's ideal is sound and is the decoupling stated crisply. Its cheapest faithful test is the **clean
always-on hybrid** (watch for anchor re-freeze); its guaranteed-correct form is **route-2** (2× cost, no
re-freeze, no truncation). Route-1 remains the cheap single-pass fallback but is the WORST fit to the stated
ideal — note this when choosing the build order.

**Schedule-tail expression of knob-B coupling (σ-shift convexity, source-confirmed):** step-space `k_d` maps
to hugely superlinear start-σ via H3's shift (d=0.2→σ≈0.75), so a low-d inject presents as ~75%-noise to
neighbors through the structure-setting window — treated as a peer, not an anchor. Only d=0.0 gets anchor
treatment (GPU-confirmed 2026-08-27). The earlier "late-delta" timing theory is DEMOTED; the σ-shift account
subsumes it. Proposed fix = select k_d in σ-space (UNVERIFIED). See [schedule-tail-late-delta](schedule-tail-late-delta.md).
