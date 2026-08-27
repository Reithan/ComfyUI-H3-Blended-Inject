<!-- provenance: status (GPU result + code analysis; next problem after the provenance fix landed) -->
<!-- verified: 2026-08-25 · GPU A/B (md=0.0 sanity + r40 md=0.5 hold=0.5) + sampler.py re-noise/correction trace @proto-latent-hold-release -->
# Anchor under-denoise after the clean keyframe-only hold (Finding 12)

Follows [hold-mechanism-and-confounds](hold-mechanism-and-confounds.md) (Findings 7–11, provenance
confound + fix). That fix is now GPU-CONFIRMED; this doc opens the NEXT problem: the held keyframe
denoises too little.

## GPU result (2026-08-25) — provenance fix confirmed, r40 under-denoises

Same gen: euler / linear_quadratic / 20 steps / one fixed seed. Config = opening video fade-out +
1-frame r40 + 1-frame r60.

- **Sanity (md=0.0, hold=0.5):** logs `hold-and-release inert` (no fractional TRUE-keyframe rows);
  output bit-matches the no-hold run. ⇒ the fade-out is no longer caught. **Provenance fix works.**
- **Real (r40 md=0.5, hold=0.5):** `armed: 97,920 keyframe anchor row-elems` (down from ~1.08M when
  the fade-out was caught — Finding 10). Timeline attracts/blends toward r40 (steps 2, 5, 13); final
  shows a **solid** blend from neighbors in/out of r40. Release log: hold 10/20, release σ=0.975,
  `|clean|=|x0|=0.7771`, post-hold `|x_mid|=0.7771`, re-noised to `|x|=0.5554` at `level=0.4875`
  (= m_release 0.5 · σ_sw 0.975). Every hold-side number is as designed.
- **The problem:** r40 ITSELF comes out **under-denoised** — residual texture + hard edges, and an
  abrupt/poor "visual fit" into the timeline. Because the blend is now correct, r40's flaws
  propagate to its neighbors. User goal: **make r40 denoise more** and see if a better r40
  propagates a better neighborhood. (r60 is a second 1-frame keyframe and is armed/held too — the
  97,920 elems span both r40 and r60 rows; r40 is just the diagnostic row with clearer visual tells,
  so it stands in for both here.)

## Why r40 under-denoises — the tail is fractional-m img2img, NOT the hold

Two code facts (`sampler.py`) that separate the hold from the tail quality:

1. **The correction caps redraw at m.** After release the row runs ordinary per-row img2img:
   `corrected = m·denoised + (1−m)·clean`. At m=0.5 the row's output is *structurally* a 50/50
   blend of the model's free denoise and the original clean still — it CANNOT move more than
   halfway off the source. "Under-denoised at m=0.5" is the DESIGNED meaning of m=0.5, not a hold
   defect. Direct lever to denoise more = **raise min_denoise**.
2. **The hold covers a near-flat sigma region.** k_sw=10 → release at σ=0.975; steps 0→10 move
   global σ 1.000→0.975 (**2.5%** of the range), steps 10→20 move it 0.975→0 (**97.5%**). The real
   denoising is entirely in the post-release tail, and the release re-noise (`eps` recovered from
   the ORIGINAL x_global, line 465) places r40 onto exactly the m=0.5 level it would hold at σ=0.975
   with no hold. ⇒ **PREDICTION (untested): r40's OWN final quality ≈ the no-hold run at the same
   m.** The hold changes the NEIGHBORS' early composition, not r40's tail denoise.

## Candidate explanations for "hard edges / poor fit" (not yet discriminated)

- **(A) magnitude** — m=0.5 simply keeps too much source; the 50/50 blend reads as under-denoised.
  Raising m cleans it up.
- **(B) blend drift** — the model's redraw drifts spatially from the clean still, so the 50/50
  blend is a faint double-image → edges. Higher m shrinks the clean half but the redraw may still
  not register.
- **(C) route-1 inconsistency** — during the hold, neighbors compose around **clean** r40; after
  release r40 redraws and moves AWAY from that clean field, so the neighborhood was fit to a target
  that no longer exists → boundary conflict / abrupt fit. This one PREDICTS fit gets WORSE the more
  r40 redraws — the opposite of what raising m is meant to achieve.
- **(D) resolution** — if r40 stays under-denoised even at high m, this is the known single-frame
  high-res under-denoise ([highres-underdenoise-model](../highres-underdenoise-model/index.md)),
  orthogonal to the hold.

## Discriminating tests (settings vs mechanism — the user's open question)

1. **Raise min_denoise** (0.5 → 0.7 → 0.85), hold on. Cleaner AND better fit ⇒ (A)/(B), a
   **settings** answer. Fit gets WORSE as r40 redraws further ⇒ (C), a **mechanism** answer
   (route-1 self-defeats: it optimizes attraction-to-clean at the cost of redraw-consistency).
2. **Hold vs no-hold at the SAME m** — compare r40's OWN quality separately from neighbor fit.
   Equal r40 quality ⇒ hold is orthogonal to the anchor (confirms the prediction above), so "denoise
   more" is purely a min_denoise question and the hold's only job is neighbor composition. Hold
   worse ⇒ the hold is degrading the anchor and (C) is live.
3. **If r40 stays under-denoised even at high m** ⇒ (D); re-test at lower resolution to confirm
   resolution-dependence, and treat as the separate high-res track.

Recommend running T1 and T2 before touching the mechanism: they cost only a min_denoise or hold toggle
and tell us whether the fix is a setting or a redesign. Do NOT pre-emptively change the re-noise
level or k_sw — that only matters if T2 shows the hold itself is the culprit.

## Findings 13–14 + the schedule-consistency argument → sibling doc

The m=0.99 probe (contagion AFFECTS the amount), the m=0.5-vs-0.99 cleanliness A/B, the res-compression
vs attention-dilution mechanism dispute, and why "init low noise + m=1" isn't a clean test now live in
[anchor-denoise-m-vs-res](anchor-denoise-m-vs-res.md) — is `m` even the right lever for a single-frame
keyframe, or is it resolution?
