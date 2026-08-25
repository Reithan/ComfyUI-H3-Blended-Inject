<!-- provenance: status (candidate-fix scoreboard + reframing implications; each verdict carries its own epistemic label) -->
<!-- verified: 2026-08-24 · user GPU runs 0.5MP; 1MP window-closed data in ../highres-underdenoise-model/data-runs.md -->
# Candidate fixes (scoreboard) & what the reframing changes

Parent: [isolated-frame-attention-support](../isolated-frame-attention-support.md).

## Candidate fixes

Ordered cheap→invasive. All aim at the same target: **restore the isolated frame's attention weight**.

1. **Clean cond token (aug≈0.999)** — ⚠ TESTED: clean blend in/out but the keyframe comes out
   **unchanged (not denoised)** = **re-freeze CONFIRMED** — a clean guide is a hard "this frame is set"
   reference. Good for blend, useless for anchor denoise. Use it for the blend, get the denoise elsewhere
   (route-2 / timed-removal). The **always-on hybrid** from
   [keyframe-two-views](../keyframe-two-views-and-knobs.md) in clean form.
2. **Cond token with TIMED REMOVAL** (user) — inject ONLY the cond row at normal `aug≈0.999`, then
   **remove that cond row at `s·(1−d)`** (or when the frame's measured noise level reaches `d`). Gives
   full clean support early, releases before it can re-freeze the anchor. Open problem: no clean metric
   for "noise level reached d" mid-sample; step-count proxy `s·(1−d)` is the tractable version.
   *Structural note:* timed-removal IS route-1's hold-and-release on knob **C** (cond token) not knob **B**
   (latent mask), so it inherits the [data-runs.md](../highres-underdenoise-model/data-runs.md) Ψ=+3 result:
   anchor commits last at EVERY d under B-only control → the impossibility **backs** timed-removal. Knob-B
   attractor ≡ Fable's λ(σ) source-spring ([crux-and-mechanism-2](../highres-underdenoise-model/crux-and-mechanism-2.md));
   ghost-free condition (λ→0 before render) is timed-removal's safety bound; k_comp gives k_sw.
   Design + build: [timed-cond-removal-prototype](../timed-cond-removal-prototype.md).
3. **Temporal duplication into support frames** — ⚠ TESTED (0.5MP, [gpu-test-0.5mp](gpu-test-0.5mp.md)):
   fixes the blend but the model reads identical rows as a **FREEZE** (motion pause + double-exposure)
   and it does NOT resolve the anchor. Self-defeating for anchor-resolution — a demoted dead-end, kept
   as a negative result.
4. **Route-3 attention-logit boost** — bias neighbor→anchor logits directly. Caveat (per two-views doc):
   this boosts attention to the anchor's ACTUAL partially-noised state, so it aids support but not the
   "see it as clean" half of the ideal.
5. **img2img crop/upscale/denoise/downscale** — ⚠ REFUTED (paradox — see
   [in-context-paths](in-context-paths.md); H3 has no single-frame img2img mode).

## What this reframing changes

- Two distinguishable failures: neighbor-blend (FIXED by support) and anchor-resolution. Anchor-
  resolution is helped by NON-identical coherent context (r60) but NOT by self-duplication (r40 froze).
- Blend: a **clean cond token** works (confirmed). Duplication freezes (route 3); fractional cond
  contaminates. The cond `aug` scalar can't give blend + anchor-denoise + no-contagion at once.
- Anchor-denoise must happen IN-CONTEXT (bake-beforehand is REFUTED — paradox; single-pass decouple is
  FALSIFIED — clean cond + raw latent = freeze + contagion). Surviving in-context options: route-2 two-pass
  (blocked at 1MP unless pass A gets coherent context — r60) and timed-removal / mode-switch hold-and-release.
  Cond-channel timed-removal builds first (H3AddGuide), but a LATENT-resident hold-and-release remains an
  active parallel goal (user, 2026-08-24) — see [status-and-open-paths](../status-and-open-paths.md) path 1.
  LATENT-path d-tuning (T_N-corrected realized-m): window CLOSED @1MP per
  [data-runs.md](../highres-underdenoise-model/data-runs.md) — those runs had no cond token, so they WERE the
  latent-only probe; 0.68 locks, 0.75/0.78 smear, no coherent middle. Valid only at ≤0.5MP or as a component
  inside a release phase.
- **Blurred-reference: REFUTED standalone (user, 2026-08-24).** Blurred reference = blurred output;
  attractor pulls every step including render phase. At most an ablation inside timed-removal.
- **Evenly-spaced-keyframes reframe (user, 2026-08-24).** More evenly-spaced high-quality keyframes
  convert the broken isolated case into the working series case. When no extra stills exist, cross-res
  pass A can manufacture realized neighbors as faded support (2d-via-2a synthesis).
- Anchor-resolution ceiling is still governed by T_N; if in-context support is insufficient, a dedicated
  fix (T_N-corrected realized-m or route-2 two-pass) is the fallback. Track it in
  [highres-underdenoise-model](../highres-underdenoise-model.md) (the T_N model + fix-strategies).
