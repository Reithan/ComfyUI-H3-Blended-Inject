<!-- provenance: theory (UNVERIFIED — reframing from user GPU observations 2026-08-24; the attention-dilution mechanism is analytical; child docs carry the test data and per-path verdicts) -->
<!-- verified: 2026-08-24 · comfy-ref @b78cec87; 1MP window-closed data in highres-underdenoise-model/data-runs.md -->
# The failure is an ISOLATED single fractional frame, not fractional-denoise in general

Reframing (user, 2026-08-24). Sibling docs: [highres-underdenoise-model](highres-underdenoise-model.md)
(the crux / T_N transfer function), [keyframe-two-views-and-knobs](keyframe-two-views-and-knobs.md)
(the four knobs), [conditioning-row-inject](conditioning-row-inject.md) (aug/cond facts).

## The key observation

Our per-row img2img solution **already works well** on:
- **Video injects** (a contiguous run of injected rows), and
- **A SERIES of keyframes with fade regions** around them.

It fails **specifically** on the **isolated single frame at `0 < d < 1`** — the "pop"/"smear". So the
bug is NOT "fractional denoise is broken." It is narrower: a *single unsupported* fractional row.

## Why — attention dilution ∝ tokens ∝ resolution

A single fractional row at `0<d<1` needs two things — originally hypothesized to BOTH depend on
attending to its neighbors (⚠ the GPU test REFINES this: only #1 does):
1. **Support** — neighbors must attend TO it strongly enough to blend toward it (neighbor-view).
2. **Its own denoise** — the row resolves its noise (hypothesized: by attending to surrounding context;
   GPU-refined — support does NOT fix this, so anchor-resolution is the row's own T_N compression).

When the frame is **isolated**, it has no adjacent same-content rows reinforcing it. As resolution
rises, **tokens/frame rises** (1MP ≈ 4128 tok/frame vs 0.2MP ≈ 836), so any single frame is a smaller
fraction of the total attention mass → its mutual attention with neighbors is **diluted**. Below some
token budget the frame can neither pull neighbors nor pull enough context to denoise itself → it stays
near its injected (noised) state = the pop. Fade-regions / keyframe-series don't hit this because the
**neighbors provide mutual support** — multiple rows share the content, so the attention mass survives
dilution.

This is the same phenomenon the [T_N(d) transfer function](highres-underdenoise-model.md) captures
(near-identity at 0.2MP, near-STEP at 1MP): the isolation + dilution IS the basin-sharpening seen at
high res. (The SDXL small-patch img2img analogy is an intuition pump ONLY — crop/upscale is REFUTED
by the paradox; H3 has no single-frame img2img mode.)

## Verdicts at a glance

- **Neighbor-blend: FIXED by support** (clean cond token). **Anchor-resolution: NOT fixed** by
  self-duplication (freeze) — but HELPED by non-identical coherent context (the r60 cross-inject).
- **Dead:** bake-beforehand (paradox), single-pass decouple (freeze + contagion), self-duplication
  (freeze), blurred-reference standalone, latent d-tuning @1MP (window CLOSED).
- **Surviving:** timed-removal (cond channel — GPU-viable @0.5MP, builds first as H3AddGuide) and
  route-2 two-pass. A **LATENT-resident hold-and-release remains an active parallel goal**
  (user, 2026-08-24) — see [status-and-open-paths](status-and-open-paths.md) path 1.

## Detail (child docs)

- [gpu-test-0.5mp](isolated-frame-attention-support/gpu-test-0.5mp.md) — the 0.5MP support-expansion
  runs: support fixes BLEND not anchor; duplication reads as FREEZE; r60 cross-inject side-effect;
  cond-`aug` one-knob-two-jobs result.
- [in-context-paths](isolated-frame-attention-support/in-context-paths.md) — why anchor denoise must
  be IN-CONTEXT; bake-beforehand + single-pass-decouple refutations; route-2, cross-res back-pocket,
  timed-removal.
- [candidate-fixes-and-reframing](isolated-frame-attention-support/candidate-fixes-and-reframing.md) —
  the 5 candidate fixes with test verdicts + everything the reframing changes (incl. the two-track
  decision and the evenly-spaced-keyframes reframe).
