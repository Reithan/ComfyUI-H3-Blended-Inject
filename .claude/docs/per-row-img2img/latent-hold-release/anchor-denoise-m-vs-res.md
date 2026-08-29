<!-- provenance: status (SUPERSEDED — route-1; GPU probes + analysis; is m the lever for a single-frame keyframe, or resolution?) -->
<!-- verified: 2026-08-25 · GPU r40 m=0.5/0.99 hold=0.5 probes + sampler.py re-noise/correction trace @proto-latent-hold-release -->
# Anchor denoise: is `m` the lever, or resolution? (Findings 13–14, SUPERSEDED)

Continues [anchor-denoise-after-clean-fix](anchor-denoise-after-clean-fix.md) (Finding 12).
Index: [index](index.md).

## Finding 13: the m=0.99 probe — contagion AFFECTS the amount

GPU probe (2026-08-25): r40 **m=0.99**, hold=0.5 (release step 10, same sigma schedule). Re-noise level =
`0.99·0.975 ≈ 0.965` (near-full noise) — r40 does a near-full redraw. **Result: final r40 "looks like a
traditional 0.5 denoise."**

Interpretation: r40 was ~fully re-drawn (level 0.965 ≈ full), yet the final looked ~50%. So the ~50% is NOT
purely r40's own denoise strength. What this shows: **temporal attention to co-denoising neighbors (contagion)
strongly AFFECTS r40's final result** — r40's redraw is shaped to be consistent with a neighborhood that is
itself mid-denoise (~50% through the shared tail, steps 10–20). It does NOT show contagion *sets* or clamps
r40 to a fixed amount. Correction logged 2026-08-25: "contagion sets the amount" was an overclaim.

The m=0.5-vs-m=0.99 A/B (both hold=0.5, same seed/schedule) is the cleanliness comparison:
- **m=0.5:** r40 under-denoised, texture + hard edges, poor fit (Finding 12).
- **m=0.99:** r40 "looks like a traditional 0.5 denoise" — the desired result.

HYPOTHESIS for WHY: a half-denoise (m=0.5) is an off-manifold blend of source-still + partial-redraw,
while a near-full denoise lands on-manifold and neighbor-coupling supplies the ~50% consistency. Mechanism
(on- vs off-manifold) is inferred from two runs, not demonstrated.

## Finding 14: the res-compression alternative (leading explanation) + mechanism dispute

The 3-point m-to-visual mapping (m=0.5→~0.1, m=0.99→~0.5, hold-independent) is more parsimoniously explained
by documented high-res underdenoise (nominal m res-compressed at 1MP) than by "hold > m" or contagion.
Under it **m is the content lever** (res-compensate its nominal value) and **hold is the seam lever** — do NOT
collapse them; any remap should be `f(m, resolution)`, not `f(m, hold)`.

Runs were 0.5MP (a proven 1MP proxy, same error modes, ~1/4 runtime), so the res-compression reading is in-scope.

**Mechanism is contested:**
- **Per-frame basin-sharpening:** r40's own denoise is res-dependent; a solo frame underdenoises at high res.
- **Attention dilution/contagion (user's leading theory):** r40's own denoise is res-INVARIANT; the effect is
  neighbors holding the single-frame inject harder at high res. NOT raw token ratio (all frames scale together)
  but raw token COUNT — more neighbor tokens pin the inject's form more rigidly. A good-sized fade DILUTES this
  (faded injects work); single-frame/no-fade injects get it worst. "Raw-count has the same scaling hole as
  ratio" caveat RETRACTED: ratio-invariance says nothing about attention quality, which degrades with token
  count (softmax entropy, RoPE range stretch). Count-based basis is well-grounded.

**Still open = sign/pathway (P1 vs P2):** both stories end in under-denoise but predict opposite internals.
(P1) neighbors PIN r40 — dilution weakens the pin at high res (wrong sign). (P2) r40's own clean signal
can't assert itself, spread thin across many weak high-res tokens; dilution worsens with res (right sign).

**The fade observation discriminates:** pure per-frame basin-sharpening still res-compresses each faded row's
own realized denoise — it does NOT predict "faded injects work." The attention family does predict it (graded
neighbors = smooth handoff). So "fades help, single-frame worst" favors attention over pure per-frame.
Confound: a video inject differs from a single-frame still in TWO ways — fade envelope AND motion content.

**Still-repeat-with-fade on r40 — ALREADY RUN (user, 2026-08-25):**
1. The fade promotes a better blend/seam; m=0 mid-keyframes still behave.
2. NOT viable: (a) spreading one static frame anchors its own artifacts across the span; (b) the model
   correctly reads the static repeat as a FREEZE-FRAME, destroying motion continuity.

Payoff: temporal support fixes the seam (d_blend) but NOT the anchor's own redraw/artifacts (d_content).
**Hard new constraint: no fix may literally replicate the frame** — this cautions route-3 too.
**Surviving direction:** raise the single frame's OWN effective denoise (high/res-compensated m) to clean
artifacts on-manifold, while real moving neighbors supply the ~50% look.

Canonical model: [../highres-underdenoise-model/the-real-bug](../highres-underdenoise-model/the-real-bug.md).

## Why "init low noise + full (m=1) denoise" is not a clean test — schedule/label consistency

*(Route-1/SUPERSEDED; preserved as a design-safety note so this mis-designed test isn't re-proposed.)*

Proposed scheme: init r40 at `hold·m`=25%, hold to step 10, then denoise at m=1 (mask m→1). Two traps:

1. The per-row DiT label is `t_row = m·σ`. **m=1 labels r40 at the FULL σ** (0.975 at release).
2. The release re-noise level is `m·σ_sw`. Setting m=1 makes it re-noise to `≈0.965` (near-full),
   **overwriting the 25% init** ⇒ collapses to the m=0.99 probe (Finding 13) above.
3. If you also DISABLE the re-noise to keep 25%, r40 is a **25%-noisy frame LABELED 97.5%** → OOD; the
   model strips ~full noise off a near-clean frame → **washes r40 out**. That answers "what does the model
   do when lied to," not the intended question.

The m-compression EXISTS to keep noise-level and `t_row` label consistent — it already *is* a
schedule-consistent m-strength SDEdit that reaches 0 noise. A genuine "init-noise instead of m" test must be
proper SDEdit: re-noise to level L **and** release at the step where σ≈L, then m=1 — tying the release step
to L (not `hold_frac`); likely unnecessary since m-compression gives the equivalent without moving the
release step. Also: `hold·m` is not the right level — SDEdit strength-s starts at σ≈s, so a 0.5 look wants
~50% init, not 25%; `hold` governs anchor exposure, not noise level.
