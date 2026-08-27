<!-- provenance: status (GPU probes + analysis; is m the lever for a single-frame keyframe, or resolution?) -->
<!-- verified: 2026-08-25 · GPU r40 m=0.5/0.99 hold=0.5 probes + sampler.py re-noise/correction trace @proto-latent-hold-release -->
# Anchor denoise — is `m` the lever, or resolution? (Findings 13–14)

Continues [anchor-denoise-after-clean-fix](anchor-denoise-after-clean-fix.md) (Finding 12: the held
keyframe under-denoises; candidates A–D; tests T1–T3). This doc holds the m-probe evidence and the two
competing explanations. Index: [index](index.md).

## Finding 13 — the m=0.99 probe: contagion AFFECTS the amount; m sets manifold-cleanliness

GPU probe (2026-08-25): r40 **m=0.99**, hold=0.5 (release step 10, same σ schedule). Re-noise level =
`0.99·0.975 ≈ 0.965` (near-FULL noise) → r40 does a near-FULL redraw. By SDEdit logic that should give
a *fresh* frame nearly independent of the inject. **Result (user): final r40 "looks like a traditional
0.5 denoise."**

Interpretation (careful) — r40 was **~fully re-drawn** (level 0.965 ≈ full), yet the final looked ~50%.
So the ~50% is NOT purely r40's own denoise strength (neither m≈1 nor noise≈full would yield 50% alone).
What this shows: **temporal attention to r40's co-denoising neighbors (contagion) strongly AFFECTS its
final result** — r40's redraw is shaped to be consistent with a neighborhood that is itself mid-denoise
(~50% through the shared tail, steps 10–20). It does NOT show contagion *sets* or clamps r40 to a fixed
amount — that (whether/how the final amount is controllable) is still open. Correction logged 2026-08-25
(the "contagion sets the amount" phrasing was an overclaim).

The m=0.5-vs-m=0.99 A/B (both hold=0.5, same seed/schedule) is ALREADY the cleanliness comparison:
- **m=0.5:** r40 under-denoised, texture / hard edges, poor fit (Finding 12).
- **m=0.99:** r40 "looks like a traditional 0.5 denoise" — the desired result.
⇒ **higher m gave the better-looking r40** in this config. HYPOTHESIS (unconfirmed) for WHY: a
half-denoise (m=0.5) is an **off-manifold blend** of source-still + partial-redraw (→ artifacts), while a
near-full denoise lands on-manifold (clean) and neighbor-coupling supplies the ~50% consistency. If that
holds, the practical recipe is **high m + let attention blend**, not low-m compression — but the
mechanism (on- vs off-manifold) is not yet demonstrated, only inferred from the two runs.

OPEN (genuinely new, not a re-run): (1) does high-m generalize — does **r60** behave like r40? (2) is the
final amount **controllable**? If neighbor-coupling holds r40 near ~50% regardless of m, then dialing a
30%/70% result may need a different lever than m; unproven either way.

## Finding 14 — the res-compression alternative (leading explanation) + the mechanism dispute

The 3-point m→visual mapping (m=0.5→~0.1, m=0.99→~0.5, hold-independent) is more parsimoniously explained
by the DOCUMENTED high-res underdenoise (nominal m res-compressed: realized ≪ nominal @1MP) than by
"hold > m" or contagion. Under it **m is the content lever** (res-compensate its nominal value) and **hold
stays the separate seam lever** — so do NOT collapse them (`new_m=f(m,hold)` re-conflates two knobs the
model needs apart; any remap should be `f(m, resolution)`).

**Runs were 0.5MP** (user, 2026-08-25) — a proven 1MP proxy (same error modes, milder, ~1/4 runtime), i.e.
the high-res regime. So the res-compression reading is in-scope.

**Mechanism is contested (this is the open axis):**
- **Per-frame basin-sharpening** (the-real-bug's framing): r40's OWN denoise is res-dependent; a solo
  frame would underdenoise at high res.
- **Attention dilution / contagion** (user's leading theory): r40's own denoise is res-INVARIANT; the
  effect is entirely neighbors "holding" the single-frame inject harder at high res / longer timelines.
  Caveat: raw token *ratio* is res-invariant (all frames scale together → r40's share stays `1/N`), so the
  res-dependence needs a subtler attention effect (spatial-locality/softmax), while **timeline length** is
  the cleaner axis. Not disproven; user reports supporting experiments.

**Refined dilution mechanism (user, 2026-08-25):** NOT ratio — **raw number**. At higher res an inject is
surrounded by (raw count) more "contagious" neighbor tokens pinning its form more rigidly; a good-sized
fade **dilutes** this across the ramp (so faded video injects still work), while **single-frame / no-fade
injects** get it worst — exactly what the mechanism predicts.
**CORRECTION (2026-08-25): my earlier "raw-count has the same scaling hole as ratio" caveat is RETRACTED.**
The `(N-1)/N` argument shows only that the *share* of attention on neighbors is scale-invariant — it says
NOTHING about attention *quality*, which is exactly what degrades with raw token count (softmax entropy
grows over more keys; per-key resolution flattens; RoPE ranges stretch). High-res DiT generation and
long-context attention both empirically suffer from *number of tokens*, independent of ratio. So the
count-based basis is well-grounded; ratio-invariance does not refute it.
**Still OPEN = the sign/pathway** (the "exact mechanics undetermined" part): two stories both end in
under-denoise but predict opposite internals — **(P1)** neighbors PIN r40 (dilution would *weaken* the pin
at high res → wrong sign); **(P2)** r40's own clean signal can't ASSERT itself, spread thin across many weak
high-res tokens so it fails to resolve within the truncated m-schedule (dilution worsens with res → right
sign). Also on the table as res-knobs: spectral-detail (coarse@0.2MP vs fine@1MP) + RoPE-locality.

**The fade observation is the strongest evidence — and it discriminates.** A pure per-frame mechanism
(basin-sharpening) would still res-compress EACH faded row's own realized denoise → it does NOT predict
"faded injects work." The attention family DOES (graded neighbors = smooth handoff; the inject's own frames
mutually anchor vs the timeline). So "fades help, single-frame worst" favors attention over pure per-frame.
Confound: a video inject differs from a still single-frame in TWO ways — fade envelope AND motion content.

**Discriminators.** (a) **0.2MP on the current setup** confirms res IS the cause (m=0.5,hold=0.5 → ~0.5).
(b) **r60** is a free second single-frame point already in the run. (c) A **timeline-length sweep at fixed
res & m** (dilution ⇒ length-dependent; basin ⇒ length-invariant) is the cleanest mechanism split but is
**DEFERRED by the user** — a coherent multi-inject timeline is not free; extract from the current setup first.

**Still-repeat-with-fade on r40 — ALREADY RUN (result, user 2026-08-25).** Replacing the 1-frame still with
a multi-frame repeat of the SAME still + fade:
1. The fade DOES promote a better blend/seam — same as a real fade-in/out inject; m=0 mid keyframes still
   behave; a repeated-still fade region works generally like a normal multi-frame inject.
2. NOT viable, two reasons: (a) spreading one static frame **anchors its OWN artifacts** (bad textures,
   harsh geometry) across the whole span — denoise/blend removes only some; (b) the model **correctly reads
   the static repeat as a FREEZE-FRAME**, destroying motion continuity in the region.

Payoff: temporal support fixes the **seam (d_blend)** but NOT the anchor's own **redraw/artifacts
(d_content)** — the two-axis split is confirmed. It does NOT resolve P1-vs-P2 (the m-cap keeps source
artifacts at m<1 regardless → confounds the anchor read). **Hard new constraint: no fix may literally
replicate the frame** (freeze-frame). This CAUTIONS route-3 too — boosting attention *to* a lone anchor can
make neighbors conform to it → the same static-region read. **Surviving direction:** raise the *single
frame's OWN* effective denoise (high / res-compensated m — the m=0.99 single-frame run already looked best)
to clean its artifacts on-manifold, while real moving neighbors supply the ~50% look and motion is preserved.
Open: is that ~50% look controllable, does r60 generalize (Finding 13). Canonical model:
[../highres-underdenoise-model/the-real-bug](../highres-underdenoise-model/the-real-bug.md).

## Why "init low noise + full (m=1) denoise" is not a clean test — schedule/label consistency

Proposed scheme: init r40 at `hold·m`=25%, hold to step 10, then denoise at m=1 (mask m→1). Two traps:

1. The per-row DiT label is `t_row = m·σ`. **m=1 labels r40 at the FULL σ** (0.975 at release).
2. The release re-noise level is `m·σ_sw`. Setting m=1 makes it re-noise to `≈0.965` (near-full),
   **overwriting the 25% init** ⇒ collapses to the m=0.99 probe above.
3. If you also DISABLE the re-noise to keep 25%, r40 is a **25%-noisy frame LABELED 97.5%** → OOD; the
   model strips ~full noise off a near-clean frame → **washes r40 out**. That answers "what does the
   model do when lied to," not the intended question.

The m-compression EXISTS to keep noise-level and `t_row` label consistent — it already *is* a
schedule-consistent m-strength SDEdit that reaches 0 noise. A genuine "init-noise instead of m" test
must be proper SDEdit: re-noise to level L **and** release at the step where σ≈L, then m=1 — which ties
the release step to L (not to `hold_frac`). Likely unnecessary, since m-compression gives the
equivalent without moving the release step. Also: `hold·m` is not the right level — SDEdit strength-s
starts at σ≈s, so a 0.5 look wants ~50% init, not 25%; `hold` governs anchor exposure, not noise level.
