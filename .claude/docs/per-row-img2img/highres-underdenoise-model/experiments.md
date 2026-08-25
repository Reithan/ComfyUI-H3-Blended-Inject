<!-- provenance: theory (Fable analytical model + 1MP GPU validation 2026-08-24; experiment plan post-closure) -->
<!-- verified: 2026-08-24 · 1MP GPU runs (0.45/0.60/0.68 lock, 0.75/0.78 smear=window closed, 0.83 chaos) + Fable adjudication · data in ../highres-singleframe-underdenoise.md -->
# Next experiments & the confounding axes

Index: [highres-underdenoise-model](../highres-underdenoise-model.md). The crux these experiments address:
[crux-and-mechanism](crux-and-mechanism.md). Build order from Fable: [crux-and-mechanism-2](crux-and-mechanism-2.md).
Fix strategies referenced below: [fix-strategies](fix-strategies.md).

## NEXT EXPERIMENTS (ordered; REWRITTEN 2026-08-24 post-closure / post-Fable-adjudication)
The d-sweep is DONE (window closed @1MP). d* is no longer a target. New plan:
0. **BUILD & RUN: anchor-then-release, single ISOLATED anchor, 1MP** — THE highest-information run; every
   outcome discriminates. Hold to measured k_comp; release with m′ targeting the 0.2MP realized-content
   fingerprint (calibrate m′ in REALIZED units — T_N applies to m′ too, so its "wide tolerance" is
   unproven); correlated re-noise; same seed discipline. Isolated (not two-anchor) so it ALSO gives the
   clean single-anchor confirmation of closure (both prior runs had anchor-spacing interference).
   Outcomes: (a) seam z<2 AND fingerprint-match → decoupler works, product path. (b) neighbors compose
   around the held anchor but release smears/locks → trust wasn't the release problem → calibrate m′ in
   realized units (cheap, no new mechanism). (c) neighbors DON'T compose around a held/resolved/consistent
   anchor → latent route to attraction insufficient → dilution/OOD proven → go route 3 (attention boost),
   run the two-pass oracle FIRST to establish the ceiling.
1. **0.2MP/d=0.45 fingerprint run (new instrument)** — record the coherent `(T_realized, p̂∞, amp, seam z)`
   in normalized units. This is BOTH the threshold-calibration run AND the content-target for #0's m′.
   (Do this before/with #0.)
1b. **0.2MP DENOISE SWEEP (window-WIDTH measurement; user re-verification of prior memory)** — same still,
   single isolated anchor, sweep d∈{0.2,0.45,0.7,0.9}, log seam z + T_realized at each. User RECALLS that at
   0.2MP arbitrary d all blended/resolved and the d choice was about the KEYFRAME's own content, not the
   blend. This directly measures the coherent-window WIDTH at low N — the load-bearing prediction.
   **Falsifiable prediction (basin-sharpening):** window width shrinks MONOTONICALLY with N — WIDE (most/all
   of d∈[0,1] gives seam z<2) @0.2MP → narrower @0.5MP → collapsed (cliff, no coherent d) @1MP; equivalently
   `T_0.2MP(d)` is a GENTLE near-identity slope, `T_1MP(d)` a near-STEP. **What would falsify it:** if 0.2MP
   is ALSO narrow (only a small d-band blends), the "wide-at-low-N" story is wrong and closure isn't purely
   basin sharpening — reopen mechanism B / a confound. If confirmed wide, it is the strongest support yet for
   basin-sharpening AND anchors the low-N end of `T_N(d)`.
2. **OPTIONAL science: 0.5MP crossover probe** — closure proves the lock-edge and chaos-edge have
   DIFFERENT N-scalings (they cross in N∈(836,4128)≈0.3-0.7MP). Sweep the same d-set at 0.5MP; if ANY d gives
   seam z<2 the window is still open here → crossover N\* > 0.5MP; if all closed → N\* < 0.5MP. Locates N\* =
   the max resolution at which d-only ever works. Not on the critical path, but cheap to fold into 1b.

**Framing for the user's skepticism (IMPORTANT — FL2VA claim CORRECTED 2026-08-24 from source):** "window
closed @1MP" is a statement about the *single nominal-d knob ONLY* (T_N steepens — proven), NOT that high-res
single-frame stability is unrecoverable. But be PRECISE about the existence proof — I overstated it once:
- **VERIFIED from `comfy_extras/nodes_minimax_h3.py`:** FL2VA (`MiniMaxH3ImageToVideo`) and the mid-timeline
  anchor (`MiniMaxH3AddGuide`) inject keyframes into the **CONDITIONING** (`minimax_keyframes`, lines 158/235),
  returning an EMPTY sample latent. Module docstring: condition latents are *"re-injected every step, never
  denoised."* So FL2VA is the **m=0 hard-anchor path via a separate conditioning-token stream** — the case we
  already solve. It is NOT evidence of a stable *fractional* (0<m<1) single-frame at high res.
- **What FL2VA DOES prove:** the model natively holds ONE frame at 1MP and neighbors **compose/blend around
  it** — via attention to a never-denoised token. So the *attention-anchoring half* works at high res (weak
  evidence AGAINST mechanism B being fatal; native realization of route 3). It proves NOTHING about the
  *anchor-denoises-fractionally half*. Separate token stream ≠ in-latent held row → suggestive, not conclusive
  for our held-anchor case. See [conditioning-row-inject](../conditioning-row-inject.md).
Routes 1/3 exist to reach the capability from our inject path even though no single d does. Do not let "closed"
read as "impossible" — but do not oversell FL2VA as a fractional existence proof either.
Two-pass oracle: run only if #0 is ambiguous (outcome c) — it answers "does a coherent content-correct
1MP single frame exist for this model at all," now a live question.

## Other axes — d is NOT the only knob (user, 2026-08-24)

Even a correct d\* is content- and layout-dependent; two orthogonal confounds modulate the window's
width/location and must be controlled when calibrating:
- **Anchor↔ambient distance** — how far the injected still is from what the model would generate at
  that timeline slot WITHOUT the anchor (what it blends *against*). Near-ambient ⇒ wide window (little
  to reconcile); far-from-ambient ⇒ narrow/absent window (still contradicts the motion → lock at low d,
  overwritten/smeared at high d). ⇒ **d\* = f(N, anchor-ambient distance)**, not resolution alone.
  Cheap proxy: a d=1.0 pure-gen reference at that row → `‖clean_r − ambient_r‖` predicts window width.
- **Anchor↔anchor spacing** — the resolution wave-front interference already seen (high-d on f136 held
  the region open until its front "touched" f204, dragging an unchanged anchor). Front-radius GROWS
  with d, so a neighbor anchor's d/spacing couples into this one's window. ⇒ calibrate d\* on an
  **ISOLATED** anchor first; treat multi-anchor interference as a separate study.
