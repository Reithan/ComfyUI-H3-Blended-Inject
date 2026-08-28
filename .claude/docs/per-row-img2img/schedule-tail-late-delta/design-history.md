<!-- provenance: theory (v1 withdrawn; v2 DISFAVORED; route-3 REJECTED 2026-08-27; renoise-release REJECTED 2026-08-27; label-lie DEAD 2026-08-28) -->
<!-- verified: 2026-08-27 · sampler.py @34a5925; attention.py @b78cec87 lines 171-204 (wrap_attn, optimized_attention_override) -->
# Schedule-tail: design history (v1 / v2 / route-3 / renoise-release)

Parent: [schedule-tail-late-delta](../schedule-tail-late-delta.md).

## v1 withdrawn

v1 (σ-space k_d with tail-stretch) WITHDRAWN: analytically falsified. Tail-stretch spends most
steps at small σ; lingers far less at depth than official `m·σ_g(i)` — σ_start=d + tail spacing
= weakest possible combination.

**Key decomposition:** two convexities conflated: (1) knob→σ_start (official `m·σ_g(i)` = linear
σ=m; step-space k_d = convex, d=0.2→σ0.75); (2) interior trajectory (official lingers near σ_start
most steps; tail-stretch spreads steps mostly at small σ). Per H1: BLEND governed by σ_start
(presentation depth); FELT IN-FRAME STRENGTH by σ_start + interior lingering.

## v2 disfavored for in-frame

**v2 design (DISFAVORED for in-frame, 2026-08-27):** `σ_row(i) = σ_start·σ_g(i)` — official-style
interior; values snapped to dense-grid entries. Mode-comparison GPU: official interior felt TOO WEAK
for in-frame even with composite off → 'rescheduled' interior keeps correct in-frame strength. v2 is
disfavored; 'rescheduled' interior is the keeper. Open problem = temporal side of the mid-band only.

## Route-3 attention-logit boost — REJECTED 2026-08-27

Hook source was confirmed (attention.py:171-204 @b78cec87; `wrap_attn` + `optimized_attention_override`).
Both variants rejected:

**(a) Cond-row-bias**: unnecessary. H3AddGuide already covers boosted-clean-reference. Cond tokens
consume the attention budget (strained at high resolution), while latent-side rows are "attention
free."

**(b) Latent-key-bias**: rejected for perf. The `[1,1,1,Lk]` bias vector is tiny, but ANY float
attn mask knocks SDPA off the fused flash kernel onto the materialized-attention path: a real perf
hit at H3 sequence lengths.

**Mandate:** this effort requires a LATENT-SIDE solution; attention-side is out of scope.
See [keyframe-two-views-and-knobs](../keyframe-two-views-and-knobs.md) route-3 bullet (updated REJECTED).

## Structure-window hold + renoise-and-release — REJECTED 2026-08-27

**Motivation (withdrawn):** H1's conflict — legibility ≡ low-σ content, rewrite amplitude ≡ high
σ_start — seems SEPARABLE IN TIME: blend is decided in the early structure window; rewrite only needs
the full σ traversal somewhere in the run.

**Proposed design:** for the first W steps (σ_g > ~0.8) present the row clean (mask-drop-style hold).
At step W: renoise via RF interpolation `x = (1−σ_start)·clean + σ_start·ε`. Run the normal
rescheduled dense-grid tail over the remaining N−W steps.

**REJECTED (user, 2026-08-27):** hold-and-release variants were already explored extensively; any
time-split approach sacrifices in-frame steps by construction. Holding for a substantial fraction of
steps leaves too few steps for proper in-frame denoising (likely also affected by the global sigma
having advanced). Holding for ANY number of steps trades in-frame quality for blend quality — the
goal is BOTH high. Time-split approaches are rejected as a class; the correct path is a
single-trajectory solution.

**Prior-experiment disambiguation:** this IS hold-and-release again — prior holds were either
confounded (provenance-blind `anchor_mask` froze wrong rows; see
[hold-mechanism-and-confounds](../latent-hold-release/hold-mechanism-and-confounds.md)) or held most
of the run then released into truncated free-fall. Short-window + renoise + rescheduled-tail was
structurally untested before rejection; the rejection stands on the class-level argument, not a
failed GPU run.

**What SCHED-4 does and doesn't prove:** mask-drop d≈0.2 held clean mid-run; frame substantially
redrawn afterward; temporal blend VERY GOOD throughout. Proves clean-during-window buys blend AND
mid-run release does not break blend. SCHED-4's bad in-frame = only the missing fractional tail.
This evidence motivates the renoise-release design but does not override the class-level rejection:
the in-frame step sacrifice remains by construction.

## Label-lie (official_labels): DEAD 2026-08-28

**OFFLABEL-1 GPU result:** TOTALLY BROKEN — abstract/psychedelic patterns in injected frames.
Label is load-bearing for the row's own velocity prediction; any lie big enough to affect
neighbor anchoring corrupts the row's denoising by the same mismatch. Structural, not tunable.
See [label-channel-probe](label-channel-probe.md) for full result and conclusions.
