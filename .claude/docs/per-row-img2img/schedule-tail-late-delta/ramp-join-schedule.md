<!-- provenance: design (REJECTED: analytical falsification, user, 2026-08-28; never built) -->
<!-- verified: n/a: rejected pre-build; no experimental run -->
# Ramp-join schedule (REJECTED)

**Proposed:** 2026-08-28, branch `proto-schedule-tail-release`.
**REJECTED pre-build** (analytical falsification, user, 2026-08-28; never built). This doc is a historical record.

Cross-links: [parent index](../schedule-tail-late-delta.md) · [design-history](design-history.md) ·
[data-and-hypotheses](data-and-hypotheses.md) ·
[keyframe-two-views-and-knobs](../keyframe-two-views-and-knobs.md) ·
[status-and-open-paths](../status-and-open-paths.md).

## Rejection argument (2026-08-28, user, analytical)

Each σ range does structurally different work on the latent. During the official-consistent rise
(steps 0..j), the denoising work belonging to the σ range between the two start values
(σ_res(0) down to σ_off(j)) never happens at proper strength — the inject row is held at
official-level noise while that deeper range passes by unworked. The compressed rescheduled tail
(σ_res(j)→0) cannot recover that work: it was already skipped at the noise level where it
belonged. Best case = a cleanly blended timeline with an under-denoised inject — exactly the shape
every hold variant has shown empirically. Ramp-join is structurally a hold variant. The hold
family stays closed; ramp-join joins it.

## Problem recap (historical context — design rejected)

Three families exhausted at the time of proposal:

- **Label-lie** (`official_labels`): DEAD: label is load-bearing for the row's own velocity
  prediction; any manipulation corrupts the anchor's denoising (OFFLABEL-1, GPU 2026-08-28).
- **Hold-and-release** (structure-window hold + renoise-release): REJECTED; pays cost
  elsewhere (renoise-release is a jump not a ramp; clean-freeze during hold is not evolution-consistent).
- **Schedule remap** (pure rescheduled σ_res): best-so-far but imperfect: in-frame strength right,
  mid-band blend weak because the dense tail starts at high noise before structure is shared.

Official schedule: σ_off(i) = d · σ_g(i); blend good (label ≈ d, neighbor-legible), frame weak.
Rescheduled: σ_res(i) = dense-tail from k_d; frame right, mid-band blend weak (label shoots high).
Proposal was an in-between that inherits the blend regime early and the rescheduled tail late.

## Ramp-join formulation (historical: rejected design)

```
u(i)      = min(i / j, 1)
σ_row(i)  = lerp(σ_off(i), σ_res(i), u(i))
```

For i < j: official-consistent levels (label ≈ d, blend-confirmed regime). The rise meets the
rescheduled tail while σ_res is still high (peak ≈ σ_res(j)), then the normal dense-grid tail
runs from there to 0 (full σ_res(j)→0 traversal, only j steps diverted).

Join knob j = structure-window length, default ~3–5 of 20 steps or σ_g threshold ~0.8.
Plain-lerp (weight = i/n globally) is a separate variant — rejected, see below.

## Plain-lerp trap (rejected variant of rejected design)

lerp(σ_off, σ_res, i/n) peaks far below σ_res(0) because the tail decays before the weight grows.
At 20 steps, shift-12, d=0.2: σ_mix ≈ 0.20 → peak ≈ 0.38 mid-run → 0. Peak 0.38 ≈ rescheduled
at d≈0.05 rewrite amplitude — same conflation trap as the withdrawn v1 schedule. Mildest at high
d (d=0.5 peak ≈ 0.65 vs σ_res(0) ≈ 0.92), worst in the problem mid-band (d≈0.1–0.2).

## Viability argument (historical: pre-rejection analysis)

Both σ_off and σ_res satisfy σ_row ≤ σ_g pointwise (rescheduled: tail position k_d+i·span ≥ i),
so any pointwise blend keeps the label ratio w = σ_row/σ_g in [0,1]. Rising-then-falling σ_row
is legal.

Content consistency during the rise (OFFLABEL-1 law: content must track label) is handled by the
existing both-mode phase-1 composite: w·x + (1−w)·clean. This places content at any σ_row ≤ σ_g
using the global x's noise — no fresh noise injected. Clean-anchoring during the rise is
phase-appropriate (structure window; SCHED-4: late rewrite doesn't break blend).

## Distinction from rejected renoise-release (historical comparison; both designs rejected)

Three differences from the rejected structure-window hold + renoise-release design:

1. **Hold phase**: official-consistent σ_row evolution (content tracks label via w·x+(1-w)·clean),
   not a clean freeze. Clean-freeze was OFFLABEL-1's failure mechanism; evolution is OFFLABEL-1-safe.
2. **Release**: a consistent multi-step ramp (j steps of linear weight growth), not a one-step jump.
3. **Tail**: full σ_res(j) → 0 traversal covering the entire dense grid from k_d+j·span, not a
   truncated free-fall from a mid-tail join point.

## H1 prediction (MOOT: pre-rejection rationale only)

Pre-rejection rationale composed two GPU-confirmed pieces:

- Neighbors see legible d-level content through the structure window (official regime = blend good).
- Deep rewrite happens after structure is set (SCHED-4: late rewrite doesn't break neighbor blend).

The analytical rejection overrides this: the σ range skipped during the hold phase cannot be
recovered in the compressed tail regardless of how well structure is shared. Both bullets remain
true; they do not rescue the design.

## Open unknowns (MOOT: design rejected pre-build)

These were open calibration questions for a design that was never built. Preserved for record only.
