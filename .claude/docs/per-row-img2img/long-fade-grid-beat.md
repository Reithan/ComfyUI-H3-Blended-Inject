<!-- provenance: theory (UNVERIFIED — 4 confounded models (M-A/B/C/D), shift-series decoupler pending; S2 GPU-falsified 2026-08-28) -->
<!-- verified: 2026-08-28 · GPU noise-midpoint observation: noise peaks at fade midpoint (frames ~60-64), not endpoints; M-D added; shift-series pending -->
# Bug E: long-fade video interference — ramp-length band (S2 GPU-falsified; 4-model confound; shift-series pending)

Bug E symptom: moiré / streamers / ribbons / electric-like patterns in video latent with long
fades; sampler-independent; audio tracks via joint attention.
Established in [bugs.md](bugs.md) and [audio-axis-verdict.md](audio-axis-verdict.md).

Note: σ̃/H2 and Fix A do NOT address Bug E — present on main before either fix, sampler-independent.

## S2 cell-alignment — GPU-FALSIFIED 2026-08-28 (tombstone)

**S2 predicted:** artifact iff `ramp-start global_row % 5 ∈ {1, 4}` (adjacent to chunk reset row).
S2 fit 13/13 CPU-sweep configs — all from the same `efo=90` family, where ramp-start row (local_row)
was a proxy for ramp length L. That single-family fit was a small-sample coincidence.

**GPU out-of-sample ran 4 new configs; S2 broke on 3 of 4:**

| Config (a/b/c/d) | local_row | S2 predicts | GPU result |
|---|---|---|---|
| 0/0/34/90 | 0 (reset) | CLEAN | ERROR |
| 0/0/47/90 | 4 | ERROR | CLEAN |
| 0/0/38/90 | 1 | ERROR | ERROR (S2 right) |
| 0/0/39/90 | 2 | CLEAN | ERROR |

**Do not revisit S2.** The prior retirement of "held+ramp" as a confound (commit 4f695e0) was based
on S2's apparent validity and is REVERSED — held+ramp is back as an equally-fitting hypothesis.

## Descriptive pattern — ramp length L = d − c

Sorting all 17 tested configs by L = fade_out − end_keyframe, ERROR cases form a contiguous block.

**ERROR iff L ∈ [51, ~64]; CLEAN iff L ≤ 50 or L ≥ 68.**

Notation: a/b/c/d = fade_in/skf/ekf/fade_out.

| Config | L | Result |
|--------|---|--------|
| 0/0/0/17 | 17 | CLEAN |
| 0/0/60/90 | 30 | CLEAN |
| 0/0/44/90 | 46 | CLEAN |
| 0/0/47/90 | 43 | CLEAN |
| 0/0/40/90 | 50 | CLEAN |
| 0/0/39/90 | 51 | ERROR |
| 0/0/38/90 | 52 | ERROR |
| 0/0/37/90 | 53 | ERROR |
| 0/0/36/90 | 54 | ERROR |
| 0/0/35/90 | 55 | ERROR |
| 0/0/34/90 | 56 | ERROR |
| 0/0/30/90 | 60 | ERROR |
| 0/0/7/75 | 68 | CLEAN |
| 0/0/0/69 | 69 | CLEAN |
| 0/0/0/73 | 73 | CLEAN |
| 0/0/0/75 | 75 | CLEAN |
| 0/0/0/85 | 85 | CLEAN |

The c=39→c=40 flip (L 51→50) is ONE frame within the SAME latent row — a continuous frame-length
threshold, not grid-quantized. This directly contradicts S2's grid-position rule.

Upper-edge evidence: 0/0/7/75 (L=68) is the only data point above L=64 that is not a trivial
c=0 case. The upper-band edge is empirically thin; a single data point.

## GPU observation — noise localizes to fade midpoint (2026-08-28)

Dissecting 0/0/34/90: noise is STRONGEST at output frames ~60–64 (~2.5–2.66 s).
The fade spans [34, 90]; midpoint = frame 62.
The degeneracy forms mid-ramp where m≈0.5 and rows release at mid-schedule (~step 10).
Consistent with the original "noise generated ~1/4–1/2 through the steps" observation.

This motivates M-D (below). Caveat: noise at the midpoint is expected under M-A too
(midpoint = relative midpoint of the ramp regardless of absolute position) — so the
noise-location datum narrows timing-within-schedule, but does NOT decide length vs position.

## Four-way confound — models not yet distinguishable

In every `efo=90` test, held = 90 − ramp, so held length, ramp length L, presence of a
trailing free (m=1) region, and fade midpoint position all covary. Four models fit all 17
points identically:

**M-A ramp band:** error iff L ∈ ~[51, 64].
Both long and short ramps are safe; it is a BAND, bounded on both sides. Long ramps (L≥68) are
presumed safe again — the upper bound is the novel, thinly-evidenced claim.

**M-B held+ramp:** error iff held ≥ ~30 AND ramp ≥ ~51.
The un-retired original hypothesis. In every efo=90 test, held = 90 − L, so held≥30 ⟺ L≤60 —
exactly consistent with all ERROR cases. Long ramps without a long held block would be safe.

**M-C trailing-free-heals:** error iff ramp ≥ ~51 AND no fully-free (m=1) region follows the fade.
Explains 0/0/7/75 (L=68) clean: frames 75–90 are free and heal the artifact. Matches the user
observation of "noise generated ~1/4–1/2 through then healed/crystallized in later steps." All
efo=90 tests have zero trailing free region; clean/error split still driven by ramp length there.

**M-D fade-midpoint absolute position:** error iff (ekf+efo)/2 ∈ ~[60, 64.5].
Evidence: error-case midpoints = 60.0–64.5 (0/0/30/90→60.0, 0/0/34/90→62.0,
0/0/35-39/90→62.5-64.5); clean cases all miss it (0/0/40/90→65.0, 0/0/44/90→67.0,
0/0/60/90→75.0, 0/0/7/75→48.5, 0/0/0/*→8.5-42.5). M-D separates all 17 tested configs.
CRUCIAL CAVEAT: within efo=90, midpoint = 90 − L/2, so M-D ("midpoint∈[60,64]") and
M-A ("L∈[51,60]") are ALGEBRAICALLY IDENTICAL statements — NOT independent evidence.
The pass/fail data cannot distinguish length from midpoint-position on the efo=90 family.
The noise-location datum adds that degeneracy forms mid-ramp, but under M-A the noise
would also sit at the relative midpoint — so it does not decide the causal variable.

All four fit the 17-point dataset. None is ruled out.

## Pending decoupling factorial (UNRUN)

Each run moves ONE variable off the known-error baseline 0/0/35/90 (L=55, held=35, no free tail).
Per-model predictions (M-D midpoint for each: clip=124→62.5, d=107→68.5, d=56→30.5):

| Run | What changes | M-A | M-B | M-C | M-D |
|-----|--------------|-----|-----|-----|-----|
| 0/0/35/90 at clip=124 | adds ~34 trailing free frames | ERROR | ERROR | CLEAN | ERROR |
| 0/0/30/107 (d=107=clip, L=77, no free tail) | long ramp, no free tail | CLEAN | ERROR | ERROR | CLEAN |
| 0/0/5/56 (d=56=clip, L=51, held=5) | short held, same ramp L | ERROR | CLEAN | ERROR | CLEAN |

Clip lengths snap to 17n+5; valid clips used: 56, 90, 107, 124.
Note 0/0/30/107 also tests the M-A upper-band edge (L=77 above the L≥68 safe threshold).
Run all three; each flips exactly one model's prediction relative to the others.

## Shift-series experiment — position vs length decoupler (UNRUN)

Fix L=55 (known-error length); slide the whole fade window to decouple ramp length (M-A)
from absolute midpoint position (M-D) and held length (M-B). Anchor: 0/0/35/90 (KNOWN ERROR,
all models predict ERROR). Clip lengths snap to 17n+5.

| Config (a/b/c/d) | L | Midpoint | Held | M-A | M-B | M-D |
|---|---|---|---|---|---|---|
| 0/0/15/70 → snap clip=73 | 55 | 42.5 | 15 | ERROR | CLEAN | CLEAN |
| 0/0/25/80 → clip=90 | 55 | 52.5 | 25 | ERROR | CLEAN | CLEAN |
| 0/0/35/90 (ANCHOR, KNOWN ERROR) | 55 | 62.5 | 35 | ERROR | ERROR | ERROR |
| 0/0/52/107 (shift fwd +17 grid period) | 55 | 79.5 | 52 | ERROR | ERROR | CLEAN |

Read-off:
- 0/0/15/70 and 0/0/25/80 ERROR → length drives it (M-A; absolute position irrelevant).
- 0/0/15/70 and 0/0/25/80 CLEAN, anchor ERROR → absolute midpoint ~frame 62 drives it (M-D).
- 0/0/52/107 splits M-A+M-B (ERROR) from M-D (CLEAN) — key tiebreaker if the above disagree.
- Co-readout: if noise tracks each fade's OWN midpoint → relative/length (M-A); if noise stays
  pinned near frame 62 regardless of window shift → absolute grid position (M-D).

Recommend 0/0/15/70 first (largest midpoint shift, shortest clip). The 0/0/30/107 test from the
factorial remains valid (tests M-A upper edge, orthogonal to the shift series).

## Epistemic note — mono-causal rules over covariable sweeps

This is the third time a geometric root cause was committed based on a single-family efo=90 sweep
where the candidate variables covary:
1. Grid-cycle-count → falsified.
2. Held+ramp — then retired as a confound for S2.
3. S2 cell-alignment → GPU-falsified 2026-08-28.

**Lesson: do not declare a root cause from a sweep where candidate variables covary. Require a
decoupling factorial before treating any model as confirmed.** Record all competitive models;
commit only after the factorial data rules one in.

## Fix direction (hold until factorial + shift-series resolve)

Strategy depends on which model the experiments confirm:

- M-A: snap ramp length L outside ~[51, 64] (e.g. enforce L ≤ 50 or L ≥ 68).
- M-B: snap either held or ramp below the respective threshold.
- M-C: ensure a trailing free (m=1) region follows any long ramp.
- M-D: snap the fade midpoint (ekf+efo)/2 outside ~[60, 64.5] — or equivalently, outside
  the absolute grid danger zone around frame 62. Note: M-D fix = M-A fix algebraically
  within efo=90; they diverge only in user-exposed constraint for multi-sweep configs.

Do not implement any fix until both experiment sets are run.

## Speculative mechanism note (M-D framing)

M-D reframes the hunt: what is special about the output-latent grid around frames 60–64?
Frame 64 = chunk3 offset 13, approaching the chunk3/chunk4 reset at frame 68. This is
speculative; treat as a prior for experiment design, not an established finding. The shift
series decides it empirically.
