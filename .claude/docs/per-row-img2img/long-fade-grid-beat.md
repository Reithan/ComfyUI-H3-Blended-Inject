<!-- provenance: theory (UNVERIFIED — 3 confounded models, decoupling factorial pending; S2 GPU-falsified 2026-08-28) -->
<!-- verified: 2026-08-28 · GPU out-of-sample runs (4 configs) falsified S2 (3/4 mispredicted); ramp-length band descriptive across all 17 tested configs -->
# Bug E: long-fade video interference — ramp-length band (S2 GPU-falsified; 3-model confound pending)

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

## Three-way confound — models not yet distinguishable

In every `efo=90` test, held = 90 − ramp, so held length, ramp length L, and presence of a
trailing free (m=1) region all covary. Three models fit all 17 points identically:

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

All three fit the 17-point dataset. None is ruled out.

## Pending decoupling factorial (UNRUN)

Each run moves ONE variable off the known-error baseline 0/0/35/90 (L=55, held=35, no free tail).
Per-model predictions:

| Run | What changes | M-A | M-B | M-C |
|-----|--------------|-----|-----|-----|
| 0/0/35/90 at clip=124 | adds ~34 trailing free frames | ERROR | ERROR | CLEAN |
| 0/0/30/107 (d=107=clip, L=77, no free tail) | long ramp, no free tail | CLEAN | ERROR | ERROR |
| 0/0/5/56 (d=56=clip, L=51, held=5) | short held, same ramp L | ERROR | CLEAN | ERROR |

Clip lengths snap to 17n+5; valid clips used: 56, 90, 107, 124.

Run all three. Each distinguishes one model from the other two by flipping EXACTLY that model's
predicted outcome. Three runs together uniquely identify the root cause.

## Epistemic note — mono-causal rules over covariable sweeps

This is the third time a geometric root cause was committed based on a single-family efo=90 sweep
where the candidate variables covary:
1. Grid-cycle-count → falsified.
2. Held+ramp — then retired as a confound for S2.
3. S2 cell-alignment → GPU-falsified 2026-08-28.

**Lesson: do not declare a root cause from a sweep where candidate variables covary. Require a
decoupling factorial before treating any model as confirmed.** Record all competitive models;
commit only after the factorial data rules one in.

## Fix direction (hold until factorial resolves)

Strategy depends on which model the factorial confirms:

- M-A: snap ramp length L outside ~[51, 64] (e.g. enforce L ≤ 50 or L ≥ 68).
- M-B: snap either held or ramp below the respective threshold.
- M-C: ensure a trailing free (m=1) region follows any long ramp.

Do not implement any fix until the factorial experiments are run.
