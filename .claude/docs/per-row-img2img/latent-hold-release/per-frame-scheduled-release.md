<!-- provenance: status (SUPERSEDED — route-1; HOLD-24 FALSIFIED schedule-sigma pin, HOLD-25 CONFIRMED neighbors-see-release; branch proto-latent-hold-release) -->
<!-- verified: 2026-08-27 · HOLD-24/25 GPU runs; HOLD-26 floor GPU-FALSIFIED by HOLD-27; branch proto-latent-hold-release -->
# Per-frame scheduled release (SUPERSEDED)

Each row gets its own release step derived from its own per-row denoise `d_row = m_row`. This dissolves
two concerns from the earlier c=1.0 discussion: (a) can't reach lower d via label compression — resolved
by choosing WHEN to release; (b) blanket c=1.0 clobbers fade rows — resolved since a fade row is a row
with its own release step. Label `c = 1.0` stays consistent throughout.

## Mechanism

Correlated noise endpoint recovered once at start:

```
eps = (x − (1 − σ_max) · clean) / σ_max
```

Per-row release step: `k_row = floor(steps · (1 − d_row))`.

**Unified rule:**

| d_row | k_row | Behavior |
|---|---|---|
| 0 (m=0) | steps | Never releases → PRESERVE |
| 1 (m=1) | 0 | Releases at step 0 → FREE |
| fractional | mid | Mid-schedule release → BLEND |

The `m · denoised + (1−m) · inp` tail correction is NOT used as the amount knob in this mode; c=1.0 for all rows.

Per step the label: `c_i = clamp(sig_L / max(σ_i, σ_{k_rel}), max=1)`. On held rows `t_row = L`; on released
rows `t_row` descends L→0 over `[k_rel, steps_n]`.

NOTE: the per_frame path sets NO denoised correction — a caveat that became the HOLD-27 failure (see below).

The per-step loop re-enters `base_fn` one sigma-interval at a time, re-pinning held rows between steps.
This loses multistep-sampler history (dpmpp_2m degrades to first-order); acceptable for the deterministic-Euler path.

## HOLD-24: FALSIFIED on structural coherence (schedule-sigma pin)

**Standard test** (0.2MP, fade-in video inject @ frame 0, single-frame injects @ rows 40 and 60):
output well-blended and well-denoised locally, but STRUCTURALLY INCOHERENT (strange camera moves, morphing).

**Mechanism (inverted influence ordering):** Broad-strokes temporal structure sets very early (~step 3).
With schedule-sigma pin, held rows descend from σ_max toward σ_k, so the strongest anchors (m≈0,
k_row≈steps) present as near-full noise during the early window. Weak fade rows (m≈0.5) seed structure
with half-noised content. Late in the run, low-m anchors release and pull back toward originals, producing
incoherence. Blend was clean; global structure broken.

**Lesson:** hiding strong anchors during the early structure-setting window is fatal regardless of local blend quality.
This supports the "neighbors see clean" framing over route-1 hold/scheduled-release.

## HOLD-25: GPU-CONFIRMED (neighbors see release)

Fix: pin each held row at its OWN release sigma `L = sigmas[k_row]` from step 0 (constant throughout the
hold), rather than descending with the global schedule. Rows present intended-strength content during the
early structure window; at release step `k` the row enters at `sigmas[k]` — continuity preserved.

**Content pin:** `pin_release = sigmas[k_row]·eps + (1−sigmas[k_row])·clean`.
**Label pin:** `c = min(1, L/σ_i)` — pins effective per-row timestep at `L` for the hold. Content and label
both sit at `σ_k` on held rows; no mismatch. m≈0 rows land at `L≈0` (clean) automatically.

**GPU result (HOLD-25, 2026-08-25):** structural incoherence resolved; blend and denoise solid.

**Fractional refinement (confirmed needed):** sweet spot (~d=0.27) is not on the 20-step integer grid.
Fractional release decouples entry level from free-step count: `σ* = lerp(sigmas[k], sigmas[k+1], frac)`,
`t_row = σ*`. Carries a per-row-sigma-vs-global-Euler mismatch on the release step (acceptable for
deterministic Euler, NOT free for stochastic samplers).

*Coupling diagnosis + caveat (archival — MOOT, the whole per_frame path is HOLD-27-falsified below; kept as
the design constraint if fractional release is ever revived).* Step math `k=floor(steps·(1−d))`:
d=0.30→k=14/6 free steps; d=0.25→k=15/5 free; d=0.20→k=16/4 free. Dropping one integer rung does TWO things
at once — lowers entry sigma `sigmas[k]` AND subtracts a free step; that lost resolve step is the smudge.
Fractional breaks the coupling (d=0.27: raw=14.6, k=14, frac=0.6 → releases at step 14 with all 6 free steps
but enters at the lerp'd lower level). **Honest caveat:** fractional preserves the higher-d neighbor's
free-step count but cannot manufacture steps beyond the schedule tail — if softness persists even at 6 free
steps, the real lever is total step count, not fractional.

## HOLD-26 design + HOLD-27 GPU-FALSIFIED

HOLD-26 (`074e443`) decouples release LEVEL (`sig_L`, always intended-d) from release TIMING (`k_rel`,
what `min_ratio` moves), so a row always releases with the noise its own `d` asked for. 580 tests pass.
See [min-free-steps-floor](min-free-steps-floor.md) for full design.

**HOLD-27 (2026-08-27): GPU-FALSIFIED.** All 5 runs over-denoise; the previously-accurate Option-1 config
regresses. Two failures: (1) per_frame path has no denoised correction — free-step count governs realized
redraw, not the level pin; (2) floor is provenance-blind, clamping opening fade-out ramp rows like keyframes
(same bug class as hold-mechanism-and-confounds Findings 7–11). New fade pop and color flash in runs 4 and 5.
Full table: [min-free-steps-floor](min-free-steps-floor.md).

See also: [amount-floor-and-step0-redesign](amount-floor-and-step0-redesign.md),
[knob-design-open-questions](knob-design-open-questions.md),
[keyframe-two-views-and-knobs](../keyframe-two-views-and-knobs.md).
