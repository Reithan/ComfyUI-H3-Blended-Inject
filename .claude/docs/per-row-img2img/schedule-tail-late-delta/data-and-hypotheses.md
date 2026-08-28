<!-- provenance: theory (H1 sole-primary SOURCE-CONFIRMED; H2 FALSIFIED 2026-08-27; label-cliff FALSIFIED 2026-08-27; H1 label-confound conceded 2026-08-27; H1 confound now PERMANENT — label-lie DEAD 2026-08-28) -->
<!-- verified: 2026-08-27 · sampler.py @34a5925 lines ~404/487-516; comfy/ldm/minimax/model.py @b78cec87 -->
# Schedule-tail: GPU data, hypotheses, falsification matrix

Parent: [schedule-tail-late-delta](../schedule-tail-late-delta.md).
See also: [label-channel-probe](label-channel-probe.md): the experiment designed to cleanly separate label from content.

## GPU data ladder ('both' mode, 2026-08-27)

| d | early label (w) | content σ | anchoring |
|---|---|---|---|
| 0.0 | 0.999 (pin) | ≈0 | strong — clean temporal blend |
| 0.05 | ~0.61 | ≈0.39 | STRONG conformance/blend (2026-08-27) |
| ≈0.2 | ~0.25 | ≈0.75 | weak on inject-distinct regions |

Falloff sits between σ≈0.39 (good) and σ≈0.75 (weak) of presented noise during structure-setting.

**SCHED-4 (mask-drop d≈0.2, 2026-08-27):** structure window presented official label + per-step clean
composite → temporal blend VERY GOOD. In-frame quality BAD (expected: post-release steps run
deep-noise label on now-clean content — the label-content mismatch the remap was built to avoid).
Follow-up datum (2026-08-27): injected frame departed substantially from source — large
post-structure-window redraw confirmed.
**H2 FALSIFIED:** blend does not require neighbors to track the inject's final content; conformance
to the structure-window presentation suffices even when the row is substantially rewritten afterward.

**Mode-comparison GPU data (2026-08-27):** Under 'rescheduled' mode, in-frame denoise strength FEELS
CORRECT. Under 'official' mode it felt TOO WEAK even with the per-step alpha-composite disabled.
Temporal blend: right at d=1 and d=0, correct-ish at d≈0.05–0.1; too weak across the mid-band.
User summary: temporal strength drops off with rising d FASTER than in-frame strength rises.
**Consequence:** v2 (σ_start·σ_g official-style interior) now DISFAVORED for in-frame strength;
'rescheduled' interior is the keeper. Open problem = temporal side of the mid-band only.

## Source facts (σ-shift convexity, source-confirmed)

`k_d = round(steps × (1−d))` (sampler.py ~line 404) interprets `d` as a step fraction.
H3's video σ-shift (shift≈12, `σ_shift = 12σ_lin/(1 + 11σ_lin)`) makes σ_start superlinear in d.
Row start sigmas: `d=0.05→σ≈0.39 · d=0.1→σ≈0.57 · d=0.2→σ≈0.75 · d=0.3→σ≈0.84 · d=0.5→σ≈0.92`.

In `both` mode: held row LABEL = `w = σ_row/σ_g` (sampler.py ~line 487) AND held CONTENT composited
at σ_row (sampler.py ~line 516); label and content present matching noise depth to neighbors.

Consistency check: pre-schedule-tail official build used `t_row = 1 − m·σ` (LINEAR) and GPU-tested
good blend at min_denoise 0.2–0.3 (@06c6bda). The regression appeared when the remap moved the
label onto shifted-σ (σ-shift convexity broke it); source facts remain sound.

## FALSIFIED: label-category cliff as dominant cause

Source: `comfy/ldm/minimax/model.py` @b78cec87; 0.999 = trained cond-timestep pin; d>0 drops into
noisy-peer label range. **FALSIFIED by d=0.05 GPU (2026-08-27):** strong anchoring at label ~0.61
(well outside pin) — category loss is a real distinction but NOT the primary driver; demoted to
contributing factor.

## Surviving hypotheses

**H1 — structure-window legibility (content-depth) [SUPPORTED — SOLE PRIMARY]:** neighbors read
inject structure via pure content Q·K (no attention mask; see
[dit-forward](../native-h3-mechanism/dit-forward.md)). Legibility degrades with the ACTUAL noise on
the row during structure-setting. σ≈0.39 legible; σ≈0.75 submerged. Non-linearity in d-space =
σ-shift convexity; H1 predicts a smooth falloff in σ-space. SCHED-4 follow-up: large post-structure
redraw + VERY GOOD blend throughout → **H1 is the SOLE primary driver** (H2 FALSIFIED — large redraw
did not degrade blend).

**H1 CONFOUND NOTE (2026-08-27):** The d=0.05 data (content σ≈0.39, truthful label ~0.61) was used
to claim "content dominates over label." This was WITHDRAWN (user challenge, accepted): truthful
labels track the same convex curve as content, so every data point moves both channels together. The
label channel's independent anchoring weight is UNTESTED by any existing data. See
[label-channel-probe](label-channel-probe.md) for the experiment designed to cleanly isolate them.
**OFFLABEL-1 (2026-08-28):** the isolation attempt confirmed the confound is PERMANENT — label
mismatch at the scale needed to separate channels corrupts the row's own velocity predictions first.
No latent-side label manipulation can cleanly isolate the label channel. This is a structural
property of the model, not a tunable parameter.

**H2 — late-delta drift [FALSIFIED 2026-08-27]:** inject departed substantially from source;
temporal blend VERY GOOD throughout. H2 is not an independent driver.

**H3 — content-conflict locality [WEAKENED as standalone]:** same region blends at d=0.05, fails at
d=0.2 ('both') → locality is d-gated within H1's margin; not an independent driver.

**H4 — label-ratio / attention-differential [THEORY, UNVERIFIED — 2026-08-28]:** cross-row
attention diffusion is a function of the ratio between inject and neighbor labels (neighbor always
1), not label absolute value. Fits all existing data: d:1 differential constant under official
schedule → good blend; rescheduled mid-band label shoots near 1 early → weak blend; d=0 (≈1:1
ratio) anchors perfectly. Observationally equivalent to H1 on all existing data — lock-breaking
experiment needed. See [label-ratio-and-observer-split](label-ratio-and-observer-split.md).

## Falsification matrix

1. **`prototype_mode='mask-drop'` @ d≈0.2: H1 vs H2 discriminator. [DONE 2026-08-27]**
   Result: temporal blend VERY GOOD / in-frame quality BAD (expected label-content mismatch).
   H1 SUPPORTED. Follow-up: inject substantially redrawn; **H2 FALSIFIED**.
2. **Redraw-log correlation [MOOT: H2 FALSIFIED].** Log line remains useful as general instrumentation.
3. **Label/content split toggle. [OFFLABEL-1: DEAD 2026-08-28]** TOTALLY BROKEN; label load-bearing for own velocity prediction; label-lie family CLOSED. See [label-channel-probe](label-channel-probe.md).
4. **Content-locality probe [MOOTED].** H3-standalone weakened; skip.

**Immediate experiment (H4, built 2026-08-28, GPU-pending):** observer-label K/V split.
See [label-ratio-and-observer-split](label-ratio-and-observer-split.md).
Result discriminates H4 vs H1: mid-band improvement = H4 confirmed; no change = H1 sole driver.

**Follow-up (if H4 inconclusive or H1 sole survivor): H1 threshold bisection (zero-code).**
d=0.06→σ0.43 · 0.08→0.51 · 0.10→0.57 · 0.12→0.62 · 0.15→0.68.
Known: 0.39 blends (SCHED-2), 0.75 fails (SCHED-3). Where blend breaks = max presentable depth.
