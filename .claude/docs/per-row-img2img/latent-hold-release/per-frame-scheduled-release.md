<!-- provenance: status (two GPU runs: HOLD-24 FALSIFIED schedule-sigma pin; HOLD-25 CONFIRMED neighbors-see-release; structural coherence restored; fade/denoise-level tuning open) -->
<!-- verified: 2026-08-26 · HOLD-24/25 GPU runs; HOLD-26 min-free-steps floor code-confirmed @074e443 (NOT GPU-verified), branch proto-latent-hold-release -->
# Per-frame scheduled release (integer-quantized) — prototype mode

**What it generalizes.** The "scheduled-hold with c=1.0" idea (from the knob-design discussion) applied to a
single held keyframe extends naturally to ALL rows: each row gets its own release step derived from its own
per-row denoise value `d_row = m_row`. This dissolves two open concerns from the c=1.0 discussion:

- (a) "With a full-1.0 label you land at σ_sw and can't cleanly reach lower d" — resolved: each row LANDS at
  its target level by choosing WHEN to release, not by compressing the label.
- (b) "A blanket c=1.0 clobbers fade rows" — resolved: a fade row is simply a row with its own
  mid-schedule release step.

Label `c = 1.0` (natural t_row) stays consistent for every row throughout the entire run.

## Mechanism

Recover the correlated noise endpoint once at the start of sampling:

```
eps = (x − (1 − σ_max) · clean) / σ_max        # sampler.py:481 form
```

At each step `i`, rows still held are PINNED to the RF-schedule state:

```
x_pin_i = sigmas[i] · eps + (1 − sigmas[i]) · clean   # RF mix, sampler.py:484 form
```

The `(1 − σ) · clean` weight is REQUIRED. An earlier `clean + σ · eps` sketch dropped it and would inflate
the anchor magnitude. At σ = σ_max the RF mix correctly recovers the original input `x`.

Per-row release step:

```
k_row = floor(steps · (1 − d_row))    with d_row = m_row
```

A row is pinned at the START of every step up to and including its release step, so it enters that release
step at exactly `sigmas[k_row]`. From the step after release it denoises freely to the end.

**No content/label mismatch.** In the original schedule-sigma pin, held rows sat at the exact schedule sigma
and the released row entered at `sigmas[k_row]` integrated consistently by the global Euler interval. The
"neighbors see release" pin below preserves the no-mismatch property by ALSO pinning the per-row label: held
content sits at `L = sigmas[k_row]` AND the effective per-row timestep `t_row = c·σ_i` is pinned at `L` via
`c = min(1, L/σ_i)`. Content and label agree on held rows; no content-vs-label mismatch under either pin.

As of HOLD-26 (`074e443`), `c = min(1, L/σ_i)` is the BASE CASE of the generalized label
`c_i = clamp(L / max(σ_i, σ_{k_rel}), max=1)`: identical when release timing `k_rel` equals the
intended-d level step, and a descending L→0 sub-schedule once the min-free-steps floor pulls `k_rel`
earlier. See [min-free-steps-floor](min-free-steps-floor.md).

## Unifying rule

| d_row | k_row | Behavior |
|---|---|---|
| 0 (m=0) | steps | Never releases → exact PRESERVE |
| 1 (m=1) | 0 | Releases at step 0 → FREE generation |
| fractional m | mid | Mid-schedule release → FADE / keyframe blend |

## Structural change from the current hold path

The current hold path performs a GLOBAL two-phase split — two `base_fn` calls over `sigmas[:k_sw+1]` and
`sigmas[k_sw:]` (`sampler.py:505` and `:526`) driven by one scalar `k_sw`. Per-row release cannot use that
structure because rows release at different steps. It needs a per-step loop that re-enters `base_fn` one
sigma-interval at a time, re-pinning held rows between steps.

**Limitation:** Re-entering per single step loses multistep-sampler history (dpmpp_2m / res_multistep degrade
to first-order). Acceptable for the prototype since the deterministic-Euler path is the target anyway.

## Label and tail correction

`c` (per-row denoise mask) is set to 1.0 for all rows. Denoise amount is governed entirely by the release
schedule. The `m · denoised + (1 − m) · inp` tail correction is NOT used as the amount knob in this mode.

## Fractional refinement — CONFIRMED-NEEDED (HOLD-25 follow-up QA, 2026-08-25)

**Evidence (0.2MP, 20 steps, row-60 single-frame inject).** d=0.30 slightly too high; d=0.25 or d=0.20 smudges.
Sweet spot (~d=0.27) is NOT on the integer grid.

**Step math** (`k = floor(steps·(1−d))`): d=0.30→k=14, 6 free steps; d=0.25→k=15, 5 free; d=0.20→k=16, 4 free.

**Key diagnosis.** Dropping one integer rung does TWO things at once: lowers the entry sigma (`sigmas[k]`) AND
subtracts a free step. Entry level and free-step count are coupled on the integer grid; the smudge is the cost
of that lost resolve step.

**Why fractional fixes it.** Fractional release decouples entry level from free-step count.
For d=0.27: raw=14.6, k=14, frac=0.6 → releases at step 14 (6 free steps) but enters at
`σ* = lerp(sigmas[14], sigmas[15], 0.6)` with `t_row = σ*`. Lower entry level WITHOUT surrendering resolve budget.

**Mechanism** (`raw = steps·(1−d)`, `k = floor(raw)`, `frac = raw−k`): release at step `k`, inject at
`σ* = lerp(sigmas[k], sigmas[k+1], frac)`, set `t_row = σ*`. Carries a per-row-σ-vs-global-Euler mismatch
on the release step (same family as existing per-row t_row compression; likely absorbed for deterministic
samplers but NOT free). NOTE: the `0.5·eps` shorthand simplified only because the LAST schedule sigma is 0;
mid-schedule releases require the full lerp.

**Honest caveat.** Fractional preserves the higher-d neighbor's free-step count but cannot manufacture steps
beyond the schedule tail. If softness persists even at 6 free steps, the lever is total step count, not fractional.

## Hypothesis under test — FALSIFIED on structural coherence (GPU, ~2026-08-25)

The blend-quality claim held; the GLOBAL-STRUCTURE claim did not. Standard test (0.2MP, fade-in video inject
@ frame 0, single-frame injects @ rows 40 & 60, per-frame release mode): output is well-blended AND
well-denoised locally — but STRUCTURALLY INCOHERENT (strange camera moves, objects morphing). So this
falsifies the hypothesis on the **global-structural-coherence** axis, NOT the seam/blend axis (the local
blend was clean, exactly as predicted).

**Mechanism (per-step scrub).** The failure is an inverted-influence-ordering problem:

- Broad-strokes temporal structure sets VERY early (~step 3) and then propagates via the
  wavefront/contagion/attractor dynamic.
- In that early window the only anchored signal present is the m ≈ 0.5 fade frames — and it is their
  50%-NOISED content, a weak/noisy seed, not their clean content. The timeline commits its structure to that
  weak noisy seed. (Be explicit: "structure forms around the m ≈ 0.5 frames" does NOT mean their clean
  version contributes — the contribution is still their half-noised version.)
- Strong anchors (m ≈ 0 keyframes) are pinned as near-full noise through the early window, so neighbors read
  them as noise nothing coalesces into. Free rows (m ≈ 1) release at step 0 but carry no anchor. Influence
  ordering is thus INVERTED vs structural importance: the strongest anchors speak LATEST, but structure is
  decided EARLIEST.
- Late in the run the timeline course-corrects as low-m anchors release and restore toward their originals →
  the incoherence. Blend clean; global structure broken.

**Generalized lesson.** Hold / scheduled-release is structurally wrong for any timeline where real anchors
must drive structure: it hides the strongest anchors during the window that decides everything. This is
supporting evidence for the [keyframe-two-views-and-knobs](../keyframe-two-views-and-knobs.md) "neighbors see
clean" framing over route-1 hold/scheduled-release.

## GPU-CONFIRMED — "neighbors see release" restores structural coherence (HOLD-25, 2026-08-25)

Instead of pinning held rows to the descending schedule sigma (`sigmas[i]·eps + (1−sigmas[i])·clean`, i.e.
noise[step0] ≈ σ_max — the falsified behavior above) or to clean (noise[step0] ≈ 0 — "neighbors see clean",
rejected as over-constraining), pin each held row at its OWN release sigma `L = sigmas[k_row]` from step 0,
constant across the hold. This flattens the reveal ramp so the anchor presents its intended-strength content
during the early (~step 3) structure-setting window, then denoises freely from release.

It GENERALIZES "neighbors see clean": m ≈ 0 rows land at `L = sigmas[steps] = 0` (clean) automatically;
fractional rows present their partial reveal. Continuity at release is preserved — a row enters release step
`k` at `sigmas[k]` exactly.

**Code change (commit `8d4eb20`, `sampler.py` `per_frame_release` loop; GPU-CONFIRMED — see below).**
Two coupled pins:

- *Content:* the `torch.where(held, ...)` pin uses a per-row
  `pin_release = sigmas[k_row]·eps + (1−sigmas[k_row])·clean` instead of `_rf(sigmas[i])`, so held content
  sits at `L = sigmas[k_row]`.
- *Label:* per step the sampler stashes a `pooled_current` (built from a `make_pooled` closure = the model's
  `_denoise_mask_values`) that the conditioning wrapper reads, setting the per-row mask `c = min(1, L/σ_i)`.
  This pins the effective per-row timestep `t_row = c·σ_i` at `L` for the whole hold. When every row is
  released (`c = 1` everywhere) it naturally falls back to the full-denoise label (`{}`).

**No content/label mismatch.** Content AND label both sit at `σ_k` on held rows. The held row's own
per-step denoised output is discarded (re-pinned each step), so the mechanism's ONLY effect is how the
model reads the anchor for neighbor attention; free/released rows integrate normally.

**GPU result (HOLD-25, 2026-08-25).** Standard test (0.2MP, fade-in video inject @ frame 0, single-frame
injects @ rows 40 & 60, per-frame release mode): blends and denoises are solid, and STRUCTURAL INCOHERENCE
(the HOLD-24 failure) is RESOLVED. Confirms the mechanism prediction: presenting each anchor at its intended
`L` during the early structure-setting window fixes the inverted-influence-ordering failure.

**Remaining honest caveats:** fade-envelope and denoise-level tuning still open (user noted); fractional refinement now CONFIRMED-NEEDED (see section above).

## HOLD-26 — min-free-steps floor (design, IMPLEMENTED `074e443`, NOT GPU-verified)

The min-free-steps floor decouples release LEVEL (always intended-d `sig_L`, driving content pin and
label numerator) from release TIMING `k_rel` (the only thing `min_ratio`/`rescale_step_release`
move). This fixes the pre-refinement (`e5996c0`) defect where rescaling `d` also deepened the entry
noise, making `d=0.05` look like `d≈0.3`. Released rows now ride a descending L→0 sub-schedule over
`[k_rel, steps_n]`. Strict generalization of this mode's label; 580 tests pass unchanged. Full
design + the motivating A/B sweep: [min-free-steps-floor](min-free-steps-floor.md).

See also: [amount-floor-and-step0-redesign](amount-floor-and-step0-redesign.md),
[knob-design-open-questions](knob-design-open-questions.md),
[keyframe-two-views-and-knobs](../keyframe-two-views-and-knobs.md),
[min-free-steps-floor](min-free-steps-floor.md) (floor depends on fractional — UNVERIFIED design).
