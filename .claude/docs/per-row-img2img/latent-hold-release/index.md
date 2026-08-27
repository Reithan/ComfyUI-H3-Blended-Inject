<!-- provenance: status (prototype design + GPU debug log; route-1 latent hold-and-release, branch proto-latent-hold-release, NOT for merge) -->
<!-- verified: 2026-08-27 · HOLD-27 GPU run; prior: GPU runs on user test deployment + Fable audits + comfy-ref source -->
# Route-1 latent hold-and-release — prototype & debug log (index)

Quick-and-dirty prototype (branch `proto-latent-hold-release`, NOT for merge) testing whether a
latent-resident anchor-then-release can give a mid-timeline keyframe intuitive img2img denoise with
neighbor blend (no ghost, no pop). All HOLD-* experiments serve the prototype-drive north star — ONE
user-facing knob mapping visually to img2img denoise `d` (via `hold`+`m`), resolution-invariant (IDEAL) or
with a res-aware internal mapping (ACCEPTABLE fallback) — see
[resolution-invariance-goal](resolution-invariance-goal.md). Design derives from
[crux-and-mechanism-2](../highres-underdenoise-model/crux-and-mechanism-2.md). Parent open-path:
[status-and-open-paths](../status-and-open-paths.md) path 1.

**Through-line (2026-08-25):** hold-residency works; the "hold CREATES attraction" premise is challenged
(a co-evolving inject already blends; a permanent freeze `denoise=0.0` attracts too). The hold-ON cut was
EXPLAINED and FIXED: `anchor_mask=(m>0)&(m<1)` was provenance-blind and froze the **opening video inject's
fade-out**, not the r40/r60 keyframes; that wrong-row freeze **propagated forward via H3's global
attention** and corrupted r40's blend. The provenance fix (hold only 1-frame keyframes at fractional
denoise) is now **GPU-CONFIRMED**: the clean keyframe-only hold attracts/blends correctly. **NEXT PROBLEM**
(Findings 12–14): the held keyframe itself under-denoises. The m=0.99 probe shows a ~full redraw still
"looks like 0.5" — so **neighbor attention (contagion) AFFECTS r40's result** (does not, on this evidence,
*set* the amount). The m=0.5-vs-m=0.99 A/B is done: **higher m gave the better-looking r40**; HYPOTHESIS
(unproven) is m=0.5 artifacts = off-manifold half-denoise. The leading read is **res-compression** (runs
are 0.5MP ≈ 1MP), but the *mechanism* is contested — per-frame basin-sharpening vs the user's
**attention/contagion** theory (raw-count degrades attention quality; open = P1/P2 sign); a
**timeline-length sweep at fixed res** would discriminate but is DEFERRED. **Still-repeat-with-fade is RUN**:
fade fixes the seam but smears the anchor's own artifacts AND reads as a **freeze-frame** — not viable, and a
CAUTION for route-3. Surviving lever = the single frame's OWN high/res-compensated m (m=0.99 looked best).
**UPDATE (2026-08-25):** high-m GENERALIZES (r60 too, HOLD-15). HOLD-16/17 separated re-noise level from
release-step: `m` at a fixed early release is a clean amount knob (m∈{0.8,0.9,0.99} good); `frac` tracks
level m·σ_sw monotonically. The initial "release-step = quality gate, NOT amount knob" conclusion from
HOLD-16 is **RETRACTED (user, 2026-08-25)** — the hold=0.75 run is non-diagnostic (confounded). See
[held-keyframe-m-vs-sdedit](held-keyframe-m-vs-sdedit.md) + [knob-design-open-questions](knob-design-open-questions.md).

**RESOLUTION-INVARIANCE — the real axis (user, 2026-08-25), and the go/no-go PASSED (HOLD-18).** m-only is
res-unstable (1MP window closed without hold). But the good HOLD recipe SURVIVES to 1MP: hold=0.25/m=0.99
frac **0.551→0.553** (0.5↔1.0MP, Δ0.002), hold=0.5/m=0.99 →frac 0.499, both clean+blended. **Hold REOPENS the
1MP window; the route is VIABLE.** Hypothesis: res-invariance = **m≈1 native denoise** (hold gives BLEND, not
res-stability) → the LOW-amount end is the hard part. **Next: m-sweep (0.8/0.9) at 1MP** — is the high-m band
a res-invariant amount knob? See [resolution-invariance-goal](resolution-invariance-goal.md). Not a dead end.
**Knob design + retraction (2026-08-25):** "release-step = quality gate" RETRACTED (see above). Current
design: step0=clean; release=const-early σ≥~0.95; **`m`=d single knob** (Fable dissolved decoupled-L design B).
Perceptual-`d` yardstick, reading-lens failure-mode split, and current best anchor hold=0.5/m~1 ≈ "looks
like 0.5 denoise" across 0.5–1MP + open problem reaching 0.25/0.75 cleanly:
[knob-design-open-questions](knob-design-open-questions.md).

**HOLD-19 (1MP m-sweep) — the amount-floor is real + the step[0] redesign.** m=0.8@1MP (frac 0.362) SMEARS the
keyframe with a fine blend; m=0.9 (0.392) clean → a low-amount failure at a GOOD release ⇒ **amount-floor (~frac
0.39) CONFIRMED, separate from the blend/release failure** (two distinct symptoms; hold=0.75 hit BOTH). `m` is
NOT res-invariant (compresses; only m≈0.99 res-stable-clean). The clean band is narrow (m∈[0.9,0.99]). A
**trilemma** (clean keyframe wants native denoise; native+partial wants late release=bad blend; early+partial
uses m<1=smear) → the user's **step[0] redesign** (present the anticipated release-state to neighbors, not the
clean input) is the lever to break it: [amount-floor-and-step0-redesign](amount-floor-and-step0-redesign.md).

**HOLD-20 (prenoise ON, 0.5MP, m=0.99) — the `hold_prenoise_step0` toggle is BUILT and tested.** No regression
(hold 0.25/0.5 clean); frac rose **+0.09–0.14** vs clean-hold at every hold. Because the anchor enters the tail at
`renoised` in BOTH modes (release-step `where` forces it), that delta can ONLY travel through attention ⇒ **direct
evidence neighbors SHAPE the anchor's realized redraw** (contagion), and prenoise is a contagion-mediated amount
lever that rescales frac UPWARD in the HEALTHY high-m band. hold=0.75 still bad (release timing unchanged).

**HOLD-21 (prenoise ON, 1MP, m=0.8) — the redesign is FALSIFIED as a smear-fix.** At the exact HOLD-19 smear config,
prenoise gave frac 0.372 (Δ+0.01 vs OFF, FLAT) and still smudgey/bad. So the 1MP low-m smear is NOT the
converge-then-jump tear; it is the **anchor's OWN low-m partial denoise** (the `m·denoised+(1−m)·inp` correction
clamps redraw; neighbors can't overcome it) — which ties to **RES-1** (base-model can't cleanly partial-denoise a
single 1MP frame). Neighbor-side levers (hold, prenoise) can't repair it. **Net: clean res-invariant output is
HIGH-amount only (m≈0.9–0.99, near-full redraw); low-d partial redraws smear at high res.** Two paths (USER
decides): A ship the knob over the supported high band; B attack the root via res-corrected effective-m
(highres-underdenoise-model γ≈1.6), orthogonal to hold-release. See [amount-floor-and-step0-redesign](amount-floor-and-step0-redesign.md).

**HOLD-22/23 (0.2MP) — anchor confirmed at a THIRD res + the no-hold ruler.** The best anchor (hold=0.5/m≈0.99,
±prenoise) now reads ~0.5 denoise at 0.2MP too (frac 0.379 OFF / 0.608 ON), so **d=0.5 is res-robust across
0.2/0.5/1.0MP**. At 0.2MP no machinery is needed — **m IS the knob (d≈m)**: TRUE no-hold m=0.5 → frac 0.273
(the perceptual "ruler"), and it's a DEAD HEAT with the old hold=0.01 proxy (0.272) — proxy VALID, prior
"they diverge" prediction FALSIFIED (same tail-entry magnitude). `frac` stays DECOUPLED from perceived-`d`
(0.273→0.608 all read ~0.5). Open frontier = low-`d` at HIGH res; leading untested lever = SPLIT release-level
`L` from tail `m` (keep m≈1 correction, lower L). NEW CODE (6e2810b): the no-hold path now derives the same
provenance-filtered anchor mask and logs its realized redraw ("calibration ruler") so no-hold `m` runs compare
directly to hold runs on the same rows (read-only, fires only when fractional keyframe rows exist). See
[knob-design-open-questions](knob-design-open-questions.md).

**HOLD-27 (2026-08-27): HOLD-26 GPU-FALSIFIED.** Every run over-denoises incl. the previously-accurate
Option-1 config (regression). Two failures: (1) per_frame path has no denoised correction — free-step
count governs realized redraw, not the level pin; (2) floor is provenance-blind, hitting the opening
fade-out ramp rows like keyframes (same class as Findings 7–11 confound), producing new fade pops.
Schedule-matched descent is NOT the next step; failure is level-semantics and provenance.
Full table + analysis: [min-free-steps-floor.md](min-free-steps-floor.md).

## Child docs
- [amount-floor-and-step0-redesign](amount-floor-and-step0-redesign.md) — HOLD-19: the amount-floor confirmed
  (keyframe-smear vs abrupt-blend symptoms), m res-compression re-confirmed, the trilemma, and the user's
  step[0]-prenoise redesign + its A/B. *Read for the current frontier + the smear fix.*
- [mechanism-and-early-findings](mechanism-and-early-findings.md) — Design (v1 simplifications);
  Finding 1 (sigma-shift makes `latent_hold_frac` the wrong unit); Finding 2 (residency confirmed +
  bugs); Finding 3 (early-diagnosis corrections). *Read for the build + how we got here.*
- [attraction-and-envelope](attraction-and-envelope.md) — Findings 4–6: the hold-vs-no-hold A/B,
  freeze-blends (`denoise=0.0`), and Fable's envelope/+0.5-row-center analysis that reframes the whole
  puzzle. *Read for the attraction/envelope thread.*
- [hold-mechanism-and-confounds](hold-mechanism-and-confounds.md) — Findings 7–11: mechanism trace
  (what the hold does to a targeted row), the code-identical-prefix puzzle, the min_denoise=0.0
  no-op correction, and the **provenance-blind `anchor_mask`** confound (froze the opening video
  fade-out → propagated to corrupt r40) + the provenance FIX. *Read for why the hold cut and how the
  fix works.*
- [anchor-denoise-after-clean-fix](anchor-denoise-after-clean-fix.md) — Finding 12: the clean
  keyframe-only hold is GPU-confirmed (attracts/blends), but the held keyframe **under-denoises**;
  candidates A–D and discriminating tests T1–T3. *Read for the current open problem.*
- [anchor-denoise-m-vs-res](anchor-denoise-m-vs-res.md) — Findings 13–14: the m=0.99 probe (a ~full
  redraw still "looks like 0.5", so **neighbor attention AFFECTS r40's result**, not shown to *set* it);
  higher m looked better; res-compression vs the user's **attention-dilution/contagion** mechanism (the
  live dispute) + its discriminators; why "init-noise + m=1" isn't a clean test. *Read for is-`m`-the-lever.*
- [resolution-invariance-goal](resolution-invariance-goal.md) — **the governing goal:** ONE res-invariant
  user knob (internal mapping may be res-aware); why the 0.5MP sweeps don't settle it; m-only dead + 1MP
  window-closed; the DECISIVE unrun test = the good hold recipe at 1MP. *Read to plan the next run / the
  final knob design.*
- [knob-design-open-questions](knob-design-open-questions.md) — RETRACTION of "hold = quality gate" +
  perceptual-`d` yardstick (Block B) + reading-lens failure-mode split (Block C) + current best anchor
  hold=0.5/m~1 and open problem reaching 0.25/0.75 cleanly (Block D); mechanism model; design conclusion
  (m=d, σ-threshold, Design B dissolved). *Read to plan the single-knob mapping and next experiments.*
- [held-keyframe-m-vs-sdedit](held-keyframe-m-vs-sdedit.md) — the r60 generalization (HOLD-15); the
  re-reading that hold=0.5/m=0.99 IS already pin-`m=1` SDEdit; the HOLD-16/17 sweep table; RETRACTION of
  "step=quality-gate" (correction block added 2026-08-25); REVERSAL to variable-m at early release.
  *Read for the sweep data and retraction trail.*
- [per-frame-scheduled-release](per-frame-scheduled-release.md) — GPU-CONFIRMED (HOLD-24/25):
  correlated noise pinning, per-row release step `k_row = floor(steps·(1−d))`, the unified
  preserve/free/blend rule, structural change from global two-phase to per-step loop.
  *Read for the per-frame release mechanism.*
- [min-free-steps-floor](min-free-steps-floor.md) — GPU-FALSIFIED (HOLD-27): min-free-steps floor
  design for per-frame scheduled release; the motivating A/B sweep (pre-refit e5996c0);
  the HOLD-27 5-run results table + failure analysis (no denoised correction + provenance-blind floor).
  *Read for the HOLD-26/27 design and its failure mode.*
