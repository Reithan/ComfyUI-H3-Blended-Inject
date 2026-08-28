<!-- provenance: reference (experiment pointer table — HOLD-15 onward, STILL, MC, AUG, AUD, BUG-B, DD, VER; child of experiments-run.md) -->
<!-- verified: 2026-08-27 · HOLD-27 added; prior cross-checked against home docs @proto-latent-hold-release -->
# Experiment Run — Hold Continued + Other Series (HOLD-15 through HOLD-27, STILL, MC, AUG, AUD, BUG-B, DD, VER)

Child of [experiments-run.md](../experiments-run.md). Results live in the linked home docs.
Fields per entry: **ID** · config | result | home link.

## HOLD series (continued)

- **HOLD-15** · 0.5MP · r60 (r40=m=0) · m=0.99 · hold=0.5 · euler/20-step
  Result: r60 ALSO well-blended, ~0.5 look → high-m GENERALIZES across position.
  [held-keyframe-m-vs-sdedit.md](../latent-hold-release/held-keyframe-m-vs-sdedit.md)
- **HOLD-16** · 0.5MP · r40 · m=0.99 · hold={0.25,0.75} · euler/20-step
  Result: 0.25→frac.55 good; 0.75→frac.24 BAD (non-diagnostic: confounded); step=quality-gate claim RETRACTED.
  [held-keyframe-m-vs-sdedit.md](../latent-hold-release/held-keyframe-m-vs-sdedit.md)
- **HOLD-17** · 0.5MP · r40 · m={0.8,0.9} · hold=0.5 · euler/20-step
  Result: frac 0.39/0.44, BOTH clean → variable m∈[0.8,1] is a clean amount knob; do NOT pin m=1.
  [held-keyframe-m-vs-sdedit.md](../latent-hold-release/held-keyframe-m-vs-sdedit.md)
- **HOLD-18** · 1.0MP · r40 · m=0.99 · hold={0.25,0.5} · euler/20-step
  Result: frac 0.553/0.499 MATCHES 0.5MP → hold recipe RES-INVARIANT; 1MP window REOPENED.
  [resolution-invariance-goal.md](../latent-hold-release/resolution-invariance-goal.md)
- **HOLD-19** · 1.0MP · r40 · m={0.8,0.9} · hold=0.5
  Result: frac 0.362 SMEARS keyframe (blend OK) / 0.392 clean → amount-floor ~0.39 CONFIRMED; m res-compresses.
  [amount-floor-and-step0-redesign.md](../latent-hold-release/amount-floor-and-step0-redesign.md)
- **HOLD-20** · 0.5MP · r40 · m=0.99 · hold={.25,.5,.75} · prenoise_step0=ON
  Result: frac +0.09–0.14 vs OFF (anchor IC identical) → prenoise is a CONTAGION-mediated amount lever; 0.75 still bad.
  [amount-floor-and-step0-redesign.md](../latent-hold-release/amount-floor-and-step0-redesign.md)
- **HOLD-21** · 1.0MP · r40 · m=0.8 · hold=0.5 · prenoise_step0=ON
  Result: frac 0.372 (Δ+0.01 vs HOLD-19 OFF) STILL smears → step[0] redesign FALSIFIED as smear-fix;
  floor = anchor's own low-m denoise (base-model, ties RES-1).
  [amount-floor-and-step0-redesign.md](../latent-hold-release/amount-floor-and-step0-redesign.md)
- **HOLD-22** · 0.2MP · r40 · true no-hold m=0.5 · hold=0 · euler
  Result: frac 0.273 (`|x0|`=0.5362, `|clean|`=0.7319) ≈ hold=0.01 proxy 0.272 → proxy VALID;
  @0.2MP m IS the knob (d≈m ruler); divergence prediction FALSIFIED.
  [knob-design-open-questions.md](../latent-hold-release/knob-design-open-questions.md)
- **HOLD-23** · 0.2MP · r40 · m=0.99 · hold=0.5 · prenoise OFF/ON · euler
  Result: frac 0.379/0.608, BOTH read ~0.5 denoise → anchor CONFIRMED at 3rd res (0.2/0.5/1.0MP);
  prenoise +0.23 frac yet perceived-d unchanged.
  [knob-design-open-questions.md](../latent-hold-release/knob-design-open-questions.md)
- **HOLD-24** · 0.2MP · fade-in f0 + r40/r60 · per-frame release · schedule-sigma pin · euler
  Result: Blend clean, STRUCTURALLY INCOHERENT; schedule-sigma pin FALSIFIED
  (strong anchors hidden as noise in early structure window).
  [per-frame-scheduled-release.md](../latent-hold-release/per-frame-scheduled-release.md)
- **HOLD-25** · 0.2MP · fade-in f0 + r40/r60 · per-frame release · neighbors-see-release pin+label · euler
  Result: Structural coherence RESTORED; blend/denoise solid; HOLD-24 incoherence resolved;
  neighbors-see-release CONFIRMED.
  [per-frame-scheduled-release.md](../latent-hold-release/per-frame-scheduled-release.md)
- **HOLD-26** · 0.2/0.5MP · per-frame release + min-free-steps floor · min_ratio/rescale A/B · euler
  Result: DESIGN (impl `074e443`, code-confirmed, NOT GPU-verified): decouple LEVEL (intended-d) from TIMING; L→0 sub-schedule.
  Sweep (pre-refit `e5996c0`): min_ratio=0 pop persists; rescale kills d=0.2 pop @0.1, floor-only ~0.3;
  d=0.05 clean @0.2–0.3 but rescale made it LOOK like d≈0.3 → motivated intended-d refit.
  [min-free-steps-floor.md](../latent-hold-release/min-free-steps-floor.md)
- **HOLD-27** · 0.2/0.5MP · per-frame release + min-free-steps floor (`074e443`) · min_ratio/rescale 5-run sweep · euler
  Result: GPU-FALSIFIED: all 5 runs over-denoise incl. Option-1 regression (run 4 accurate-0.05/no-pop on `e5996c0`).
  (1) per_frame path has no denoised correction → free-step count governs redraw, not level pin.
  (2) provenance-blind floor hits opening fade-out ramp rows → new fade pops (same class as Findings 7–11 confound).
  [min-free-steps-floor.md](../latent-hold-release/min-free-steps-floor.md)

## Other series

- **STILL-1** · 0.5MP · r40 5f/22f repeated still · d=0.5 ease fade · euler
  Result: Seam blends; anchor under-denoised + freeze-read; cross-inject r60 resolved (contagion).
  [gpu-test-0.5mp.md](../isolated-frame-attention-support/gpu-test-0.5mp.md)
- **MC-1** · ~0.5MP · keyframe md>0 · Blended vs MC side-by-side · euler @06c6bda
  Result: MC pops/ghosts; Blended SMOOTH; headline raison d'être confirmed.
  [motion-context-comparison.md](../motion-context-comparison.md)
- **MC-2** · 1MP · r40+r60 · MC "H3 Custom Keyframes" cond-row · stock KSampler
  Result: Neighbors blend well @1MP; anchor rows copy-clean (wrong content).
  [fade-and-decoupler.md](../conditioning-row-inject/fade-and-decoupler.md)
- **AUG-1** · 0.2MP · r40/r60 · per-frame latent-only strength 0.5/0.45 · MC KSampler
  Result: Anchors → static (noise+timestep decoupled); per-frame lever-1 FALSIFIED.
  [aug-mechanism.md](../conditioning-row-inject/aug-mechanism.md)
- **AUG-2** · 0.2MP · r40/r60 · global aug=0.5 · MC KSampler
  Result: Both anchors blend (r40 slight residual noise, r60 slightly off-inject).
  [aug-mechanism.md](../conditioning-row-inject/aug-mechanism.md)
- **AUG-3** · 0.2MP · r40 · our inject + frac mask + cond token aug=0.4 · our sampler
  Result: Anchor blends; contagion noise blobs spread into neighbors.
  [aug-mechanism.md](../conditioning-row-inject/aug-mechanism.md)
- **AUG-4** · 0.2MP · r40 · clean cond token aug≈0.999 + latent inject · our sampler
  Result: Clean blend; anchor FROZEN/unchanged (clean cond = hard anchor, no denoise).
  [aug-mechanism.md](../conditioning-row-inject/aug-mechanism.md)
- **AUG-5** · 0.2MP · r40 · frac latent d>0 + clean cond row · our sampler
  Result: Great blend; inject FROZEN + hard-edge contagion → single-pass decouple DEAD.
  [aug-mechanism.md](../conditioning-row-inject/aug-mechanism.md)
- **AUD-A** · ~0.5MP · f0 · d=0.01, audio fade · euler
  Result: Audio garbled (×4 mismatch); ×S fix → CLEAN; retest incl. non-default shift → CONFIRMED.
  [bugs.md](../bugs.md)
- **BUG-B** · any · frac rows · any m · euler_ancestral
  Result: Frac rows run in reverse → grey static (RF renoise not scale-invariant; deferred).
  [bugs.md](../bugs.md)
- **DD-1** · any · anchor+fade rows · any m · euler + DD
  Result: Fine through step 7/10; cracks step 7→10; euler_a survives same case.
  [differential-diffusion.md](../differential-diffusion.md)
- **VER-1** · ~0.5MP · multi-inject · md=0.2–0.3 · euler @06c6bda
  Result: Full workflow checklist PASS: headline + audio + keep-mode + non-default shift all good.
  [status-and-open-paths.md](../status-and-open-paths.md)
- **SCHED-1** · inject · d=0.0 · both-mode · euler @34a5925
  Result: Strong anchor, clean temporal blend; anchor conformance confirmed.
  [schedule-tail-late-delta.md](../schedule-tail-late-delta.md)
- **SCHED-2** · inject · d=0.05 · both-mode · euler @34a5925
  Result: STRONG conformance/blend (label~0.61 / σ~0.39); well outside pin, still anchors.
  [schedule-tail-late-delta.md](../schedule-tail-late-delta.md)
- **SCHED-3** · inject · d≈0.2 · both-mode · euler @34a5925
  Result: Weak anchoring on inject-distinct regions (label~0.25 / σ~0.75); falloff confirmed.
  [schedule-tail-late-delta.md](../schedule-tail-late-delta.md)
- **SCHED-4** · inject · d≈0.2 · mask-drop-mode · euler @34a5925 · same scene
  Result: Temporal blend VERY GOOD / in-frame BAD (expected) / inject substantially redrawn → H2 FALSIFIED, H1 sole primary.
  [schedule-tail-late-delta.md](../schedule-tail-late-delta.md)
- **OFFLABEL-1** · inject · d≈0.2 · both-mode · official_labels=ON · euler @48b846e
  Result: TOTALLY BROKEN: abstract/psychedelic patterns; label load-bearing for own velocity prediction;
  label-lie family CLOSED.
  [label-channel-probe.md](../schedule-tail-late-delta/label-channel-probe.md)
