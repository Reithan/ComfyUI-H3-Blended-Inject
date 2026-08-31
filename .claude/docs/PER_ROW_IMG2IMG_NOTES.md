<!-- provenance: status (top-level index & direction; child docs carry their own tags) -->
<!-- verified: 2026-08-27 · HOLD-27 GPU result recorded -->
# Per-Row img2img for H3: Index & Direction

**Purpose:** the durable, token-lean map of this effort. Read this file first every session;
drill into a detail doc under [`per-row-img2img/`](per-row-img2img/) only when the current task
needs it. If code contradicts a doc, fix one of them; don't silently diverge.

Last updated: 2026-08-31 (branch `wiki-per-row-img2img-c2-findings`). Reconciled the fade noise into a
TWO-LAYER STACK — accepted decoupling Δ residue (both modalities, floor) + C2 carry-compression
(audio-only, correctable) + ancestral ρ-drift (amplifier); PRIMARY candidate = decoupling-residue vs
raw `_euler_step` r-lerp arithmetic (not yet distinguished). See
c2-rho-fix-paths/residual-accounting.md + long-fade-grid-beat/kv-observer-mismatch.md.

⚠ **Code comments/docstrings/tooltips are likely STALE mid-rework** (hold-and-release language,
"(inclusive)" on the exclusive `end_keyframes`, "compatible with all samplers"). Trust the wiki +
tests over in-code prose until the post-prototype docs pass. Cite commits/PRs, not task-list numbers.

## The goal

Make `min_denoise:X` behave like **intuitive img2img denoise** (no ghosting/artifacts for
`0 < min_denoise < 1`), anchoring a keyframed section the rest of the video blends into.
Immediate bar: **as good as Motion Context's fade mask, minus the ghosting.** Constraint: do NOT
recreate Motion Context 1:1 (would reset the repo instead).

**Why this repo exists (user-confirmed).** Motion Context does everything wanted, including
temporally-faded masks on stochastic samplers, as long as `min_denoise = 0.0`. At
`0 < min_denoise < 1` on the anchor keyframe it fails two ways: (a) it **ghosts**, and (b) the
knob is **semantically broken** (the per-step re-composite drags the frame back to the original,
so even 0.15–0.2 yields near-original frames + ghost). Fixing both is the PRIMARY reason for this
repo. Also-better axes: (1) **injection flexibility** — MC is built for f0 clip-continuation;
Blended does arbitrary inject lengths/points via principled frame-space snapping; (2) **QoL** —
fewer nodes vs MC's node-sprawl (MC upstream accepts no PRs, heavy churn). See
[motion-context-comparison.md](per-row-img2img/motion-context-comparison.md).

**Prototype north star:** ONE knob mapping visually to img2img denoise `d` (via `hold`+`m`),
resolution-invariant (IDEAL) or via `lever = f(d, res)` (ACCEPTABLE); no per-resolution re-tune.
Full statement: [latent-hold-release/resolution-invariance-goal.md](per-row-img2img/latent-hold-release/resolution-invariance-goal.md).

## The core finding (why this isn't trivial)

"img2img via a mask" exists as a general primitive: **Differential Diffusion (DD)**.
On H3, DD and our synthesized approach are **exact duals**, and neither covers both sampler
classes:

| | Deterministic (euler, res_multistep, dpmpp_2m) | Stochastic (euler_a, SDE) |
|---|---|---|
| **Our per-row compression** | ✅ correct (GPU-confirmed video + audio) | ❌ RF renoise not scale-invariant |
| **DD / native inpaint** | ❌ cracks (H3 conditioning-injection) | ✅ ancestral re-noise hides it |

No single native mechanism covers both on H3; supporting both via native paths requires two
engines selected by sampler type (special-casing smell). Current build is **deterministic-only**.
Details + source proof: [differential-diffusion.md](per-row-img2img/differential-diffusion.md).

**But this may be reopenable:** a THEORY (unverified) argues stochastic is recoverable inside our
single engine via a per-row ancestral step. See
[stochastic-recovery-theory.md](per-row-img2img/stochastic-recovery-theory.md).

## Current direction

- **SHIPPED (production, branch `implement-inject-schedule-remap`, off `main`):** the schedule-tail
  remap + observer-label K/V split is the SOLE, always-on per-row img2img mechanism — no toggles.
  Mode is `rescheduled`: per-row schedule-tail remap (dense `steps²+1` σ-grid, exact stretched-tail
  sigma) with per-row `r`-scaling onto each row's own compressed σ-tail, init-only clean composite
  at step 0, truthful `w` labels, observer KV split always on. **Audio port completed:** audio rows
  run the remap on the sigma-shifted audio schedule via `time_shift_sigma` (only σ VALUES differ by
  modality; `k_d`/`span` shared). Canonical detail:
  [label-ratio-and-observer-split](per-row-img2img/schedule-tail-late-delta/label-ratio-and-observer-split.md).
- **RETIRED:** the stock three-lever path (init-lerp + fractional denoise_mask conditioning +
  denoised correction) and the deferred stochastic-sampler shim (Bug B) were DELETED; the remap's
  per-row `r`-scaling replaces the denoised correction. [our-architecture.md](per-row-img2img/our-architecture.md)
  now describes the retired stock path only.
- **Deterministic-only** still holds: deterministic correct; stochastic warns (RF renoise not
  scale-invariant, gate deferred; multistep runs first-order under remap — see [sampler-class-support.md](per-row-img2img/sampler-class-support.md)).
- **GPU-VERIFIED (2026-08-23, full user checklist @06c6bda):** fractional audio clean after the
  ×S fix, keyframe `min_denoise` 0.2–0.3 good, keep-mode audio good, preview working. HEADLINE: at
  `min_denoise>0` from a keyframe, **MC pops over a ghost frame; Blended is smooth** — repo raison
  d'être confirmed. Chaining widgets hidden @72b61c6 (unsupported, revisit post-prototype). See
  [status-and-open-paths.md](per-row-img2img/status-and-open-paths.md).

## Detail docs: drill down only as needed

- [our-architecture.md](per-row-img2img/our-architecture.md): the three levers + sampler code.
  *Read when editing `sampler.py`/`composite.py`/`_run_sampler`.*
- [native-h3-mechanism.md](per-row-img2img/native-h3-mechanism.md): comfy/H3 internals reference
  map (audio scaling, sampler loop, DiT per-row timestep, k_diffusion samplers). *Read when
  debugging why comfy/H3 behaves a certain way.*
- [differential-diffusion.md](per-row-img2img/differential-diffusion.md): DD mechanism, ghost
  math, why native mask paths fail on H3, the duality. *Read before considering any mask/DD/inpaint approach.*
- [bugs.md](per-row-img2img/bugs.md): Bug A (audio scale, fixed), B (stochastic, open), C (free-audio
  ancestral axis, FIXED by Fix A), D (optional inject_list, fixed), E (long-fade video interference,
  open). *Read when debugging fractional-region artifacts.*
- [long-fade-grid-beat.md](per-row-img2img/long-fade-grid-beat.md): **THEORY (UNVERIFIED):** Bug E
  DECOUPLED — M-B (held ≥ ~28 AND ramp ≥ 51) unique survivor; M-A/M-C/M-D/M-E refuted; refined to
  FORMATION ∧ NOT-HEALED; full data table + mechanism + children. *Read for Bug E.*
- [audio-carry-identity.md](per-row-img2img/audio-carry-identity.md): **CONFIRMED (source-derived
  math):** why the ×S audio fix is exact globally but leaks per-row for m<1 (C2 audibility
  AMBIGUOUS); candidate wrapper compensation. *Read when fractional AUDIO artifacts appear.*
- [audio-axis-verdict.md](per-row-img2img/audio-axis-verdict.md): **Fix A VALIDATED free audio
  (ancestral integration → σ_v axis); H2 FALSIFIED (fade-length confound); σ_a-LABEL proof valid;
  primary long-fade VIDEO bug open.** *Read for euler_a audio behavior.*
- [stochastic-recovery-theory.md](per-row-img2img/stochastic-recovery-theory.md): **THEORY
  (unverified):** recover stochastic samplers via a per-row ancestral step. *Read when revisiting
  the stochastic gate.*
- [sampler-class-support.md](per-row-img2img/sampler-class-support.md): **THEORY + 1 confirmed:**
  multistep degrades to first-order under remap; step-function design for stochastic + 2nd-order.
  *Read when revisiting sampler support.*
- [motion-context-comparison.md](per-row-img2img/motion-context-comparison.md): how Motion Context
  works (composite-blend; ghost diagnosis), why it's stochastic-robust, and the 3 design points.
  *Read when comparing to MC or deciding the fractional-vs-stochastic tradeoff.*
- [conditioning-row-inject.md](per-row-img2img/conditioning-row-inject.md): MC's non-masked "H3
  Custom Keyframes" (`minimax_keyframes` cond rows); verdict: different tool, not substitute;
  interop free. *Read when considering a conditioning-target inject variant/toggle.*
- [highres-singleframe-underdenoise.md](per-row-img2img/highres-singleframe-underdenoise.md):
  **THEORY (UNVERIFIED):** single-frame stills at fractional `min_denoise` come out
  source-identical @1MP but clean @0.2MP; leading cause: fixed sigma-shift giving
  resolution-dependent effective denoise (H1); fix: resolution-corrected effective-m. *Read when
  debugging the high-res single-frame pop.*
- [highres-underdenoise-model.md](per-row-img2img/highres-underdenoise-model.md): **THEORY + 1MP
  GPU validation (Fable):** α=ρ up-map FALSIFIED; refit single-exponent **γ≈1.6 → d\*≈0.75-0.78
  @1MP**; FOUR régimes; seam z-score primary gate + ρ_ret / φ̄ tellers. *Read when
  building/calibrating the resolution-corrected effective-m fix.*
- [keyframe-two-views-and-knobs.md](per-row-img2img/keyframe-two-views-and-knobs.md): **JOINT
  MODEL:** the two questions (neighbor-view vs anchor-resolution) + four knobs (A latent-content /
  B mask / C cond-aug / D composite); maps the "neighbors see clean, keyframe denoises full-time"
  ideal onto hybrid/route-2 (route-1 = worst fit). *Read to reason about cond+latent composition.*
- [timed-cond-removal-prototype.md](per-row-img2img/timed-cond-removal-prototype.md): **DESIGN:**
  timed cond-removal on knob C; source-verified (payload COPY + layout drop); surface
  (`cond_hold_frac`). *Read when building/tuning the timed-removal prototype.*
- [lanpaint-langevin-corrector.md](per-row-img2img/lanpaint-langevin-corrector.md): **REFERENCE
  (external code read):** LanPaint's Langevin inpainting corrector; verdict: new **route 4
  (per-σ inner-loop equilibration)** and two transplants; attention-logit boost (route-3) REJECTED
  2026-08-27. KILL RISK: if neighbor conditioning is t_row-label-gated not content-gated, both
  collapse to single-pass. *Read when considering a corrector/equilibration mechanism.*
- [latent-hold-release/](per-row-img2img/latent-hold-release/index.md): **STATUS (SUPERSEDED,
  route-1):** hold-and-release prototype (branch `proto-latent-hold-release`); see subfolder index
  for child docs. *Read only when working the superseded prototype.*
- [status-and-open-paths.md](per-row-img2img/status-and-open-paths.md): confirmed-working state +
  open paths. *Read when planning.*
- [per-row-img2img/experiments-run.md](per-row-img2img/experiments-run.md): run-results index (40
  rows, stable IDs; split into two child tables). **Grep this before proposing any experiment.**
- [file-line-index.md](per-row-img2img/file-line-index.md): bare source locations. *Read when
  you just need a file:line.*
- [code-quality-audit-2026-08-23.md](per-row-img2img/code-quality-audit-2026-08-23.md): module
  grades + prioritized cleanup findings from the end-of-prototype audit. *Read when doing
  stabilization/cleanup work.*
- [schedule-tail-late-delta](per-row-img2img/schedule-tail-late-delta.md): **SHIPPED (observer-label
  K/V split = production mechanism); H1 sole primary; H2/route-3/renoise-release/label-lie/ramp-join
  FALSIFIED or DEAD; OFFLABEL-1 GPU 2026-08-28.** *Read for schedule-tail blend-fight.*

## comfy-ref access & verification stamps (meta)

How to reach comfy source (sparse checkout) and interpret each doc's line-2 verified stamp are
split into [comfy-ref-access.md](per-row-img2img/comfy-ref-access.md). *Read when you need a comfy
file that isn't checked out, or when refreshing a verification stamp.*
