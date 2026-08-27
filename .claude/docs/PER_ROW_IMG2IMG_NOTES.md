<!-- provenance: status (top-level index & direction; child docs carry their own tags) -->
<!-- verified: 2026-08-27 · HOLD-27 GPU result recorded -->
# Per-Row img2img for H3 — Index & Direction

**Purpose:** the durable, token-lean map of this effort. Read THIS file first every session;
drill into a detail doc under [`per-row-img2img/`](per-row-img2img/) only when the current task
needs it. If code contradicts a doc, fix one of them — don't silently diverge.

Last updated: 2026-08-27 (branch `record-schedule-tail-design`). Mode `rescheduled` audited and a
per-row multistep/stochastic sampler design sketched, in
[schedule-tail-composite-release](per-row-img2img/latent-hold-release/schedule-tail-composite-release.md)
children `consistency-audit` and `multistep-stochastic-support`.

⚠ **Code comments/docstrings/tooltips are likely STALE mid-rework** (e.g. hold-and-release
language, "(inclusive)" on the exclusive `end_keyframes`, "compatible with all samplers"). Trust
the wiki + tests over in-code prose until the post-prototype docs pass. Do not anchor on task-list
numbers anywhere in the wiki — the task list is cleared during prototyping; cite commits/PRs.

## The goal

Make `min_denoise:X` behave like **intuitive img2img denoise** — no ghosting/artifacts for
`0 < min_denoise < 1` — anchoring a keyframed section the rest of the video blends into.
Immediate bar: **as good as Motion Context's fade mask, minus the ghosting.** Constraint: do NOT
recreate Motion Context 1:1 (would reset the repo instead).

**Why this repo exists (user-confirmed).** Motion Context already does everything wanted —
including temporally-faded masks on stochastic samplers — **as long as `min_denoise = 0.0`.** At
`0 < min_denoise < 1` on the anchor keyframe it fails two ways: (a) it **ghosts**, and (b) the knob
is **semantically broken** — the per-step re-composite drags the frame back to the original, so
even 0.15–0.2 yields near-original frames + ghost instead of "mild regeneration." Fixing both
(intuitive img2img `min_denoise`) is the PRIMARY reason for this repo. Also-better axes: (1)
**injection flexibility** — MC is built for f0 clip-continuation; Blended does arbitrary inject
lengths/points (mid-timeline keyframes, mid-context video guiding) via principled frame-space
snapping; (2) **QoL** — cleaner, fewer nodes vs MC's node-sprawl/messy workflows. And MC upstream
isn't taking PRs + has heavy churn. See
[motion-context-comparison.md](per-row-img2img/motion-context-comparison.md).

**Prototype-drive north star:** ONE user knob mapping visually to img2img denoise `d` (via `hold`+`m`),
resolution-invariant (IDEAL) or via `lever = f(d, res)` (ACCEPTABLE). User sets one value; must NOT re-tune
per resolution. Full statement: [latent-hold-release/resolution-invariance-goal.md](per-row-img2img/latent-hold-release/resolution-invariance-goal.md).

## The core finding (why this isn't trivial)

"img2img via a mask" DOES exist as a general primitive — it's **Differential Diffusion (DD)**.
But on H3, DD and our synthesized approach are **exact duals**, and neither covers both sampler
classes:

| | Deterministic (euler, res_multistep, dpmpp_2m) | Stochastic (euler_a, SDE) |
|---|---|---|
| **Our per-row compression** | ✅ correct (GPU-confirmed video + audio) | ❌ RF renoise not scale-invariant |
| **DD / native inpaint** | ❌ cracks (H3 conditioning-injection) | ✅ ancestral re-noise hides it |

There is no single *native* mechanism covering both on H3; supporting both via native paths = two
engines selected by sampler type = special-casing smell. ⇒ current build is **deterministic-only**.
Details + source proof: [differential-diffusion.md](per-row-img2img/differential-diffusion.md).

**But this may be reopenable:** a THEORY (unverified) argues stochastic is recoverable *inside our
single engine* via a per-row ancestral step — see
[stochastic-recovery-theory.md](per-row-img2img/stochastic-recovery-theory.md).

## Current direction

- **Deterministic-only**, prototype mode. Stochastic-sampler gate DEFERRED; the stochastic shim
  may rot (user doesn't care).
- Our approach = synthesize per-row img2img via three levers (init-lerp + fractional denoise_mask
  conditioning + REQUIRED denoised correction) with `noise_mask=None` and a post-composite for
  exact-preserve rows. See [our-architecture.md](per-row-img2img/our-architecture.md).
- **GPU-VERIFIED (2026-08-23, full user checklist @06c6bda):** fractional audio clean after the
  ×S fix (incl. non-default shift), keyframe `min_denoise` 0.2–0.3 good, keep-mode audio good,
  preview working — and the HEADLINE: at `min_denoise>0` from a keyframe, **MC pops over a ghost
  frame; Blended is smooth**. The repo's raison d'être is confirmed on GPU. Chaining widgets
  hidden @72b61c6 (unsupported, revisit post-prototype). See
  [status-and-open-paths.md](per-row-img2img/status-and-open-paths.md).

## Detail docs — drill down only as needed

- [our-architecture.md](per-row-img2img/our-architecture.md) — the three levers + our sampler
  code. *Read when editing `sampler.py`/`composite.py`/`_run_sampler`.*
- [native-h3-mechanism.md](per-row-img2img/native-h3-mechanism.md) — comfy/H3 internals reference
  map (audio scaling, sampler loop, DiT per-row timestep, k_diffusion samplers). *Read when
  debugging why comfy/H3 behaves a certain way.*
- [differential-diffusion.md](per-row-img2img/differential-diffusion.md) — DD mechanism, the
  ghost math, why native mask paths fail on H3, the duality. *Read before considering any
  mask/DD/inpaint approach.*
- [bugs.md](per-row-img2img/bugs.md) — Bug A (audio scale, fixed) & Bug B (stochastic). *Read
  when debugging fractional-region artifacts.*
- [audio-carry-identity.md](per-row-img2img/audio-carry-identity.md) — **THEORY (source-derived
  math):** why the ×S audio fix is exact globally but leaks per-row for m<1; candidate wrapper
  compensation. *Read when fractional AUDIO artifacts appear.*
- [stochastic-recovery-theory.md](per-row-img2img/stochastic-recovery-theory.md) — **THEORY
  (unverified):** recover stochastic samplers via a per-row ancestral step. *Read when revisiting
  the stochastic gate.*
- [motion-context-comparison.md](per-row-img2img/motion-context-comparison.md) — how Motion Context
  does it (composite-blend; ghost diagnosis) & why it's stochastic-robust; the 3 design points. *Read
  when comparing to MC or deciding the fractional-vs-stochastic tradeoff.*
- [conditioning-row-inject.md](per-row-img2img/conditioning-row-inject.md) — MC's non-masked "H3
  Custom Keyframes" injects `minimax_keyframes` cond rows (native context tokens, no denoise/preserve);
  verdict = different tool, not a substitute; interop is already free through our sampler. *Read when
  considering a conditioning-target inject variant/toggle.*
- [highres-singleframe-underdenoise.md](per-row-img2img/highres-singleframe-underdenoise.md) —
  **THEORY (UNVERIFIED):** single-frame stills at fractional `min_denoise` come out
  source-identical @1MP but clean @0.2MP; leading cause = fixed sigma-shift → resolution-dependent
  effective denoise (H1), fix = resolution-corrected effective-m. *Read when debugging the
  high-res single-frame pop.*
- [highres-underdenoise-model.md](per-row-img2img/highres-underdenoise-model.md) — **THEORY + 1MP
  GPU validation (Fable):** α=ρ up-map FALSIFIED (0.83=chaos, 0.45=lock); refit single-exponent
  **γ≈1.6 → d\*≈0.75-0.78 @1MP**; FOUR régimes (lock→coherent→chaos→generic-gen). Discriminator: Ψ &
  p-cross-1 DEAD → **seam z-score** primary gate + ρ_ret (lock) / φ̄ (chaos) tellers; content &
  anchor-spacing confounds; anchor-then-release fallback if the window is closed. *Read when
  building/calibrating the resolution-corrected effective-m fix.*
- [keyframe-two-views-and-knobs.md](per-row-img2img/keyframe-two-views-and-knobs.md) — **JOINT MODEL:**
  the two questions (neighbor-view vs anchor-resolution) + four knobs (A latent-content / B mask / C cond-aug
  / D composite); maps the user's "neighbors see clean, keyframe denoises full-time" ideal onto
  hybrid/route-2/route-3 (route-1 = worst fit). *Read to reason about cond+latent composition.*
- [timed-cond-removal-prototype.md](per-row-img2img/timed-cond-removal-prototype.md) — **DESIGN (build-first):**
  timed cond-removal — route-1 on knob C; mechanism source-verified (payload COPY + layout drop); gating,
  surface (`cond_hold_frac`). *Read when building/tuning the timed-removal prototype.*
- [lanpaint-langevin-corrector.md](per-row-img2img/lanpaint-langevin-corrector.md) — **REFERENCE
  (external code read):** LanPaint's training-free Langevin inpainting corrector. Verdict: NOT a
  fractional-anchor fix (it binarizes masks → hard-preserves known = the m=0 case we already solve;
  fractionalized it becomes our ghost), but a new **route 4 (per-σ inner-loop equilibration)** and two
  transplants — BiG model-consistency counterweight on Fable's continuous spring (FREE, deterministic)
  + early-step-only re-equilibration. KILL RISK: if neighbor conditioning is t_row-label-gated not
  content-gated, both collapse to single-pass. *Read when considering a corrector/equilibration
  mechanism or the continuous λ(σ) spring.*
- [latent-hold-release/](per-row-img2img/latent-hold-release/index.md) — **STATUS (prototype + GPU
  debug log):** route-1 latent hold-and-release (branch `proto-latent-hold-release`). Subfolder:
  index + `mechanism-and-early-findings` (design, sigma-shift knob, residency + bugs) +
  `attraction-and-envelope` (Findings 4–6: hold-vs-no-hold A/B, freeze-blends, envelope/+0.5 fork) +
  `hold-mechanism-and-confounds` (Findings 7–10: mechanism trace, the provenance-blind `anchor_mask`
  confound that froze the opening fade-out and propagated to corrupt r40, + the provenance fix).
  *Read when working the latent hold-and-release prototype.*
- [status-and-open-paths.md](per-row-img2img/status-and-open-paths.md) — confirmed-working +
  what's next. *Read when planning.*
- [per-row-img2img/experiments-run.md](per-row-img2img/experiments-run.md) — run-results index (40
  rows, stable IDs; split into two child tables); **grep this before proposing any experiment**.
- [file-line-index.md](per-row-img2img/file-line-index.md) — bare source locations. *Read when
  you just need a file:line.*
- [code-quality-audit-2026-08-23.md](per-row-img2img/code-quality-audit-2026-08-23.md) — module
  grades + prioritized cleanup findings from the end-of-prototype audit. *Read when doing
  stabilization/cleanup work.*

## comfy-ref access (meta)

`/home/reithan/projects/comfy-ref` is a SPARSE checkout — files we reference, ON DISK. When a
needed file is missing, ADD it (user-confirmed policy 2026-08-24):
`cd /home/reithan/projects/comfy-ref && git sparse-checkout add /path/to/file` (leading slash
for a single file). Checked out now:
`comfy/{sample,samplers,utils,nested_tensor,model_base,model_sampling,latent_formats}.py`,
`comfy/k_diffusion/sampling.py`, `comfy/ldm/minimax/`,
`comfy_extras/{nodes_differential_diffusion,nodes_minimax_h3}.py`.

**Verification stamps:** every wiki doc carries a `<!-- verified: <date> · <source> @<sha> -->`
comment on line 2. It names the date and the exact HEAD SHA(s) of each source repo its file:line
refs were checked against. Refresh the stamp (date + SHA) whenever you re-verify a doc. When a
cited SHA no longer matches the current checkout, treat line numbers as approximate hints and
navigate by symbol name instead.
