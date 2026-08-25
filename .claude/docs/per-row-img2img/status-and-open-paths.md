<!-- provenance: status (current state / open paths) -->
<!-- verified: 2026-08-24 · repo @72b61c6 -->
<!-- last updated: 2026-08-24, branch: debug-single-frame-underdenoise -->
<!-- anchoring: reference commits/PRs, never task numbers — the task list is cleared/churned
     during prototyping and is not a durable artifact -->
# Status & open paths

Read this when planning what to do next.

## Blended vs Motion Context — the fork (current build)

The stochastic limit comes from **fractional (fade) rows**, not from `min_denoise>0` — any faded
blend has `0<m<1` ramp rows even at `min_denoise=0`, and those break under stochastic
([bugs · Bug B](bugs.md#bug-b)). So Blended is deterministic-only for any real blend.

| You need… | Use | Why |
|---|---|---|
| `min_denoise = 0.0`, stochastic samplers | **Motion Context** | MC handles fade fractionals on stochastic (transient → no visible ghost); Blended is deterministic-only. |
| `min_denoise = 0.0`, cleaner workflow / fewer nodes | **Blended** | QoL/cleanliness win; deterministic samplers only. |
| `min_denoise > 0.0` (true fractional-keyframe denoise) | **Blended (only correct option)** | MC ghosts AND its knob is semantically broken (see [motion-context-comparison](motion-context-comparison.md)); Blended = true img2img, but deterministic only. |

Not a total order at `min_denoise=0`: MC wins **sampler coverage**, Blended wins **QoL/cleanliness**.
The [stochastic-recovery theory](stochastic-recovery-theory.md) would remove the tradeoff entirely.

## Confirmed WORKING (user GPU tests)

Full checklist pass @06c6bda (user GPU run, 2026-08-23) — **every item good**:

- **HEADLINE — raison d'être confirmed:** fade from a **keyframe with `min_denoise > 0`** — MC
  pops over a ghost frame; **Blended is smooth** (no ghost, no pop). The exact failure this repo
  exists to fix, now GPU-verified. Fade from a clip at `min_denoise=0.0` looks visually similar
  between MC and Blended (parity, as targeted).
- 39f fade f0, euler: fractional **audio clean** after the ×S fix ([Bug A](bugs.md#bug-a)
  CONFIRMED fixed; [audio-carry-identity](audio-carry-identity.md) per-row leak not perceptible
  in this run).
- `min_denoise` 0.2–0.3 on a keyframe: good (true img2img behavior).
- `audio_mode='keep'`: good.
- Non-default sigma shift: good (**×S generalization confirmed**, not just S=4).
- Stock latent preview (`latent_preview.prepare_callback` @06c6bda): confirmed working.

Earlier spot checks: 1f inject f187 denoise 0.2 euler perfect; deterministic per-row denoised
correction confirmed ([our-architecture](our-architecture.md)).

## Open paths

1. **Isolated-single-frame fractional-keyframe underdenoise bug** — isolated single frame at `0<d<1` pops at
   high res (works at 0.2MP; d-only latent-noise-boost window **CLOSED @1MP**, DATA); neighbor-blend half
   FIXED by support/clean cond; anchor-resolution must be denoised IN-CONTEXT. Falsified approaches:
   bake-beforehand (paradox), single-pass decouple (freeze+contagion), self-duplication (freeze), per-frame
   latent-only aug (lever 1). Surviving: **timed-removal (build first)**, route-2 two-pass. Instrumentation
   (Ψ/seam-z/k_comp) on branch `debug-single-frame-underdenoise`. See
   [isolated-frame-attention-support](isolated-frame-attention-support.md),
   [highres-underdenoise-model](highres-underdenoise-model.md),
   [keyframe-two-views-and-knobs](keyframe-two-views-and-knobs.md).
2. ~~GPU-verify the current build~~ **DONE** — full checklist passed (see Confirmed WORKING
   above).
3. **Stochastic: warning shipped (@06c6bda), hard gate deferred.** `sampler_is_stochastic`
   (eta-default>0 signature heuristic, no hardcoded list) drives a `UserWarning` when fractional
   rows meet an ancestral/SDE sampler. The dead magnitude shim (`make_per_row_noise_sampler`,
   `scale_stochastic_noise`) can still be deleted when hardening. **Alternative to gating:** the
   per-row ancestral step in [stochastic-recovery-theory](stochastic-recovery-theory.md) would
   instead SUPPORT stochastic (spike euler_ancestral first, after the GPU pass).
4. **Chaining widgets — RESOLVED: hidden (@72b61c6), revisit post-prototype.** User decision:
   returning leftover noise is pointless if no follow-up node can consume it, and the resume side
   (add_noise=disable feeding init-lerp an already-noised x) is the hard part — so the whole
   surface (`add_noise`, `start_at_step`, `end_at_step`, `return_with_leftover_noise`) is removed
   from `INPUT_TYPES` and hardcoded internally to add_noise=enable / full range / no leftover.
   If chained sampling is ever wanted: output side is a cheap per-row rescale
   `×(1−σ_end)/(1−m·σ_end)`; resume side is substantially harder (see
   [our-architecture · sampler-surface limitations](our-architecture.md)).
5. **DD path — RESOLVED, dead end for our goal.** See [differential-diffusion](differential-diffusion.md):
   it's the dual of ours (stochastic-only on H3). Not worth building unless we decide to support
   stochastic via a separate engine.
6. **Cleanup (independent):** remove unused `derive_mask`/`apply_derived_mask` (+ their tests)
   and the dead `region`/`classify_row_region` schedule leftovers — dead since the per-row
   rework (PR #4).
