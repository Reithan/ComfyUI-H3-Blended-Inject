<!-- provenance: status (current state / open paths) -->
<!-- verified: 2026-08-24 · repo @72b61c6 -->
<!-- last updated: 2026-08-27, branch: proto-schedule-tail-release -->
<!-- anchoring: reference commits/PRs, never task numbers — the task list is cleared/churned
     during prototyping and is not a durable artifact -->
# Status & open paths

Read this when planning what to do next.

## Blended vs Motion Context — the fork (current build)

The stochastic limit comes from **fractional (fade) rows**, not from `min_denoise>0`; any faded
blend has `0<m<1` ramp rows even at `min_denoise=0`, and those break under stochastic
([bugs · Bug B](bugs.md#bug-b)). So Blended is deterministic-only for any real blend.

| You need… | Use | Why |
|---|---|---|
| `min_denoise = 0.0`, stochastic samplers | **Motion Context** | MC handles fade fractionals on stochastic (transient → no visible ghost); Blended is deterministic-only. |
| `min_denoise = 0.0`, cleaner workflow / fewer nodes | **Blended** | QoL/cleanliness win; deterministic samplers only. |
| `min_denoise > 0.0` (true fractional-keyframe denoise) | **Blended (only correct option)** | MC ghosts ([knob semantically broken](motion-context-comparison.md)); true img2img, deterministic only. |

Not a total order at `min_denoise=0`: MC wins **sampler coverage**, Blended wins **QoL/cleanliness**.
The [stochastic-recovery theory](stochastic-recovery-theory.md) would remove the tradeoff entirely.

## Confirmed WORKING (user GPU tests)

Full checklist pass @06c6bda (user GPU run, 2026-08-23), **every item good**:

- **HEADLINE: raison d'être confirmed:** fade from a **keyframe with `min_denoise > 0`**; MC
  pops over a ghost frame; **Blended is smooth** (no ghost, no pop). The exact failure this repo
  exists to fix, now GPU-verified. Fade from a clip at `min_denoise=0.0` looks visually similar
  between MC and Blended (parity, as targeted).
- 39f fade f0, euler: fractional **audio clean** after the ×S fix ([Bug A](bugs.md#bug-a)
  CONFIRMED fixed; [audio-carry-identity](audio-carry-identity.md) per-row leak not perceptible
  in this run).
- `min_denoise` 0.2–0.3 on a keyframe: good (true img2img behavior).
- `audio_mode='keep'`: good.
- Non-default sigma shift: good (**×S generalization confirmed** beyond S=4).
- Stock latent preview (`latent_preview.prepare_callback` @06c6bda): confirmed working.

Earlier spot checks: 1f inject f187 denoise 0.2 euler perfect; deterministic per-row denoised
correction confirmed ([our-architecture](our-architecture.md)).

## Open paths

1. **Isolated-single-frame fractional-keyframe underdenoise bug**: isolated single frame at `0<d<1` pops at
   high res (works at 0.2MP; d-only latent-noise-boost window **CLOSED @1MP**, DATA); neighbor-blend half
   FIXED by support/clean cond; anchor-resolution must be denoised IN-CONTEXT. Falsified approaches:
   bake-beforehand (paradox), single-pass decouple (freeze+contagion), self-duplication (freeze), per-frame
   latent-only aug (lever 1). Surviving: **timed-removal (build first)**, route-2 two-pass. Instrumentation
   (Ψ/seam-z/k_comp) on branch `debug-single-frame-underdenoise`. See
   [isolated-frame-attention-support](isolated-frame-attention-support.md),
   [highres-underdenoise-model](highres-underdenoise-model.md),
   [keyframe-two-views-and-knobs](keyframe-two-views-and-knobs.md).
   **TWO PARALLEL TRACKS (user decision, 2026-08-24):** the H3AddGuide node (per-guide timed cond
   removal) is being built first (a COND-channel solution, useful in its own right), but it does NOT
   retire this problem. A **LATENT-resident solution is still an active goal**:
   likely one of the hold-and-release strategies on the latent/mask side (knob B), needing new tests
   and possibly a new node. OPEN: whether 1-frame keyframe injects stay in `H3AddInject` or get a
   dedicated node; decide when the latent-resident work starts.

   **PROTOTYPE (branch `proto-latent-hold-release`, NOT for merge): design + GPU debug log in
   [latent-hold-release/](latent-hold-release/index.md).** Route-1 anchor-then-release on knob **B** (latent).
   State as of 2026-08-25:
   - Hold-residency **CONFIRMED** on GPU (bit-identical clean through the hold).
   - A/B + denoise=0.0: hold OFF blends; hold ON cuts; freeze BLENDS. Attraction is a baseline property,
     no hold needed. Envelope SETTLED under none-interp: `{row40: md, row39: 1.0}`; Fable +0.5 trap
     does NOT apply.
   - **Hold-ON cut EXPLAINED:** `anchor_mask=(m>0)&(m<1)` is provenance-blind → froze the opening
     fade-out (1.08M elems), not the keyframes; wrong-row freeze propagated via H3 global attention,
     corrupting r40's blend. FIX = tag intended inject rows at construction, then re-run keyframe-only A/B.
     Route-1 is NOT a dead end. Full log:
     [latent-hold-release/hold-mechanism-and-confounds](latent-hold-release/hold-mechanism-and-confounds.md)
     Findings 7–10 (attraction/envelope Findings 4–6 in sibling doc).
2. ~~GPU-verify the current build~~ **DONE**: full checklist passed (see Confirmed WORKING
   above).
3. **Stochastic: warning shipped (@06c6bda), hard gate deferred.** `sampler_is_stochastic`
   (eta-default>0 signature heuristic, no hardcoded list) drives a `UserWarning` when fractional
   rows meet an ancestral/SDE sampler. The dead magnitude shim (`make_per_row_noise_sampler`,
   `scale_stochastic_noise`) can still be deleted when hardening. **Alternative to gating:** the
   per-row ancestral step in [stochastic-recovery-theory](stochastic-recovery-theory.md) would
   instead SUPPORT stochastic (spike euler_ancestral first, after the GPU pass).
4. **Chaining widgets: RESOLVED (hidden @72b61c6, revisit post-prototype).** User decision:
   returning leftover noise is pointless if no follow-up node can consume it, and the resume side
   (add_noise=disable feeding init-lerp an already-noised x) is the hard part, so the whole
   surface (`add_noise`, `start_at_step`, `end_at_step`, `return_with_leftover_noise`) is removed
   from `INPUT_TYPES` and hardcoded internally to add_noise=enable / full range / no leftover.
   If chained sampling is ever wanted: output side is a cheap per-row rescale
   `×(1−σ_end)/(1−m·σ_end)`; resume side is substantially harder (see
   [our-architecture · sampler-surface limitations](our-architecture.md)).
5. **DD path: RESOLVED, dead end for our goal.** See [differential-diffusion](differential-diffusion.md):
   it's the dual of ours (stochastic-only on H3). Not worth building unless we decide to support
   stochastic via a separate engine.
6. **H4 observer-label K/V split (PROTOTYPE, GPU-pending 2026-08-28).** SCHED experiments confirmed
   H1 (late-delta anchoring) and falsified H2 (mask-drop via label re-routing). OFFLABEL-1 closed the
   label-lie family: official labels are load-bearing for velocity prediction and cannot be replaced.
   Active prototype on `proto-schedule-tail-release`: inject a separate observer label row that reads
   H4 K/V context without contributing to the velocity label, probing whether H4 label-ratio is the
   stochastic-recovery lever. GPU run pending. Design:
   [label-ratio-and-observer-split](schedule-tail-late-delta/label-ratio-and-observer-split.md).
7. **Cleanup (independent):** remove unused `derive_mask`/`apply_derived_mask` (+ their tests)
   and the dead `region`/`classify_row_region` schedule leftovers, dead since the per-row
   rework (PR #4).
