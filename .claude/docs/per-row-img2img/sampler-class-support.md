<!-- provenance: theory + confirmed (Finding 2 GPU-CONFIRMED for euler_ancestral VIDEO 2026-08-27;
     Finding 1 FIXED by PR3 2026-09-02, GPU-CONFIRMED (user local test dpmpp_2m + res_multistep both good);
     audio AXIS-BLIND post-#33 (no σ_a projection); PR4 SDE BUILT + GPU-CONFIRMED user local 2026-09-02 all three, PR #36 open; PR5 dpmpp_2s_ancestral BUILT + CPU-tested, GPU pending) -->
<!-- verified: 2026-09-02 · PR4 SDE BUILT + GPU-CONFIRMED (add-per-row-dpmpp-sde-steps, PR #36 open; user local all three good: 2m_sde/3m_sde/sde); PR3 multistep BUILT + GPU-CONFIRMED (@6a5e786, user local dpmpp_2m + res_multistep both good; CPU m=1 tests); PR2 GPU pass task #68 @ede2d8c; PR5 dpmpp_2s_ancestral BUILT + CPU-tested (same PR #36 branch), GPU pending in task #73 -->
<!-- source: comfy-ref k_diffusion/sampling.py @b78cec87: 240-266 euler_ancestral_RF, 738-792 dpmpp_sde,
     796-818 dpmpp_2m, 822-873 dpmpp_2m_sde, 68-176 helpers, 1394+ res_multistep; repo sampler.py, observer_split.py -->
# Sampler-class support under the schedule-tail remap (stochastic + multi-step)

**Status: PR2 shipped + GPU-CONFIRMED (task #68, 2026-08-27); PR3 multistep BUILT + GPU-CONFIRMED
(branch `add-per-row-multistep-steps` @6a5e786, durable + CPU tests, Finding 1 FIXED, user local
GPU test 2026-09-02 dpmpp_2m + res_multistep both good); PR1 refactor pending; PR4 (SDE family)
BUILT + GPU-CONFIRMED (branch `add-per-row-dpmpp-sde-steps`, PR #36 open; user local quality check
2026-09-02, all three good: dpmpp_2m_sde/dpmpp_3m_sde/dpmpp_sde). PR5 `dpmpp_2s_ancestral` BUILT +
CPU-tested on the same PR #36 branch (folded in, not a separate PR), GPU still owed in task #73.**
See also: [bugs.md Bug B](bugs.md#bug-b) · [stochastic-recovery-theory.md](stochastic-recovery-theory.md)
· [audio-carry-identity.md](audio-carry-identity.md) · [PER_ROW_IMG2IMG_NOTES.md](../PER_ROW_IMG2IMG_NOTES.md)

## Finding 1 (SOURCE-CONFIRMED, 2026-08-27 → FIXED 2026-09-02 by PR3): multistep degraded to first order

The old remap loop (`sampler.py::build_per_row_sampler_function`) called
`base_fn(model, x, sigmas[i:i+2], ...)` one global interval at a time.
History-based samplers re-initialize `old_denoised = None` at function entry —
`dpmpp_2m` (sampling.py:796-818) and `res_multistep` (1417-1456) both follow this pattern —
so with a 2-sigma slice they took their first-order fallback branch on **every step**.

Under that remap, dpmpp_2m and res_multistep ran as first-order (euler/DDIM-like) **everywhere**,
including free m=1 rows. Not a correctness bug — outputs were valid first-order trajectories, true
for all GPU validation runs of the shipped mechanism. But "supported: dpmpp_2m/res_multistep" then
meant "runs as first-order," an accuracy regression vs the retired three-lever path.

**FIXED (PR3, branch `add-per-row-multistep-steps` @6a5e786):** native per-row step functions
`_dpmpp_2m_step` / `_res_multistep_step` now live in the `_NATIVE_ROW_STEPS` registry. Both carry
per-row `old_denoised` history (res_multistep also `old_sigma_down`/`old_sig_row`) across the outer
loop via `ctx.state`, restoring true 2nd order per row. CPU tests reproduce the stock samplers
BIT-FOR-BIT for all-m=1 (atol=1e-6), and a `test_dpmpp_2m_not_first_order` guard proves the native
path DIFFERS from the first-order fallback. GPU-CONFIRMED: user local quality check 2026-09-02 —
both dpmpp_2m and res_multistep look good.

## Finding 2 (GPU-CONFIRMED 2026-08-27): Bug B killed for euler_ancestral VIDEO; audio hiss OPEN

**CONFIRMED (VIDEO):** PR2 (`add-per-row-rf-ancestral-step`, shipped ede2d8c) GPU-validated in
task #68. Config: 0.2MP, two injects at fractional min_denoise d=0.20 and d=0.15, fade-in at
start, 20 steps; both `euler` (deterministic baseline) and `euler_ancestral` tested.
VIDEO result: no grey static / no corruption on fractional or preserved rows, no ghosting;
euler_ancestral video quality matches euler. The per-row velocity-recovery identity
`v = (x − denoised)/σ_carrier`, `denoised_r = x − σ_row·v` holds on the real H3 model with
NO hidden σ-dependence beyond the label channel for the VIDEO stream. Bug B is killed for video.

**AUDIO (was OPEN → FIXED by Fix A, 2026-08-28):** euler_a gen showed faint audio noise ("slight
microphone feedback"); euler clean. Root cause is now known — ancestral renoise in
`_euler_ancestral_rf_step` ran on the σ_a axis while packed audio lives on σ_v; **Fix A** moves it
to σ_v and makes free-audio euler_a clean (GPU-validated; see [bugs.md](bugs.md#bug-c) Bug C and
[audio-carry-identity.md](audio-carry-identity.md)). Task #76 tracks the fix (pending merge).

*Historical context (why the remap form cracked before PR2):* the per-row r-scaling
`x_cur = x_prev + r·(x_cur − x_prev)` linearly rescales a displacement containing the renoise term,
but `sample_euler_ancestral_RF`'s affine alpha terms and `renoise_coeff` (sampling.py:240-266) do
NOT scale linearly under `σ → σ_row`. Same root cause as [Bug B](bugs.md#bug-b); PR2 closes it for
VIDEO (single-eval RF-ancestral); audio stochastic and multistep/DPM++ not verified. The competing
`/sig_g` divisor fix was GPU-FALSIFIED (fix-audio-carrier-recovery @2483914; free m=1 audio got
LOUDER) → `/carrier` (σ_v) load-bearing, corroborating Fix A; abandon PR #20. Detail:
[audio-carry-identity.md](audio-carry-identity.md) Consequence 3.

## Design (RF-ancestral + multistep now BUILT): per-row step functions replace the black-box base_fn → [native-step-design.md](sampler-class-support/native-step-design.md)

Replace the per-interval `base_fn` call with in-house step functions written elementwise over the
loop's exact per-row sigma tensors (`sig_row`, `sig_row_next`); r-scaling then applies only to a
generic-fallback path. This simplifies [stochastic-recovery-theory.md](stochastic-recovery-theory.md):
the truthful `w` labels already make the model return per-row velocity, recovered as
`v = (x − denoised)/σ_g`, `denoised_r = x − σ_row·v`. Full design — per-row RF-ancestral formulas,
per-row multistep, dispatch/whitelist, and deferred multi-eval samplers — in the child doc
[sampler-class-support/native-step-design.md](sampler-class-support/native-step-design.md).

## Risks and verification notes

- **Leak risk (GPU-CONFIRMED clear for the euler_ancestral VIDEO stream, task #68):** no hidden
  σ-dependence detected beyond the label channel for the single-eval RF-ancestral VIDEO path.
  PR3 (multistep, BUILT) reuses the same recovery identity but adds multistep history — CPU tests
  pass and its GPU quality check is CONFIRMED (user local 2026-09-02, dpmpp_2m + res_multistep both
  good). PR4 (DPM++ SDE family, BUILT) adds a second in-step eval + an SNR-mapping leak surface
  larger than PR2's — the task-#73 GPU gate is now CLEARED: user local quality check 2026-09-02, all
  three (dpmpp_2m_sde/dpmpp_3m_sde/dpmpp_sde) ran clean, no leak on the 39f fade / min_denoise
  0.2–0.3 surrogate; PR #36 open.
- **PR5 (`dpmpp_2s_ancestral`, BUILT + CPU-tested, GPU pending):** native
  `_dpmpp_2s_ancestral_step` is now registered in `_NATIVE_ROW_STEPS` (folded into the PR #36 branch
  `add-per-row-dpmpp-sde-steps`), so it no longer routes through the Bug-B-prone `_fallback_step`.
  Structurally it is PR2's ancestral renoise (shared `_default_row_noise_sampler`, DETERMINISTIC
  inner midpoint) ∩ PR4's `_dpmpp_sde_step` 2-eval midpoint + `publish_labels`. CPU tests pass (665
  suite, 100% diff coverage); GPU still owed — user must add it to the task-#73 all-samplers leak
  spike (the SDE trio is GPU-confirmed, 2s_ancestral is NOT yet GPU-run).
- **Audio (AXIS-BLIND post-#33):** the shipped architecture returns the VIDEO σ_v for EVERY row
  (`sampler.py::row_sigma`/`global_sigma`), and audio's fade rides the official KSamplerX0Inpaint
  noise_mask composite ([audio-native-composite.md](audio-native-composite.md)). There is NO σ_a/σ_v
  projection hazard for PR3: audio integrates on σ_v exactly as stock, and audio m=1 rows reproduce
  stock (row σ == carrier σ). PR4's own axis handling should be re-checked against this axis-blind
  design when it is built. The old euler_a σ_a-axis hiss (Fix A / [bugs.md](bugs.md#bug-c) Bug C) was
  superseded by #33; the custom σ_a audio path (task #76) was dropped, so PR3 is NOT blocked on it.
- **Maintenance:** each native step is a small reimpl that can drift from comfy upstream.
- **Status:** discussion-stage design; nothing built; direction not yet user-confirmed.

## Delivery plan (4 PRs, tasks #66–#73) → [delivery-plan.md](sampler-class-support/delivery-plan.md)

Ordering rationale: stochastic first — it is the user-visible failure and validates the leak risk
before investing in multistep work. The full per-PR specs live in the child doc
[sampler-class-support/delivery-plan.md](sampler-class-support/delivery-plan.md); headlines:

- **PR1 `refactor-per-row-step-dispatch` (#66)** — behavior-identical refactor to a per-row
  step-function protocol + dispatch by sampler name. No GPU gate.
- **PR2 `add-per-row-rf-ancestral-step` (#67) — SHIPPED (ede2d8c) + GPU-VALIDATED (#68).**
  Elementwise `sample_euler_ancestral_RF` over tensor σ_row; whitelisted euler_ancestral.
- **PR3 `add-per-row-multistep-steps` (#69) — BUILT (@6a5e786) + GPU-CONFIRMED (user local
  2026-09-02, dpmpp_2m + res_multistep both good), CPU tests pass.** Per-row dpmpp_2m +
  res_multistep, deterministic (eta=0), per-row `old_denoised`
  history via `ctx.state` (fixes Finding 1). Shares the `_recover_row_denoised(ctx, carrier)` spine
  (one model eval at the global carrier σ; `denoised_r = x_prev − sig_row·v`). Axis-blind post-#33:
  NO σ_a projection hazard, audio m=1 reproduces stock, NOT blocked on task #76. Full detail in the
  child doc's PR3 note.
- **PR4 `add-per-row-dpmpp-sde-steps` (#72) — BUILT + GPU-CONFIRMED (user local 2026-09-02, all
  three good), PR #36 open.** Per-row `dpmpp_sde` + `dpmpp_2m_sde` + `dpmpp_3m_sde` on one
  branch, sequenced after PR3 (2M SDE = PR2 renoise ∩ PR3 history). RF IMPLICIT (α=1−σ inside the
  SNR helpers), `dpmpp_sde` 2-eval + `publish_labels` refresh, axis-blind (σ_v every row, σ_a
  machinery dropped). Task-#73 GPU gate CLEARED: all three ran clean on H3 (dpmpp_2m_sde good,
  dpmpp_3m_sde good, dpmpp_sde good). Full SDE spine spec + source-line map in the child doc.
- **PR5 `dpmpp_2s_ancestral` — BUILT + CPU-tested, GPU pending (folded into the PR #36 branch).**
  Native `_dpmpp_2s_ancestral_step` in `_NATIVE_ROW_STEPS` = PR2 ancestral renoise (shared
  `_default_row_noise_sampler`, deterministic inner midpoint) ∩ PR4 2-eval midpoint; GPU owed in
  task #73. Detail in the child doc.
