<!-- provenance: theory + confirmed (Finding 2 GPU-CONFIRMED for euler_ancestral VIDEO 2026-08-27;
     Finding 1 SOURCE-CONFIRMED; audio euler_a hiss = σ_a-axis renoise bug, FIXED by Fix A
     GPU-validated 2026-08-28; /sig_g divisor fix GPU-FALSIFIED 2026-08-27; PR3/PR4 design UNVERIFIED) -->
<!-- verified: 2026-08-28 · PR2 GPU pass task #68 @ede2d8c; Fix A validated free audio (audio-axis-verdict.md); /sig_g fix FALSIFIED by A/B GPU test branch fix-audio-carrier-recovery @2483914 -->
<!-- source: comfy-ref k_diffusion/sampling.py @b78cec87: 240-266 euler_ancestral_RF, 738-792 dpmpp_sde,
     796-818 dpmpp_2m, 822-873 dpmpp_2m_sde, 68-176 helpers, 1394+ res_multistep; repo sampler.py, observer_split.py -->
# Sampler-class support under the schedule-tail remap (stochastic + multi-step)

**Status: PR2 shipped + GPU-CONFIRMED (task #68, 2026-08-27); PR1 refactor pending; PR3/PR4 design stage.**
See also: [bugs.md Bug B](bugs.md#bug-b) · [stochastic-recovery-theory.md](stochastic-recovery-theory.md)
· [audio-carry-identity.md](audio-carry-identity.md) · [PER_ROW_IMG2IMG_NOTES.md](../PER_ROW_IMG2IMG_NOTES.md)

## Finding 1 (SOURCE-CONFIRMED, 2026-08-27): multistep samplers silently degrade to first order

The shipped remap loop (`sampler.py::build_per_row_sampler_function`) calls
`base_fn(model, x, sigmas[i:i+2], ...)` one global interval at a time.
History-based samplers re-initialize `old_denoised = None` at function entry —
`dpmpp_2m` (sampling.py:796-818) and `res_multistep` (1394+) both follow this pattern —
so with a 2-sigma slice they take their first-order fallback branch on **every step**.

Under the remap, dpmpp_2m and res_multistep therefore run as first-order (euler/DDIM-like)
**everywhere**, including free m=1 rows. Not a correctness bug — outputs are valid first-order
trajectories, and this was true for all GPU validation runs of the shipped mechanism. But
"supported: dpmpp_2m/res_multistep" currently means "runs as first-order," an accuracy
regression vs the retired three-lever path where `base_fn` ran the full schedule once.

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

## Design (UNVERIFIED): per-row step functions replace the black-box base_fn → [native-step-design.md](sampler-class-support/native-step-design.md)

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
  The AUDIO stochastic path has an OBSERVED residual-noise defect (task #76) — NOT clear.
  PR3 (multistep) and PR4 (DPM++ SDE) reuse the same recovery identity but add multistep
  history / a second in-step eval — their leak surface is larger and they remain UNVERIFIED.
- **Audio:** free-audio euler_a hiss = σ_a-axis renoise bug, FIXED on σ_v by Fix A (see Finding 2 /
  [bugs.md](bugs.md#bug-c) Bug C). **PR3/PR4 inherit the same axis dependency — see PR3 note below.**
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
- **PR3 `add-per-row-multistep-steps` (#69)** — per-row dpmpp_2m + res_multistep, deterministic
  (eta=0), per-row `old_denoised` history (fixes Finding 1). Introduces the shared DPM++ spine
  helpers reused by PR4. ⚠ **σ_v-axis coherence dependency on Fix A (decision 2026-08-29):** the
  spine's `_recover_row_denoised` must project audio onto `sig_row_v`, not the σ_a-shifted
  `sig_row`; the "m=1 reproduces stock bit-for-bit" guarantee holds for VIDEO rows only. Full
  detail (3 consequences, spine seeding PR4) in the child doc's PR3 note.
- **PR4 `add-per-row-dpmpp-sde-steps` (#72)** — per-row `dpmpp_sde` + `dpmpp_2m_sde` +
  `dpmpp_3m_sde`, sequenced after PR3 (2M SDE = PR2 renoise ∩ PR3 history). SDE spine spec,
  source-line map, label-refresh plumbing, and the task-#76 block are in the child doc.
