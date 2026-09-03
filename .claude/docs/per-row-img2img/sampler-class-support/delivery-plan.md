<!-- provenance: status + confirmed (PR2 SHIPPED + GPU-CONFIRMED task #68; PR3 BUILT @6a5e786 + GPU-CONFIRMED user local 2026-09-02 dpmpp_2m + res_multistep; PR1 refactor pending; PR4 SDE BUILT + GPU-CONFIRMED user local 2026-09-02 all three, PR #36 open; PR5 dpmpp_2s_ancestral BUILT + CPU-tested, GPU pending) -->
<!-- verified: 2026-09-02 · PR4 SDE BUILT + GPU-CONFIRMED (add-per-row-dpmpp-sde-steps, PR #36 open; user local all three good: 2m_sde/3m_sde/sde); PR3 multistep BUILT + GPU-CONFIRMED (@6a5e786, user local both good, CPU m=1 tests); PR2 GPU pass task #68 @ede2d8c; audio AXIS-BLIND post-#33; PR5 dpmpp_2s_ancestral BUILT + CPU-tested (same PR #36 branch), GPU pending in task #73 -->
# Delivery plan (4 PRs, tasks #66–#73)

Child of [sampler-class-support.md](../sampler-class-support.md) — detailed per-PR specs
(incl. PR3 axis-blind note and PR4 SDE spine spec).

Ordering rationale: stochastic first — it is the user-visible failure and validates the leak risk
before investing in multistep work.

**PR1 `refactor-per-row-step-dispatch` (task #66)** — behavior-identical refactor only.
Restructures `build_per_row_sampler_function` to a per-row step-function protocol + dispatch by
sampler name. Native per-row euler step is numerically equivalent to the current wrap+r-scale for
euler; all other samplers keep the wrap+r-scale fallback unchanged. CPU equivalence tests:
m=0 / fractional / m=1, audio-shifted rows. No GPU gate — zero behavior change.

**PR2 `add-per-row-rf-ancestral-step` (task #67) — SHIPPED (ede2d8c) + GPU-VALIDATED (task #68).**
Elementwise `sample_euler_ancestral_RF` formulas over tensor σ_row; σ clamped ≥ε so m=0 rows
freeze; seeded `default_noise_sampler`; `s_noise·noise_scale` replicated. Whitelisted
euler_ancestral stops the stochastic warning; non-whitelisted stochastic still warns.
CPU tests vs scalar reference; analytical audio-carry-identity check for shifted-σ audio rows.
GPU spike (task #68) passed: 0.2MP, d=0.20/0.15 injects, fade-in, 20 steps, both euler +
euler_ancestral — no corruption, no ghosting; leak-risk test clear.

**PR3 `add-per-row-multistep-steps` (task #69) — BUILT (@6a5e786) + GPU-CONFIRMED (user local
quality check 2026-09-02: dpmpp_2m + res_multistep both good), durable feature + tests.**
Per-row dpmpp_2m + res_multistep, deterministic (eta=0), in the
`_NATIVE_ROW_STEPS` registry (`_dpmpp_2m_step` under `sample_dpmpp_2m`, `_res_multistep_step` under
`sample_res_multistep`). Per-row `old_denoised` history (res_multistep also `old_sigma_down`/
`old_sig_row`) carried across the outer loop via `ctx.state` — fixes Finding 1's first-order
degradation, restoring true 2nd order per row. m=0 rows freeze (`where(never, clean, ·)`); terminal
rows take the first-order branch; fractional VIDEO rows still route through `_single_forward_denoised`
(clean-K/V observer splice, Bug F). Shared spine `_recover_row_denoised(ctx, carrier)`: ONE model
eval at the global carrier σ, recover per-row x0 via `v = (x_prev − denoised)/carrier`,
`denoised_r = x_prev − sig_row·v` (same recovery as `_euler_ancestral_rf_step`); reused by PR4.
Tests (`tests/test_sampler.py::TestMultistepStepEquivalence`, local scalar refs
`_local_sample_dpmpp_2m`/`_local_sample_res_multistep`): all-m=1 reproduces stock BIT-FOR-BIT
(atol=1e-6); `test_dpmpp_2m_not_first_order` proves the native path DIFFERS from the first-order
fallback; m=0 exact-preserve; fractional-finite; audio-finite; callback-per-step. Full suite 655
passed, 100% diff coverage. GPU-CONFIRMED: user local quality check 2026-09-02 — both dpmpp_2m and
res_multistep look good.

**Axis-blind (post-#33, commit 60fa207) — the old σ_v-axis coherence ⚠ warning is MOOT/SUPERSEDED.**
The shipped architecture is AXIS-BLIND: `sampler.py::row_sigma`/`global_sigma` return the VIDEO σ_v
for EVERY row, and audio's fade rides the official KSamplerX0Inpaint noise_mask composite
([audio-native-composite.md](../audio-native-composite.md)). Consequences for PR3: (1) there is NO
σ_a/σ_v projection hazard — the spine recovers on σ_v and audio integrates on σ_v exactly as stock;
(2) audio m=1 rows DO reproduce stock (row σ == carrier σ); (3) PR3 is NOT blocked on task #76 — the
custom σ_a audio path was DROPPED in #33. Source note: plain `sample_dpmpp_2m` (sampling.py:796-818)
and `res_multistep` (1417-1456) use the plain log-σ form `t_fn = sigma.log().neg()`, NOT the
logit/half-log-SNR form — that complication is confined to the SDE family (PR4).

**PR4 `add-per-row-dpmpp-sde-steps` (task #72) — BUILT + GPU-CONFIRMED (user local quality check
2026-09-02, all three good), PR #36 open.** Per-row `dpmpp_sde` + `dpmpp_2m_sde` + `dpmpp_3m_sde`,
all on one branch (user-confirmed). GPU-CONFIRMED on H3: `dpmpp_2m_sde` good (252.89s cold),
`dpmpp_3m_sde` good (193.89s warm), `dpmpp_sde` good (326.19s warm) — clears the task-#73 GPU merge
gate (39f fade / min_denoise 0.2–0.3 leak surrogate, all three samplers ran clean). Sequenced AFTER PR3: 2M SDE is literally PR2's stochastic renoise ∩ PR3's
history carry. Source map (sampling.py @b78cec87): `sample_dpmpp_sde` (738-792) = TWO evals/step,
NO history; `sample_dpmpp_2m_sde` (822-873) = ONE eval/step, `old_denoised` + `h_last`; both use
BrownianTreeNoiseSampler + `s_noise·noise_scale`.

**RF is IMPLICIT (SOURCE-CONFIRMED b78cec87):** H3 is CONST (rectified flow), so
`sigma_to_half_log_snr(σ) = logit(σ).neg() = log((1−σ)/σ)` (152), and in all three SDE samplers
`alpha_t = sigmas[i+1]·λ_t.exp() = (1−σ)` — the `α = 1−σ` identity lives INSIDE the SNR helpers,
no `_RF` variants (confirms the plan's "RF IMPLICIT" note). Per-row ports run the logit form
elementwise over `sig_row`, clamped to `[ε, 1−ε]` at BOTH ends (comfy's `offset_first_sigma_for_snr`
(168) clamps only the global scalar first-σ; per-row needs both since logit(0)/logit(1) = ±inf).

**Port shapes.**
- `_dpmpp_2m_sde_step`: 1 eval/step, 1-deep history `(denoised_r, h)` in `ctx.state`; midpoint
  solver default; renoise `+ noise·sip1·(−2·h·eta).expm1().neg().sqrt()·s_noise`. Direct analog of
  PR3's `_dpmpp_2m_step` with logit-form λ + SDE renoise.
- `_dpmpp_3m_sde_step`: 1 eval/step, 2-deep history `(denoised_1, denoised_2, h_1, h_2)`; 3M
  combiner elementwise, 2M branch as one-history fallback, Euler as no-history fallback.
- `_dpmpp_sde_step`: 2 evals/step, NO history, needs a mid-eval label refresh. Eval1 → `denoised_r`
  (the recover spine). Per-row midpoint `λ_s_1 = λ_s + r·h` (r=½), `σ_row_mid = sigmoid(−λ_s_1)`;
  step-1 ancestral `get_ancestral_step` reimplemented with `torch.minimum` (comfy uses python
  `min()`, 73). Eval2 at a GLOBAL midpoint carrier `σ_mid_g = sigmoid(−(λ_s_g + r·h_g))` after
  publishing `w_mid = σ_row_mid/σ_mid_g`; recover `denoised_2_r` via the same velocity identity.

**Plumbing:** a `publish_labels(w)` closure (wraps the loop's existing
`make_pooled`→`schedule_tail["pooled_current"]`, sampler.py:558), stashed in `step_state`, lets
`_dpmpp_sde_step` refresh pooled labels between its two evals.

**Noise sampler:** `BrownianTreeNoiseSampler` is scalar-interval-only (`BatchedBrownianTree.sort`
uses `a < b`, 114-116) → query ONCE at the GLOBAL carrier interval; output is unit-variance (149),
so ALL per-row scaling lives in the elementwise coefficients (noise correlation follows the global
schedule — accepted approximation).

**Axis-blind (do NOT resurrect σ_a machinery):** σ_v for every row + official noise_mask composite
([audio-native-composite.md](../audio-native-composite.md)). The pre-#33 draft's σ_a `w_mid` audio
plumbing is DROPPED; PR4 runs axis-blind (`w_mid` on σ_v). Task #76 is closed; PR4 is NOT blocked on it.

**Fractional-row cost:** `2m_sde`/`3m_sde` route fractional VIDEO rows through
`_single_forward_denoised` for free via `_recover_row_denoised` (same as PR3). `dpmpp_sde`'s SECOND
eval on fractional rows needs its own side-stream priming at the midpoint σ — the one genuinely NEW
fractional-row path and the main GPU risk.

**Merge gate CLEARED (USER GPU spike, task #73):** leak surface LARGER than PR2's (these lean on the
SNR mapping, not the clean RF alpha identity) → user reran the label→timestep leak test (39f fade,
min_denoise 0.2–0.3, ALL THREE samplers). User local quality check 2026-09-02: all three good, no
leak — gate cleared. PR #36 open; mark completed only after it merges to `main`.

**PR5 `dpmpp_2s_ancestral` — BUILT (folded into the PR #36 branch `add-per-row-dpmpp-sde-steps`;
user asked to keep it here, NOT a separate PR) + CPU-tested, GPU PENDING.** New
`_dpmpp_2s_ancestral_step` registered under `sample_dpmpp_2s_ancestral` in `_NATIVE_ROW_STEPS`.
Ports comfy `sample_dpmpp_2s_ancestral_RF` (sampling.py:686-734): PR2's exact RF-ancestral renoise
algebra (`downstep_ratio`→`sigma_down`, `alpha_ip1`/`alpha_down`, `renoise_coeff`,
`default_noise_sampler` — NOT the SDE BrownianTree) wrapped around a 2-eval DPM++(2S) midpoint
refine shaped like `_dpmpp_sde_step` (per-row + global midpoint via the half-log-SNR identity,
`publish_labels(w_mid)` refresh between the two evals). Difference vs `dpmpp_sde`: the inner
midpoint point `u` is DETERMINISTIC (no ancestral noise on the inner solve); noise is added only in
the final per-interval renoise. Shared `_default_row_noise_sampler(ctx, extra_args)` helper
EXTRACTED, now used by BOTH `_euler_ancestral_rf_step` and `_dpmpp_2s_ancestral_step` (the
euler_ancestral change is a behavior-identical refactor). 2 evals/step, no history; terminal rows
(`sig_row_next==0`) → denoised_r (matches stock's Euler terminal); m=0 rows freeze; same
fractional-row caveat as `dpmpp_sde` (eval-2 side stream primed at step σ, not midpoint — the
task-#73 GPU risk). Tests (`tests/test_sampler.py::TestSDEStepEquivalence`):
`_local_sample_dpmpp_2s_ancestral` CONST/RF logit reference (`__name__`-routed), m1-equals-stock,
folded into the eta=0/m0-preserve/fractional/registered-stochastic loops + a callback-once-per-step
test. Full suite 665 passed, sampler.py diff coverage 100%. GPU PENDING: user must add
`dpmpp_2s_ancestral` to the task-#73 GPU spike (all-samplers leak test); the SDE trio
(2m_sde/3m_sde/sde) is GPU-confirmed, 2s_ancestral is NOT yet GPU-run.
