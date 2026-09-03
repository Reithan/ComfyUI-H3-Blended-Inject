<!-- provenance: status + confirmed (PR2 SHIPPED + GPU-CONFIRMED task #68 2026-08-27; PR3 BUILT @6a5e786 + GPU-CONFIRMED user local 2026-09-02 dpmpp_2m + res_multistep both good; PR1 refactor pending; PR4 SDE design UNVERIFIED) -->
<!-- verified: 2026-09-02 · PR3 multistep BUILT + GPU-CONFIRMED (add-per-row-multistep-steps @6a5e786, user local both good, CPU bit-for-bit m=1 tests, Finding 1 FIXED); PR2 GPU pass task #68 @ede2d8c; audio AXIS-BLIND post-#33 -->
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

**PR4 `add-per-row-dpmpp-sde-steps` (task #72)** — per-row `dpmpp_sde` + `dpmpp_2m_sde` + `dpmpp_3m_sde`. Sequenced
AFTER PR3: 2M SDE is literally PR2's stochastic renoise ∩ PR3's history carry. Source facts
(sampling.py @b78cec87): `sample_dpmpp_sde` (738-792) = TWO model evals/step (denoised @σ_i, then
denoised_2 @midpoint σ_s_1, r=1/2 default), NO history, noise injected after EACH sub-step (781,
791); `sample_dpmpp_2m_sde` (822-873) = ONE eval/step, carries `old_denoised` AND `h_last`, SDE
noise `σ_{i+1}·(−2hη).expm1().neg().sqrt()·s_noise` (869), solver_type midpoint (default) / heun
(877). Both: BrownianTreeNoiseSampler, `s_noise·noise_scale`, `offset_first_sigma_for_snr`.
Corrections vs the draft plan: no `_RF` variants, but RF is IMPLICIT — `sigma_to_half_log_snr`
(152) branches on CONST → λ = −logit(σ), so α = 1−σ lives INSIDE the helpers; per-row steps run
the logit form elementwise with σ_row clamped to [ε, 1−ε] (the first-σ offset at 168 fixes only
the global scalar). `get_ancestral_step` operates in exp(−λ) = σ/α space (776, 785), python
`min()` (73) → per-row reimpl needs `torch.minimum`. Label refresh is pooled-only: the observer self-refreshes
per model call from the CALL's σ (`t_obs = 1 − m·σ`, observer_split.py:38-59), but the pooled `w`
stash publishes once per outer step (sampler.py:558) → `_StepContext` gains a `publish_labels(w)`
closure so the step sets `w_mid = σ_row_mid/σ_s_1` before eval 2 (elementwise midpoint:
`λ_row_s1 = λ_row + r·h_row`, `σ_row_mid = sigma_fn(λ_row_s1)`). BrownianTreeNoiseSampler is
scalar-interval-only (`sort` uses `a < b`, 116) → query at global carrier σ; output is
unit-variance (149), so all per-row scaling stays in the elementwise coefficients (noise
correlation structure follows the global schedule — accepted approximation).
**Merge gated on NEW USER GPU spike (task #73):** leak surface LARGER than PR2's — these lean on
the SNR mapping rather than the clean RF alpha identity, so rerun the label→timestep leak test
(39f fade, min_denoise 0.2–0.3) with both samplers before merge.
Spine composition (PR3 helpers): 2M SDE = recover → time_coeffs → 2nd-order combiner + stochastic
renoise (1 eval/step, 2M history). 3M SDE = recover → time_coeffs → 3-term 2-deep combiner +
stochastic renoise (1 eval/step). `dpmpp_sde` = recover → time_coeffs → mid-eval label refresh +
BrownianTree (2 evals/step, no history). **Axis re-check when built:** the σ_a/σ_v audio plumbing
above (`w_mid` on σ_a, label refresh) predates #33's axis-blind design — the custom σ_a audio path
was dropped, so PR4's own axis handling must be re-verified against the axis-blind base (σ_v for
every row + official noise_mask composite) before implementing, not the old σ_a machinery. Task #76
is closed/dropped; PR4 is no longer blocked on it.
