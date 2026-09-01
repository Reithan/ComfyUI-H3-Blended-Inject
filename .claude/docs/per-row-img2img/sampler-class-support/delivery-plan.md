<!-- provenance: status + confirmed (PR2 SHIPPED + GPU-CONFIRMED task #68 2026-08-27; PR1 refactor pending; PR3/PR4 design UNVERIFIED) -->
<!-- verified: 2026-08-28 · PR2 GPU pass task #68 @ede2d8c; Fix A validated free audio (../audio-axis-verdict.md); /sig_g fix FALSIFIED by A/B GPU test branch fix-audio-carrier-recovery @2483914 -->
# Delivery plan (4 PRs, tasks #66–#73)

Child of [sampler-class-support.md](../sampler-class-support.md) — detailed per-PR specs
(incl. PR3 σ_v-axis coherence detail and PR4 SDE spine spec).

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

**PR3 `add-per-row-multistep-steps` (task #69)** — per-row dpmpp_2m + res_multistep.
Deterministic (eta=0) form; per-row `old_denoised` history carried across the outer loop — fixes
Finding 1's first-order degradation. Key test: all-m=1 rows over a full schedule must reproduce
the stock sampler bit-for-bit on a toy model. Gated on USER GPU quality check (task #70) at the
d=0.2/0.15 sweet-spot config.
Introduces shared DPM++ spine helpers (reused by PR4 SDE family): `_recover_row_denoised(ctx)`
(velocity-recovery + model-eval block), `_dpmpp_time_coeffs(σ_a, σ_b)` (t = −log σ, h, sigma_ratio),
`_dpmpp_2m_second_order(denoised_r, old_denoised_r, r)` ((1+1/2r)·d − (1/2r)·d_old multistep combiner).

⚠ **σ_v-axis coherence dependency on Fix A (decision 2026-08-29, branch `add-per-row-multistep-steps`):**
PR3 PREDATES Fix A ([audio-axis-verdict.md](../audio-axis-verdict.md)) and is being ported onto its σ_v
axis. Audio-only (video rows: sig_row == sig_row_v). The spine `_recover_row_denoised` recovers
`denoised_r = x_prev − sig_row·v` on the σ_a-SHIFTED `sig_row` while packed audio rides σ_v — the
pre-Fix-A ancestral mis-scale (σ_a/σ_v ≈ 0.27→1.0) — so it MUST project onto `sig_row_v`.
Consequences: (1) the "m=1 reproduces stock bit-for-bit" guarantee holds for VIDEO rows ONLY (audio
m=1: sig_row = σ_a ≠ carrier σ_v ⇒ denoised_r ≠ denoised, diverging from stock as main's ancestral
did on free audio); (2) `_dpmpp_2m_step` / `_res_multistep_step` run the full DPM++/RES integration
(t, h, r, sigma_down, phi1/phi2) on sig_row/next → must use sig_row_v/next for audio, while the σ_a
LABEL `w = sig_row/sig_g` (outer loop) stays on σ_a, correct/untouched; (3) the same spine seeds
PR4's SDE family (#72) → fix at the spine or the bug propagates to every future sampler. eta=0 adds
no renoise, so these escape the LOUD renoise component, but the #1 x0-recovery mis-scale is
renoise-independent and hits ALL audio rows incl. free m=1. Inherits Fix A's GPU-pending status
(GPU-validated for the ancestral case; branch #77 unmerged).

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
BrownianTree (2 evals/step, no history). **All three SDE variants blocked on task #76** (the audio-carry
renoise miscalibration — the euler_a hiss now fixed by Fix A on the σ_v axis, pending merge).
