<!-- provenance: confirmed (CPU) / UNVERIFIED (GPU) — δ-residual CONFIRMED Branch 1 (ret_clean_corr); noise-carry fix IMPLEMENTED commit a28a62b; GPU cross-check pre-registered -->
<!-- verified: 2026-09-02 (branch fix-euler-ancestral-per-row-renoise) · CPU: 117 tests pass, noise-carry init |corr|<0.05 at low m; GPU pre-registered (ret_clean_corr low-m must drop from +0.94→~0) -->
# δ re-injection — round-10 record + noise-carry fix

Parent: [euler-ancestral-per-row-fix.md](../euler-ancestral-per-row-fix.md).
Competing primary: [../c2-rho-fix-paths/residual-accounting.md](../c2-rho-fix-paths/residual-accounting.md).

## Mechanism (CONFIRMED Branch 1 — δ-as-RESIDUAL)

In `_c2_audio_ancestral_update`, `ε̂ = (y − a·ĉ)/σ_c` recovered by inverting through network `ĉ`.
Denoiser shrinks: `ĉ = γ·C_true` (γ<1, set by GLOBAL noise level, not local σ_c).
So `a·ĉ` under-removes content; `ε̂` retains `[a(1−γ)/σ_c]·C_true` leak, amplified 1/σ_c,
re-injected coherently every step → low-m preserved-keyframe static.

**Branch 1 CONFIRMED** (fixed-logger CSV h3bi_c2_debug-normal-2, 20 steps, PLANT_AXIS="row"):
`ret_clean_corr` climbs monotonically as m→0 at every sampling step.
Step-0 exemplar: k_d=1 (m=0.95) −0.07; k_d=12 (m=0.38) +0.47; k_d=16 (m=0.19) +0.94; k_d=19 (m=0.05) +0.95.
k_d=19 holds ~+0.94 across steps 0→18 (terminal step 19: retained=0 by schedule, corr=0 trivially).

**Sign correction vs pre-registration:** predicted negative (anti-correlation); actual POSITIVE.
`a·ĉ` too small to cancel y's content at low m → `ε̂ ≈ y/σ_c` carries content positively;
1/σ_c amplifies the leftover at small σ_c. retained_rms ≈ 0.03–0.15 vs clean_rms ≈ 1–8 at k_d≥16
(a few % per step but STRUCTURED/coherent → 19-step accumulation → audible static).

**δ-as-C2-GENERATOR FALSIFIED** (Test B, `H3BI_DISABLE_C2=1`): static got LOUDER/LONGER (persisted
past 2 s, obscured all dialogue) → C2 is net-corrective; δ lives inside C2 as residual, not generator.
**Branch 2** (white noise, corr≈0 at low m) FALSIFIED.

### Historical notes (GPU T1 run + logger bug)

T1 PLANT_AXIS="row" revert: quiet 0–0.25 s, LOUD static 0.25–1.5 s, original voice ~1.5–2.0 s.
Logger bug (fixed): k_d was read from wrong key (`ctx.state.get("total_steps")` → 0);
fixed to `ctx.state["schedule_tail"]["total_steps"]` → bins by m correctly.
δ-as-GENERATOR falsification and Branch-1 exploration complete; mechanism is now settled.

## THE FIX — noise-carry (commit a28a62b)

`sampler.py`: `_c2_audio_ancestral_update` + its caller `_euler_ancestral_rf_step`.
State key: `ctx.state["c2_eps_carry"]`.

Carry TRUE unit noise `εc` as sampler state instead of re-inverting shrunk `ĉ` each step.

### Init (i=0)

```
εc_init = (y − (1−w)·clean_raw) / σ_c
```

- `w = (sig_row/sig_g).clamp(max=1)` — plant's OWN coefficient (PLANT_AXIS="row"; equals σ_a(m) at i=0).
- `clean_raw = state["clean"]` — RAW un-shrunk injected composite C_true.

The plant books `y = w·x_noise + (1−w)·clean_raw`; subtracting plant's `(1−w)·clean_raw`
leaves `w·x_noise`; dividing by σ_c gives `εc_init = (w/σ_c)·x_noise` — pure noise, zero
content by construction. Under RF σ_max=1: `w = σ_c ⇒ εc_init = x_noise` (unit noise).

**MANDATORY: use plant's `(1−w)`, NOT C2's `a`.**
For audio: `a=(1−s)/S ≠ (1−w)=1−s` at i=0 → a-based init leaves `((1−w)−a)·clean` content residual.
Also NOT `(y−clean)/σ_c` = `x_noise − clean` (carries a −clean seed into the carry).

### Carry recurrence (i>0)

```
εc' = (r_ret·εc + c_fresh·noise) / σ_c'
```

Unit-variance by construction: `var = r_ret² + c_fresh² = σ_c'²` → dividing by σ_c' gives unit std.
Returned via `carry_out["eps_next"]`; caller persists to state, gated to frac_audio rows via `torch.where`.
Stochastic noise (`c_fresh·noise`) fully preserved — does NOT disable stochastic sampling.
`εc` never touches shrunk `ĉ` again after i=0; the clean channel `a_next·ĉ` is unchanged.

### Regression tests (tests/test_sampler.py::TestC2AudioAncestralUpdate)

1. `test_noise_carry_init_is_content_orthogonal_vs_leaky_inversion` — builds y via actual plant
   lerp, injects γ=0.4 shrinkage; asserts old-inversion corr(retained,clean)>0.5 vs fixed carry
   |corr|<0.05 at low m. Fail-then-pass verified.
2. `test_noise_carry_advance_stays_unit_and_content_free` — next-carry ~unit std and content-orthogonal.

Full suite: 117 pass, ruff clean, 99% sampler coverage.

### GPU cross-check (pre-registered)

Rerun with `H3BI_C2_DEBUG` logging → `ret_clean_corr` at low m (high k_d) must drop from ~+0.94 toward ~0.
Ear test: mid-fade audio static/hiss (0.25–1.5 s band) should be gone.
High-m rows unchanged (corr already≈0 → no regression expected there).

## Competing explanation

residual-accounting.md: deterministic per-row injection error in BOTH modalities, node-specific,
present in euler@5 steps; exists in VIDEO (S=1 → C2≡0) and in euler (no C2 branch).
δ cannot explain video/euler cases → either two errors coexist, or mode-independent error is
primary and δ is an audio-only rider on top.

## Unexplained seam (open)

T1 "1.5–2.0 s sounds like the ORIGINAL." Neither δ nor mode-independent error predicts
return-to-source mid-ramp. Either the ear reads low static as "original," or there is a real null
at m≈0.5 (a≈a', σ_c'/σ_c≈1). A spectrogram would settle it.
