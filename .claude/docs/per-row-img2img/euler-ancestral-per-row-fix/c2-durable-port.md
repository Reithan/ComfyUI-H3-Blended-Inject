<!-- provenance: theory + decision (user 2026-09-01) + SHIPPED (commits e3e167f+parent, GPU A/B CONFIRMED; round-8 anchor fix fe0343a+91078cc, GPU pending) -->
<!-- verified: 2026-09-01 — Fable round-6 re-affirmation; shipped e3e167f; 98 CPU tests pass; GPU CONFIRMED partial (residual localized low-m band, round 8; anchor fix fe0343a+91078cc pending) -->
# C2 carry-compression durable port — PR #32 decision

Decision record and implementation spec for porting the C2 x0/trajectory correction into the
durable `_euler_ancestral_rf_step`, folded into PR #32.

Context: [../c2-rho-fix-paths/index.md](../c2-rho-fix-paths/index.md) (prototype ladder).
Read with: [content-axis.md](content-axis.md) (GPU PARTIAL result that prompted this decision).

## Why the C2 fix stayed out of durable until now

The C2 ρ-correction was validated on the prototype ladder (v3 @f06a84a → v4 @12ea3b6 → v6
@02fee22; loud→faint→very-quiet on the same fade-ramp euler_a buzz). That ladder lived on a
proto lineage and was dropped when the durable base (PR #28, `clean-kv-observer-splice`) was
cut. `_euler_ancestral_rf_step` on branch `fix-euler-ancestral-per-row-renoise` is that ladder's
BASELINE — no correction applied.

## GPU PARTIAL result that revealed the residual (content-axis A/B, 2026-09-01)

Content-axis fix retained (it is not the residual cause, but it helps):

- Fade-audio hiss: SHORTER and SLIGHTLY QUIETER; ring NARROWED toward mid-m rows; both ends clean.
- Deterministic euler fade: UNCHANGED / CLEAN → no regression from the shared `prime_side_stream`.
- Residual ring: mid-m rows, stochastic-path-specific (ancestral only). NOT a v̂ floor.

## Fable round-5 residual verdict

The residual is the known C2 carry-compression error in the ancestral x0 recovery
(`denoised_r = x_prev − sig_row_v·v`, sampler.py ~539-540). The GPU-confirmed prototype ladder
corrected this; it was never ported to durable.

## Mechanism (source-derived, comfy-ref minimax/model.py:530-551)

Forward multiplies packed audio by global carry `k = σ_a/σ_v` (:535-538) and un-transforms
with global `σ_a` (:549-550). With S = shift_v/shift_a (= `model_sampling.audio_scale`);
identity S·k ≡ 1+(S−1)σ_a exact:

```
packed velocity  v = (1−S)·k·y + S·k·u
```

Exact packed clean estimate for ANY network output:

```
S·Â = y − σ_c·v − (S−1)·(σ_row_a − σ_a)·y
```

where `σ_c = σ_row_a·carrier/σ_a` (row's true packed noise level; σ_c < sig_row_v).

Current recovery carries two exact errors:

1. State-proportional: `−(S−1)·(σ_a − σ_row_a)·y` (≈ −0.37·y mid-fade at S=4; includes
   the current noise realization).
2. Velocity-proportional: `−(sig_row_v − σ_c)·v`.

Truthful packed clean coefficient for a fractional audio row: `a = (1−σ_row_a) / (1+(S−1)σ_a)`.
NOT `1−sig_row_v` and NOT `1−σ_a`. Both errors are exactly 0 at m=1 (stock bit-exact) and →0
at m→0 → mid-m peak, both ends clean.

**Why stochastic-only:** `_euler_step` never forms `denoised_r` (carrier-axis ODE step + r-lerp;
ODE self-corrects). The ancestral step blends `(1−ratio)·denoised_r` (carrying a fraction of
retained ε) then adds INDEPENDENT fresh noise sized to sig_row_v' > σ_c' → row leaves every step
over-noised ≈ k/k_row (~1.3 mid-fade) → persistent re-excitation.

`noise_scale`, `noise_sampler(carrier,…)`, and the `w` label (σ_a) are axis-clean. Observer
adaLN path (`_observer_timestep` = `clamp(1−m·σ_a, pin)`, native-identical model.py:604-605)
RULED OUT: deterministic, euler-clean.

Note: the content-axis ratio was in effect a partial compensation for this. Once main-stream
packed noise is truthfully σ_c, the exact band ratio may revert to `m·σ_a/σ_row_a` — revisit.

## Residual error ranking

1. C2 x0/trajectory error — largest, exact, ancestral-only, GPU precedent. **(This fix.)**
2. ρ-drift across steps (ladder "v7") — included free in the exact form.
3. m=0 audio-context heat (v6, deterministic, both samplers) — follow-up.
4. Observer clean-anchor heat — follow-up.
5. v̂ floor — only after 1–3 cleared.

Audio-band splice opt-out (skip "audio" stream in install/prime) remains a valid one-line
discriminator of the observer path; run only if the C2 port fails. If that is unchanged too
→ declare floor.

## DECISION (user, 2026-09-01) — REVERSAL of proto-only policy

Port the C2 correction into the durable ancestral step in exact generalized form, folded into
PR #32. This **reverses** the earlier "C2/ancestral experiments stay proto-only" decision that
was made when the durable base was cut.

## Implementation spec (fractional-audio rows only)

Gate: `frac_audio = audio_mask & (0 < m < 1)`. m=1 bit-exact + video byte-identical by gate.

Per row with s=sig_row, s'=sig_row_next, σ_a=sig_g, σ_a'=sig_g_next:

```
σ_c  = s · carrier / σ_a
σ_c' = where(σ_a' > 0, s' · sigmas[i+1] / σ_a', 0)
a    = (1 − s)  / (1 + (S−1)·σ_a)
a'   = (1 − s') / (1 + (S−1)·σ_a')

C_hat = y − σ_c·v − (S−1)·(s − σ_a)·y
ε_hat = (y − a·C_hat) / σ_c
sd    = σ_c' · (1 + (σ_c'/σ_c − 1)·η)
r_ret = sd · (1 − σ_c') / (1 − sd)
x     = a'·C_hat + r_ret·ε_hat + noise·s_noise·sqrt(max(σ_c'² − r_ret², 0))
```

Reduces bit-for-bit to stock RF-ancestral when σ_c=σ_v, a=1−σ. Stochastic-preserving
(same η semantics on σ_c axis). eta-gating remains REJECTED.

Fallback: cherry-pick v4 @12ea3b6 verbatim.

## GPU A/B plan

Same seed/fade/euler_a as the content-axis A/B (e4a9940 vs e3e167f). PASS = mid-fade ring drops.

- **Ring drops markedly** → C2 correction CONFIRMED; done designing.
- **Unchanged** → splice opt-out (skip "audio" stream in install/prime; one-line discriminator).
- **Splice opt-out unchanged too** → declare v̂ floor.

## SHIPPED (2026-09-01) — implementation record

Commits: e3e167f (tests) + its parent (sampler). Implementation:

- `_c2_audio_ancestral_update` — pure helper containing the correction math above.
- `_euler_ancestral_rf_step` override: gate `frac_audio = audio_mask & (0 < sig_row < sig_g)`.
- `_StepContext` gains `audio_mask` (bool row-tensor) and `audio_scale` (S = shift_v/shift_a).

**Fable round-6 re-affirmation:** `a·Ĉ` is term-for-term identical to v4's `(1−σ_c)·S·Â/ρ_true`.
Delta vs v4 = the ladder's v7 ρ-drift: truthful `a'` vs v4's frozen `a·(1−σ_c')/(1−σ_c)`. No v5
content included. Does not re-run input-side pre-comp (63b291e). m=1 bit-exactness comes from
the `where`-gate, not the formulas (formulas yield the same result at m=1 by construction).

**CPU tests:** `TestC2AudioAncestralUpdate` — 5 tests including fail-then-pass vs v4's frozen
coefficient; 98 sampler tests total pass. ruff clean.

## GPU result (2026-09-01)

C2 durable port CONFIRMED: mid-fade audio noise now much shorter and quieter (ladder precedent
held, loud→faint). Residual changed character to a brief crackle.

**Residual localization revised (round 8):** the 1 s fade was too coarse to localize. With a
91-f fade (same seed/euler_a), the residual spans ≈0.75–1.0 s → m ≈ 0.11–0.18 BAND at LOW m,
NOT a mid-fade point. All C2-path singularities at m≈0.1–0.2 checked benign.
Top candidate: audio band anchor `h_clean` S× too hot; fix implemented fe0343a+91078cc.
Full analysis: [audio-anchor-scale.md](audio-anchor-scale.md).

**Round-9 addendum (2026-09-02):** anchor fix FALSIFIED for the peak — introduces muffling.
Root cause: `PLANT_AXIS = "v"` makes i=0 plant use σ_v axis while C2 books on σ_c,
over-noising every fractional audio row by F₀ = σ_v(m)/σ_a(m).
Fix = revert PLANT_AXIS to "row". Full analysis: [plant-over-noise.md](plant-over-noise.md).
