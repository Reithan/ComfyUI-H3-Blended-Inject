<!-- provenance: confirmed (muffling fix GPU CONFIRMED round 10; static falsified as plant-caused) -->
<!-- verified: 2026-09-02 (branch fix-euler-ancestral-per-row-renoise) · round-10: PLANT_AXIS "row" revert GPU confirmed — muffling fixed, static persists 0.25–1.5 s (not plant-caused) -->
# Plant-over-noise: PLANT_AXIS "v" untruthful under C2 — round 9

Round-9 analysis by Fable (`fable-ancestral-design`) and claude-opus-4-8 review/impl.
All runs: euler_ancestral, same seed, 20 steps, min_denoise=0.
Code = 0a9da30 (C2 port + content-axis ratio + plant-axis "v" + anchor fix fe0343a + splice toggle 91078cc).

Context: [c2-durable-port.md](c2-durable-port.md); [audio-anchor-scale.md](audio-anchor-scale.md) (anchor fix, FALSIFIED here).

## Round-9 GPU runs

| Run | Config | Result |
|---|---|---|
| 0 | 0/0/49/73, ease_in_out, splice ON, anchor fix | small pop during fade |
| 1 | 0/0/0/90, ease_in_out, splice ON, anchor fix | same static peak ~0.75–1.0 s; rest of fade muffled/garbled |
| 2 | 0/0/0/90, ease_in_out, splice OFF (`H3BI_SPLICE_AUDIO=0`) | same peak ~0.75–1.0 s; rest muffled/muted |
| 3 | 0/0/0/90, LINEAR, splice OFF | static peaks at 0.00 s, decays through fade, post-fade normal |

**Conclusions:** anchor fix FALSIFIED for the peak (and introduces muffling); splice is NOT the
peak's cause (ON/OFF identical); the S×-hot anchor was COMPENSATING a main-stream deficit.

## Root cause: i=0 plant uses σ_v axis but C2 books on σ_c

`sampler.py ~670`: `_euler_ancestral_rf_step.PLANT_AXIS = "v"` → i=0 plant uses
`w_plant = sig_row_v/σ_v[0]`. C2 books the row on the packed axis σ_c = s·σ_v/g, which at
i=0 (k₀=1) equals σ_a(m). Every fractional audio row enters C2 over-noised by
F₀ = σ_v(m)/σ_a(m). Video rows have sig_row_v ≡ sig_row → unaffected (audio-only defect).

Selected F₀ values and integrated excess (20-step schedule, shift 12/3, η=1):

| k_d | F₀ | excess at i=0 | peak_m |
|---|---|---|---|
| 19 | 2.84× | 0.251 | low-m plateau |
| 17 | 1.96× | 0.333 | ← peak absolute |
| 16 | 1.75× | 0.321 | |
| 15 | 1.60× | 0.300 | |
| 10 | 1.23× | 0.173 | |
| 5  | 1.08× | 0.043 | |

Integrated excess forms a plateau over m≈0.1–0.25, peaking k_d 16–17 (rows 6–7 = 0.75–1.0 s
under ease_in_out). ~65% of peak at m=0.4; ~10% at m=0.75.

**Why the error persists:** clean-coefficient errors self-correct each step (v5 GPU — "no change").
Noise-level errors enter ε̂ = (y − aĈ)/σ_c and are RETAINED by r_ret·ε̂ every step.
Fresh noise only replaces the variance fraction 1−(r_ret/σ_c′)².

Static = excess noise passing into Ĉ (network cannot attribute F·s noise to signal).
Muffling = network estimating content from a row 1.3–2× noisier than its label →
heavier posterior mean → fine temporal detail stripped (audio's content IS fine temporal detail;
video tolerates it).

## Per-result table: why each prior result follows from F₀

1. **Baseline hiss (pre-C2):** per-step update on σ_v → row re-over-noised EVERY step; peaks mid-m.
2. **Plant-axis "v" (pre-round-9):** self-consistent with old σ_v step → "minor change"; became
   an error only when C2 moved bookkeeping to σ_c (sampler.py comment is stale).
3. **Content-axis fix:** deterministic observer K/V content fix; never touched the row's noise level.
4. **C2 confirmed-partial:** removed per-step over-noising; left i=0 over-plant intact;
   retained-noise chain preserves it every step.
5. **Anchor fix FALSIFIED + muffling:** anchor scale is K/V CONTENT amplitude, orthogonal to
   noise level → peak unchanged; cooling removed the compensation.
6. **Splice-off muffling:** same peak (main stream), same unmasking of muffling.
7. **Linear shape:** excess(m) is 0.25 at 0.05 s, peaks m≈0.15–0.2 (0.37–0.7 s), decays →
   loud from the very start, decaying. Never-row count plays no role.

## Why the S×-hot anchor helped (and muffling reveals the truth)

The S×-hot band K/V, via self-reception (clean-kv-split.md Finding 1), fed the row's own
attention a loud clean copy of its content every block.
This pulled the network's estimate toward the clip audio, masking the F-driven blur —
but it could not remove the excess noise.
Truthful anchor + over-noised row = honest, muffled estimate.

Rejected alternative (claude-opus-4-8): "i=0 plant clean term is S× hot on both axes" —
the clean-term part self-corrects (v5 GPU), so only the noise-level part matters;
also contradicted by euler being clean with the same plant.

## Fix (user-approved 2026-09-02)

Set `PLANT_AXIS = "row"`: `w_plant = sig_row/sig_g = σ_row_a(0) = σ_c(0)` — truthful packed
noise under C2. The clean coefficient stays `(1−w)·clean_packed` (S× hot in model space,
harmless per v5). Keep anchor fix fe0343a and splice ON.

Regression tests `tests/test_sampler.py::TestAncestralPlantAxis` inverted: ancestral audio
plant == σ_a composite == euler's; ≠ σ_v composite. Run fail-then-pass to verify.

## Predictions (0/0/0/90, ease_in_out, 20 steps, η=1)

- Peak band 0.75–1.0 s gone or reduced to the euler floor.
- Muffling substantially reduced across the fade.
- 0/0/49/73 pop gone; linear early static gone; curve dependence vanishes.
- Static gone but muffling stays → muffling is a separate model-level deficit
  (euler 0/0/0/90 should show the same muffling — that is the follow-up run).
- Peak persists unchanged → over-plant falsified; residue is the deterministic per-row floor
  ([../c2-rho-fix-paths/residual-accounting.md](../c2-rho-fix-paths/residual-accounting.md))
  amplified by ancestral; euler 0/0/0/90 should show a faint band at the same place.

## Follow-ups

1. `_audio_observer_ratio` (shift⁻¹(m·σ_a)/σ_row_v) was derived for the v-plant content level;
   under σ_c-consistent plant the native ratio m·g/s may be the consistent choice — re-check
   after the GPU run.
2. Optional exact plant (clean coefficient (1−s)/(1+(S−1)g), v5 form) deferred.
3. v6 never-row heat port remains the next principled lever if post-revert lands at "faint".

## Stated weak point

Model predicts audible shoulders at ~0.55–0.75 s (rows 4–5) and ~1.0–1.1 s (row 8) that
the user described as "contained" to 0.75–1.0 s. A spectrogram would settle it.

## Round-10 update (GPU T1 result, 2026-09-02)

- **Muffling CONFIRMED fixed:** PLANT_AXIS="row" revert removed the upper-band muffling.
  This doc's root-cause analysis is validated.
- **Static FALSIFIED as plant-caused:** over-noise was masking static by converting it to
  muffling. Truthful "row" plant leaves the static raw: louder and wider (0.75–1.0 s peak
  expands to 0.25–1.5 s band). The plant axis only redistributes the energy; neither setting
  addresses the source.
- **CONFIRMED δ-residual (round-10 ret_clean_corr, Branch 1):** the remaining static is δ re-injection
  operating WITHIN the C2 update (audio-only, clean-estimate error leak at low m) — NOT C2-generated.
  Test B (`H3BI_DISABLE_C2=1`) made the static WORSE → C2 is net-corrective (δ-as-generator falsified).
  The fixed-logger run then confirmed δ-as-residual: ret_clean_corr climbs monotonically as m→0 (+0.94
  at k_d=19) — content re-injected via ε̂ under-cancellation + 1/σ_c amplification. See
  [delta-reinjection.md](delta-reinjection.md) for the full record.
