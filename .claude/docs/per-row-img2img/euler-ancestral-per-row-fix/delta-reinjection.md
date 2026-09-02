<!-- provenance: confirmed (δ-as-RESIDUAL CONFIRMED via ret_clean_corr Branch 1 — content re-injection via ε̂ under-cancellation, amplified 1/σ_c; δ-as-C2-GENERATOR stays FALSIFIED, Test B) -->
<!-- verified: 2026-09-02 (branch fix-euler-ancestral-per-row-renoise) · fixed-logger CSV h3bi_c2_debug-normal-2: ret_clean_corr climbs monotonically as m→0 (+0.94 at k_d=19) → δ-residual CONFIRMED (Branch 1) -->
# δ re-injection theory — round-10 canonical record

Parent: [euler-ancestral-per-row-fix.md](../euler-ancestral-per-row-fix.md).
Competing primary: [../c2-rho-fix-paths/residual-accounting.md](../c2-rho-fix-paths/residual-accounting.md).

## T1 GPU result (round 10)

Config: PLANT_AXIS="row" revert, 0/0/0/90 ease_in_out euler_ancestral, same seed, no env var.

| Timeline | Description |
|---|---|
| 0.0–0.25 s | quiet/normal (m≈0, near-frozen) |
| 0.25–1.5 s | LOUD static |
| 1.5–2.0 s | very similar/identical to original audio |
| 2.0 s–end | new audio |

Fade-OUT: early timeline = low m (preserved), late = high m (new).
Round-9 verdict (plant-over-noise.md): muffling CONFIRMED fixed by the "row" revert; static
FALSIFIED as plant-caused (over-noise MASKED it as muffling; truthful plant leaves it raw, louder,
0.25–1.5 s). The plant axis only redistributes the energy; neither setting addresses the source.

## δ re-injection theory (original analytical form — confirmed below, sign corrected)

In `_c2_audio_ancestral_update`, ε̂ = (y − a·ĉ)/σ_c is recovered by inverting through the network
clean estimate ĉ. The 1/σ_c cancels ONLY if ĉ is exact; the network carries error δ = ĉ − C_true,
plus content the a·ĉ term fails to cancel. The leak coefficient at η=1 (r_ret ≈ σ_c'²/σ_c) is
(1 − (σ_c'/σ_c)²) > 0, GROWING as m→0 → an m-graded, content-correlated leak, largest at low m →
structured static in the low-m band. Fresh stochastic noise (c_fresh = √(σ_c'² − r_ret²), >0 at η=1,
→0 only as m→0) washes it out as m rises. NOTE: the original prediction of NEGATIVE ret_clean_corr was
corrected to POSITIVE by the result below (under-cancellation, not anti-cancellation); the low-m
concentration held.

## Round-10 continued: Test B + logger (claude-opus-4-8 + user GPU, 2026-09-02)

**Test B (`H3BI_DISABLE_C2=1`):** static got LOUDER and LONGER — persisted past 2 s, obscuring ALL
original dialogue; worse than the C2-on run. No CSV produced (the logger only fires inside the C2 path).

**Overturns the pre-registered table.** The pre-registered options were "static drops→δ /
persists→residual floor"; the actual "static WORSE" outcome was in neither branch. Conclusion:
**C2 is net-CORRECTIVE, not the static's generator.**
- **δ-reinjection as a C2-internal GENERATOR is FALSIFIED.** If C2 generated the static, disabling
  it would remove it; instead it amplifies it.
- What survives: error-retention through the ancestral ε̂ is a GENERAL per-row audio-ancestral
  mechanism; C2 MITIGATES it by compressing the σ_a/σ_v carry (plant-over-noise.md: without
  carry-compression every fractional audio row re-enters over-noised → bigger retained error →
  louder static). The residual heard WITH C2 on is the portion C2's compression doesn't reach. δ
  operating WITHIN the (necessary) C2 update — the ĉ→ε̂ re-inversion re-injecting network error,
  amplified at low m by the 1/σ_c division — is now CONFIRMED as that RESIDUAL (see RESULT below).
  Test B did NOT falsify δ-as-residual; only δ-as-wholesale-generator.

**Normal-run CSV (C2 on, 20 steps):** retained_rms dominates fresh_rms through steps 0–9 (0.79 vs
0.27 at step 0, ~2.9×), crossing over ~step 9–10; sig_c decays 0.69→0.09. Consistent with
schedule-level error retention.

**Logger bug (fixed):** total_steps read as `ctx.state.get("total_steps")` came through 0 (it lives at
`ctx.state["schedule_tail"]["total_steps"]`) → k_d collapsed to 0, pooling all 7680 fractional rows into
ONE m-bin (first CSV measured leak-vs-STEP, not leak-vs-m). Fixed to `max(1, len(ctx.sigmas)−1)`.

**Discriminating column `ret_clean_corr`** (prior `leak_to_signal = retained_rms/clean_rms` is just
ancestral SNR): Pearson corr of the SIGNED retained vs clean terms across each bin's rows. Three
branches were pre-registered — (1) corr grows toward m→0 ⇒ δ confirmed as the residual; (2) corr≈0 ⇒
white ancestral noise, δ absent; (3) flat/mid-m-peak ⇒ redirect to the observer content path.

## RESULT — Branch 1 CONFIRMED (fixed-logger run, CSV h3bi_c2_debug-normal-2, 20 steps)

Logger fix worked: k_d now bins by m (k_d 1→19 = m 0.95→0.05); `ret_clean_corr` present. **`ret_clean_corr`
climbs MONOTONICALLY as m falls at every one of the 19 sampling steps.** Step-0 exemplar: k_d=1 (m=0.95)
−0.07; k_d=4 (m=0.78) −0.35; k_d=10 (m=0.51) −0.13; k_d=12 (m=0.38) +0.47; k_d=14 (m=0.30) +0.88; k_d=16
(m=0.19) +0.94; k_d=19 (m=0.05) +0.95. Stable across steps: k_d=19 holds ~+0.94 step 0→18; k_d=1 hovers
near 0. Terminal step 19: retained=0 (leak_coeff=1) → corr=0 trivially.
- **Branch 2 (white noise, corr≈0) FALSIFIED:** at low m the retained "noise" is +0.94 correlated with
  clean content ⇒ it IS content, not noise.
- **Branch 3 (flat / mid-m peak) FALSIFIED:** clean monotonic climb toward low m.

**SIGN CORRECTION (sharpens the mechanism).** Pre-registration predicted strongly NEGATIVE corr (retained
anti-correlating via the `−a·ĉ` term). Reality is strongly POSITIVE. `retained = (r_ret/σ_c)·(y − a·ĉ)`
correlating POSITIVELY with `clean = a'·ĉ` ⇒ the `a·ĉ` term is TOO SMALL to cancel y's content at low m:
the ε̂ inversion UNDER-subtracts the clean signal, so ε̂ ≈ y/σ_c carries content POSITIVELY, and the 1/σ_c
division AMPLIFIES the leftover at small σ_c. **Confirmed mechanism: content (+ network error δ) leaks into
ε̂ via under-cancellation at low σ_c, amplified by 1/σ_c, re-injected as the retained term every step →
low-m preserved-keyframe static.** Magnitude: retained_rms ≈ 0.03–0.15 vs clean_rms ≈ 1–8 at k_d≥16 (a few
% per step) but STRUCTURED/coherent ⇒ 19 steps accumulate it into audible static instead of averaging out.
Matches "original voice legible under static" and the 0–0.25 s quiet / 0.25–1.5 s static timeline (corr
high AND retained magnitude larger at k_d≈14–16).

## Competing explanation

residual-accounting.md records a deterministic per-row injection error in BOTH modalities,
our-node-specific, present in euler@5 steps. It exists in VIDEO (S=1 → C2≡0) and in euler (no C2
branch). δ lives inside the C2 audio branch and cannot explain video/euler noise. So either there
are two errors, or the mode-independent error is PRIMARY and δ is an audio-only residual rider on top.

## Unexplained seam (open)

T1 "1.5–2.0 s sounds like the ORIGINAL." Neither δ nor the mode-independent error predicts a
return-to-source mid-ramp. Either the ear reads low static as "original," or there is a real null at
m≈0.5 (a≈a', σ_c'/σ_c≈1). Needs a spectrogram to settle.

## Fix direction (δ-residual CONFIRMED; needs a Fable design pass for exact packed-axis algebra)

Stop re-deriving retained noise by inverting ĉ. Either persist the actual injected stochastic noise as
sampler state and carry IT forward, or reformulate the packed update so y's content cancels ĉ's content
exactly (retained carries only the true residual). Stochasticity untouched — respects the hard constraint:
do NOT disable/eta-gate stochastic noise. Pre-registered expectation for the fix: low-m `ret_clean_corr`
collapses toward ~0 (retained becomes noise-like), audible low-m static drops to the euler floor, high-m
rows unchanged (already corr≈0). Design pending (fable-ancestral-design).

**Status:** δ-reinjection-as-RESIDUAL CONFIRMED as the low-m static mechanism (content re-injection via
ε̂ under-cancellation + 1/σ_c amplification). δ-as-generator stays FALSIFIED (Test B).
