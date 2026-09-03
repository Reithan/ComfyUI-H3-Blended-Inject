<!-- provenance: confirmed (GPU) — noise-carry fix partial: δ low-m leak eliminated; mid-m ANCESTRAL-SPECIFIC (Round-11b falsifies residual-accounting attribution; fix = ancestral noise calibration on packed audio axis) -->
<!-- verified: 2026-09-02 (branch fix-euler-ancestral-per-row-renoise) · GPU run h3bi_c2_debug-normal-3.csv + Round-11b euler-deterministic control (MiniMax_H3_00001 vs MiniMax_H3_00002, spectral-flatness tool) -->
# Noise-carry fix — GPU result (round 11)

Parent: [euler-ancestral-per-row-fix.md](../euler-ancestral-per-row-fix.md).
Sibling: [delta-reinjection.md](delta-reinjection.md).
Competing: [../c2-rho-fix-paths/residual-accounting.md](../c2-rho-fix-paths/residual-accounting.md).

## Summary: PARTIAL fix

The noise-carry fix (commit a28a62b) achieves the pre-registered low-m target but reveals a
distinct mid-m residual band not reachable by the carry. Two mechanisms coexist.

## Finding 1 — TARGET HIT: δ init leak eliminated at low m

Rows k_d 17–19 (m ≈ 0.05–0.14) — where the pre-fix run (h3bi_c2_debug-normal-2) held
ret_clean_corr +0.94/+0.95 across all steps — now read +0.33 / +0.22 / +0.15.
Coherent leak energy (√Σ retained² over steps where corr > 0.3) at k_d 18, 19: 0.000; k_d 17: 0.204.
The 1/σ_c-amplified init content leak is gone.

## Finding 2 — RESIDUAL RELOCATED to mid-m (k_d 6–14, m ≈ 0.30–0.72)

Peak ret_clean_corr rises to +0.58–+0.73 in this band; coherent leak energy concentrates at
k_d 9–13 = 1.1–1.6. Full per-row table (GPU run h3bi_c2_debug-normal-3):

| k_d | m    | peak corr | @ step | leak energy |
|-----|------|-----------|--------|-------------|
| 1   | 0.95 | +0.36     | s17    | 0.579       |
| 4   | 0.78 | +0.36     | s17    | 0.867       |
| 6   | 0.72 | +0.46     | s12    | 0.919       |
| 7   | 0.66 | +0.56     | s12    | 1.092       |
| 8   | 0.62 | +0.56     | s10    | 1.202       |
| 9   | 0.57 | +0.58     | s10    | 1.433       |
| 10  | 0.51 | +0.58     | s6     | 1.305       |
| 11  | 0.45 | +0.70     | s6     | 1.556       |
| 12  | 0.38 | +0.71     | s4     | 1.390       |
| 13  | 0.34 | +0.73     | s5     | 1.121       |
| 14  | 0.30 | +0.60     | s4     | 0.879       |
| 15  | 0.24 | +0.55     | s11    | 0.749       |
| 16  | 0.19 | +0.51     | s11    | 0.470       |
| 17  | 0.14 | +0.33     | s14    | 0.204       |
| 18  | 0.09 | +0.22     | s16    | 0.000       |
| 19  | 0.05 | +0.15     | s17    | 0.000       |

## Finding 3 — Accumulation-over-steps signature

Per mid-m row, corr grows from ~0 (or negative) at step 0 to a peak at mid-trajectory (steps
4–12), then decays toward the terminal step. This differs from the pre-fix low-m leak, which was
high at ALL steps.

Since the carried εc is content-free by construction at init and (by derivation) throughout the
recurrence, this trajectory-build means content enters the retained channel via a path the carry
does not reach. Mechanism is UNDER ANALYSIS (handed to Fable). Do not assert a cause.

**Candidate hypotheses (all unconfirmed):**
- (a) The untouched clean channel a_next·ĉ shrinkage undershoot is no longer compensated by the removed leak.
- (b) x_cur pre-plant at i=0 is not pure noise for these rows.
- (c) FALSIFIED (Round-11b) — residual-accounting deterministic per-row error as primary: euler-deterministic control is clean (flatness ratio 1.10), so the mechanism is not mode-independent.
- (d) Carry recurrence variance/normalization drift.

## Finding 4 — EAR

Static is quieter and dissolves back into generated audio more smoothly than before the fix, but
NOT fully gone. Original audio is not discernible through the static. Static range roughly 0.3–2 s.

## Verdict

Matches the pre-registered fork: "corr collapses at the δ target but static persists → competing
residual-accounting mechanism (or a carry-unreached path), not the δ init leak."
The δ-as-init-leak theory is CONFIRMED; its fix is VALIDATED at the low-m target.
The mid-m residual band (k_d 6–14) is now the primary open item.

## Mid-m residual — mechanism verdict (Fable round-11)

**Carry is provably content-free (CPU probe).** Threading `_c2_audio_ancestral_update` over 20
steps against a synthesized content-proportional shrinking denoiser (ĉ = γ·C_true, γ ramp 0.4→1,
white εc by construction) gives corr(retained, clean) ≈ 0 at every step (−0.01 to +0.00),
including steps where fresh_rms = 0. The noise-carry fix is CORRECT AND COMPLETE for its target.

**GPU +0.7 mid-m band cannot originate in the carry.** ret_clean_corr = corr(r_ret·εc, a_next·ĉ).
Since εc is white by construction, +0.7 across thousands of elements is only possible if the real
network's ĉ contains a component correlated with the per-row injected noise εc in x. The network
naturalizes injected/carried noise into content-shaped static.

**[FALSIFIED Round-11b — see §Round-11b below] This was attributed as the residual-accounting PRIMARY mechanism** (was GPU-confirmed 2026-08-31;
see [../c2-rho-fix-paths/residual-accounting.md](../c2-rho-fix-paths/residual-accounting.md)):
deterministic per-row injection error in BOTH modalities, present in our euler@5, absent in
stock euler@5. Ancestral renoise only amplifies it. Attribution withdrawn: euler-deterministic control is clean in Round-11b.

**Why mid-m + mid-trajectory.** Per-row σ_c schedule is maximally off-distribution at mid-m.
Network noise-to-content naturalization is strongest at mid-σ_c, mid-trajectory. A pure init
leak peaks at step 0; a pure 1/σ_c leak at lowest m; only an off-distribution model error
peaks at intermediate σ_c AND intermediate step. Fits the data.

**Ranked mechanisms:**

1. FALSIFIED (Round-11b) — model per-row injection error as PRIMARY: euler-deterministic control is clean; the static is ancestral-specific, not a sampler-agnostic deterministic error.
2. Clean channel now relies on the noise-leaking ĉ — flip side of #1. The old δ leak was partly
   masking by re-adding content; removing it (correctly) exposes the model's own ĉ static.
   Matches ear: quieter + smoother, but raw model static now shows.
3. r_ret > σ_c' clamp → fresh = 0. CONFIRMED firing (steps 0–5 low m, 0–4 mid, 0–2 high; never
   at m = 0.95). This is exact stock RF-ancestral behavior — content-neutral, not a defect.
4. FALSIFIED — init contamination. Probe step-0 corr ≈ 0. Plant's clean = state["clean"] so
   (y − (1−w)·clean_raw) cancels exactly to εc_init = (w/σ_c)·x_noise; x_cur pre-plant is
   the initial Gaussian.

**[SUPERSEDED by Round-11b] Fix direction:** the fix IS on the ancestral axis — noise calibration
on the packed audio axis at mid-m, preserving the fresh stochastic term (design in progress,
Fable). The noise-carry fix stays as-is (low-m δ target is complete). Clean-estimate axis levers
remain valid secondary approaches but are not the primary direction:

- (a) Clean-K/V observer path — feed cleaner K/V so ĉ hallucinates less from compressed noise.
- (b) Extend v6 mid-m context heat clean/(1+(S−1)σ_g) into the mid-m band, not just frozen rows.
- (c) Deeper cure: reduce mid-m per-row σ_c schedule compression (architectural remap change).

**FLAGGED AS INFERRED (now resolved):** the proposed confirming test — euler-vs-stock A/B on this
config — was run in Round-11b. It REFUTED the hypothesis (euler deterministic is clean;
see §Round-11b below). The network-leak-into-ĉ inference is WITHDRAWN; source is ancestral-specific.

### Round-11b: euler-deterministic control falsifies residual-accounting attribution

Config identical to round-11 (0/0/0/90 ease_in_out, 20 steps, min_denoise=0, fixed seed).
Sampler swapped euler_ancestral → euler. Flatness measured with `.claude/tmp/audio_flatness.py`,
window 0.3–2.0 s vs reference 2.0 s→end.

- euler_ancestral (MiniMax_H3_00001): flatness 0.0407 / ref 0.0144 → ratio **2.83** (hiss present).
- euler deterministic (MiniMax_H3_00002): flatness 0.0212 / ref 0.0192 → ratio **1.10** (CLEAN).

User confirms: audio AND video both fine on the euler run. Static GONE with deterministic sampler.

**Conclusion:** mid-m static (0.3–2.0 s) is ANCESTRAL-SPECIFIC. Residual-accounting is
sampler-agnostic — a deterministic injection error would appear in BOTH samplers. Euler is clean;
the residual-accounting PRIMARY attribution is FALSIFIED.

**What still stands:** carry-is-content-free CPU proof (corr ≈ 0 at every step). The noise-carry
fix is correct and complete for its low-m δ target. The impossibility proof excludes the carry as
the source of mid-m content; it does not determine which path actually injects it.

**New direction:** the culprit is the ancestral noise injection itself (retained+fresh magnitude/
calibration on the packed audio axis at mid-m). Re-derivation + calibration-fix design preserves
the fresh stochastic term. Design handed to Fable.
