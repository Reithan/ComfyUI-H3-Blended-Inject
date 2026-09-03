<!-- provenance: theory (mechanism UNVERIFIED on GPU; pooled leak-removal design CPU-CONFIRMED — see CPU-probe results §; GPU confirmation pending) -->
<!-- verified: 2026-09-02 (branch fix-euler-ancestral-per-row-renoise) · reconciled from normal-3 CSV by claude-opus-4-8 + fable-ancestral-design -->
# mid-m denoiser noise-leak — mechanism + fix candidates

GPU logger run: `euler_ancestral`, 20 steps, `min_denoise=0`, full fade 0/0/0/90, η=1.
CSV: `.claude/tmp/h3bi_c2_debug-normal-3.csv`.
Two findings were cross-verified by claude-opus-4-8 + fable-ancestral-design against the logged data.

---

## Over-injection (C2 packed-axis clamp) — CORRECTED claim, minor effect

Earlier claim (Fable): the packed-axis retained/fresh split over-injects across m 0.05–0.72
by up to 1.19×/step, 1.32× compounded.
**This is RETRACTED** — it came from a reconstructed schedule that doesn't match the real
per-row-remapped σ_c.

Verified against LOGGED σ_c: `r_ret` reconstructed from consecutive σ_c via the current formula
`sd = σ_c′·(1+(σ_c′/σ_c−1)·η)`, `r_ret = sd·(1−σ_c′)/(1−sd)`.

The clamp (`r_ret > σ_c′`, `c_fresh` forced to 0) fires ONLY at k_d 6–9 (m≈0.57–0.72),
early steps only:
- k_d 6: steps 0–4; k_d 7: steps 0–3; k_d 8: steps 0–2; k_d 9: steps 0–1.

Max ratio r_ret/σ_c′ = 1.018–1.056 (≤6%).
Nothing fires at k_d 10+.
The logged `fresh_rms==0` rows match these clamp steps exactly.
(The extra step-19 zero is the terminal step — normal for every row.)

**Why only there:** `r_ret > σ_c′` ⟺ `σ_c′ > σ_c`, i.e. the packed σ level rises step-to-step.
At k_d 6–9 the logged σ_c rises early (e.g. k_d 6: 0.9426→0.9455 over steps 0–5)
because `1/k = σ_v/g` grows faster than `s` falls.
Stock's η formula assumes a monotone axis and yields `sd > σ_c′` there.
Low-m rows never see a rising σ_c → no clamp.

**Verdict:** a real bookkeeping wart, but MINOR (≤6%, ≤5 steps, m≈0.6–0.7).
NOT the cause of the audible static.
The native-axis split fix (do the η split on the monotone `s` axis, then pack by `1/k′`)
removes only this clamp.
At k_d 11–13, packed vs native split agree to within 2–5% in fresh-variance share.
Correct bookkeeping, minor effect.

---

## THE mechanism — denoiser noise-leak committed by ancestral replacement

The static maps to the `ret_clean_corr` peak: k_d 11–13 (m 0.34–0.45),
corr +0.70/+0.71/+0.73, where variance injection is ALREADY CORRECT (no clamp, fresh>0).

This matches the user's spectrogram: static/reference spectral-flatness ratio 2.83
("tonal but flatter than reference — mixed/broadband"), vs the clean euler control at 1.10.

`corr > 0` is NOT peculiar to fractional rows:
even near-stock k_d=1 (m=0.95) reaches +0.357.

Since the noise-carry fix (commit a28a62b) makes εc content-free by construction
(`init clean_raw = clean_packed = ground-truth injected latent`, sampler.py:949;
advance mixes only carried noise + fresh white noise),
`corr(εc, ĉ) > 0` can only mean ĉ CONTAINS the row's own input noise:
`ĉ ≈ γ·C + λ·σ_c·εc`.
This is the ordinary MMSE denoiser posterior-mean shrinkage — legitimate —
and the exact reverse-SDE tolerates it.

**Why mid-m + mid-trajectory:**
at steps ~2–8, those rows sit at s≈0.4–0.7 while global g≈0.9–0.97.
The row's CONTEXT is still largely noise, so the network has no prior to shrink toward
and passes the row's own noise through (λ high, ~0.7;
corr² ≈ noise-energy-fraction of ĉ ≈ 0.5 there vs ≈0.1 for stock rows).
corr rises from ≈0 at step 0 (plant noise, content-free εc)
to a peak at steps 4–6, then decays as clean context arrives —
exactly the per-step shape in the CSV.

**Why ancestral-only (euler control is clean):**
the leak λσ_c·εc lives in the CONTENT slot `a′·ĉ` every step.
Under EULER (η=0) the noise slot keeps the SAME εc realization,
so the committed leak stays perfectly correlated with the row's noise
and is re-denoised on later steps — a coherent fixed point, clean.
Under ANCESTRAL (η>0) `c_fresh` replaces a fraction of εc each step;
the part of the leak whose matching noise got replaced is now orthogonal
to anything the net will treat as noise — it has become committed content.
Per-step commit ≈ `a′·λ·σ_c·(1−r_ret/σ_c′)·(fresh share)`;
independent across steps → random-walk accumulation → broadband static
(consistent with flatness 2.83, i.e. accumulated noise, NOT naturalized content).
Stock ancestral shows the same effect at small λ — its familiar faint grain.

The advance re-correlates because ĉ ITSELF carries εc;
the carry is the victim, not the cause.
This is NOT an init leak (step-0 corr≈0) and NOT carry drift (εc is provably content-free).

---

## Candidate fix directions (CPU-probe-gated, NOT yet implemented)

**Fix A — excess-leak cancellation (per fractional audio row, before recomposition):**
Cancel the committed EXCESS leak using the carried εc
(now available thanks to the noise-carry fix):

`λ̂ = ⟨ĉ, εc⟩ / (σ_c·‖εc‖²)`
`ĉ_corr = ĉ − (λ̂ − λ_ref(σ_c))·σ_c·εc`

where `λ_ref` is the in-distribution stock-row pass-through at the same σ_c
(measurable from m=1 rows in the same run, or a fitted curve).
Only the EXCESS (context-starved ~2× stock) is off-distribution and accumulates as static.

**Fix B — cheaper variant (no λ_ref needed, euler-invariant by construction):**
`ĉ − λ̂·σ_c·(1−r_ret/σ_c′)·εc`
Corrects only the replaced share; vanishes under η=0/euler (self-consistent with the clean euler control).

**Properties of Fix A/B:**
Projection along one known random direction → content bias O(1/√N) per row
(N = audio elements/row — a RISK to check on realistic audio shapes).
Retained/fresh/η untouched; gated on `frac_audio` so m=1 stays bit-exact.

**Structural lever (independent):**
Reduce λ itself by feeding those rows cleaner context —
v6 never-row presentation / observer K/V at a cleaner level for mid-m rows.
Changes what the net sees, not the sampler.

**VERIFICATION GATE (before any GPU):**
CPU synthetic-leaky-denoiser probe:
set `ĉ = γC + λσ_c·εc` with λ≈0.7 for steps 2–8,
run through `_c2_audio_ancestral_update` at η=1 vs η=0,
and confirm:
(a) the accumulation appears (η=1 residual > η=0), and
(b) Fix A/B removes it while preserving the fresh term.

**CPU-probe RESULT (2026-09-02):** run — per-row projection is DEAD (content-biased,
stuck at ~0.30); the POOLED (group-averaged) content-blind projection recovers the euler
floor and is the landing fix. Full numbers, real bin occupancy, fix form, and the
build decision live in
[mid-m-denoiser-leak/cpu-probe-results.md](mid-m-denoiser-leak/cpu-probe-results.md).
