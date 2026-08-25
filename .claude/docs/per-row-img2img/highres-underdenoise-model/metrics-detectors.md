<!-- provenance: theory (seam z-score GPU-confirmed 2026-08-24 as primary gate; Ψ/p-cross-1 killed; instrumentation live on debug branch) -->
<!-- verified: 2026-08-24 · 1MP GPU runs (0.45/0.60/0.68 lock, 0.75/0.78 smear=window closed, 0.83 chaos) + Fable adjudication · data in highres-singleframe-underdenoise.md -->
# Metrics and detectors — seam z-score, lock detector, instrumentation

Index: [highres-underdenoise-model](../highres-underdenoise-model.md). Data these metrics were validated on: [data-runs](data-runs.md). Experiments using these metrics: [experiments](experiments.md) / [fix-strategies](fix-strategies.md).

## FREE, RIGOROUS lock detector — step-count invariance (Q4 framing CONFIRMED)

For CONST/RF: x_σ=(1−σ)x₀+σn, ODE dx/dσ=(x−D)/σ. If the posterior D collapses on source (constant
along path), the trajectory is a STRAIGHT LINE in σ, and **euler is exact on linear trajectories at
any step count.** So 1MP popping identically @10 AND @40 steps is the *signature* of basin-lock, not a
coincidence. **Test:** any res, run 10 vs 40 steps, compare the injected row only — near-identical ⇒
locked ⇒ will pop; step-count-sensitive ⇒ curved trajectory ⇒ blending. Free pop detector, no eyeball.
(Also explains why SMOOTH has the LOWEST R: R = scalar distance from source, but the axis is
DIRECTION/destination not distance — a coherent blend keeps source low-freqs & revises mid/high bands,
staying L1-near source; chaos is far-incoherent; lock is near. R conflates all three → drop it.)

## THE metric — seam z-score (PRIMARY gate); Ψ & p-cross-1 are DEAD

**GPU-KILLED 2026-08-24.** Ψ (commitment-timing) and "p crosses below 1" both FAILED the visual: the
1MP lock (d=0.45) had t_c=17 (LATE) and p starting at 1.50, and the chaos (d=0.83) had p∞=0.92<1 yet
was incoherent. Diagnosis: (i) lock here is NOT a constant-D basin hold — it's an
**excursion-and-RETURN** to source (neighbors DO perturb the row mid-run, then it re-attracts), so
commitment TIMING cannot see the destination; (ii) p(0)=1.5 was **amplitude gain** (early D overshoots
clean's amplitude ~1.5×, a Wiener artifact), conflated with direction. Split them.

**PRIMARY success gate — seam z-score** (final latent; directly encodes "no pop"):
`g_r = ‖x_final[r] − x_final[r±1]‖₁` for both temporal seams; `z = (g_r − μ)/s` where μ,s are the
mean/std of ‖x[r']−x[r'+1]‖₁ over all consecutive **generative** row pairs in the same run.
**COHERENT ⟺ max(z over both seams) ≲ 1.5-2.** Both failure modes pop ⇒ both raise z ⇒ z is **U-shaped
in d** (high at lock AND chaos, dips in the window) → its **minimum over d LOCATES d\*** label-free, and
min-z above threshold is the quantitative "window closed" verdict. Golden-section / binary-search d on z.

**Mode-tellers (which wall, once z fails):**
- **LOCK:** p̂∞=cos(D,clean) high AND `ρ_ret = ‖D(final)−clean‖ / max_k‖D(k)−clean‖ ≪ 1` (it came back).
- **CHAOS:** ρ_ret ≈ 1 (monotone departure); high mid-run revision ΣΔ_r with low coupling
  φ̄ = mean_{k∈[¼K,¾K]} cos(ΔD_r, ½(ΔD_{r−1}+ΔD_{r+1})); + spectral-slope mismatch of D_r(final) vs
  neighbors (deferred — not yet instrumented).

Split `p` into **p̂=cos(D,clean)** (direction) and **amplitude ‖D‖/‖clean‖** — the old
p=⟨D,clean⟩/‖clean‖² mixed them. Calibrate every threshold on the **known-coherent 0.2MP/d=0.45 run**
(rerun with full instrumentation) → the coherent fingerprint. `|out|` is only a weak secondary flag.

## Reading the instrumentation (debug branch, live)

The debug build (`nodes.py` `_run_sampler` + `sampler.py` `build_conditioning_wrapper`) accumulates
the raw denoised prediction `D_r(k)` (pre-correction) per tracked row per step (CPU fp16) and prints
at run-end — always on in this branch, no flag. **Read `seam` FIRST (the gate); the rest are tellers:**
- `[H3_DEBUG] seam row=R zmax=… [COHERENT|POP]  r{nb}:g=… z=…  (gen-pair mu/sd/n)` — **PRIMARY gate.**
  zmax ≲ 1.5-2 ⇒ coherent. Verdict tag flips at z=2.0. Track zmax vs d: golden-section its minimum.
- `[H3_DEBUG] rho_ret row=R = …` — LOCK-teller. ≪1 = excursion-then-return to source (lock); ~1 =
  departed (chaos or blend).
- `[H3_DEBUG] phi_bar row=R = … over k∈[…]` — CHAOS-teller. >0 = row follows the neighbor front
  (coherent); ~0 with large revision = anchor-conflict smear.
- `[H3_DEBUG] p_hat row=R cos(D,clean) …` + `[H3_DEBUG] amp row=R |D|/|clean| …` — direction vs
  amplitude split of the old source-projection (p̂ high + ρ_ret≪1 = lock).
- `[H3_DEBUG] commit … / PSI …` — Ψ/t_c retained as DIAGNOSTICS only (killed as a gate; still handy
  for seeing the excursion-return timing). Neighbors auto-added as `NBR(of R)` rows (R±1,R±2).
- `[H3_DEBUG] k_comp row=R ~K/… (per-nbr […]) => suggested k_sw` — **composition-lock step** = first k
  where the neighbors' coarse layout (spatial-lowpass D, avg-pool 8×) matches their final, mean over
  R±1,R±2. This is the AUTO-CALIBRATED **k_sw** (hold-until) for anchor-then-release; record it every run
  so the hold length is measured, not tuned. Wrapper stashes `den_lofreq_by_row`; post-hoc cosine>0.9.

**Calibration loop:** golden-section d on `seam zmax` (minimum = d\*); confirm with the visual and the
tellers (which wall when z is high). FIRST rerun the known-coherent 0.2MP/d=0.45 to record its
`seam zmax` / `rho_ret` / `phi_bar` fingerprint so the thresholds are anchored. Watch for chunk-boundary
(per-17 grid reset) neighbors polluting the gen-pair baseline — sanity-check `n` and the seam detail.
