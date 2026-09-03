<!-- provenance: confirmed (surviving GPU-backed claims from prior rungs, untouched by the
     2026-08-31 overturn; round-9 boundary condition added 2026-09-02). -->
<!-- verified: GPU rungs B/v3/v4/v5/v6 + comfy-ref (model.py:529-541, 549-550;
     model_sampling.py:90-92; model_base.py:2262-2266); round-9 boundary: analytical. -->
# Surviving PROVEN claims (post-overturn)

Parent: [../residual-accounting.md](../residual-accounting.md)

These tie to GPU rungs untouched by the overturn:

- **Monotone same-seed descent B→v3→v4 in the ancestral path.** The C2 carry-compression
  clean-coeff correction on active fractional audio rows removed the large majority of the
  ancestral buzz (loud→faint). Forward carries all audio by global `k=σ_a/σ_v`
  (model.py:529-541), so a row labeled `sig_row` sits at packed coeff `σ_c=sig_row·carrier/sig_g`;
  v4 subtracts the exact residual `(S−1)(sig_row−sig_g)·x_prev`
  (model.py:549-550 + CONST `denoised=x−σ·v`, model_sampling.py:90-92).

- **NOT init-state-driven.** v5 corrected the i==0 ~S× over-plant exactly; GPU showed no change
  ⇒ the DiT self-corrects one-shot state errors early.

  COROLLARY: only PERSISTENT, re-applied-every-step errors can sustain the residual.

  *(Round-9 boundary condition, 2026-09-02): the COROLLARY holds for the CLEAN coefficient (Ĉ
  is re-estimated by the network every step). It is FALSE for the NOISE coefficient: stock
  RF-ancestral's r_ret·ε̂ term carries the initial noise level forward every step, relaxing only
  through the fresh-noise fraction (F² → ρ²F² + (1−ρ²); e.g. F=1.96→1.39 by i=10 at k_d 17).
  An i=0 over-plant on the NOISE axis therefore PERSISTS across all steps.
  Full record: [../euler-ancestral-per-row-fix/plant-over-noise.md](../../euler-ancestral-per-row-fix/plant-over-noise.md).*

- **m=0 audio-context heat is a real contributor.** v6 presents frozen audio rows at
  `clean/(1+(S−1)·sig_g)` instead of `S·A` (native `scale_latent_inpaint` `1/(S·k)`,
  model_base.py:2262-2266); GPU quieter than v4.

- **Input-side static-reference correction is unsound.** 63b291e was LOUDER (`S·A` truthful only
  at step 0; subtracting from a regenerating state injects garbage). Rules out any simple fixed
  x-space clean offset.
