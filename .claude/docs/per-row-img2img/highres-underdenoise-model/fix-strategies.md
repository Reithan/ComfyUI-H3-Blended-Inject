<!-- provenance: theory (fix strategies + feasibility argument 2026-08-24; own-code checks are CONFIRMED where marked CHECKED) -->
<!-- verified: 2026-08-24 · 1MP GPU runs (0.45/0.60/0.68 lock, 0.75/0.78 smear=window closed, 0.83 chaos) + Fable adjudication · data in ../highres-singleframe-underdenoise.md -->
# Fix strategies, feasibility, and own-code checks

Index: [highres-underdenoise-model](../highres-underdenoise-model.md). The experiments that feed these:
[experiments](experiments.md). The crux: [crux-and-mechanism](crux-and-mechanism.md).

## The fix if the window is closed — anchor-then-release (schedule split)

If 0.75 also fails, d-tuning is provably insufficient (all three levers modulate the ROW; neighbor→row
attraction is left to attention, whose per-key mass on the inject dilutes ~∝1/total-tokens — plausibly
part of the resolution effect itself). Minimal attraction mechanism, preferred:
**anchor-then-release** — for early steps k<k_sw (while composition is decided) present the row RESOLVED
(x_row=clean, low t_row) so neighbors compose AROUND it (inpainting-style attraction during exactly the
composition-deciding steps); at k_sw release into the per-row img2img machinery with an effective m
sized to the REMAINING schedule + the denoised correction (the piece the OLD hold-and-release lacked —
that's why this differs from the reverted design). One new param (k_sw / its σ). Alternative: an
attention-logit boost (+log(b), b≈token-ratio) on neighbor→inject keys during early steps — orthogonal,
deterministic, directly tests the dilution hypothesis, but more invasive to hook.

## Is a stable fix even possible? (strategy, 2026-08-24)

Concern (user): with resolution + anchor↔ambient + anchor↔spacing confounds, maybe no stable single-
frame fix exists. Counter-case, three points:
- **FL2VA is an EXISTENCE PROOF.** H3 demonstrably CAN hold a single frame and blend neighbors robustly
  — the user confirmed native keyframes attract and look right across a broad range. So a stable
  single-frame keyframe blend is real *on this model*; the open question is narrower — can our
  latent-inject path reach it, or must we borrow the attraction mechanism.
- **The confounds likely share ONE root:** a passive anchor whose attention mass on neighbors dilutes
  ~∝1/tokens. The bimodal NARROW window is the signature of solving an **attraction** problem with a
  **denoise-balance** knob — the wrong axis. Real attraction (anchor-then-release) should COLLAPSE all
  three at once: undo dilution (resolution), pull neighbors to the anchor regardless of how far it sits
  from ambient (distance), and resolve the anchor early so it stops holding the region open (spacing).
- **MC's per-step re-composite, re-read:** it's not just compositing — it's a **stability mechanism**.
  Re-anchoring x toward noised-source every step clamps the row on the source manifold so the front
  never tears it into chaos; MC therefore NEVER smears — it trades chaos for the GHOST (+ semantic
  break at low d). It's "good enough" for MC's real target (m=0 clip continuation) and incidentally
  chaos-proof, but it does NOT solve fractional single-frame. Crucially, **anchor-then-release ≈
  MC-recomposite-EARLY (stabilize composition) + img2img-release-LATE (semantic + no ghost)** — the two
  design lineages converge. The empirical test for "stable fix exists on our path": does attraction
  WIDEN the coherent basin (robust across d and anchor-ambient distance)? Basin WIDTH is the answer to
  the pessimism, and we have not yet tested the right-axis mechanism at all — only mapped the failure
  walls of the wrong one.

## Loose ends to control (own-code checks)
- **Noise identity — CHECKED 2026-08-24, correct as-is (no action).** `per_row_init_lerp`
  (sampler.py:47,76) blends `x_r = m·x_global + (1−m)·clean` where `x_global = σ_max·eps+(1−σ_max)·clean`
  uses the SINGLE `prepare_noise(samples,noise_seed)` field (nodes.py:319) shared by the whole gen →
  injected row's noise is CORRELATED with neighbors (eases blending, as desired). Not independent.
- **t_pin clamp collision — CHECKED 2026-08-24, NO collision at high d (Fable's concern inverted).**
  `model.py:589,596`: `t_pin_v = max(1−σ_v, 0.999)`; `rows_t = (1−m·σ_v).clamp(max=t_pin_v)` is an
  UPPER cap. `1−m·σ_v` DECREASES as m grows → raising d moves the label AWAY from the 0.999 ceiling.
  Clamp only bites when `m·σ_v<0.001` (near-preserve m→0 rows = intended pinning; or terminal σ→0 where
  the cap has risen to ~1.0 = no-op). At d≈0.83 the label is 0.17 early, ~0.99 only at the last step —
  never meaningfully clamped. `quantize_denoise` 1/256 grid has ~212 levels below 0.83 → no quant loss.
  ⇒ the odds-linear d′=0.83 drops straight into the existing lever arithmetic; nothing flattens it.
- **Audio re-anchoring:** shift_audio≈3 vs shift_video 10-16 → audio row commits on a different σ(t);
  if A/V cross-attn re-injects source identity after the video row de-commits, pops survive correct
  video-d. Check t_c on the audio row too.
- **Measure β directly:** FFT the clean latent row, fit per-band power slope → predict Λ_lock/Λ_id
  rather than fit; check whether the H3 VAE whitens (β<2 narrows the window further).
