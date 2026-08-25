<!-- provenance: theory (H1 CONFIRMED by elimination; bimodality GPU-confirmed 2026-08-24; R REFUTED as discriminator; coherence/phase is the real signal) -->
<!-- verified: 2026-08-24 · repo @debug-single-frame-underdenoise · 1MP run (row40@0.83=chaos, row60@0.45=lock) -->
# Resolution ladder — bimodal failure and coherence findings

Index: [highres-singleframe-underdenoise](../highres-singleframe-underdenoise.md). Kill-shot data and hypotheses: [hypotheses-and-data](hypotheses-and-data.md). Temporal effects and contagion: [temporal-and-contagion](temporal-and-contagion.md).

### RESOLUTION LADDER (all d=0.5/0.45; user) — R REFUTED as discriminator; bimodal failure
| res / steps | R₄₀ | R₆₀ | \|out\|₄₀ | \|out\|₆₀ | `|inp|` rebound | visual |
|---|---|---|---|---|---|---|
| 1MP /10 | 0.188 | 0.169 | 0.800 | 0.817 | 0.70–0.71 | **POP** source-identical |
| 0.1MP /20 | 0.239 | 0.225 | 0.735 | 0.767 | — | **POP** *chaotic* |
| 0.2MP /20 | 0.215 | 0.162 | 0.735 | 0.772 | 0.671 | **SMOOTH** ✓ |
(R = |out−clean| / the d=1.00 companion row 39/59. 1MP also popped identically @10 AND 40 steps →
**step-INDEPENDENT**, spot-check confirmed.)

**FINDING 1 — displacement R does NOT discriminate pop from smooth.** Smooth 0.2MP has R₆₀=0.162,
*below* the 1MP pop's 0.169; R₄₀ 0.215(smooth) vs 0.188(pop) barely differ; 0.1MP pops with the
HIGHEST R. **How far the frame moves from source is not the cause.** ⇒ "hit a target R" / α=√5
magnitude *calibration* is DEAD (raising m still helps — the reason was wrong, see Finding 3).

**FINDING 2 — bimodal failure; 0.2MP is a true goldilocks.** Too-high res (1MP): frame
self-reconstructs to SOURCE → under-denoise pop. Too-low res (0.1MP): frame departs source the SAME
distance but INCOHERENTLY → chaotic pop ("opposite direction"). 0.2MP: departs source COHERENTLY
with neighbors. Same distance, different DESTINATION — only content-coherence separates them, no
magnitude does. (`|inp|` rebound happens even at smooth 0.2MP → rebound per se isn't the tell; the
tell is rebound-to-source-CONTENT vs to-a-neighbor-coherent state.)

**FINDING 3 — the real discriminator is COHERENCE / phase with the denoising front, not magnitude.**
The fix (raise effective-m at high res) works because it keeps the keyframe UNRESOLVED LONGER so it
re-syncs with the neighbor front (a TIMING effect), NOT because it hits a displacement target. So
calibrate on coherence, not R. **Usable corrected targets** (from the ladder): (a) `|out|` DOES flag
under-denoise — 0.80 @1MP-pop vs 0.735 @both-denoised — so raise d @1MP until `|out|₄₀`→~0.735;
(b) `|inp|` rebound level: target the 0.2MP-smooth ~0.67, not the 1MP-pop 0.70–0.71; (c) then CONFIRM
VISUALLY it's a 0.2-style coherent blend, not a 0.1-style chaotic departure (neither `|out|` nor
`|inp|` can tell those apart — same magnitudes). Best pure-objective metric would be a temporal-
coherence measure (keyframe row vs its temporal-neighbor trend), not yet instrumented.

**H1 — fixed sigma-shift ≠ constant perceptual corruption across resolution (top).** RF injects
white noise per latent element, but natural-image latent power is ~1/f. At 1MP a frame spreads the
same content over ~5× the latent tokens (~1024 vs ~196) → content power concentrates in low
frequencies while injected noise stays white → per-band SNR at `x=0.5·noise+0.5·clean` is *higher*
@1MP → model recovers the input (artifacts included) almost exactly. SD3/Flux grow timestep shift
with token count precisely to hold perceptual corruption constant; **H3's shift is a fixed *user*
value not coupled to token count** → fails to compensate. Explains A/B directly; reframes **C as an observability artifact** (an
under-denoised multi-frame row still looks coherent; only a lone still whose artifacts must be
*hallucinated away* exposes weak effective strength). The exoneration ruled out the schedule
*varying* with res — this is the inverse: the schedule *failing to vary* is the bug.
- **Kill-shot = fix prototype in ONE run:** SNR-match `m` via SD3-style shift, α=√(token ratio)≈√5≈2.24,
  so `m=0.5@0.2MP ≈ m≈0.69–0.70@1MP`. One 1MP run @denoise≈0.7 matching the 0.2MP result confirms;
  the fix is a **resolution-corrected effective-m mapping** dropped into the existing lever
  arithmetic, no change to the invariance proofs.
- **Free diagnostic:** if per-row `|out−clean|` drops ~uniformly across ALL rows @1MP (not just the
  keyframe), asymmetry is observational → H1 stands.

**H2 — neighbor bleed (D) is VAE decode-overlap, not attention.** Confirmed mechanism (memory
`h3-vae-decode-overlap`): decoder blends overlapping token windows → bad row 40 contaminates
decoded pixels ±~2 frames even if latent rows 39/41 are pristine. **Displaces H-ATTENTION-BLEED** —
no DiT mechanism needed for D. CPU diagnostic: diff saved latent rows 39/41 vs clean ref;
clean-in-latent + dirty-in-pixels → decode overlap.

**H3 — narrow-token coincidence (cheapest discriminator).** f136=8·17, f204=12·17 both land on
chunk-start `local_row 0`, the `FRAME_PER_TOKEN[0]=1` narrow token (span 5/3 vs 20/3, asymmetric
gaps; model.py:95–103) — weakest cross-temporal attention support. @1MP the 5× spatial-token
softmax competition could push *only* the isolated narrow token over the under-denoise threshold;
video injects flank it with anchored neighbors → immune. **One 1MP run injecting mid-chunk
(f138/f140 → local_row 1, wide token):** cleans up → narrow-token-specific; still fails → narrow
token exonerated, H1 strengthened. Single run splits the space.

**H4 — resolution-dependent VAE *encode* stats — REFUTED (build log @1MP).** Predicted smoother,
lower-variance single-frame latents @1MP. Measured: inject latent std=0.97 (row 40) / 0.99 (row 60)
vs video rows std 0.94–1.12, matching min/max ranges — statistically indistinguishable. The lerp
starts from a well-formed latent; the under-denoise emerges DURING sampling, not from a bad input.
Kills H4 AND the VAE-`shape[2]==1`-special-case hypothesis.

**H5 (long shot) — per-token adaLN modulation diluted at high token count** (model.py:619–626).
No such reduction spotted in layout code; killable with one hooked forward dump of (shift,scale,gate).
