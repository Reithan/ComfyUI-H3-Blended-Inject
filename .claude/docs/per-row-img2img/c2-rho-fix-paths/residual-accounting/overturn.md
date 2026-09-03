<!-- provenance: confirmed (GPU 2026-08-31 overturn: deterministic injection noise in BOTH
     modalities, isolated to OUR node; "euler CLEAN" premise is FALSE). -->
<!-- verified: 2026-08-31 · branch proto-c2-rho-denoised-r-comp · GPU: euler-not-clean,
     silence-prompt, 5-step official-vs-ours; code read sampler.py:305-335. -->
# Overturn: NOT ancestral-specific — deterministic, in OUR node

Parent: [../residual-accounting.md](../residual-accounting.md)

## OVERTURNED (GPU, 2026-08-31): NOT ancestral-specific — deterministic, in OUR node

The prior "euler CLEAN, so the artifact is ancestral-specific" premise (which had ruled out every
sampler-INDEPENDENT mechanism) is FALSE. Three user GPU tests overturn it:
- Re-listening the deterministic **euler** render at the fade: sound IS present (ambiguous alone —
  reads as SCUBA/bubbles).
- Forcing silence by prompt (`overall_soundscape: N/A`, even a blank prompt) STILL yields diegetic
  sound in the fade region.
- Cutting steps **20→5** (so the model can't naturalize noise into plausible sound), original
  prompt, OUR inject node, euler: VISUAL noise becomes VISIBLE and unrequested diegetic AUDIO
  remains. The **OFFICIAL/stock** sampler at the SAME 5 steps (no injection): NO visual noise, NO
  audio noise, NO diegetic sound.

CONCLUSION (user, moderate certainty): a DETERMINISTIC injection-path noise exists in BOTH the
audio fade and the visual fade. The many-step model NATURALIZES it into diegetic sound (why it read
"clean" at 20 steps); low steps expose it raw. It is isolated to OUR node (present in our euler @5,
absent in official euler @5). Ancestral renoise does NOT create the artifact — it merely AMPLIFIES
this deterministic injection error.

## Code fact (verified, sampler.py:305-335)

`_euler_step` — the deterministic per-row euler registered for `sample_euler` — applies NONE of
the C2 corrections: it takes a FULL euler step on the carrier axis (`d=(x−denoised)/σ_i`,
`x_base=x+d·(σ_{i+1}−σ_i)`) then lerps by the legacy r-scale
`r = clamp((sig_row−sig_row_next)/(sig_g−sig_g_next), min=0)`. No σ_c projection, no ρ_true, no
residual cancel, no m=0 context rescale, no per-row sigma integration — ALL of v3/v4/v6 live ONLY
in `_euler_ancestral_rf_step`. So the euler tests exercised the FULLY-UNCORRECTED injection path;
the noise is the raw carry-compression error (audio) + raw per-row injection on fractional rows,
unnaturalized.

## Video-noise pointer (≠ C2)

Video rows have no audio carry (S=1, C2 does not apply), so the VISUAL fade noise is NOT the C2
error — it points to a per-row-remap/injection noise common to BOTH modalities, on top of which
C2 is an audio-only EXTRA. Endgame re-derivation (unified carry correction ported to `_euler_step`,
video-noise mechanism, next discriminator) is IN PROGRESS on a separate thread (Fable) — do NOT
derive it here.
