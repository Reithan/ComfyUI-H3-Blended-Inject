<!-- provenance: theory (analysis + source-grounded; recommendation UNVERIFIED on GPU) -->
<!-- verified: 2026-08-24 · mc @d299ea5 · comfy-ref @b78cec87 (+ GPU: MC keyframes blend r40/r60 @1MP, anchors m=0-wrong) -->
# Conditioning-row inject (MC "H3 Custom Keyframes") — what it is & should we add it

**Question (user, 2026-08-23):** MC injects keyframes not only into the latent but also into
**conditioning rows** — see the non-masked `MiniMaxH3CustomKeyframes` node. Is there any benefit
to a Blended variant/toggle that targets conditioning rows instead of the latent?

## Detail docs

- [what-and-verdict](conditioning-row-inject/what-and-verdict.md) — what the cond-row path actually is (source-confirmed); verdict (different tool, not a substitute); interop already free (no new node needed)
- [fade-and-decoupler](conditioning-row-inject/fade-and-decoupler.md) — blended fade design analysis (lever 1 FALSIFIED, lever 2 attention-cost trade); open hybrid hypothesis; SHARPENED: cond token as native crux-decoupler (GPU-confirmed @1MP)
- [aug-mechanism](conditioning-row-inject/aug-mechanism.md) — `aug` scope (global, not per-frame); FALSIFIED lever 1 (coupled noise+timestep, per-row impossible); CLARIFIED mask relabels OUTPUT row not guide; fractional-aug contagion data
- [experiments-and-nodes](conditioning-row-inject/experiments-and-nodes.md) — fractional denoise vs fractional strength comparison; experiment order (do NOT jump to hybrid); proto nodes built (`H3SetCondAug` global, `H3SetKeyframeStrength` BROKEN)
