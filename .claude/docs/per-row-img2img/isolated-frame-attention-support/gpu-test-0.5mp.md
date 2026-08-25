<!-- provenance: confirmed (user GPU runs @0.5MP 2026-08-24; run data confirmed, mechanism interpretation partly theory) -->
<!-- verified: 2026-08-24 · user GPU runs 0.5MP · repo @b0efef8 era -->
# GPU TEST (0.5MP, 2026-08-24) — support fixes the BLEND, not the anchor; duplication FREEZES

Parent: [isolated-frame-attention-support](../isolated-frame-attention-support.md).

User expanded the single still into supporting rows (= candidate 3), at 0.5MP (a known-failing res that
gens ~4× faster than 1MP — cost is quadratic in resolution):
- **5-frame, f1 keyframe @d=0.5, ease_in_out fade OUT only:** the keyframe popped in over 1-2 frames and
  stayed **basically fully original (under-denoised)**, but then **blended seamlessly** into the trailing
  faded frames.
- **22-frame chunk, fade UP to f9 keyframe @d=0.5 then back OUT to f17:** blend in/out cleaner; blend-IN
  a bit abrupt with **double-exposure/blur** (possibly model motion-blur reconciling motion→still);
  **middle under-denoised** more than expected; fade-out best of all.
- **Both:** expanding a still into 5-17 identical rows makes the model treat it as a **hold/freeze frame
  → motion momentarily pauses.**
- **CROSS-INJECT SIDE-EFFECT (both runs, IMPORTANT):** a SEPARATE single-frame inject at r60 (NOT
  expanded, still isolated) **resolved cleanly — clean blend in/out AND proper denoise** — as a
  side-effect of expanding r40. Expanding ONE anchor helped a DIFFERENT isolated anchor resolve. See
  [temporal-and-contagion](../highres-singleframe-underdenoise/temporal-and-contagion.md) (injects are not
  independent).

**What the two runs show together:**
- **Neighbor-view (blend/contagion): FIXED by support.** Faded neighbor rows blend seamlessly; contagion
  gone. The support half of the hypothesis is CONFIRMED.
- **Anchor-resolution: NOT fixed by SELF-duplication, but HELPED by coherent context.** r40's OWN
  expansion rows (identical) → read as freeze → r40 under-denoises. But r60 RESOLVED once the sequence
  gained the coherent r40 region to attend to. ⇒ anchor-resolution needs **non-identical coherent
  context**; T_N still governs the lone-row ceiling.
- **Duplication is the WRONG KIND of support:** identical rows → (a) motion pause, (b) freeze the
  duplicated anchor. Useful support is non-identical coherent context: cond-token reference, real fade
  region, or (as with r60) other anchored structure.

**Cond-token support: TESTED — the cond `aug` scalar trades anchor-denoise AGAINST contagion, can't give
both** (clean 0.999 → perfect blend + FROZEN anchor; fractional 0.4 → denoises + contagion; one knob,
two jobs). Full data: [aug-mechanism](../conditioning-row-inject/aug-mechanism.md). Clean-cond always-on
hybrid = NOT the missing experiment; it was run.
