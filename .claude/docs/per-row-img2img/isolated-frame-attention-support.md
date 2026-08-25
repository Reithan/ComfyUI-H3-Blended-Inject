<!-- provenance: theory (UNVERIFIED — reframing from user GPU observations 2026-08-24; the attention-dilution mechanism and all candidate fixes are analytical, no experimental confirmation) -->
<!-- verified: 2026-08-24 · comfy-ref @b78cec87; 1MP window-closed data in data-runs.md -->
# The failure is an ISOLATED single fractional frame, not fractional-denoise in general

Reframing (user, 2026-08-24). Sibling docs: [highres-underdenoise-model](highres-underdenoise-model.md)
(the crux / T_N transfer function), [keyframe-two-views-and-knobs](keyframe-two-views-and-knobs.md)
(the four knobs), [conditioning-row-inject](conditioning-row-inject.md) (aug/cond facts).

## The key observation

Our per-row img2img solution **already works well** on:
- **Video injects** (a contiguous run of injected rows), and
- **A SERIES of keyframes with fade regions** around them.

It fails **specifically** on the **isolated single frame at `0 < d < 1`** — the "pop"/"smear". So the
bug is NOT "fractional denoise is broken." It is narrower: a *single unsupported* fractional row.

## Why — attention dilution ∝ tokens ∝ resolution

A single fractional row at `0<d<1` needs two things — originally hypothesized to BOTH depend on
attending to its neighbors (⚠ the GPU test below REFINES this: only #1 does):
1. **Support** — neighbors must attend TO it strongly enough to blend toward it (neighbor-view).
2. **Its own denoise** — the row resolves its noise (hypothesized: by attending to surrounding context;
   GPU-refined below — support does NOT fix this, so anchor-resolution is the row's own T_N compression).

When the frame is **isolated**, it has no adjacent same-content rows reinforcing it. As resolution
rises, **tokens/frame rises** (1MP ≈ 4128 tok/frame vs 0.2MP ≈ 836), so any single frame is a smaller
fraction of the total attention mass → its mutual attention with neighbors is **diluted**. Below some
token budget the frame can neither pull neighbors nor pull enough context to denoise itself → it stays
near its injected (noised) state = the pop. Fade-regions / keyframe-series don't hit this because the
**neighbors provide mutual support** — multiple rows share the content, so the attention mass survives
dilution.

This is the same phenomenon the [T_N(d) transfer function](highres-underdenoise-model.md) captures
(near-identity at 0.2MP, near-STEP at 1MP): the isolation + dilution IS the basin-sharpening seen at
high res.

## The img2img analogy — intuition pump only

Like an SDXL patch too small for faithful inpainting → the lone frame lacks attention mass. H3 has no
single-frame img2img mode; crop/upscale is REFUTED by the paradox (keyframe can't be denoised against a
video that doesn't exist yet). Keep the intuition, not the mechanism.

## GPU TEST (0.5MP, 2026-08-24) — support fixes the BLEND, not the anchor; duplication FREEZES

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
  [temporal-and-contagion](highres-singleframe-underdenoise/temporal-and-contagion.md) (injects are not
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
two jobs). Full data: [aug-mechanism](conditioning-row-inject/aug-mechanism.md). Clean-cond always-on
hybrid = NOT the missing experiment; it was run.

**Implication — the cond channel alone can't solve it; DECOUPLE. Anchor partial denoise MUST be
IN-CONTEXT** (user, 2026-08-24) — the keyframe generates its own temporal context, so pre-denoise
is paradoxical:
- **⚠ Bake-beforehand REFUTED (paradox).** You can't partially-denoise the keyframe against a video that
  doesn't exist yet — the keyframe is what generates that video. A pre-denoised still is denoised against
  nothing (no temporal neighbors), so it has no timeline blend and is just a different clean keyframe.
  The img2img crop/upscale story was an SDXL illustration; **H3 has NO single-frame img2img mode.**
  Cyclic/paradoxical — dead end.

The remaining paths keep the clean cond for BLEND and denoise the anchor IN-CONTEXT:
- **⚠ Single-pass decouple FALSIFIED (GPU, 2026-08-24):** clean cond + OUR fractional latent+mask on the
  SAME row, together, was TESTED — great in/out blend but the anchor came out **non-denoised-looking AND its
  hard edges contaminated neighbors**. Mechanism NOT settled — likely a row-pinned ATTRACTOR (the latent DOES
  denoise but the cond token re-pulls it + neighbors toward the raw inject each step; favored by `|inp|` moving
  while x0 tracks source), possibly suppression. Either way dead as a single-pass. Data:
  [aug-mechanism](conditioning-row-inject/aug-mechanism.md) ("SINGLE-PASS DECOUPLE FALSIFIED").
- **Route-2 two-pass:** pass A the anchor denoises IN-CONTEXT (fractional, real gen) → capture its
  realized content; pass B re-inject that as a CLEAN anchor → clean neighbor blend, no contagion, no
  re-freeze. Denoises against the real (pass-A) video → NOT paradoxical. Caveat: if pass A under-denoises
  via T_N, pass B just re-injects an under-denoised frame; gain only if pass A's in-context denoise is
  adequate (r60 hints context helps).
- **Cross-resolution two-pass (workaround, back pocket):** pass A at 0.2MP (window open) → extract
  realized keyframe → upscale → re-inject CLEAN into the 1MP run. Avoids the paradox and T_N; cost ≈
  few %. USER VERDICT (2026-08-24): viable but workaround family; back pocket while a direct 1MP fix
  is pursued. Risks: upscale softness; realized frame must read compatible in the fresh 1MP context.
- **Timed removal / anchor-then-release:** clean reference EARLY so neighbors compose around it, release
  to the latent+mask fractional denoise in the tail. Single-pass; truncated-tail denoise + a release
  transition to manage.

## Candidate fixes (all UNVERIFIED)

Ordered cheap→invasive. All aim at the same target: **restore the isolated frame's attention weight**.

1. **Clean cond token (aug≈0.999)** — ⚠ TESTED: clean blend in/out but the keyframe comes out
   **unchanged (not denoised)** = **re-freeze CONFIRMED** — a clean guide is a hard "this frame is set"
   reference. Good for blend, useless for anchor denoise. Use it for the blend, get the denoise elsewhere
   (bake-beforehand / route-2 / timed-removal). The **always-on hybrid** from
   [keyframe-two-views](keyframe-two-views-and-knobs.md) in clean form.
2. **Cond token with TIMED REMOVAL** (user) — inject ONLY the cond row at normal `aug≈0.999`, then
   **remove that cond row at `s·(1−d)`** (or when the frame's measured noise level reaches `d`). Gives
   full clean support early, releases before it can re-freeze the anchor. Open problem: no clean metric
   for "noise level reached d" mid-sample; step-count proxy `s·(1−d)` is the tractable version.
   *Structural note:* timed-removal IS route-1's hold-and-release on knob **C** (cond token) not knob **B**
   (latent mask), so it inherits the [data-runs.md](highres-underdenoise-model/data-runs.md) Ψ=+3 result:
   anchor commits last at EVERY d under B-only control → the impossibility **backs** timed-removal. Knob-B
   attractor ≡ Fable's λ(σ) source-spring ([crux-and-mechanism-2](highres-underdenoise-model/crux-and-mechanism-2.md));
   ghost-free condition (λ→0 before render) is timed-removal's safety bound; k_comp gives k_sw.
   Design + build: [timed-cond-removal-prototype](timed-cond-removal-prototype.md).
3. **Temporal duplication into support frames** — ⚠ TESTED (0.5MP, above): fixes the blend but the model
   reads identical rows as a **FREEZE** (motion pause + double-exposure) and it does NOT resolve the
   anchor. Self-defeating for anchor-resolution — a demoted dead-end, kept as a negative result.
4. **Route-3 attention-logit boost** — bias neighbor→anchor logits directly. Caveat (per two-views doc):
   this boosts attention to the anchor's ACTUAL partially-noised state, so it aids support but not the
   "see it as clean" half of the ideal.
5. **img2img crop/upscale/denoise/downscale** — ⚠ REFUTED (paradox — see analogy note above;
   H3 has no single-frame img2img mode).

## What this reframing changes

- Two distinguishable failures: neighbor-blend (FIXED by support) and anchor-resolution. Anchor-
  resolution is helped by NON-identical coherent context (r60) but NOT by self-duplication (r40 froze).
- Blend: a **clean cond token** works (confirmed). Duplication freezes (route 3); fractional cond
  contaminates. The cond `aug` scalar can't give blend + anchor-denoise + no-contagion at once.
- Anchor-denoise must happen IN-CONTEXT (bake-beforehand is REFUTED — paradox; single-pass decouple is
  FALSIFIED — clean cond + raw latent = freeze + contagion). Surviving in-context options: route-2 two-pass
  (blocked at 1MP unless pass A gets coherent context — r60) and timed-removal / mode-switch hold-and-release.
  LATENT-path d-tuning (T_N-corrected realized-m): window CLOSED @1MP per
  [data-runs.md](highres-underdenoise-model/data-runs.md) — those runs had no cond token, so they WERE the
  latent-only probe; 0.68 locks, 0.75/0.78 smear, no coherent middle. Valid only at ≤0.5MP or as a component
  inside a release phase.
- **Blurred-reference: REFUTED standalone (user, 2026-08-24).** Blurred reference = blurred output;
  attractor pulls every step including render phase. At most an ablation inside timed-removal.
- **Evenly-spaced-keyframes reframe (user, 2026-08-24).** More evenly-spaced high-quality keyframes
  convert the broken isolated case into the working series case. When no extra stills exist, cross-res
  pass A can manufacture realized neighbors as faded support (2d-via-2a synthesis).
- Anchor-resolution ceiling is still governed by T_N; if in-context support is insufficient, a dedicated
  fix (T_N-corrected realized-m or route-2 two-pass) is the fallback. Track it in
  [highres-underdenoise-model](highres-underdenoise-model.md) (the T_N model + fix-strategies).
