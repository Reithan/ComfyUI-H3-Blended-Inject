<!-- provenance: theory (aug scope source-confirmed @b78cec87; lever 1 GPU-FALSIFIED 2026-08-24; CLARIFIED mask/output-row distinction; fractional-aug contagion GPU data) -->
<!-- verified: 2026-08-24 · mc @d299ea5 · comfy-ref @b78cec87 (+ GPU: MC keyframes blend r40/r60 @1MP, anchors m=0-wrong) -->
# Aug scope, falsified lever 1, mask clarification, and contagion data

Index: [conditioning-row-inject](../conditioning-row-inject.md). What the cond path is and verdict: [what-and-verdict](what-and-verdict.md). Experiments and nodes: [experiments-and-nodes](experiments-and-nodes.md).

## `aug` SCOPE — confirmed from source (model.py @b78cec87)

`visual_cond_noise_aug` is read ONLY in `_cond_video_rows` (line 502) + the segment-timestep table
(581-585). It sets the timestep for the `"cond"` AND `"ref_img"` segments; the **`"text"` segment uses
`t_v` — the USER PROMPT is NOT touched by the global** (answers user worry, 2026-08-24). BUT the global
is SHARED across every keyframe cond row AND every ref2va reference image uniformly. ⚠ **lever 1 (bake
per-keyframe strength into each latent) is FALSIFIED — see the ⚠ FALSIFIED section:** the timestep half of
`aug` is global and not per-row-settable for cond rows, so a latent-only per-keyframe pre-blend renders as
static. There is NO working per-keyframe cond-strength knob; global `aug` is the only native lever.

## ⚠ FALSIFIED (GPU, 2026-08-24) — `aug` is a COUPLED (noise+timestep) pair; lever 1 is BROKEN

**GPU result (user, 2026-08-24):** `H3SetKeyframeStrength` set r40/r60 to strength 0.5/0.45 (per-keyframe
latent noise pre-blend, lever 1). Both anchors trended over the 20 steps toward a **noised/static version
of the injected frame**; from f136 onward the video faded into colored static through the end. Meanwhile a
follow-up setting the **GLOBAL** `visual_cond_noise_aug = 0.5` (both halves, one scalar) blended BOTH
anchors well (r40 slight residual noise, r60 slightly over-denoised/off-inject). **Per-frame latent-only =
broken; global = promising.** This **falsifies the claim that a per-keyframe latent pre-blend approximates
`aug`.**

**Why (source-confirmed, model.py @b78cec87):** `aug` does TWO things with ONE shared scalar, and they must
move together:
1. **Latent noising** (`_cond_video_rows`, model.py:502-510): `r = aug·r + (1−aug)·noise` on the normalized
   cond latent.
2. **Timestep relabel** (model.py:581-584): `seg_t["cond"] = seg_t["ref_img"] = max(t_v, vis_aug)` — the cond
   segment's adaLN timestep label IS `aug` (t-space). This is the half that tells the DiT "this reference
   lives at noise-level (1−aug)," making the noised latent *legible*.

`H3SetKeyframeStrength` / lever 1 noise the latent (half 1) but leave the timestep at the global
`max(t_v, 0.999)≈0.999` = "fully clean" → **the model reads noised content as CLEAN signal and faithfully
reproduces it → persistent colored static.** Exactly the observed failure. The global `aug=0.5` works because
it moves label AND noise together to 0.5, so the DiT correctly reads a half-noised reference.

**Structural fact (no per-cond-row timestep exists):** per-row timestep (`video_rows_t`/`audio_rows_t` via
`rows_to_mod_index`) is built ONLY for the **video/audio TARGET segments** (model.py:636-639). `cond`/`ref_img`
take the `else` branch (model.py:640-641): a SINGLE uniform `row_base` for the whole segment. Combined with the
single-scalar `aug` and the single concatenated `cond_video_latents` list (model_base.py:2183-2196), there is
**no per-keyframe aug and no per-keyframe timestep anywhere in the conditioning path.**

**Consequence — the correction to our plan:**
- Per-frame **fractional denoise CANNOT live in the conditioning path** — it is structurally global there
  (one `aug`, one cond-segment timestep). Lever 1, `H3SetKeyframeStrength`, and "solo per-frame cond-strength"
  (experiment order §1 below) are **not achievable natively**.
- The ONLY native **per-row fractional** control is the **video target rows** via `denoise_mask → video_rows_t`
  (model.py:593-600, 636-637) = **our existing per-row img2img latent path.** That is the home for per-frame
  fractional, confirming the repo goal (guidance in the latent, not conditioning).
- **`aug` treats all 5 visual inject types identically** — F/L keyframes, guide keyframes, image refs, video
  refs all land in ONE `cond_video_latents` list (model_base.py:2186+2191) under one `vis_aug`; audio refs under
  one `aud_aug`. "Row-anchored" (keyframes/guide, positioned by `resolved_frame_index`) vs "not" (refs) affects
  POSITION only, never aug.
- To get the *coupled* behavior per-frame you'd need either (a) a forward-wrapper overriding the cond segment's
  per-row mod index (invasive core patch), or (b) accept **global** `aug` when no conflicting refs are present.
  Global `aug` is the only native lever that actually works — it just isn't per-frame.

## ⚠ CLARIFIED (GPU+source, 2026-08-24) — the mask relabels the OUTPUT row, NOT the guide

**User source-read (correct):** the per-row VIDEO timestep is `rows_t = (1 − m·σ_v).clamp(max=max(t_v,
0.999))` = `min(1 − m·σ_v, 0.999)` (model.py:596; fed by `process_timestep` `v_timestep = denoise_mask·
timestep`, model_base.py:1233). So the denoise_mask IS a genuine **per-frame timestep lever**. BUT it acts on
the **video/OUTPUT segment**, not the cond/guide segment — `cond`/`ref_img` take the uniform `else` branch
(model.py:640-641) and stay at global `max(t_v, vis_aug)`. **So the mask cannot set "per-frame guide aug"; it
sets the per-frame denoise level of the OUTPUT row co-located with the guide.** Different tokens.

Design note the source bakes in: a fully-preserved output row (m→0) lands at `min(1, 0.999) = 0.999` = the
SAME label a default guide cond token carries (model.py:588-589 comment "clamped at the cond timestep for
fully preserved rows"). The output-row-preserve and the cond-timestep are designed to be *consistent*, not to
be the same knob.

**The composite is the other half (and the ghost source).** With a `noise_mask` set, `KSamplerX0Inpaint`
composites the clean latent each step: PRE `x = x·m + scale_latent_inpaint(...)·(1−m)`, POST `out = out·m +
latent_image·(1−m)` — and H3's `scale_latent_inpaint` returns `latent_image` **unscaled** (model_base.py:1240),
so it blends the CLEAN (un-noised) still at weight (1−m) all the way to σ=0 = the ghost. Canonical:
[sampler-loop](../native-h3-mechanism/sampler-loop.md) · [ghost math](../differential-diffusion.md).

**Explains both GPU tests (user, 2026-08-24):**
- **Test 1 — MC latent-inject + mask + SAME still as guide:** ghost AMPLIFIED — anchor near-identical to
  source AND bleeds into neighbors. = the `KSamplerX0Inpaint` composite (clean still each step) PLUS a second
  clean-source pull from the guide cond token. Two clean sources → double ghost.
- **Test 2 — mask + guide, NO latent inject:** injected frames half-smeared + grey-noise "contagion" in
  strips/blobs into neighbors. = the mask relabels those output rows "nearly preserved" (`min(1−m·σ, 0.999)`)
  but there is NO clean latent there to preserve (only gen noise) → model told "low-noise" over actual-noise
  content → grey-noise mismatch that attention smears outward.

**Design-intent recontextualization (user, 2026-08-24 — refined):** correct that official inject nodes write
ONLY to conditioning (guides/keyframes, cond rows). The `denoise_mask` is a SEPARATE per-row **output-inpaint**
lever (H3 A/V temporal inpaint; LanPaint's m=0 masks) whose composite preserves a REAL latent — it was
designed for hard preserve (m∈{0,1}), where clean-composite + clean-label is self-consistent. It is **not**
"meant to augment the guide," and it needs a real clean latent in the row (not a guide) to be meaningful.

**RESOLUTION — where per-frame fractional denoise actually lives:** OUTPUT-row per-frame timestep (mask) **+
clean still injected into that row's latent**, WITHOUT the composite ghost = **exactly our per-row img2img**
(`noise_mask=None` so `KSamplerX0Inpaint` is skipped, per-row timestep supplied through the model path, +
manual denoised correction — [our-architecture](../our-architecture.md)). The guide (global-aug cond token) is an
optional, separate neighbor-anchor. So the fractional home is confirmed to be the **latent + mask** path, and
the high-res under-denoise bug we chase is a property of THAT path (T_N collapse), not of the missing cond knob.

## ⚠ DATA — fractional-aug cond token is itself a CONTAGION SOURCE (GPU, 2026-08-24)

**Test (user):** 0.2MP, OUR latent inject + fractional mask (ghost-free) + the SAME still as a cond token at
**global `aug=0.4`**. Result: neighbors AND anchor blend/denoise well (better than the mask-only guide test),
BUT "contagion" noise blobs propagate out from the injected frames — milder than test 2 (our faithful latent
denoise partly counteracts) but present.

**Diagnosis:** `aug=0.4` means the cond latent is `0.4·clean + 0.6·noise` labeled at `t=0.4` (= "40% denoised"
in the user's correct mental model). So the guide is a **60%-noise reference** and neighbors attending to it
pick up its noise structure → contagion. **Rule: a fractional-aug cond token SPREADS its own noise via
attention.** If a cond token is used at all it should be **clean (`aug→0.999`)**; fractional strength does NOT
belong on the cond token (it belongs on the latent row, where the denoise trajectory lives). This prunes the
cond/latent composition space: **no fractional-aug cond in any hybrid.**

## ⚠ DATA — clean-aug (0.999) cond token FREEZES the anchor (GPU, 2026-08-24)

**Test (user):** clean cond token at `aug≈0.999`. Result: **clean blend into and out of the keyframes, but
the keyframes come out basically UNCHANGED (not denoised)** — exactly as expected for a 0.999 guide: a clean
reference = "this frame is already set" = a hard anchor neighbors blend toward but that does not itself denoise.

**Consequence — the cond `aug` scalar is one-knob-two-jobs (same coupling as the latent mask):** clean
(0.999) → perfect blend, ZERO anchor denoise (re-freeze); fractional (0.4) → anchor denoises, contagion. No
single aug gives blend + anchor-denoise + no-contagion. ⇒ the cond channel alone cannot solve the single
frame — use a clean cond for the BLEND and source the anchor's partial denoise from OUTSIDE the cond channel
(route-2 two-pass, or timed removal). See
[isolated-frame-attention-support](../isolated-frame-attention-support.md).

## ⚠ DATA — SINGLE-PASS DECOUPLE FALSIFIED: clean cond + raw latent = freeze AND contagion (GPU, 2026-08-24)

**Test (user):** the pivotal single-pass decouple — OUR fractional latent inject at `d>0` **co-present** with a
clean cond row on the SAME anchor. Result: **great in/out blend, BUT the inject came out non-denoised (frozen)
AND its hard edges + rough textures spread into the surrounding frames** (contagion). So both halves failed at
once.

**Answers the pivotal question NO** (single-pass decouple is dead — the anchor's denoise cannot be co-driven
in one pass alongside a cond token). But the MECHANISM is NOT settled — two candidates both fit the observable
(anchor looks non-denoised + hard edges spread to neighbors):

- **(A) suppression/freeze** *(weaker)* — the clean cond PREVENTS the latent from denoising; the raw latent
  leaks separately. ⚠ **Disfavored by data:** a true freeze would show anchor `|inp|` NOT moving.
- **(B) row-pinned ATTRACTOR** *(user, 2026-08-24 — preferred)* — the latent DOES undergo the fractional
  denoise, but the row-pinned cond token is a **persistent attractor**: every denoise step pulls the anchor's
  trajectory back toward the originally-injected latent, AND drags neighbors toward it too. The pre-denoise
  textures/shapes spreading through the timeline are the attractor winning each step, not denoise being absent.
  **Supported by the 1MP debug data:** the anchor row `|inp|` moved `0.56→0.70` ("denoises fully, not frozen")
  while predicted x0 kept tracking `clean`=source — the row IS denoising but is continuously re-pulled to source.

**Why the distinction matters for the fix:** under (B) the denoise machinery works — the cond token is a
standing pull. Removing/weakening that cond token late (mode-switch hold-and-release) should let the
already-happening denoise stand → the release path is well-motivated. Under (A) a late release would have to
start denoise from scratch. Either way, a cond token co-present with a raw latent gives freeze-look + contagion;
fractional aug gives denoise-with-contagion — never clean anchor + no contagion.
