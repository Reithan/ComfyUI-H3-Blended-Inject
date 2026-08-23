<!-- provenance: confirmed (MC source at ~/projects/ComfyUI-H3-Motion-Context-Arbitary-Inserts verified; behavioral claims user-confirmed) -->
<!-- verified: 2026-08-23 · mc @d299ea5 -->
# How Motion Context does it — and why it's stochastic-robust

Source: `~/projects/ComfyUI-H3-Motion-Context-Arbitary-Inserts` (a fork of Motion Context).
Citations are file:line **in the MC repo**.

## MC's mechanism — composite-blend, NOT schedule compression

- **No sampler in the pack.** MC nodes only *attach* a per-token `noise_mask` onto the `LATENT`
  dict (`NestedTensor((video_mask, audio_mask))` — `existing_video_extension.py:633`,
  `h3_masked_bridge.py:442`, `nodes.py:1457`) and hand off to the **stock ComfyUI KSampler** →
  `comfy.sample` → `KSamplerX0Inpaint`. `noise_mask` is **non-None** — the exact opposite of our
  `noise_mask=None` bypass.
- **Every row marches at the FULL global sigma in x-space.** Because a real `noise_mask` is
  present, `KSamplerX0Inpaint` re-composites the known region **every step** (noise clean to the
  current *global* sigma, blend by `1−mask`). MC never compresses the sampler's x-space schedule.
- **The `m·sigma` compression is adaln-label ONLY.** `rows_t = (1 − m·sigma_v).clamp(max=t_pin_v)`
  via `mask_row_values` (`h3_mask_compat.py:145`) feeds only the DiT adaln timestep embedding — how
  the model *interprets* a row — never the x-space schedule. **This is exactly our
  [lever 2](our-architecture.md#the-three-levers).** MC does **not** use our lever 1 (init-lerp)
  or lever 3 (denoised correction).
- H3 specifics: `scale_latent_inpaint` injects preserved rows at cond strength
  `cleans[0] = aug·clean + (1−aug)·noise`, `aug = VISUAL_COND_TIMESTEP ≈0.999`, and undoes the
  audio carry `factor = (sigma_v/sigma_a)/scale` (`h3_mask_compat.py:162`); forward strips/re-adds
  the audio carry (`:144`); preserved rows pinned near t=1 (`:145`). Payload compat emits
  `denoise_mask` + `audio_denoise_mask` CONDRegular (`h3_mask_payload_compat.py:140-159`).

## Why stochastic-robust

Ancestral/SDE renoise (its affine `(1−sigma)` terms) always runs on `x` at the **true global
sigma** for every row; the known content is re-frozen each step by the native composite. No row's
x-space schedule is ever `m·sigma`, so nothing has to survive `sigma → m·sigma`. Our path pushes
the `m·sigma` compression **into** x-space (lever 3 `m·denoised + (1−m)·x`, `noise_mask=None`) — so
ancestral renoise runs on a fabricated per-row schedule that isn't scale-invariant
([bugs · Bug B](bugs.md#bug-b)). Same diagnosis the
[stochastic-recovery theory](stochastic-recovery-theory.md) reaches from the other direction.

## The ghost — where it actually appears (user-confirmed; corrects earlier claim)

**MC DOES support fractional / temporally-faded masks.** The fork adds `H3 Set AV Noise Mask` /
`H3 Clear AV Noise Mask`, so you can paint arbitrary mask values; a temporal FADE works very well
**including on stochastic samplers** (user's own workflow). The subagent's "video binary only" was
the DEFAULT mask builder (`h3_mask_compat.py:161` thresholds ≥0.995→1, ≤0.05→0, quantize 1/256),
**not** the capability. MC is NOT limited to binary video.

The ghost is **narrow (user-confirmed):** it appears only when **`min_denoise` is 0<value<1, and
generally only in the KEYFRAMES.** It is NOT fractional masks in general — it's **sustained
fractional denoise on coherent held anchor content.**

Mechanistic read: MC's fractionality is composite-blend (`x·m + noised_clean·(1−m)` each step at
global σ), which double-exposes the evolving latent against the clean. For a **transient temporal
fade** the offending frames are few and motion hides it → looks great. For a **sustained keyframe
held at 0<min_denoise<1** the same double-exposure persists on a coherent image → visible ghost.
(m=0 preserved rows never ghost: injected at `aug≈clean` consistent with the DiT pin —
`h3_mask_compat.py:162,145`.) Interior softness is VAE round-trip; fixed by pixel-splicing
(`H3 Assemble Interior Insert`, `nodes.py`), not a denoise ghost.

## `min_denoise` is also SEMANTICALLY broken under composite-blend (user-confirmed)

Because the inject is re-composited **every step** (`x·m + noised_clean·(1−m)`), a low value like
`min_denoise = 0.15–0.2` drags the latent ~80% back to the original at every step. The frame can
never diverge far from the source ⇒ output frames look **nearly unchanged from the original**,
*and* the small generated fraction overlays as ghost. So MC's `min_denoise` doesn't behave as an
img2img **strength** knob at all — 0.2 means "stay ≈original + ghost," not "mildly regenerate."
True img2img (our schedule-compression) noises **once** to `d·σ_max` then denoises freely ⇒ the
knob means what it says: change proportional to `d`, coherent, no ghost.

## Raison d'être of THIS repo (user-confirmed)

MC does everything the user wants — temporally-faded masks, stochastic samplers — **as long as
`min_denoise = 0.0`.** The gap is `0 < min_denoise < 1` on the anchor keyframe, which fails **two
ways** at once: (a) it **ghosts**, and (b) the knob is **counter-intuitive** (re-anchors to clean
each step, so it doesn't act like img2img strength). Fixing both — making `min_denoise:X` behave
like intuitive img2img denoise — is the PRIMARY reason this repo exists.

**GPU-CONFIRMED 2026-08-23 (user side-by-side, build @06c6bda):** fading from a keyframe with
`min_denoise > 0`, **MC pops over a ghost frame while Blended stays smooth** — the exact gap,
closed. At `min_denoise = 0.0` (fade from a clip) the two look visually similar (parity achieved).

Also: MC upstream isn't
taking PRs and has heavy churn; MC's node-sprawl and workflows are a QoL problem. See also
[differential-diffusion · ghost](differential-diffusion.md#why-native-inpaint-ghosts-the-math).

## Scope & injection flexibility (design divergence, user-confirmed)

MC was built around **clip continuation** — extending one clip into the next — so it targets injects
at **f0 almost exclusively**, and heavily prioritizes tight A/V line-up to stop audio pops from
**compounding** as the clip grows with each successive extension.

Blended targets **arbitrary injects**: a wider variety of inject lengths at more arbitrary timeline
points, via principled frame-space snapping (per-17 latent-grid reset, single-frame injects,
half-open envelope, tail-local A/V join, `17n+5` length snap).

**The real enabler is better BLENDING, not the lack of extension-chaining.** MC's binary masking
exposes A/V-misalignment pops at hard boundaries, so MC must enforce tight A/V line-up. Blended
fades the boundary, hiding the pop. The guardrail (`warn_audio_tail_alignment` / `is_faded_through`)
warns/snaps only when a misalignment is NOT covered by a fade. This opens uses MC isn't shaped for:
**mid-timeline keyframe injection, mid-context video guiding**, etc. (Design intent — GPU
verification pending; see [status-and-open-paths](status-and-open-paths.md#open-paths).)

## The three design points (the real fork)

| | fractional video | ghost | stochastic |
|---|---|---|---|
| **MC** (composite-blend) | ✅ via composite-blend; temporal fades work | none for transient fades; **ghosts for sustained 0<min_denoise<1 on keyframes** | ✅ free |
| **Ours** (schedule compression, lever 1+2+3, `noise_mask=None`) | ✅ true per-row img2img | none (even sustained) | ❌ breaks |
| **[Theory](stochastic-recovery-theory.md)** (schedule + per-row ancestral) | ✅ true per-row | none | ✅ (if it holds) |

MC gets stochastic + fractional fades for free because it never compresses x-space — but its
composite-blend double-exposes: invisible on transient fades, a visible ghost on a sustained
keyframe (`0<min_denoise<1`). Ours removes the ghost via TRUE per-row img2img (no composite) at
the cost of stochastic; the theory aims for both. "Become MC" would re-introduce the exact ghost
this repo was built to kill.
