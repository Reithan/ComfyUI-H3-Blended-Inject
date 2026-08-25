<!-- provenance: theory (analysis + source-grounded; what the cond-row path is, verdict, and interop finding; source confirmed @b78cec87) -->
<!-- verified: 2026-08-24 · mc @d299ea5 · comfy-ref @b78cec87 (+ GPU: MC keyframes blend r40/r60 @1MP, anchors m=0-wrong) -->
# What the conditioning-row path is, verdict, and interop

Index: [conditioning-row-inject](../conditioning-row-inject.md). Fade and decoupler options: [fade-and-decoupler](fade-and-decoupler.md). Aug mechanism and falsifications: [aug-mechanism](aug-mechanism.md).

## What the conditioning-row path actually is

MC's **`MiniMaxH3CustomKeyframes`** (non-masked) is a pure **CONDITIONING→CONDITIONING** node
(`mc nodes.py:844-846`). It VAE-encodes each still to a `[B,C,1,H,W]` still-latent and writes
`{"resolved_frame_index", "latent"}` dicts into the conditioning under the key **`minimax_keyframes`**
via `conditioning_set_values` (`mc nodes.py:964-970`). It writes **no** `samples`/`noise_mask` and
does **not** touch the target latent (the `latent` input is only a shape/frame-count template).

Native consumption (comfy-ref):
- `comfy/model_base.py:2183-2212` — `extra_conds` lifts `minimax_keyframes` into `minimax_payload`
  (`keyframes`, `cond_video_latents`, `cond_audio_latents`) as a `CONDConstant`.
- `comfy/ldm/minimax/model.py:340-361` — `PackedLayout` appends a **`"cond"` segment** of video
  rows on the target spatial grid at `cond_t = cursor + FRAME_RESCALE·resolved_frame_index`
  ("fl2va: keyframe cond rows right after text, sharing the target spatial grid"). Consumed as cond
  rows in `_forward` via `_cond_video_rows` (`model.py:499-512, 648`).

**So a conditioning-row inject = extra native context tokens the DiT attends to — NOT part of the
denoised output grid.** But "context" undersells it (user-corrected, 2026-08-23): with the default
`visual_cond_noise_aug = VISUAL_COND_TIMESTEP ≈ 0.999` (`model.py:32,499-512`) the cond row is a
**near-clean** latent carrying a near-1.0 timestep label (`max(t_v, 0.999)`, `model.py:583-585`).
The DiT runs **full self-attention with `mask=None`** (`model.py:195`), so the *generated* target
row at the same time position (`cond_t`) attends to that clean reference and **copies it — a
near-perfect reproduction where directed** (user-observed; ≈ as strong as a masked latent inject).
Cond rows are never read out (`img_update` marks them `False`, `model.py:652-655`); only the
`video`/`audio` target segments are (`model.py:715-726`). So the frame isn't *frozen* like the
masked latent path (`mc nodes.py:1190-1462`) — it's *regenerated to match*, and that same clean
reference **also steers surrounding output holistically** (user-observed strength). The cost, per
the user: cond rows are ordinary full-attention tokens, so they **consume H3's finite attention
budget** — too many refs / too long a prompt and adherence degrades. Latent injects spend **zero**
extra attention (they ride output rows we already generate), which is why the user defaults to
latent and reserves conditioning for storyboard / character / anatomy refs.

**Strength control is a SINGLE GLOBAL scalar, not per-keyframe.** `_cond_video_rows` applies one
`aug` uniformly to every entry in `cond_video_latents` (`model.py:499-512`; loop var `z` is
per-keyframe but `aug` is loop-invariant). The keyframe dicts carry only `resolved_frame_index` +
`latent` (`mc nodes.py:964-967`) — no per-entry strength/weight. There is **no attention mask and
no per-token weight** over the cond segment (`mask=None`, `model.py:195`), and standard ComfyUI
conditioning modifiers do **not** reach it: `strength` scales only `c_crossattn` CONDRegular, not
the `CONDConstant` payload (`model_base.py:2171,2212`); area/mask flow only via `denoise_mask` to
*target* rows; `start_percent`/`end_percent` gate the whole payload (layout + all keyframes) as one
all-or-nothing swap, not a smooth per-row fade.

## Verdict: it's a different tool, not a substitute — and building it duplicates MC

1. **It cannot serve this repo's raison d'être.** Our goal is *intuitive img2img `min_denoise`* —
   anchor keyframed pixels and blend around them ([repo raison d'être](../../PER_ROW_IMG2IMG_NOTES.md)).
   The cond-row path has **no denoise/preserve semantics at all**; it's soft reference guidance.
   Toggling our latent inject over to cond rows would *delete* the one capability we exist to
   provide.
2. **It's sampler-agnostic & ghost-free for free** (no x-space composite → nothing to survive
   `σ→m·σ`), i.e. it sidesteps [Bug B](../bugs.md#bug-b). But it buys that by giving up pixel fidelity,
   not by solving our problem.
3. **Reimplementing it = "become MC."** MC already ships this exact node, and its output is plain
   CONDITIONING keyed on a NATIVE H3 key. That collides with the standing constraint *do not recreate
   MC* ([motion-context-comparison](../motion-context-comparison.md)).

## The actually-useful conclusion: interop is already free (no new node needed)

Because `minimax_keyframes` is a **native** conditioning key and our `H3InjectSampler` forwards
`positive` **unmodified** into `sample_custom` (`nodes.py:922` → `_run_sampler`; no key stripping),
a user can wire **`H3 Custom Keyframes (MC) → positive → H3 Inject Sampler (ours)`** TODAY. The cond
rows ride through our sampler and are consumed by `extra_conds` regardless of our per-row latent
machinery. So the "benefit" is available without us writing a node — and it *composes* with our
latent anchor rather than replacing it.
