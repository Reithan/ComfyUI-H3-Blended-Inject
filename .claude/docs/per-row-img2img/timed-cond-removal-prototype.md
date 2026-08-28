<!-- provenance: confirmed (global prototype GPU-CONFIRMED @0.5MP 2026-08-24; per-guide H3AddGuide GPU-CONFIRMED @0.5MP 2026-08-24 — official-guide coexistence + audio/clip guides + 1MP untested) -->
<!-- verified: 2026-08-24 · comfy-ref @b78cec87 model_base.py:2162-2212, minimax/model.py:318-360,499-524,559-585 · repo @b0efef8 -->
# Timed cond-removal prototype — design + build

Siblings: [isolated-frame-attention-support](isolated-frame-attention-support.md) (why this is build-first),
[aug-mechanism](conditioning-row-inject/aug-mechanism.md) (single-pass failures),
[data-runs.md](highres-underdenoise-model/data-runs.md) (Ψ=+3 backing),
[crux-and-mechanism-2](highres-underdenoise-model/crux-and-mechanism-2.md) (λ(σ) family).

## What it is

Single-pass decoupling attempt — the build-first surviving option. The user's still is wired BOTH as our
fractional latent inject (denoise rate `d`, runs the whole schedule) AND as a native keyframe/guide cond row
(clean `aug≈0.999`). For the first `cond_hold_frac` fraction of steps the cond row provides the clean
reference that gives neighbors good blend (confirmed working in this config); at step `k_sw` the cond row
is REMOVED so it stops re-pulling the anchor toward source; the latent row's fractional denoise — running
the entire time — finishes unimpeded.

Contrast with old hold-and-release: the latent row is NEVER frozen, so there is no truncated-tail denoise.
The hold lives entirely in the cond channel.

## Why this is the priority build

(a) Cond-channel hold-and-release (knob **C**) avoids the B-only impossibility (anchor commits last at every d —
Ψ=+3 data in [data-runs.md](highres-underdenoise-model/data-runs.md)).
(b) Under mechanism (B) attractor, denoise is already ongoing each step; removal lets it stand. Also discriminates
(A) vs (B): anchor snaps post-removal ⇒ (B); must restart ⇒ (A).
(c) The λ(σ) spring family says any source-pull is safe iff gone before render
([crux-and-mechanism-2](highres-underdenoise-model/crux-and-mechanism-2.md)). Removal timing is exactly that.

## Mechanism — source-verified (comfy-ref @b78cec87)

H3 cond rows ride in one dict `out['minimax_payload']` = CONDConstant (model_base.py:2211): keyframe latents
go into `payload["cond_video_latents"]`/`["cond_audio_latents"]` (+ ref latents appended after,
model_base.py:2183-2192), originals kept in `payload["keyframes"]`/`["refs"]`. The packed-sequence
`payload["layout"]` (PackedLayout) is prebuilt ONCE per run (model_base.py:2205-2210).

⚠ TRAP: `layout.signature` does NOT encode keyframes (model.py:567-570); stripping cond latents without
popping `layout` desyncs the sequence silently.
**Safe strip:** forward rebuilds layout from `payload.get("keyframes")/get("refs")` whenever
`payload.get("layout")` is None (model.py:567-571). Per-step strip = payload COPY with `keyframes` +
`layout` popped, cond lists rebuilt from refs only.

## Gating

`model_function_wrapper` receives `timestep` as the RAW 0-1 sigma (conversion to t=σ·1000 happens later
in BaseModel.apply_model; consistent with our σ@tc≈0.88 debug values). Threshold = midpoint of
`sigmas[k_sw-1]` and `sigmas[k_sw]`; release when sig < threshold. `k_sw = round(cond_hold_frac · K)`.

Surface: `cond_hold_frac` FLOAT on `H3InjectSampler` — 1.0=never remove (inert), 0.0=full ablation,
suggested value = printed k_comp.

## Build (commit `b0efef8`)

Implemented as described; effect UNVERIFIED. No tests (pragma'd); diff-cover at 100%.
- `nodes.py`: `cond_hold_frac` optional input; arms `cond_release` dict after sigmas exist
  (`k_sw = round(frac·K)`; threshold = boundary-sigma midpoint; `k_sw<=0` → `inf` = ablation).
- `sampler.py`: wrapper swaps in cached payload COPY (keyframes+layout popped, cond lists from refs)
  when `sigma < threshold`; cached per payload id so cond/uncond streams don't cross.
- Run markers: `[H3_INJECT] timed cond removal armed: ...` and `[H3_INJECT] timed cond removal: keyframe cond rows released at sigma=...`.

## GPU result (2026-08-24, global prototype)

0.5MP; guides at rows 40 and 60 + fractional latent injects; `cond_hold_frac=0.6`. BOTH anchors:
visually-estimated **~0.4 realized denoise** AND **clean blend with neighbors**.

VERDICT: mechanism **VIABLE** — first config to deliver anchor-denoise + clean blend + no contagion.
(A-vs-B discriminator NOT settled; whether anchors "snapped" post-release was not characterized.)

## Limitation: global strip; per-guide direction (UNVERIFIED design)

**Strip is GLOBAL:** pops ALL keyframe cond rows, including FL2VA anchors. Refs are unaffected;
cond lists rebuilt from refs only (model_base.py:2183-2192).

**Per-guide hold control (design, UNVERIFIED):** generalize from "pop all" to "filter by entry" using
`kf["resolved_frame_index"]` as the key; per-entry thresholds; payload COPY with held keyframes + refs;
layout popped on held-set change; cache keyed by released-set.

*(Surface options (a)/(b) superseded — see H3AddGuide settled design below.)*

## Settled design: H3AddGuide node (user-confirmed 2026-08-24)

**Scope note:** H3AddGuide is a cond-channel mechanism; it does not retire the latent-resident
anchor-resolution pursuit (hold-and-release variant; tracked at
[status-and-open-paths](status-and-open-paths.md) open path 1).

**H3AddGuide** ("H3 Add Guide") — third non-prototype node; outward clone of comfy's
MiniMaxH3AddGuide but monadic. Drops `positive` + `latent` inputs; chains through `INJECT_LIST`
like H3AddInject. Sampler partitions the list (Inject vs Guide entries) and appends keyframe cond
dicts to positive conditioning at sample time.

**Per-guide `hold_frac`** (FLOAT, default 1.0 = never removed). Release filter matches by
**OBJECT IDENTITY** — official-node guides at the same frame are never caught; FL2VA safe by
construction. Prototype's global `cond_hold_frac` stays debug-only.

**Resolution:** exact-match, no in-node resize (raise on mismatch, like H3AddInject). User
resizes upstream with kjnodes "Resize Image v2".

**`frame_idx`:** pixel-frame index; negative counts from end. Deliberately different from
inject_at's latent-frame/17-snap — tooltips must be loud about the distinction.

**Video + audio guides** (source-verified, comfy-ref @b78cec87 nodes_minimax_h3.py:162-238):
- Multi-frame image batches anchor as clips snapped to 17k+5 (<5 frames → first image only).
- Audio anchors at same frame, cropped to remaining track (FRAME_RESCALE shared time axis).
- Encode at node time via optional vae/audio_vae; ≥1 of image/audio required.
- Official-node logic moved to sampler time: keyframe append, negative-idx resolution + bounds
  check, audio crop. (Resize dropped — we validate, not resize.)

**Testability:** filter logic factored into pure `filter_released_keyframes(payload,
released_ids)` — CPU-unit-testable dict manipulation; only the thin wrapper glue is pragma'd.

## Build (branch `add-h3-guide-node`, PR #2) — per-guide, GPU-CONFIRMED @0.5MP

GPU result (2026-08-24): 0.5MP, two guides at different `hold_frac` — working. Untested on GPU:
official-guide coexistence, audio guides (`_encode_ref_audio` pragma'd), multi-frame clips, 1MP.
- `schedule.py`: `Guide` dataclass (`eq=False` — identity IS the release key); `InjectList` widened.
- `guides.py` (NEW, pure): `partition_inject_list`, `snap_guide_length` (warns on trim),
  `resolve_frame_index`, `crop_audio_latent`, `release_threshold` (per-guide midpoint math),
  `build_keyframe`, `filter_released_keyframes` (held-keyframes-then-refs; layout popped; input
  never mutated).
- `sampler.py`: filters per step; cache keyed `(id(payload), frozenset(released))`; NOT
  pragma'd — covered by CPU tests with fake `apply_model`.
- `nodes.py`: `H3AddGuide` encodes at node time; `_run_sampler` appends keyframes to positive
  and arms per-guide thresholds after sigmas exist.
