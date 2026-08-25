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

(a) It is route-1's bang-bang hold-and-release on knob **C** (cond token) rather than knob **B** (latent
mask), inheriting the Ψ=+3 last-to-commit structural argument from
[data-runs.md](highres-underdenoise-model/data-runs.md): the anchor commits last at EVERY d under B-only
control, so that impossibility BACKS the cond-channel approach.

(b) Under mechanism **(B) row-pinned attractor** ([aug-mechanism](conditioning-row-inject/aug-mechanism.md))
the anchor's denoise is already happening every step; removal lets it stand. The run doubles as the
(A)-vs-(B) discriminator: anchor snaps denoised after removal ⇒ (B); anchor must denoise from scratch in
the tail ⇒ (A).

(c) The λ(σ) source-spring family ([crux-and-mechanism-2](highres-underdenoise-model/crux-and-mechanism-2.md))
says any source-pull is safe iff it is gone before the render phase. Removal timing is exactly that
condition; measured k_comp gives the concrete k_sw.

## Mechanism — source-verified (comfy-ref @b78cec87)

H3 cond rows ride in one dict `out['minimax_payload']` = CONDConstant (model_base.py:2211): keyframe latents
go into `payload["cond_video_latents"]`/`["cond_audio_latents"]` (+ ref latents appended after,
model_base.py:2183-2192), originals kept in `payload["keyframes"]`/`["refs"]`. The packed-sequence
`payload["layout"]` (PackedLayout) is prebuilt ONCE per run (model_base.py:2205-2210).

⚠ TRAP: `layout.signature` encodes only `(text_len, latent_t, lat_h, lat_w, audio_t)` — NOT keyframes
(model.py:567-570) — so stripping cond latents while leaving a stale layout desyncs the sequence and
passes the signature check silently.

**Safe per-step strip:** forward REBUILDS the layout from `payload.get("keyframes")/get("refs")` whenever
`payload.get("layout")` is None (model.py:567-571), and `keyframes` has no other consumer in model.py.
Per-step safe strip = hand forward a payload COPY with `keyframes` popped, `layout` popped, and cond lists
rebuilt from refs only (refs preserved). Per-step layout rebuild on release steps is minor CPU; acceptable
for the prototype.

## Gating

`model_function_wrapper` receives `timestep` as the RAW 0-1 sigma (conversion to t=σ·1000 happens later
in BaseModel.apply_model; consistent with our σ@tc≈0.88 debug values). Threshold = midpoint of
`sigmas[k_sw-1]` and `sigmas[k_sw]`; release when sig < threshold. `k_sw = round(cond_hold_frac · K)`.

## Surface (prototype, debug branch)

New optional `cond_hold_frac` FLOAT input on `H3InjectSampler`:
- default `1.0` = never remove (native behavior, code path fully inert)
- `0.0` = cond removed from step 0 (no-reference ablation)
- values near the printed k_comp suggestion = the experiment to run

## Build (commit `b0efef8`)

Implemented exactly as above; effect UNVERIFIED until GPU runs. No tests (all new paths
pragma'd / inside the pragma'd `_run_sampler`); diff-cover gate passed at 100%.
- `nodes.py` — `cond_hold_frac` optional input on `H3InjectSampler`; threaded through `sample()`
  → `_run_sampler`, which arms a mutable `cond_release` dict after the sigma schedule exists
  (`k_sw = round(frac·K)`; threshold = boundary-sigma midpoint; `k_sw<=0` → `inf` = ablation).
- `sampler.py` — `build_conditioning_wrapper(..., cond_release=...)`: when armed and
  `sigma < threshold`, swaps in a cached payload COPY (keyframes+layout popped, cond lists
  rebuilt from refs; cached per payload id so cond/uncond streams don't cross).
- Run markers to watch: `[H3_INJECT] timed cond removal armed: ...` (arming, prints k_sw +
  threshold) and `[H3_INJECT] timed cond removal: keyframe cond rows released at sigma=...`
  (first release step, once per payload).

## GPU result (2026-08-24, global prototype)

0.5MP run (previously-broken resolution, ~4x faster than 1MP); guides at rows 40 and 60 plus our
fractional latent injects; `cond_hold_frac=0.6`. BOTH anchors: visually-estimated **~0.4 realized
denoise on the anchor frame** AND **clean blend with neighbors**.

VERDICT: mechanism **VIABLE in isolation** — first config to deliver anchor-denoise + clean blend + no
contagion at a broken resolution. (A-vs-B discriminator NOT settled — user reported realized denoise;
whether anchors "snapped" post-release was not characterized.)

## Limitation: global strip; per-guide direction (UNVERIFIED design)

**Strip is GLOBAL.** Popping `payload["keyframes"]` removes ALL keyframe cond rows — including FL2VA
first/last-frame anchors (comfy labels the keyframes list "fl2va: keyframe cond rows",
model.py:342-343 area). Refs are NOT affected — the strip preserves `payload["refs"]` and rebuilds
cond lists from refs only (model_base.py:2183-2192 keeps keyframe and ref latents in separate lists).

**Per-guide hold control (design, UNVERIFIED):** each keyframe entry carries `kf["resolved_frame_index"]`
(model.py:345 area) — a stable per-guide matching key. Generalize from "pop all" to "filter by entry":
per-entry release thresholds; payload COPY with keyframes filtered to still-held entries; cond lists
rebuilt from held-keyframes + refs (preserving construction order); layout popped whenever the held set
differs from the previous step; cache keyed by released-set so cond/uncond streams don't cross.

*(Surface options (a)/(b) superseded — see H3AddGuide settled design below.)*

## Settled design: H3AddGuide node (user-confirmed 2026-08-24)

See also: local execution plan `.claude/plans/add-h3-guide-node.md` (gitignored, session-local).

**Scope note (user, 2026-08-24):** H3AddGuide is a COND-channel mechanism and useful functionality
in its own right — it does NOT retire the latent-resident pursuit of the same anchor-resolution
problem (likely a hold-and-release variant; new tests + possibly a new node). Track in
[status-and-open-paths](status-and-open-paths.md) open path 1.

**H3AddGuide** ("H3 Add Guide") — third non-prototype node; outward clone of comfy's
MiniMaxH3AddGuide but monadic. Drops `positive` + `latent` inputs; chains through `INJECT_LIST`
like H3AddInject. Sampler partitions the list (Inject vs Guide entries) and appends keyframe cond
dicts to positive conditioning at sample time.

**Per-guide `hold_frac`** (FLOAT, default 1.0 = never removed = official behavior). Sampler
computes per-guide sigma thresholds; wrapper filters `payload["keyframes"]` per step. Release
filter matches by **OBJECT IDENTITY** (not frame index) — official-node guides at the same frame
are never caught. FL2VA and official guides = hold-forever, safe by construction. No global
`cond_hold_frac` on the release branch (prototype knob stays debug-only).

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

GPU result (2026-08-24, user): 0.5MP, two guides at DIFFERENT `hold_frac` values — working fine.
User passed on larger sweeps for now. Still untested on GPU: official-guide coexistence (mixed
official Add Guide + ours in one workflow — post-release layout rebuild with a held foreign
keyframe), audio guides (`_encode_ref_audio` pragma'd), multi-frame clip guides, 1MP. Layout:
- `schedule.py` — `Guide` dataclass (`eq=False`: identity IS the release key); `InjectList`
  widened to `list[Inject | Guide]`.
- `guides.py` (NEW, pure) — `partition_inject_list`, `snap_guide_length` (warns on trim, unlike
  the silent official node), `frame_count_for_rows`, `resolve_frame_index`, `crop_audio_latent`,
  `release_threshold` (per-guide; same midpoint math as prototype), `build_keyframe`,
  `filter_released_keyframes` (held-keyframes-then-refs, exact model_base predicates incl. refs
  video `"latent" in r`; layout popped; input never mutated).
- `sampler.py` — wrapper takes `guide_release={entries: [(id(kf), threshold)], cache}`; filters
  per step, cache keyed `(id(payload), frozenset(released))` so cond/uncond never cross. NOT
  pragma'd — covered by CPU tests with fake apply_model.
- `nodes.py` — `H3AddGuide` node (encode at node time; snap/trim); `sample()` partitions chain,
  resolves guides vs target (exact-resolution gate, bounds, audio crop), builds keyframe dicts
  ONCE (identity!); `_run_sampler` appends them to positive via `conditioning_set_values` and
  arms thresholds after sigmas exist. Run markers: same `[H3_INJECT] timed cond removal` prints,
  now with per-guide counts.
