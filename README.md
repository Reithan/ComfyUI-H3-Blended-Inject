[![CI](https://github.com/Reithan/ComfyUI-H3-Blended-Inject/actions/workflows/ci.yml/badge.svg)](https://github.com/Reithan/ComfyUI-H3-Blended-Inject/actions/workflows/ci.yml)

# ComfyUI-H3-Blended-Inject

Write keyframes, clips, and audio into a MiniMax H3 audio/video generation and blend the rest of the video into them with an intuitive, img2img-style per-row denoise. You get a smooth anchor with no ghosting, the faint double-image that ComfyUI's native keyframe injection leaves when you hold a frame at partial strength.

#### TL;DR

1. Native H3 keyframe injection leaves that ghosting at any partial denoise, and its strength knob is semantically broken (it drags the frame back to the original on every step).
2. H3-Blended-Inject gives every latent row its own img2img denoise: `0` locks the content exactly, `1` regenerates it freely, and fractional values redraw that fraction.
3. Anchor a clip or still at the strength you want, and the rest of the video blends into it cleanly, with no ghosting and no visible jump where the anchor meets the generated frames.

> [!NOTE]
> **Pre-release.** The nodes work and the author validates them on GPU runs, but the pack is not yet published to the [Comfy Registry](https://registry.comfy.org/); install manually (see [Installation](#installation)).

**Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

> [!TIP]
> Skip to the [Beginner how-to](#beginner-how-to) if you want to get started.

## How it works

<details>
<summary>Expand for the per-row denoise mechanism</summary>

H3 packs a whole audio/video clip into one latent, laid out as rows of frames. This pack gives each row its own denoise value `d` in `[0, 1]`:

- `d = 0`: preserve the source row exactly.
- `d = 1`: generate that row freely (normal H3 sampling).
- `0 < d < 1`: redraw that fraction of the row, img2img-style.

The pack composites an injected clip or still into a clean reference latent, then runs the sampler so that:

1. The sampler lerps each row's initial noise toward the clean reference in proportion to `1 − d`.
2. The noise schedule is remapped per row: a fractional row samples only the tail of the schedule that matches its `d`, so it starts partway down the denoise curve as img2img would.
3. The wrapper feeds the model accurate per-row denoise labels, so the model treats each row at its own strength instead of the whole latent at one.

The sampler runs with `noise_mask = None`, so there is no per-step re-compositing. Native keyframe injection re-pins the frame to the source on every step; at partial denoise those repeated re-pins pile a faint copy of the source onto the result, which is the ghosting. Skipping them keeps the blend clean. A single post-sampling composite restores exact `d = 0` rows.

The full derivation, sampler-class analysis, and H3 internals live in the developer wiki under [`.claude/docs/`](.claude/docs/PER_ROW_IMG2IMG_NOTES.md).
</details>

## The nodes

All three nodes live in the H3 Blended Inject category and share one monadic, chainable style: each `H3 Add …` node appends one entry to an `INJECT_LIST`, and you feed the finished list into the sampler. Chain as many as you like; later entries win on any row they overlap (a hard edge, not a crossfade).

<!-- TODO: screenshot of the example workflow (nodes wired end to end) -->
> *Example workflow screenshot coming soon.*

### H3 Add Inject

The blend. Appends one inject (a video clip or single image, and/or audio) with a denoise envelope the rest of the video blends across.

<!-- TODO: screenshot of the H3 Add Inject node -->
> *Node screenshot coming soon.*

- `inject_at`: latent frame where the inject lands, snapped down to H3's 17-frame grid.
- `start_fade_in` / `start_keyframes` / `end_keyframes` / `end_fade_out`: the envelope, in the clip's own frame indices. Denoise fades in from `1.0`, holds at `min_denoise` across the keyframe region, then fades back out. Half-open `[start_fade_in, end_fade_out)`. For a single still, set all four equal.
- `min_denoise`: the denoise floor during the hold. `0` preserves the frame exactly; `1` fully regenerates it; fractional values anchor it while letting the video move around it.
- `interpolation_type`: `ease_in` / `ease_out` / `ease_in_out` / `linear` / `none` for the fade curves.
- `audio_mode`: `fade` (audio follows the video denoise envelope), `drop` (no audio inject), or `keep` (audio preserved exactly).
- Optional inputs: `images`, `audio`, `vae`, `audio_vae`, and `inject_list` (the chain input).

### H3 Add Guide

Native H3 keyframe conditioning, in the same clean workflow. A monadic clone of ComfyUI's built-in *MiniMax H3 Add Guide* that appends a native keyframe/guide cond entry to the same `INJECT_LIST` chain, so cond-token keyframes and blended injects share one graph and one sampler instead of the native node's separate wiring.

<!-- TODO: screenshot of the H3 Add Guide node -->
> *Node screenshot coming soon.*

- `inject_at`: pixel-frame index to anchor the guide (negative counts from the end), matching the official node.
- `start_percent` / `end_percent`: the step window `[start_percent, end_percent)` during which this guide's cond row is active. At the defaults (`0.0` / `1.0`) it matches the official node; lowering `end_percent` drops the guide's conditioning partway through sampling, so a co-located fractional `H3 Add Inject` keyframe can finish its own denoise without the guide pulling it back toward the source.
- Optional inputs: `image`, `audio`, `vae`, `audio_vae`, and `inject_list`.

> [!NOTE]
> No in-node resize: guide resolution must match the target latent exactly. Resize upstream (e.g. KJNodes "Resize Image v2").

### H3 Blended Sampler

The sampler. A KSampler-Advanced clone with an `inject_list` input that builds the per-row schedule, the clean reference, and the fractional denoise mask, then runs the per-row img2img sampler.

<!-- TODO: screenshot of the H3 Blended Sampler node -->
> *Node screenshot coming soon.*

- Mirrors the core KSampler surface: `model`, `latent_image`, `positive`, `sampler_name`, `scheduler`, `steps`, `cfg`, `noise_seed`.
- `inject_list` (optional): the chain from `H3 Add Inject` / `H3 Add Guide`. Leave it unconnected to sample the latent normally (all rows free, `d = 1`).
- `negative` (optional): H3 is CFG-distilled and runs with no uncond by default; connect a negative conditioning to enable CFG / NRS-style guidance.
- The KSampler-Advanced chaining widgets (add-noise, start/end step, leftover noise) are hidden on purpose: per-row compression would make each row's denoise math wrong on a partial run.

## Beginner how-to

1. Load your MiniMax H3 model, video VAE, and audio VAE as usual.
2. Add an H3 Add Inject node. Connect your clip or image (and audio, if any) plus the matching VAE(s). Set `inject_at` to the latent frame where it should land.
3. Set the envelope in the clip's own frames: `start_fade_in` → `start_keyframes` → `end_keyframes` → `end_fade_out`. For a single still image, set all four equal.
4. Set `min_denoise`. Start around `0.2` to `0.3` for a strong anchor that still blends into the video. `0.0` locks the frame exactly; higher values regenerate more of it.
5. Feed the `inject_list` output into H3 Blended Sampler, pick a sampler and step count, and generate.

> [!TIP]
> `min_denoise` follows the usual img2img feel on H3's schedule: `d ≤ 0.3` retains most of the original, `d ≥ 0.7` is a heavy redraw. Very small values (roughly `d ≲ 0.5/steps`) round to exact preserve, so keep at least a few hundredths if you want the frame to move at all.

## Sampler compatibility

Every major deterministic and stochastic sampler runs natively per-row and is GPU-validated:

| Sampler | Class | Status |
| --- | --- | --- |
| `euler`, `res_multistep`, `dpmpp_2m` | Deterministic | Supported |
| `euler_ancestral`, `dpmpp_2s_ancestral` | Ancestral | Supported |
| `dpmpp_sde`, `dpmpp_2m_sde`, `dpmpp_3m_sde` | SDE | Supported |
| Any other stochastic sampler | n/a | Warns; falls back to a generic path where per-row rows may be corrupted |

Any scheduler works; `simple` is the H3 default. If you pick a stochastic sampler with no native per-row step, the node warns at runtime and you should switch to one of the supported samplers above.

## Installation

Clone into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Reithan/ComfyUI-H3-Blended-Inject.git
```

Restart ComfyUI. This pack targets MiniMax H3 and relies on ComfyUI's bundled `torch`; no extra runtime dependencies are required.

## Contributing

Development setup, the git workflow, the coverage gate, and code style live in
[CONTRIBUTING.md](CONTRIBUTING.md). See [`RELEASING.md`](RELEASING.md) for the Comfy Registry
publishing checklist.

## License

[GPL-3.0](LICENSE).
