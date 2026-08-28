"""ComfyUI node definitions for H3 Blended Inject.

Three nodes are defined:

- :class:`H3AddInject`: Appends one inject to an ``INJECT_LIST`` chain.  Chainable.
- :class:`H3AddGuide`: Appends one native keyframe/guide cond entry to the same chain
  (monadic clone of comfy's "Add Guide for MiniMax H3", with per-guide timed cond
  removal via ``hold_frac``).  Chainable.
- :class:`H3InjectSampler`: KSampler Advanced clone that accepts an ``INJECT_LIST`` and
  applies per-row img2img inject (and native guide cond rows) during sampling.

``comfy`` imports are lazy (inside methods) so this module imports without ComfyUI present,
enabling CPU-side unit tests.

``NODE_CLASS_MAPPINGS`` and ``NODE_DISPLAY_NAME_MAPPINGS`` are defined at module bottom and
re-exported from the top-level ``__init__.py``.
"""

from __future__ import annotations

import math
from typing import Any

from comfyui_h3_blended_inject import grid, guides
from comfyui_h3_blended_inject.guidance import resolve_guidance
from comfyui_h3_blended_inject.sanitize import (
    check_resolution,
    sanitize_audio,
    snap_inject_at,
    snap_length_down,
    validate_envelope_indices,
    warn_audio_tail_alignment,
    warn_audio_tick_alignment,
)
from comfyui_h3_blended_inject.schedule import Guide, Inject, InjectList, merge_schedule

# ComfyUI custom type string for the inject-list wire type.
INJECT_LIST = "INJECT_LIST"


def _encode_ref_audio(audio_vae: Any, audio: Any) -> tuple[Any, int]:  # pragma: no cover
    """Encode an audio waveform into an H3 audio latent.

    Mirrors the ``_encode_ref_audio`` helper from
    ``comfy_extras/nodes_minimax_h3.py``.  Resamples ``audio`` to the VAE's
    native sample rate then encodes via the audio VAE.

    Parameters
    ----------
    audio_vae:
        H3 audio VAE instance (from a ``VAELoader`` node).
    audio:
        AUDIO dict ``{"waveform": Tensor, "sample_rate": int}``.

    Returns
    -------
    tuple[Any, int]
        ``(latent, latent_length)`` where ``latent`` has shape ``[1, 32, 2, T]``
        and ``latent_length = T``.

    Notes
    -----
    ``torchaudio`` is imported lazily here.  This function requires a real
    H3 audio VAE and is therefore not exercised in CPU unit tests.
    """
    waveform = audio["waveform"]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        # Lazy import: torchaudio is only present in the ComfyUI runtime; when the
        # waveform is already at the VAE rate (sanitize_audio resamples upstream)
        # this path — and the dependency — is skipped entirely.
        import torchaudio  # noqa: PLC0415

        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, z.shape[-1]


def _unpack_av(samples: Any) -> tuple[Any, Any | None]:
    """Unpack an H3 NestedTensor or plain tensor into ``(video, audio_or_None)``.

    H3 FLOW_AV latents arrive as ``NestedTensor((video, audio))``.  Plain tensors
    (non-nested CPU-test paths or video-only runs) pass through unchanged with
    ``audio=None``.
    """
    if getattr(samples, "is_nested", False):
        components = list(samples.unbind())
        video = components[0]  # [B, C_v, T, Hl, Wl]; row axis dim 2
        audio = components[1] if len(components) > 1 else None  # [B, C_a, 2, audio_t]
    else:
        video = samples
        audio = None
    return video, audio


def _run_sampler(  # pragma: no cover
    model: Any,
    schedule: Any,
    latent_image: dict[str, Any],
    noise_seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive: Any,
    negative: Any | None,
    samples: Any,
    target_rows: int,
    audio_ticks: int,
    guide_entries: list[tuple[dict[str, Any], float]] | None = None,
) -> tuple[dict[str, Any]]:
    """GPU/ComfyUI per-row img2img sampling pipeline — not CPU-testable.

    Implements the per-row img2img design (see ``sampler.py`` and ``composite.py``):

    1. Build the *clean reference* — the target latent with all inject video/audio content
       composited in (:func:`~comfyui_h3_blended_inject.composite.build_clean_reference`).
       This is passed to ``sample_custom`` as the ``latent_image`` so ComfyUI's global
       ``noise_scaling`` produces ``x_global = sigma_max*eps + (1-sigma_max)*clean``.
    2. Build the *fractional per-row denoise mask*
       (:func:`~comfyui_h3_blended_inject.mask.derive_fractional_mask`) and pack it to the
       sampler's flat layout.  The schedule-tail remap loop publishes the per-step per-row
       label mask as pooled DiT conditioning via
       :func:`~comfyui_h3_blended_inject.sampler.build_conditioning_wrapper`.
    3. Wrap the base k-diffusion ``sampler_function`` with
       :func:`~comfyui_h3_blended_inject.sampler.build_per_row_sampler_function`, which runs
       each fractional row over the last ``d``-fraction of the schedule (read from a dense
       ``steps²+1`` sigma grid), composites the clean reference once at step 0, and scales each
       row's per-step delta onto its own compressed tail.  Audio rows use the sigma-shifted
       audio schedule.  The observer-label K/V split
       (:func:`~comfyui_h3_blended_inject.observer_split.install_observer_split`) is installed
       on top when fractional rows exist.
    4. Run ``comfy.sample.sample_custom`` with ``noise_mask=None`` (no native compositing → no
       compounding re-pin ghost), then apply the binary exact-preserve overwrite
       (:func:`~comfyui_h3_blended_inject.composite.post_composite_preserve`) for ``m == 0``
       rows and audio-preserve ticks.

    Guides (``guide_entries``: pre-built keyframe dicts + per-guide ``hold_frac``) are
    appended to the positive conditioning's ``minimax_keyframes`` here (on a COPY — user
    conds are never mutated), and their per-guide release thresholds are armed on the
    conditioning wrapper once the sigma schedule exists (per-guide timed cond removal;
    see :func:`~comfyui_h3_blended_inject.sampler.build_conditioning_wrapper` and
    :mod:`~comfyui_h3_blended_inject.guides`).

    Notes
    -----
    This function is pragma'd because it requires real H3 latents, a running ComfyUI
    environment, and (typically) a GPU.  GPU verification is tracked as task #19.
    """
    # model.clone() is called before any comfy.* import so a plain-object model (CPU tests)
    # raises AttributeError before an unavailable comfy package would be imported.
    m = model.clone()

    import comfy.sample
    import comfy.samplers
    from comfy.nested_tensor import NestedTensor

    from comfyui_h3_blended_inject.composite import (
        build_clean_reference,
        post_composite_preserve,
    )
    from comfyui_h3_blended_inject.mask import derive_fractional_mask
    from comfyui_h3_blended_inject.sampler import (
        _NATIVE_ROW_STEPS,
        build_conditioning_wrapper,
        build_per_row_sampler_function,
        quantize_denoise,
        sampler_is_stochastic,
        scale_packed_audio,
    )

    # --- 0. Append guide keyframe cond rows to the positive conditioning. ---
    # conditioning_set_values returns a new conditioning list with copied dicts, so the
    # user's conds are never mutated.  The keyframe dicts keep their object identity
    # through comfy's cond processing into payload["keyframes"], which is what the
    # timed-removal release filter matches on.
    if guide_entries:
        import node_helpers

        existing = list(positive[0][1].get("minimax_keyframes", [])) if positive else []
        positive = node_helpers.conditioning_set_values(
            positive,
            {"minimax_keyframes": existing + [kf for kf, _ in guide_entries]},
        )

    # --- 1. Split the target latent into components (H3 FLOW_AV is NestedTensor). ---
    _samples = latent_image["samples"]
    is_nested = getattr(_samples, "is_nested", False)
    video, audio = _unpack_av(_samples)

    # --- 2. Clean reference: target with all inject content composited in. ---
    # Fractional rows img2img *from* this content; m==1 rows ignore it; m==0 rows are
    # restored from it after sampling by post_composite_preserve.
    clean_video, clean_audio = build_clean_reference(
        video, audio, schedule, target_rows, audio_ticks
    )
    clean_components = (clean_video,) if clean_audio is None else (clean_video, clean_audio)
    clean_nested = NestedTensor(clean_components) if is_nested else clean_video

    # The clean reference is used in TWO spaces that do NOT match:
    #   - sample_custom's latent_image: passed RAW (clean_nested); comfy applies
    #     MiniMaxH3.process_latent_in internally, which scales the AUDIO slice by audio_scale
    #     (shift/audio_shift = 4.0; video scale_factor is 1.0) → x_global carries scaled audio.
    #   - the per-row init lerp's clean term (clean_packed): must live in that SAME scaled
    #     space, since the lerp blends x_global toward it. So scale clean_packed's audio slice
    #     by audio_scale here (see sampler.scale_packed_audio). Skipping this left fractional
    #     (0<m<1) audio rows img2img-ing from a 4x-mismatched reference → decoded static.
    import comfy.utils as _comfy_utils

    clean_packed, latent_shapes = _comfy_utils.pack_latents(clean_components)
    # Audio rows run the schedule-tail remap on the sigma-shifted audio schedule; capture the
    # packed video-prefix length (the modality boundary) and the model's sigma shifts here.
    video_element_count: int | None = None
    shift_v, shift_a = 12.0, 3.0
    if clean_audio is not None:
        audio_scale = float(getattr(m.model.model_sampling, "audio_scale", 1.0))
        n_video_elems = math.prod(int(d) for d in latent_shapes[0][1:])
        clean_packed = scale_packed_audio(clean_packed, n_video_elems, audio_scale)
        video_element_count = n_video_elems
        _dm = m.get_model_object("diffusion_model")
        shift_v = float(getattr(_dm, "sigma_shift_video", 12.0))
        shift_a = float(getattr(_dm, "sigma_shift_audio", 3.0))

    # --- 3. Fractional per-row denoise mask, packed to the same flat layout. ---
    video_shape = tuple(int(d) for d in video.shape)
    audio_shape = tuple(int(d) for d in audio.shape) if audio is not None else None
    frac_components = derive_fractional_mask(
        schedule,
        target_rows,
        audio_ticks,
        video_component_shape=video_shape,
        audio_component_shape=audio_shape,
        nested_factory=lambda v, a: (v, a),
    )
    if isinstance(frac_components, dict):
        frac_components = (frac_components["video_mask"],)
    m_packed, _ = _comfy_utils.pack_latents(frac_components)
    # Snap m to the native 1/256 mask grid BEFORE it feeds any lever: the DiT's
    # _token_grid_masks quantizes with ceil(m*256)/256, so pre-quantizing keeps the
    # init lerp (lever 1), the DiT labels (lever 2), and the denoised correction
    # (lever 3) on the *identical* per-row m. See sampler.quantize_denoise.
    m_packed = quantize_denoise(m_packed).to(device=clean_packed.device)

    # --- 4. Build the schedule-tail remap config + install the conditioning wrapper. ---
    # Each fractional row runs the LAST d-fraction of the schedule's σ-values stretched over
    # ALL steps; the sampler loop publishes the per-step per-row label mask into
    # schedule_tail_cfg["pooled_current"] (truthful t_row = 1 − σ_row) and does its own per-row
    # step scaling, so the wrapper does NO denoised correction. See build_per_row_sampler_function.
    import torch

    schedule_tail_cfg: dict[str, Any] = {
        "pooled_ones": m.model._denoise_mask_values(torch.ones_like(m_packed), latent_shapes),
        "make_pooled": lambda c_vec: m.model._denoise_mask_values(c_vec, latent_shapes),
    }
    # Per-guide timed cond removal: mutable state shared with the wrapper.  Its "entries"
    # (keyframe id → sigma threshold) are filled in below once the sigma schedule exists;
    # until then — and for hold_frac=1.0 guides, forever — the wrapper path is inert.
    guide_release: dict[str, Any] = {}
    m.model_options = {
        **m.model_options,
        "model_function_wrapper": build_conditioning_wrapper(
            schedule_tail_cfg, guide_release=guide_release
        ),
    }

    # Observer-label K/V split: every DiT block presents fractional inject rows' K+V under the
    # official label m (observed ratio pinned at m:1) while Q/residual/MLP keep the truthful
    # remapped label. Populates schedule_tail_cfg["observer"]; inert (returns False) when there
    # are no fractional rows. See observer_split.py.
    from comfyui_h3_blended_inject.observer_split import install_observer_split

    installed = install_observer_split(m, schedule_tail_cfg, m_packed, latent_shapes)
    print(
        "[H3_INJECT] observer-label split: "
        + ("block patches installed" if installed else "no fractional rows — inert"),
        flush=True,
    )

    # --- 5. Wrap the base sampler_function for per-row img2img. ---
    base_ks = comfy.samplers.sampler_object(sampler_name)
    base_fn = base_ks.sampler_function
    if (
        sampler_is_stochastic(base_fn)
        and getattr(base_fn, "__name__", "") not in _NATIVE_ROW_STEPS
        and any(r.denoise < 1.0 for r in schedule)
    ):
        import warnings as _warnings

        _warnings.warn(
            f"H3InjectSampler: sampler '{sampler_name}' is stochastic (ancestral/SDE). "
            "Per-row img2img compression is NOT scale-invariant under stochastic "
            "renoise on H3's rectified-flow path; fractional/preserved rows will "
            "come out corrupted (grey static). Use a deterministic sampler "
            "(euler, res_multistep, dpmpp_2m) or euler_ancestral.",
            UserWarning,
            stacklevel=2,
        )
    wrapped_fn = build_per_row_sampler_function(
        base_fn,
        m_packed,
        clean_packed,
        schedule_tail_cfg,
        video_element_count=video_element_count,
        shift_v=shift_v,
        shift_a=shift_a,
    )
    sampler = comfy.samplers.KSAMPLER(
        wrapped_fn,
        extra_options=base_ks.extra_options,
        inpaint_options=base_ks.inpaint_options,
    )

    # --- 6. Resolve optional-negative guidance (H3 NRS-agnostic rule). ---
    #   - negative wired → forward it; set disable_cfg1_optimization so the uncond pass runs
    #     even at cfg==1.0 (required for cfg-independent hooks like NRS).
    #   - negative None + sampler_cfg_function present → warn; pass [] unchanged.
    #   - negative None + no hook → force effective_cfg=1.0 (avoids silent cond*cfg gain).
    effective_negative, effective_cfg, m.model_options = resolve_guidance(
        negative, cfg, m.model_options
    )

    # --- 7. Prepare noise and the full-range sigma schedule. ---
    # Chaining (add_noise=disable / partial step range / leftover noise) is intentionally
    # unsupported: per-row compression makes those per-row-incorrect (see INPUT_TYPES note).
    import comfy.model_management

    noise = comfy.sample.prepare_noise(samples, noise_seed)

    device = comfy.model_management.get_torch_device()
    ksampler_obj = comfy.samplers.KSampler(
        m,
        steps=steps,
        device=device,
        sampler=sampler_name,
        scheduler=scheduler,
        denoise=1.0,
        model_options=m.model_options,
    )
    sigmas = ksampler_obj.sigmas.to(device)

    # Exact per-row stretched-tail sigmas: every σ_row(i) sits at position
    # (k_d·steps + i·(steps−k_d))/steps² of the continuous schedule — an exact grid point of a
    # steps²-step run of the SAME scheduler.  Pre-generate that dense grid (schedule math only,
    # no model eval) so the sampler indexes real scheduler values instead of lerping the coarse
    # steps-point grid (inaccurate under shift-12 curvature).
    dense_ks = comfy.samplers.KSampler(
        m,
        steps=steps * steps,
        device=device,
        sampler=sampler_name,
        scheduler=scheduler,
        denoise=1.0,
        model_options=m.model_options,
    )
    schedule_tail_cfg["sigmas_dense"] = dense_ks.sigmas.to(device)

    # Arm per-guide timed cond removal now the sigma schedule exists.  hold_frac=1.0
    # guides get no threshold (release_threshold → None) and are never released — the
    # official Add Guide behavior.  Matching is by keyframe-dict object identity, so
    # official-node / fl2va keyframes riding the same conditioning are never caught.
    if guide_entries:
        sigma_list = [float(s) for s in sigmas]
        release_entries = [
            (id(kf), threshold)
            for kf, hold_frac in guide_entries
            if (threshold := guides.release_threshold(hold_frac, sigma_list)) is not None
        ]
        if release_entries:
            guide_release["entries"] = release_entries
            print(
                "[H3_INJECT] timed cond removal armed: "
                + ", ".join(f"kf@{t:.4f}" for _, t in release_entries),
                flush=True,
            )

    # --- 8. Sample with noise_mask=None (no native compositing → no compounding ghost). ---
    # Standard custom-sampler preview callback (same wiring as SamplerCustom): drives the
    # official latent preview + progress; previewer-override nodes patch inside this path.
    import latent_preview

    callback = latent_preview.prepare_callback(m, sigmas.shape[-1] - 1)
    out_samples = comfy.sample.sample_custom(
        m,
        noise,
        effective_cfg,
        sampler,
        sigmas,
        positive,
        effective_negative,
        clean_nested,
        noise_mask=None,
        callback=callback,
        disable_pbar=False,
        seed=noise_seed,
    )

    # --- 9. Exact-preserve overwrite of m==0 rows / audio-preserve ticks. ---
    out_video, out_audio = _unpack_av(out_samples)
    out_video, out_audio = post_composite_preserve(
        out_video, out_audio, clean_video, clean_audio, schedule, target_rows, audio_ticks
    )
    if is_nested:
        final = NestedTensor((out_video,) if out_audio is None else (out_video, out_audio))
    else:
        final = out_video

    out = latent_image.copy()
    out["samples"] = final
    return (out,)


class H3AddInject:
    """Append one inject configuration to an ``INJECT_LIST`` chain.

    Each call to this node creates one :class:`~comfyui_h3_blended_inject.schedule.Inject`
    and appends it to the incoming list (or starts a new list if ``inject_list`` is absent).
    Nodes are chained by connecting the ``INJECT_LIST`` output of one node to the optional
    ``inject_list`` input of the next.  The order of the list determines last-in-wins
    priority during schedule merge.

    Overlap policy (important for tooltip): a later inject in the chain overwrites earlier
    injects on every row it claims.  Overlapping injects produce a hard edge at the boundary,
    not a crossfade.  Crossfade between injects is explicitly out of scope.

    Sanitization performed by this node before constructing the :class:`Inject`:

    - ``inject_at`` is snapped down to the nearest multiple of 17; a warning is issued on
      snap (see :func:`~comfyui_h3_blended_inject.sanitize.snap_inject_at`).
    - If ``inject_at`` is not a multiple of 51, an audio-tick position warning is issued with
      the millisecond error (up to ~12.5 ms).
    - Envelope index ordering and bounds are validated; violation raises ``ValueError``.
    - If ``images`` is provided: resolution is validated against the target latent (multiple
      of 32, exact match); mismatch raises ``ValueError``.
    - If both ``images`` and ``audio`` are provided: video is the clock; audio is resampled
      to the target sample rate and trimmed or silence-padded to video duration.

    Note: images are assumed to be at 24 fps native.  No fps conversion is performed.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802
        """Return the full ComfyUI input schema for H3AddInject."""
        return {
            "required": {
                "inject_at": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10000,
                        "step": 17,
                        "tooltip": (
                            "Latent FRAME index in the target latent where this inject begins. "
                            "Snapped down to the nearest multiple of 17 frames. "
                            "The fade indices (start_fade_in, start_keyframes, end_keyframes, "
                            "end_fade_out) are CLIP frame indices — positions within the "
                            "injected clip's own content, not in the target latent. "
                            "Positions that are not multiples of 51 incur an audio-tick "
                            "rounding error of up to ~12.5 ms (a warning is issued)."
                        ),
                    },
                ),
                "start_fade_in": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10000,
                        "step": 1,
                        "tooltip": (
                            "Source clip frame index where the fade-in begins. Denoise = 1.0 here."
                        ),
                    },
                ),
                "start_keyframes": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10000,
                        "step": 1,
                        "tooltip": "Source clip frame index where the hold at min_denoise begins.",
                    },
                ),
                "end_keyframes": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10000,
                        "step": 1,
                        "tooltip": (
                            "Source clip frame index where the hold at min_denoise ends "
                            "(inclusive)."
                        ),
                    },
                ),
                "end_fade_out": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10000,
                        "step": 1,
                        "tooltip": (
                            "EXCLUSIVE upper bound of the fade-out: denoise returns to 1.0 "
                            "at this frame index; the last active clip frame is end_fade_out - 1.  "
                            "Part of the half-open interval [start_fade_in, end_fade_out).  "
                            "Set equal to source_length to fade out through the last frame."
                        ),
                    },
                ),
                "min_denoise": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Denoise floor during the hold region. 0 = exact preserve "
                            "(routed via the derived noise mask). 1 = fully regenerated. "
                            "For a still inject (single image, all fade indices equal), "
                            "this is the single frame's denoise value. "
                            "Follows img2img convention on H3's shift-12 schedule: "
                            "d <= 0.3 retains most content; d >= 0.7 is heavy redraw. "
                            "Values below ~1/steps never release due to step quantization."
                        ),
                    },
                ),
                "interpolation_type": (
                    ["ease_in", "ease_out", "ease_in_out", "linear", "none"],
                    {
                        "tooltip": (
                            "Interpolation curve applied to both the fade-in and fade-out regions."
                        ),
                    },
                ),
                "audio_mode": (
                    ["fade", "drop", "keep"],
                    {
                        "tooltip": (
                            "fade: audio envelope follows the video denoise schedule. "
                            "drop: no audio inject (audio is generated normally). "
                            "keep: audio inject at d=0 via the derived noise mask "
                            "(exact preservation)."
                        ),
                    },
                ),
            },
            "optional": {
                "inject_list": (
                    INJECT_LIST,
                    {"tooltip": "Chain input. Absent = start a new list."},
                ),
                "images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Video clip or single image to inject. Assumed 24 fps native; "
                            "no fps conversion is performed. Resolution must be a multiple "
                            "of 32 and match the target latent exactly."
                        ),
                    },
                ),
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Audio to inject. Resampled to target rate; "
                            "trimmed or padded to video duration."
                        )
                    },
                ),
                "vae": (
                    "VAE",
                    {
                        "tooltip": (
                            "Video VAE for pre-encoding inject images into the H3 video latent "
                            "space. From a VAELoader node. When provided, the encoded latent is "
                            "stored on the Inject and passed to the sampler's hold-and-release "
                            "mechanism, avoiding re-encoding at sample time."
                        ),
                    },
                ),
                "audio_vae": (
                    "VAE",
                    {
                        "tooltip": (
                            "Audio VAE for pre-encoding inject audio into the H3 audio latent "
                            "space. From a VAELoader node. When provided, the encoded audio "
                            "latent is stored on the Inject alongside the video latent. "
                            "Requires a matching audio VAE for the H3 model."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = (INJECT_LIST,)
    RETURN_NAMES = ("inject_list",)
    FUNCTION = "add_inject"
    CATEGORY = "H3 Blended Inject"

    def add_inject(
        self,
        inject_at: int,
        start_fade_in: int,
        start_keyframes: int,
        end_keyframes: int,
        end_fade_out: int,
        min_denoise: float,
        interpolation_type: str,
        audio_mode: str,
        inject_list: InjectList | None = None,
        images: Any | None = None,
        audio: Any | None = None,
        vae: Any | None = None,
        audio_vae: Any | None = None,
    ) -> tuple[InjectList]:
        """Validate inputs, construct an :class:`~comfyui_h3_blended_inject.schedule.Inject`,
        and append it to the chain.

        Parameters
        ----------
        inject_at:
            Requested latent FRAME index in the target latent where the inject begins
            (snapped down to the nearest multiple of 17 frames internally).
        start_fade_in:
            Source frame where fade-in begins.
        start_keyframes:
            Source frame where hold begins.
        end_keyframes:
            Source frame where hold ends.
        end_fade_out:
            Source frame where fade-out ends.
        min_denoise:
            Denoise floor during hold, in [0.0, 1.0].
        interpolation_type:
            Curve name: one of ``"ease_in"``, ``"ease_out"``, ``"ease_in_out"``,
            ``"linear"``, ``"none"``.
        audio_mode:
            One of ``"fade"``, ``"drop"``, ``"keep"``.
        inject_list:
            Existing chain to append to, or ``None`` to start a new list.
        images:
            IMAGE tensor or ``None``.
        audio:
            AUDIO dict or ``None``.
        vae:
            Video VAE for pre-encoding inject images, or ``None``.
        audio_vae:
            Audio VAE for pre-encoding inject audio, or ``None``.

        Returns
        -------
        tuple[InjectList]
            A 1-tuple containing the updated inject list.

        Raises
        ------
        ValueError
            If envelope index ordering is violated or image resolution does not match the
            target latent.
        """
        # a. Snap inject_at to the nearest 17-frame boundary.
        snapped = snap_inject_at(inject_at)
        # Only warn about non-51 audio-tick alignment when audio is actually being injected.
        # The %51 position error is irrelevant for audio_mode='drop' (pure noise, no audio insert).
        if audio is not None and audio_mode != "drop":
            warn_audio_tick_alignment(snapped)  # side-effect: warn if not mult of 51

        # b. Lightweight ordering validation (full bounds check with target_rows happens
        #    later in sample() once the target latent is known).
        # Deliberately repeats the ordering check in validate_envelope_indices to catch
        # mis-ordered indices early at add_inject time, before the target latent is known.
        if not (start_fade_in <= start_keyframes <= end_keyframes <= end_fade_out):
            raise ValueError(
                f"Envelope index ordering violated: "
                f"start_fade_in={start_fade_in}, start_keyframes={start_keyframes}, "
                f"end_keyframes={end_keyframes}, end_fade_out={end_fade_out}. "
                "Must satisfy start_fade_in <= start_keyframes <= end_keyframes <= end_fade_out."
            )

        # b2. Content without a matching VAE would be SILENTLY dropped downstream
        #     (build_clean_reference skips injects whose latents are None) — error early.
        if images is not None and vae is None:
            raise ValueError(
                "images were provided but no vae is wired; the inject content cannot be "
                "encoded and would be silently ignored. Connect the video VAE."
            )
        if audio is not None and audio_mode != "drop" and audio_vae is None:
            raise ValueError(
                f"audio was provided with audio_mode={audio_mode!r} but no audio_vae is "
                "wired; the inject audio cannot be encoded and would be silently ignored. "
                "Connect the audio VAE or set audio_mode='drop'."
            )

        # c. Derive source_length and resolution from images if present.
        #    Snap source content length down to the nearest valid H3 inject length.
        #    Valid set: {1} ∪ {17n+5}.  Raises ValueError for lengths 2-4 or < 1;
        #    warns and trims when source_length > 5 and is not already a valid 17n+5 value.
        if images is not None:
            _raw_length = int(images.shape[0])
            source_length = snap_length_down(_raw_length)  # raises on invalid; warns on trim
            if source_length < _raw_length:
                images = images[:source_length]  # trim trailing frames from image batch
            resolution = (int(images.shape[2]), int(images.shape[1]))  # (width, height)

            # Single-frame (F=1) keyframe guards — must use the ORIGINAL inject_at (the
            # user-requested value), NOT the post-snap value, so that a non-chunk-boundary
            # request is clearly rejected rather than silently rewritten.
            if source_length == 1:
                if inject_at % 17 != 0:
                    lo = 17 * (inject_at // 17)
                    hi = lo + 17
                    raise ValueError(
                        f"Single-frame (keyframe) injects must be placed exactly on a chunk "
                        f"boundary (inject_at a multiple of 17, a '1' edge); "
                        f"got inject_at={inject_at}. Nearest edges: {lo} or {hi}."
                    )
                if audio is not None and audio_mode != "drop":
                    raise ValueError(
                        f"Single-frame (keyframe) injects require audio_mode='drop' or no "
                        f"audio; got audio_mode={audio_mode!r} with audio present."
                    )

            # Validate fade indices against the POST-TRIM snapped length.
            # end_fade_out is EXCLUSIVE (half-open): efo == source_length is valid.
            # This is an early check; the full target-row bounds check runs at sample() time.
            if end_fade_out > source_length:
                raise ValueError(
                    f"end_fade_out={end_fade_out} exceeds the post-trim source_length="
                    f"{source_length} (snapped from {_raw_length} frames). "
                    "Fade indices must not reference frames beyond the trimmed content."
                )
            if start_fade_in >= source_length:
                raise ValueError(
                    f"start_fade_in={start_fade_in} >= post-trim source_length={source_length}. "
                    "start_fade_in must be a valid frame index (< source_length)."
                )

            # Warn when the snapped length is not audio-sync-aligned (51k+39) and the clip
            # end is exposed without a fade-out ramp that reaches the tail.
            warn_audio_tail_alignment(
                source_length, audio_mode, end_keyframes, end_fade_out, audio is not None
            )
        else:
            source_length = 0
            resolution = (0, 0)

        # d. Sanitize audio against the video duration when both are present.
        if audio is not None and images is not None:
            target_sample_rate = getattr(audio_vae, "audio_sample_rate", 32000)
            audio = sanitize_audio(
                audio,
                target_sample_rate=target_sample_rate,
                video_duration_frames=source_length,
                fps=int(grid.FPS),
            )

        # e. Pre-encode images and audio into latent space when VAEs are provided.
        #    The encode calls are guarded by vae/audio_vae None-checks; a FakeVAE can
        #    exercise the video path CPU-side.
        video_latent = vae.encode(images) if (vae is not None and images is not None) else None
        audio_latent = (
            _encode_ref_audio(audio_vae, audio)[0]  # pragma: no cover
            if (audio_vae is not None and audio is not None)
            else None
        )

        # f. Construct the Inject dataclass with all resolved fields.
        inj = Inject(
            inject_at=snapped,
            start_fade_in=start_fade_in,
            start_keyframes=start_keyframes,
            end_keyframes=end_keyframes,
            end_fade_out=end_fade_out,
            min_denoise=min_denoise,
            interpolation_type=interpolation_type,
            audio_mode=audio_mode,
            images=images,
            audio=audio,
            resolution=resolution,
            source_length=source_length,
            video_latent=video_latent,
            audio_latent=audio_latent,
        )

        # g. Append to a shallow copy of the incoming list (never mutate the input).
        new_list = list(inject_list) if inject_list else []
        new_list.append(inj)
        return (new_list,)


class H3AddGuide:
    """Append one native keyframe/guide cond entry to an ``INJECT_LIST`` chain.

    Outward clone of comfy's ``MiniMaxH3AddGuide`` ("Add Guide for MiniMax H3"), made
    monadic: instead of taking ``positive`` + ``latent`` inputs and mutating conditioning
    at node time, it appends a :class:`~comfyui_h3_blended_inject.schedule.Guide` to the
    same chain ``H3AddInject`` uses.  ``H3InjectSampler`` partitions the chain and appends
    the native keyframe cond dicts to the positive conditioning at sample time (when the
    target latent — and therefore frame count, resolution, and audio length — is known).

    Differences from the official node (deliberate):

    - **No in-node resize**: the guide resolution must match the target latent exactly;
      a mismatch raises at sample time.  Resize upstream (e.g. kjnodes "Resize Image v2").
    - **Per-guide ``hold_frac``**: at ``1.0`` (default) the guide behaves exactly like the
      official node — its cond row is held for the whole run.  Below 1.0, the cond row is
      REMOVED partway through sampling (timed cond removal), so a co-located fractional
      ``H3AddInject`` keyframe can finish its own denoise without the cond-token attractor
      re-pulling it toward the source.  Official-node and fl2va keyframes are never
      affected — the release filter matches only entries this chain appended, by object
      identity.

    ``frame_idx`` keeps the official PIXEL-frame semantics (negative = from the end) —
    NOT ``H3AddInject.inject_at``'s latent-frame/17-snap semantics.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802
        """Return the full ComfyUI input schema for H3AddGuide."""
        return {
            "required": {
                "frame_idx": (
                    "INT",
                    {
                        "default": 0,
                        "min": -9999,
                        "max": 9999,
                        "tooltip": (
                            "PIXEL-frame index to anchor the image (or the clip's first "
                            "frame) at. Negative values count from the end of the video. "
                            "NOTE: this is a pixel-frame index like the official Add Guide "
                            "node — NOT H3AddInject's inject_at, which is a latent-frame "
                            "index snapped to the 17-frame grid."
                        ),
                    },
                ),
                "hold_frac": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Fraction of the sampling schedule this guide's cond row is "
                            "held. 1.0 = held for the entire run (identical to the "
                            "official Add Guide node). Lower values REMOVE the cond row "
                            "partway through sampling (timed cond removal) so a "
                            "co-located fractional H3AddInject keyframe can finish its "
                            "own denoise without being re-pulled toward the source. "
                            "0.0 = removed from step 0 (no-reference ablation)."
                        ),
                    },
                ),
            },
            "optional": {
                "inject_list": (
                    INJECT_LIST,
                    {"tooltip": "Chain input. Absent = start a new list."},
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Image or video frames to anchor. Multi-frame batches are "
                            "anchored as a clip and snapped down to the model's valid "
                            "clip lengths: 5, 22, 39... (17k + 5) frames; batches "
                            "shorter than 5 frames use only the first image. Resolution "
                            "must match the target latent exactly (no in-node resize — "
                            "resize upstream); a mismatch raises at sample time."
                        ),
                    },
                ),
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Soundtrack to anchor starting at the same frame index, "
                            "cropped to the video's remaining duration at sample time."
                        ),
                    },
                ),
                "vae": (
                    "VAE",
                    {"tooltip": "Video VAE; required when an image is connected."},
                ),
                "audio_vae": (
                    "VAE",
                    {"tooltip": "Audio VAE; required when an audio is connected."},
                ),
            },
        }

    RETURN_TYPES = (INJECT_LIST,)
    RETURN_NAMES = ("inject_list",)
    FUNCTION = "add_guide"
    CATEGORY = "H3 Blended Inject"

    def add_guide(
        self,
        frame_idx: int,
        hold_frac: float,
        inject_list: InjectList | None = None,
        image: Any | None = None,
        audio: Any | None = None,
        vae: Any | None = None,
        audio_vae: Any | None = None,
    ) -> tuple[InjectList]:
        """Validate inputs, encode content, and append a :class:`Guide` to the chain.

        Encoding happens here at node time (mirroring the official node): images through
        ``vae``, audio through ``audio_vae``.  Frame-index resolution, bounds checks, and
        the audio crop are deferred to sample time, when the target latent is known.

        Returns
        -------
        tuple[InjectList]
            A 1-tuple containing the updated inject list.

        Raises
        ------
        ValueError
            If neither image nor audio is provided, or content is missing its VAE.
        """
        if image is None and audio is None:
            raise ValueError("H3AddGuide needs an image or an audio to anchor")
        if image is not None and vae is None:
            raise ValueError(
                "an image was provided but no vae is wired; the guide frames cannot "
                "be encoded. Connect the video VAE."
            )
        if audio is not None and audio_vae is None:
            raise ValueError(
                "audio was provided but no audio_vae is wired; the guide audio cannot "
                "be encoded. Connect the audio VAE."
            )

        video_latent = None
        resolution = (0, 0)
        guide_frames = 1
        if image is not None:
            # Snap multi-frame batches to the valid clip grid (warns on trim; <5 → 1).
            guide_frames = guides.snap_guide_length(int(image.shape[0]))
            image = image[:guide_frames]
            resolution = (int(image.shape[2]), int(image.shape[1]))  # (width, height)
            video_latent = vae.encode(image)
        audio_latent = (
            _encode_ref_audio(audio_vae, audio)[0]  # pragma: no cover — needs real VAE
            if audio is not None
            else None
        )

        g = Guide(
            frame_idx=frame_idx,
            hold_frac=hold_frac,
            video_latent=video_latent,
            audio_latent=audio_latent,
            resolution=resolution,
            guide_frames=guide_frames,
        )

        # Append to a shallow copy of the incoming list (never mutate the input).
        new_list = list(inject_list) if inject_list else []
        new_list.append(g)
        return (new_list,)


class H3InjectSampler:
    """KSampler Advanced clone that applies per-row img2img inject during sampling.

    Mirrors the core KSampler surface (model, seed, steps, cfg, sampler, scheduler, latent,
    conditioning) and adds an ``inject_list`` input.  The KSampler-Advanced chaining widgets
    (add_noise / start-end step / leftover noise) are deliberately hidden — per-row
    compression makes partial runs per-row-incorrect (see the INPUT_TYPES note).

    Responsibilities:

    1. VAE-encode inject content into per-inject video/audio latents (done by ``H3AddInject``).
    2. Build the per-row schedule from the inject list (last-in-wins merge via
       :func:`~comfyui_h3_blended_inject.schedule.merge_schedule`).
    3. Build the *clean reference* (target + composited inject content) and the *fractional
       per-row denoise mask*, then run a **custom sampler** that lerps each row's initial noise
       toward the clean reference and feeds the fractional schedule to the DiT as conditioning
       (see :func:`_run_sampler`).  ``noise_mask=None`` — no native compositing — so there is
       no compounding re-pin ghost; exact ``m == 0`` rows are restored by a post-sampling
       composite.
    4. Compatible with all samplers: deterministic ones (res_multistep, dpmpp_2m, euler) need
       no change; stochastic ones get a per-row noise-scaling shim.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802
        """Return the full ComfyUI input schema for H3InjectSampler.

        Sampler and scheduler lists are sourced from ``comfy.samplers`` lazily.  Falls back to
        empty lists if comfy is not available (test context).
        """
        try:
            import comfy.samplers

            samplers = comfy.samplers.KSampler.SAMPLERS
            schedulers = comfy.samplers.KSampler.SCHEDULERS
        except ImportError:
            samplers = []
            schedulers = []

        return {
            "required": {
                "model": ("MODEL",),
                # NOTE (prototype): the KSampler-Advanced chaining surface (add_noise,
                # start_at_step, end_at_step, return_with_leftover_noise) is HIDDEN, not
                # supported: per-row compressed schedules make partial runs / leftover
                # noise per-row-incorrect (fractional rows end at m*sigma_end, but the
                # sampler's final inverse_noise_scaling divides all rows by (1-sigma_end)).
                # Internally hardcoded to add_noise=enable / full step range / no leftover.
                # Revisit post-prototype: see wiki status-and-open-paths.
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "RNG seed for both sampler noise and per-row hold noise.",
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01},
                ),
                "sampler_name": (
                    samplers,
                    {
                        "tooltip": (
                            "Sampler algorithm.  Deterministic samplers (euler, res_multistep, "
                            "dpmpp_2m) work correctly with the per-row schedule-tail remap.  "
                            "Stochastic samplers (euler_ancestral, dpmpp_2s_ancestral, SDE "
                            "variants) will warn at runtime: the remap is not scale-invariant "
                            "under stochastic renoise on H3's RF path and results may be incorrect."
                        ),
                    },
                ),
                "scheduler": (
                    schedulers,
                    {
                        "tooltip": (
                            "Noise schedule.  Any ComfyUI scheduler works; 'simple' is the "
                            "H3 default.  The per-row img2img compression is applied on top of "
                            "whatever global schedule is chosen."
                        ),
                    },
                ),
                "positive": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "inject_list": (INJECT_LIST,),
            },
            "optional": {
                "negative": (
                    "CONDITIONING",
                    {
                        "tooltip": (
                            "Optional. H3 is CFG-distilled and runs with no uncond by default — "
                            "leave unconnected for standard H3 sampling. Connect a negative "
                            "conditioning to enable CFG / NRS-style guidance."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "H3 Blended Inject"

    def sample(
        self,
        model: Any,
        noise_seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        latent_image: dict[str, Any],
        inject_list: InjectList,
        negative: Any | None = None,
    ) -> tuple[dict[str, Any]]:
        """Run the H3 sampler with per-row img2img inject.

        Parameters
        ----------
        model:
            ComfyUI MODEL object (a ``MiniMaxH3`` instance is expected).
        noise_seed:
            RNG seed for the sampler's initial noise (and the stochastic per-row noise shim).
        steps:
            Total number of sampling steps.
        cfg:
            Classifier-free guidance scale.
        sampler_name:
            Sampler name string (from ``comfy.samplers.KSampler.SAMPLERS``).
        scheduler:
            Scheduler name string (from ``comfy.samplers.KSampler.SCHEDULERS``).
        positive:
            Positive conditioning tensor.
        latent_image:
            ComfyUI LATENT dict with ``"samples"`` key.
        inject_list:
            Ordered chain of :class:`~comfyui_h3_blended_inject.schedule.Inject` and
            :class:`~comfyui_h3_blended_inject.schedule.Guide` instances (partitioned
            by type here).
        negative:
            Optional negative conditioning.  ``None`` (the default) means "no uncond" —
            the H3 CFG-distilled default.  When provided, it is forwarded verbatim to the
            sampler for CFG / NRS-style guidance.

        Returns
        -------
        tuple[dict[str, Any]]
            A 1-tuple containing the output LATENT dict.

        Raises
        ------
        ValueError
            If any inject's image resolution does not match the latent or if envelope
            index validation fails.
        """
        # 1. Derive target pixel resolution and temporal dimensions from the latent.
        #    H3 spatial downsample is 16x, so multiply back to pixel space.
        #    Real H3 FLOW_AV latents are NestedTensor((video, audio)); plain tensors
        #    are used in CPU tests.
        samples = latent_image["samples"]

        if getattr(samples, "is_nested", False):
            # Real H3 FLOW_AV NestedTensor latent (GPU path).
            # video: [B, 24, T, Hl, Wl]  audio: [B, 32, 2, audio_t]
            _video = samples.tensors[0]
            _audio = samples.tensors[1]
            target_rows = int(_video.shape[2])
            target_h = int(_video.shape[-2]) * 16
            target_w = int(_video.shape[-1]) * 16
            # Read the real audio tick count directly from the audio latent shape.
            # _run_sampler re-derives the component shapes from the latent itself.
            _audio_ticks_from_latent: int | None = int(_audio.shape[-1])
        elif samples.dim() == 5:  # pragma: no cover
            # Plain 5-dim tensor: [N, C, T, H/16, W/16] (non-nested GPU latent).
            target_rows = int(samples.shape[2])
            target_h = int(samples.shape[-2]) * 16
            target_w = int(samples.shape[-1]) * 16
            _audio_ticks_from_latent = None
        else:
            # 4-dim synthetic tensor used in CPU tests: [N, rows, H, W].
            target_rows = int(samples.shape[1])
            target_h = int(samples.shape[-2]) * 16
            target_w = int(samples.shape[-1]) * 16
            _audio_ticks_from_latent = None

        # 2. Partition the chain: Inject entries feed the per-row img2img schedule;
        #    Guide entries feed the native keyframe cond path (appended at sample time).
        injects, guide_list = guides.partition_inject_list(inject_list)

        # 3. Validate all injects: resolution first (so the CPU test can raise before
        #    touching the model), then envelope indices.
        for inj in injects:
            if inj.images is not None:
                check_resolution(inj.images, target_w, target_h)
            validate_envelope_indices(
                inj.start_fade_in,
                inj.start_keyframes,
                inj.end_keyframes,
                inj.end_fade_out,
                inj.source_length,
                target_rows,
                inj.inject_at,
            )

        # 4. Merge the injects into a flat per-row schedule (last-in-wins).
        schedule = merge_schedule(injects, target_rows)

        # 5. Derive audio tick count (real value from nested latent when available;
        #    computed from row count as fallback for plain-tensor paths).
        if _audio_ticks_from_latent is not None:
            audio_ticks = _audio_ticks_from_latent
        else:
            audio_ticks = grid.audio_ticks_for_rows(target_rows)

        # 6. Resolve guides against the target: exact-match resolution check, pixel
        #    frame-index resolution + bounds check, audio crop, keyframe dict build.
        #    The keyframe dicts' OBJECT IDENTITY is the timed-removal tracking key —
        #    built once here, threaded through _run_sampler untouched.
        guide_entries: list[tuple[dict[str, Any], float]] = []
        if guide_list:
            frame_count = guides.frame_count_for_rows(target_rows)
            for g in guide_list:
                if g.video_latent is not None and g.resolution != (target_w, target_h):
                    raise ValueError(
                        f"Guide resolution {g.resolution[0]}x{g.resolution[1]} does not "
                        f"match the target latent {target_w}x{target_h}. H3AddGuide does "
                        "not resize; resize the guide image upstream (e.g. kjnodes "
                        "'Resize Image v2')."
                    )
                resolved = guides.resolve_frame_index(g.frame_idx, frame_count, g.guide_frames)
                guide_entries.append((guides.build_keyframe(g, resolved, audio_ticks), g.hold_frac))

        # GPU/ComfyUI-dependent per-row img2img pipeline: build the clean reference
        # and fractional denoise mask, install the conditioning wrapper, wrap the base
        # sampler for per-row init lerp, run sample_custom, and exact-preserve composite.
        return _run_sampler(  # pragma: no cover
            model=model,
            schedule=schedule,
            latent_image=latent_image,
            noise_seed=noise_seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            samples=samples,
            target_rows=target_rows,
            audio_ticks=audio_ticks,
            guide_entries=guide_entries,
        )


# ---------------------------------------------------------------------------
# ComfyUI node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "H3AddInject": H3AddInject,
    "H3AddGuide": H3AddGuide,
    "H3InjectSampler": H3InjectSampler,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "H3AddInject": "H3 Add Inject",
    "H3AddGuide": "H3 Add Guide",
    "H3InjectSampler": "H3 Inject Sampler",
}

__all__ = [
    "H3AddGuide",
    "H3AddInject",
    "H3InjectSampler",
    "INJECT_LIST",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
