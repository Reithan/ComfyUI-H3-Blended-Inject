"""ComfyUI node definitions for H3 Blended Inject.

Two nodes are defined:

- :class:`H3AddInject`: Appends one inject to an ``INJECT_LIST`` chain.  Chainable.
- :class:`H3InjectSampler`: KSampler Advanced clone that accepts an ``INJECT_LIST`` and
  applies per-row img2img inject during sampling.

``comfy`` imports are lazy (inside methods) so this module imports without ComfyUI present,
enabling CPU-side unit tests.

``NODE_CLASS_MAPPINGS`` and ``NODE_DISPLAY_NAME_MAPPINGS`` are defined at module bottom and
re-exported from the top-level ``__init__.py``.
"""

from __future__ import annotations

from typing import Any

from comfyui_h3_blended_inject import constants
from comfyui_h3_blended_inject.guidance import resolve_guidance
from comfyui_h3_blended_inject.sanitize import (
    check_resolution,
    sanitize_audio,
    snap_inject_at,
    snap_inject_at_audio_tick,
    snap_length_down,
    validate_envelope_indices,
    warn_audio_tail_alignment,
)
from comfyui_h3_blended_inject.schedule import Inject, InjectList, merge_schedule

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
    import torchaudio  # noqa: PLC0415

    waveform = audio["waveform"]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, z.shape[-1]


def _run_sampler(  # pragma: no cover
    model: Any,
    schedule: Any,
    latent_image: dict[str, Any],
    noise_seed: int,
    add_noise: str,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive: Any,
    negative: Any | None,
    samples: Any,
    start_at_step: int,
    end_at_step: int,
    return_with_leftover_noise: str,
    target_rows: int,
    audio_ticks: int,
) -> tuple[dict[str, Any]]:
    """GPU/ComfyUI per-row img2img sampling pipeline — not CPU-testable.

    Implements the per-row img2img design (see ``sampler.py`` and ``composite.py``):

    1. Build the *clean reference* — the target latent with all inject video/audio content
       composited in (:func:`~comfyui_h3_blended_inject.composite.build_clean_reference`).
       This is passed to ``sample_custom`` as the ``latent_image`` so ComfyUI's global
       ``noise_scaling`` produces ``x_global = sigma_max*eps + (1-sigma_max)*clean``.
    2. Build the *fractional per-row denoise mask*
       (:func:`~comfyui_h3_blended_inject.mask.derive_fractional_mask`) and pack it to the
       sampler's flat layout.  Its pooled token-grid values are injected as DiT conditioning
       via the :func:`~comfyui_h3_blended_inject.sampler.build_conditioning_wrapper`, so each
       row runs its own compressed schedule.
    3. Wrap the base k-diffusion ``sampler_function`` with
       :func:`~comfyui_h3_blended_inject.sampler.build_per_row_sampler_function`, which lerps
       the initial ``x`` toward the clean reference per row (``m*x + (1-m)*clean``) and — for
       stochastic samplers — installs a per-row-scaling noise_sampler shim.
    4. Run ``comfy.sample.sample_custom`` with ``noise_mask=None`` (no native compositing → no
       compounding re-pin ghost), then apply the binary exact-preserve overwrite
       (:func:`~comfyui_h3_blended_inject.composite.post_composite_preserve`) for ``m == 0``
       rows and audio-preserve ticks.

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
    import torch
    from comfy.nested_tensor import NestedTensor

    from comfyui_h3_blended_inject.composite import (
        build_clean_reference,
        post_composite_preserve,
    )
    from comfyui_h3_blended_inject.mask import derive_fractional_mask
    from comfyui_h3_blended_inject.sampler import (
        build_conditioning_wrapper,
        build_per_row_sampler_function,
        sampler_accepts_noise_sampler,
    )

    # --- 1. Split the target latent into components (H3 FLOW_AV is NestedTensor). ---
    _samples = latent_image["samples"]
    is_nested = getattr(_samples, "is_nested", False)
    if is_nested:
        _components = list(_samples.unbind())
        video = _components[0]  # [1, C_v, T, Hl, Wl]; row axis dim 2
        audio = _components[1] if len(_components) > 1 else None  # [1, C_a, 2, audio_t]
    else:
        video = _samples
        audio = None

    # --- 2. Clean reference: target with all inject content composited in. ---
    # Fractional rows img2img *from* this content; m==1 rows ignore it; m==0 rows are
    # restored from it after sampling by post_composite_preserve.
    clean_video, clean_audio = build_clean_reference(
        video, audio, schedule, target_rows, audio_ticks
    )
    clean_components = (clean_video,) if clean_audio is None else (clean_video, clean_audio)
    clean_nested = NestedTensor(clean_components) if is_nested else clean_video

    # Pack the clean reference — it is BOTH sample_custom's latent_image (noise_scaling
    # blends toward it) AND the clean term for the per-row init lerp, so the two must match.
    import comfy.utils as _comfy_utils

    clean_packed, latent_shapes = _comfy_utils.pack_latents(clean_components)

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
    m_packed = m_packed.to(device=clean_packed.device)

    # --- 4. Install the fractional-denoise conditioning + denoised-correction wrapper. ---
    # The wrapper injects the pooled per-row timestep conditioning AND corrects the denoised
    # (m*denoised + (1-m)*input) so the sampler integrates each row over its compressed
    # m*sigma interval — H3's process_timestep only compresses the network's timestep, while
    # calculate_denoised still divides by the outer sigma. See build_conditioning_wrapper.
    pooled = m.model._denoise_mask_values(m_packed, latent_shapes)
    m.model_options = {
        **m.model_options,
        "model_function_wrapper": build_conditioning_wrapper(pooled, m_packed),
    }

    # --- 5. Wrap the base sampler_function for per-row img2img. ---
    base_ks = comfy.samplers.sampler_object(sampler_name)
    base_fn = base_ks.sampler_function
    wrapped_fn = build_per_row_sampler_function(
        base_fn,
        m_packed,
        clean_packed,
        scale_stochastic_noise=sampler_accepts_noise_sampler(base_fn),
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

    # --- 7. Prepare noise and per-step sigmas (last_step then start_step slicing). ---
    import comfy.model_management

    disable_noise = add_noise == "disable"
    force_full_denoise = return_with_leftover_noise == "disable"

    noise = comfy.sample.prepare_noise(samples, noise_seed)
    if disable_noise:
        if getattr(noise, "is_nested", False):
            noise = NestedTensor(tuple(torch.zeros_like(c) for c in noise.unbind()))
        else:
            noise = torch.zeros_like(noise)

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
    sigmas = ksampler_obj.sigmas
    if end_at_step < len(sigmas) - 1:
        sigmas = sigmas[: end_at_step + 1]
        if force_full_denoise:
            sigmas[-1] = 0
    if start_at_step > 0:
        if start_at_step < len(sigmas) - 1:
            sigmas = sigmas[start_at_step:]
        else:
            # Nothing left to sample — return the clean reference unchanged.
            out = latent_image.copy()
            out["samples"] = clean_nested
            return (out,)
    sigmas = sigmas.to(device)

    # --- 8. Sample with noise_mask=None (no native compositing → no compounding ghost). ---
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
        callback=None,
        disable_pbar=False,
        seed=noise_seed,
    )

    # --- 9. Exact-preserve overwrite of m==0 rows / audio-preserve ticks. ---
    if getattr(out_samples, "is_nested", False):
        _out_components = list(out_samples.unbind())
        out_video = _out_components[0]
        out_audio = _out_components[1] if len(_out_components) > 1 else None
    else:
        out_video = out_samples
        out_audio = None
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
        """Return the full ComfyUI input schema for H3AddInject.

        Returns the real dict (not a stub); ``INPUT_TYPES`` is a static schema, not logic.
        """
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
                            "Source clip frame index where the fade-out ends. Denoise = 1.0 here."
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
            snap_inject_at_audio_tick(snapped)  # side-effect: warn if not mult of 51

        # b. Lightweight ordering validation (full bounds check with target_rows happens
        #    later in sample() once the target latent is known).
        if not (start_fade_in <= start_keyframes <= end_keyframes <= end_fade_out):
            raise ValueError(
                f"Envelope index ordering violated: "
                f"start_fade_in={start_fade_in}, start_keyframes={start_keyframes}, "
                f"end_keyframes={end_keyframes}, end_fade_out={end_fade_out}. "
                "Must satisfy start_fade_in <= start_keyframes <= end_keyframes <= end_fade_out."
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
                fps=int(constants.FPS),
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


class H3InjectSampler:
    """KSampler Advanced clone that applies per-row img2img inject during sampling.

    Mirrors the KSampler Advanced surface (model, seed, steps, cfg, sampler, scheduler,
    start/end step, latent, conditioning, add_noise, return_with_leftover_noise) and adds an
    ``inject_list`` input.

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

        Returns the real dict (not a stub); ``INPUT_TYPES`` is a static schema, not logic.
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
                "add_noise": (
                    ["enable", "disable"],
                    {"tooltip": "Whether to add noise before sampling (disable for img2img)."},
                ),
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
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "positive": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (
                    ["disable", "enable"],
                    {"tooltip": "Return the latent with leftover noise (for chained samplers)."},
                ),
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
        add_noise: str,
        noise_seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        latent_image: dict[str, Any],
        start_at_step: int,
        end_at_step: int,
        return_with_leftover_noise: str,
        inject_list: InjectList,
        negative: Any | None = None,
    ) -> tuple[dict[str, Any]]:
        """Run the H3 sampler with per-row img2img inject.

        Parameters
        ----------
        model:
            ComfyUI MODEL object (a ``MiniMaxH3`` instance is expected).
        add_noise:
            ``"enable"`` to add noise before sampling; ``"disable"`` for img2img.
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
        start_at_step:
            First step index to sample (for step-range chaining).
        end_at_step:
            Last step index to sample (exclusive).
        return_with_leftover_noise:
            ``"enable"`` to return the latent without final denoising (for chained samplers).
        inject_list:
            Ordered list of :class:`~comfyui_h3_blended_inject.schedule.Inject` instances.
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

        # 2. Validate all injects: resolution first (so the CPU test can raise before
        #    touching the model), then envelope indices.
        for inj in inject_list:
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

        # 3. Merge inject list into a flat per-row schedule (last-in-wins).
        schedule = merge_schedule(inject_list, target_rows)

        # 4. Derive audio tick count (real value from nested latent when available;
        #    computed from row count as fallback for plain-tensor paths).
        if _audio_ticks_from_latent is not None:
            audio_ticks = _audio_ticks_from_latent
        else:
            audio_ticks = constants.audio_ticks_for_rows(target_rows)

        # 5–9. GPU/ComfyUI-dependent per-row img2img pipeline: build the clean reference
        #      and fractional denoise mask, install the conditioning wrapper, wrap the base
        #      sampler for per-row init lerp, run sample_custom, and exact-preserve composite.
        return _run_sampler(  # pragma: no cover
            model=model,
            schedule=schedule,
            latent_image=latent_image,
            noise_seed=noise_seed,
            add_noise=add_noise,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            samples=samples,
            start_at_step=start_at_step,
            end_at_step=end_at_step,
            return_with_leftover_noise=return_with_leftover_noise,
            target_rows=target_rows,
            audio_ticks=audio_ticks,
        )


# ---------------------------------------------------------------------------
# ComfyUI node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "H3AddInject": H3AddInject,
    "H3InjectSampler": H3InjectSampler,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "H3AddInject": "H3 Add Inject",
    "H3InjectSampler": "H3 Inject Sampler",
}

__all__ = [
    "H3AddInject",
    "H3InjectSampler",
    "INJECT_LIST",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
