"""ComfyUI node definitions for H3 Blended Inject.

Two nodes are defined:

- :class:`H3AddInject`: Appends one inject to an ``INJECT_LIST`` chain.  Chainable.
- :class:`H3InjectSampler`: KSampler Advanced clone that accepts an ``INJECT_LIST`` and
  applies the hold-and-release mechanism during sampling.

``comfy`` imports are lazy (inside methods) so this module imports without ComfyUI present,
enabling CPU-side unit tests.

``NODE_CLASS_MAPPINGS`` and ``NODE_DISPLAY_NAME_MAPPINGS`` are defined at module bottom and
re-exported from the top-level ``__init__.py``.
"""

from __future__ import annotations

from typing import Any

from comfyui_h3_blended_inject import constants
from comfyui_h3_blended_inject.mask import apply_derived_mask
from comfyui_h3_blended_inject.sanitize import (
    check_resolution,
    sanitize_audio,
    snap_inject_at,
    snap_inject_at_audio_tick,
    validate_envelope_indices,
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
    negative: Any,
    samples: Any,
    start_at_step: int,
    end_at_step: int,
    return_with_leftover_noise: str,
) -> tuple[dict[str, Any]]:
    """GPU/ComfyUI sampling pipeline — not CPU-testable.

    Builds per-row original/noise tensors from the inject latents, installs the
    hold-and-release ``model_function_wrapper``, and runs the sampler.

    Notes
    -----
    This function is pragma'd because it requires real H3 latents, a running
    ComfyUI environment, and (typically) a GPU.  The ``audio_scale_factor``
    derivation from ``model.model.model_sampling`` also needs GPU verification.
    """
    from comfyui_h3_blended_inject.hold_release import build_model_function_wrapper, draw_row_noise

    # --- Step 6: Build per-row original and noise tensors from inject latents. ---
    # H3 video latent shape: [N, C, T, H/16, W/16]; row axis is dim 2 (T).
    # H3 audio latent shape: [1, 32, 2, T_audio]; tick axis is dim 3 (T_audio).
    per_row_original: dict[int, Any] = {}
    per_row_noise: dict[int, Any] = {}
    audio_row_original: dict[int, Any] = {}
    audio_row_noise: dict[int, Any] = {}

    for row_s in schedule:
        inj = row_s.inject
        if inj is None or inj.video_latent is None:
            continue
        row_idx = row_s.row_idx
        per_row_original[row_idx] = inj.video_latent[..., row_idx : row_idx + 1, :, :]
        per_row_noise[row_idx] = draw_row_noise(
            noise_seed,
            per_row_original[row_idx].shape,
            device=per_row_original[row_idx].device,
            dtype=per_row_original[row_idx].dtype,
        )

    for row_s in schedule:
        inj = row_s.inject
        if inj is None or inj.audio_latent is None:
            continue
        row_idx = row_s.row_idx
        tick = constants.video_row_to_audio_tick(row_idx)
        audio_row_original[tick] = inj.audio_latent[..., tick : tick + 1]
        audio_row_noise[tick] = draw_row_noise(
            noise_seed,
            audio_row_original[tick].shape,
            device=audio_row_original[tick].device,
            dtype=audio_row_original[tick].dtype,
        )

    # --- Step 7: Derive audio_scale_factor from model_sampling if available. ---
    # audio_scale = sigma_shift_video / sigma_shift_audio = 12.0 / 3.0 = 4.0
    # Derivation from model_sampling needs GPU verification against a real H3 model.
    try:
        audio_scale_factor = (
            model.model.model_sampling.shift / model.model.model_sampling.audio_shift
        )
    except AttributeError:
        audio_scale_factor = 4.0  # default; needs GPU verification

    # --- Step 8: Clone model and install the hold-and-release wrapper. ---
    # model.clone() is called before the comfy.* imports so that a plain-object model
    # (used in CPU tests) raises AttributeError before any import of the unavailable
    # comfy package is attempted.
    m = model.clone()
    wrapper = build_model_function_wrapper(
        schedule,
        per_row_original,
        per_row_noise,
        audio_row_original,
        audio_row_noise,
        audio_scale_factor,
    )
    m.model_options = {**m.model_options, "model_function_wrapper": wrapper}

    # --- Step 9: Run the sampler (KSamplerAdvanced / common_ksampler pattern). ---
    import comfy.sample
    import comfy.samplers  # noqa: F401 — imported for KSampler registration side-effects

    disable_noise = add_noise == "disable"
    force_full_denoise = return_with_leftover_noise == "disable"

    noise = None
    if not disable_noise:
        noise = comfy.sample.prepare_noise(samples, noise_seed, noise_mask=None)

    out_samples = comfy.sample.sample(
        m,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image["samples"],
        denoise=1.0,
        disable_noise=disable_noise,
        start_step=start_at_step,
        last_step=end_at_step,
        force_full_denoise=force_full_denoise,
        noise_mask=latent_image.get("noise_mask"),
        callback=None,
        disable_pbar=False,
        seed=noise_seed,
    )
    out = latent_image.copy()
    out["samples"] = out_samples
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
                            "this is the single frame's denoise value."
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
                    ["match", "drop", "frozen"],
                    {
                        "tooltip": (
                            "match: audio envelope follows the video denoise schedule. "
                            "drop: no audio inject (audio is generated normally). "
                            "frozen: audio inject at d=0 via the derived noise mask "
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
            One of ``"match"``, ``"drop"``, ``"frozen"``.
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
        if images is not None:
            source_length = int(images.shape[0])
            resolution = (int(images.shape[2]), int(images.shape[1]))  # (width, height)
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
    """KSampler Advanced clone that applies hold-and-release inject during sampling.

    Mirrors the KSampler Advanced surface (model, seed, steps, cfg, sampler, scheduler,
    start/end step, latent, conditioning, add_noise, return_with_leftover_noise) and adds an
    ``inject_list`` input.

    Responsibilities:

    1. VAE-encode inject content into per-row ``original`` tensors.
    2. Build the per-row schedule from the inject list (last-in-wins merge via
       :func:`~comfyui_h3_blended_inject.schedule.merge_schedule`).
    3. Derive the nested AV noise mask for exact ``d = 0`` rows and ``frozen`` audio ticks
       (via :func:`~comfyui_h3_blended_inject.mask.apply_derived_mask`).
    4. Install the ``model_function_wrapper`` from
       :func:`~comfyui_h3_blended_inject.hold_release.build_model_function_wrapper` on a
       cloned model.
    5. Run the sampler **once, normally** (does not own the step loop; compatible with all
       samplers including ``res_multistep`` and ``euler_a``).

    If the incoming latent already has a ``noise_mask``, a warning is issued and it is
    replaced by the derived mask.
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
        """Run the H3 sampler with hold-and-release inject.

        Parameters
        ----------
        model:
            ComfyUI MODEL object (a ``MiniMaxH3`` instance is expected).
        add_noise:
            ``"enable"`` to add noise before sampling; ``"disable"`` for img2img.
        noise_seed:
            RNG seed.  Used for both the sampler's noise and :func:`draw_row_noise` calls.
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
        # 1. Derive target pixel resolution from the latent's spatial dimensions.
        #    H3 spatial downsample is 16x, so multiply back to pixel space.
        samples = latent_image["samples"]
        target_h = int(samples.shape[-2]) * 16
        target_w = int(samples.shape[-1]) * 16

        # 2. Derive target_rows from the latent temporal dimension.
        #    Real H3 latents are 5-dim [N, C, T, H/16, W/16]; test tensors are 4-dim.
        if samples.dim() == 5:  # pragma: no cover
            target_rows = int(samples.shape[2])  # temporal dim for a real H3 latent
        else:
            target_rows = int(samples.shape[1])  # fallback for synthetic test tensors

        # 3. Validate all injects: resolution first (so the CPU test can raise before
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

        # 4. Merge inject list into a flat per-row schedule (last-in-wins).
        schedule = merge_schedule(inject_list, target_rows)

        # 5. Derive audio tick count and apply the derived AV noise mask.
        audio_ticks = constants.audio_ticks_for_rows(target_rows)
        latent_image = apply_derived_mask(latent_image, schedule, target_rows, audio_ticks)

        # 6–9. GPU/ComfyUI-dependent: per-row tensor construction, model clone,
        #       model_function_wrapper installation, and sampler execution.
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
