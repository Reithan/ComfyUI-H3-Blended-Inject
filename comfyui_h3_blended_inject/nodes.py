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

from comfyui_h3_blended_inject.schedule import InjectList

# ComfyUI custom type string for the inject-list wire type.
INJECT_LIST = "INJECT_LIST"


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
        raise NotImplementedError("H3AddInject.add_inject: sanitize, construct Inject, append")


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
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (
                    ["disable", "enable"],
                    {"tooltip": "Return the latent with leftover noise (for chained samplers)."},
                ),
                "inject_list": (INJECT_LIST,),
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
        negative: Any,
        latent_image: dict[str, Any],
        start_at_step: int,
        end_at_step: int,
        return_with_leftover_noise: str,
        inject_list: InjectList,
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
        negative:
            Negative conditioning tensor.
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
        raise NotImplementedError("H3InjectSampler.sample: encode, merge, mask, wrap, sample")


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
