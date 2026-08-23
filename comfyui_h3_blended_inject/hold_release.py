"""Hold-and-release sampler wrapper for fractional-denoise inject rows.

The hold-and-release mechanism intercepts every model evaluation via
``model_options["model_function_wrapper"]`` and, for each held row (sigma > row's denoise
value ``d``), performs two operations:

1. Overwrites the row in the working sample with
   ``(1 - sigma) * original + sigma * noise``.
2. Reports the row's denoised prediction as ``original``.

Once sigma falls to ``d`` or below, the row is released — no intervention, no injection
step.  The held path is exactly where a ``d``-noisy frame belongs at every step, so release
is the natural absence of intervention.

Audio rows use the shifted audio sigma (from
:func:`~comfyui_h3_blended_inject.constants.time_shift_sigma`) and the sampler's internal
audio scale factor, not the raw video sigma.

``torch`` may be imported at module top level.  ``comfy`` and ``comfy.utils`` must be
imported *lazily* (inside function bodies) so this module imports without ComfyUI present.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from comfyui_h3_blended_inject.schedule import RowSchedule


def draw_row_noise(
    seed: int,
    shape: tuple[int, ...],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Draw fixed per-row noise from ``seed``, returned as a standard-normal tensor.

    The noise is drawn once at the start of the sampling run and stored per row.  Using a
    fixed noise tensor (rather than re-sampling each step) keeps the hold trajectory
    deterministic and avoids per-step noise drain.

    Parameters
    ----------
    seed:
        Integer RNG seed.  The same seed produces the same noise tensor across calls.
    shape:
        Shape of the output noise tensor (matches the latent row slice shape).
    device:
        Torch device for the output tensor.  Defaults to CPU if ``None``.
    dtype:
        Torch dtype for the output tensor.  Defaults to ``torch.float32`` if ``None``.

    Returns
    -------
    torch.Tensor
        Standard-normal tensor of ``shape``, drawn from a seeded RNG.
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(shape, generator=gen, device=device, dtype=dtype)


def hold_value(
    original: torch.Tensor,
    noise: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Compute the held sample value for one row at step sigma.

    Returns ``(1 - sigma) * original + sigma * noise``.  This places the row exactly on
    the noised-original trajectory at the current sigma, matching the k-diffusion derivative
    expectation so the sampler advances the row correctly under its own math.

    Parameters
    ----------
    original:
        The encoded source content for this row (latent-space tensor).
    noise:
        Fixed per-row noise tensor (from :func:`draw_row_noise`), same shape as ``original``.
    sigma:
        Current sigma value for this stream (video sigma for video rows, shifted audio sigma
        for audio ticks — see :func:`~comfyui_h3_blended_inject.constants.time_shift_sigma`).

    Returns
    -------
    torch.Tensor
        Blended tensor of the same shape as ``original`` and ``noise``.
    """
    return (1.0 - sigma) * original + sigma * noise


def is_held(sigma: float, d: float) -> bool:
    """Return True while the current sigma exceeds the row's denoise value ``d``.

    The row is held (intervention active) when ``sigma > d``.  At ``sigma <= d`` the row is
    released and the hold-and-release mechanism stops touching it.

    Parameters
    ----------
    sigma:
        Current sigma for this stream (video sigma for video rows, shifted audio sigma for
        audio ticks).
    d:
        Row denoise value in [0.0, 1.0].  Rows with ``d == 0.0`` route to the derived mask,
        not to hold-and-release, so this predicate is only called for fractional-d rows.

    Returns
    -------
    bool
        ``True`` if ``sigma > d``, ``False`` otherwise.
    """
    return sigma > d


def audio_internal_scale(
    value: torch.Tensor,
    sigma: float,
    audio_scale_factor: float,
) -> torch.Tensor:
    """Scale an audio latent tensor from raw latent space to the sampler's internal scale.

    The H3 sampler carries audio internally rescaled by ``audio_scale_factor`` (the
    sigma-ratio factor that ``MiniMaxH3.scale_latent_inpaint`` compensates for).  Held audio
    rows must be written at this internal scale; writing at raw latent scale produces audio
    sync drift.

    Parameters
    ----------
    value:
        Raw audio latent tensor (row slice in latent space).
    sigma:
        Current shifted audio sigma for this step.
    audio_scale_factor:
        The scale factor derived from the model's audio handling (matches the
        ``scale_latent_inpaint`` ratio in ``comfy/model_base.py``).

    Returns
    -------
    torch.Tensor
        Rescaled tensor at sampler-internal audio scale, same shape as ``value``.
    """
    return value * audio_scale_factor


def build_model_function_wrapper(
    schedule: list[RowSchedule],
    per_row_original: dict[int, torch.Tensor],
    per_row_noise: dict[int, torch.Tensor],
    audio_row_original: dict[int, torch.Tensor],
    audio_row_noise: dict[int, torch.Tensor],
    audio_scale_factor: float,
    latent_shapes: list[tuple[int, ...]],
    target_rows: int,
    audio_ticks: int,
    default_shift_video: float = 12.0,
    default_shift_audio: float = 3.0,
) -> Callable[[Callable[..., Any], dict[str, Any]], Any]:
    """Return a ``model_function_wrapper`` callable implementing hold-and-release.

    The returned callable is installed as ``model_options["model_function_wrapper"]`` on a
    cloned model before the sampler is called.  It intercepts every model evaluation,
    including multistep samplers' inner evaluations, and applies the hold-and-release logic.

    Wrapper body (see plan for authoritative description):

    1. Recover sigma directly from ``timestep`` (raw k-diffusion sigma in [0, 1]).
    2. Call ``comfy.utils.unpack_latents(input, latent_shapes)`` to separate the packed AV
       latent into its video and audio streams.  ``latent_shapes`` comes from the model
       instance at call time and must be threaded in through the closure.
    3. For each video row in ``schedule`` where ``is_held(sigma_video, d)``:
       - Write ``hold_value(per_row_original[row], per_row_noise[row], sigma_video)`` into
         a copy of the video stream at the row's slice.
    4. For each audio tick corresponding to a held row where ``is_held(sigma_audio, d)``:
       - Write ``hold_value(audio_row_original[tick], audio_row_noise[tick], sigma_audio)``
         at ``audio_internal_scale`` into the audio stream copy.
       - ``sigma_audio = time_shift_sigma(sigma_video)``.
    5. Repack the edited streams with ``comfy.utils.pack_latents`` and call ``apply_model``
       with the edited input.
    6. On the returned prediction tensor, overwrite held video rows with
       ``per_row_original[row]`` and held audio ticks with ``audio_row_original[tick]``
       (at internal audio scale).
    7. Return the edited prediction.

    All writes are made to fresh copies; no in-place modifications to sampler-owned tensors.
    ``comfy`` and ``comfy.utils`` are imported lazily inside the returned callable to keep
    this module importable without ComfyUI.

    Parameters
    ----------
    schedule:
        Merged per-row schedule from
        :func:`~comfyui_h3_blended_inject.schedule.merge_schedule`.
    per_row_original:
        Mapping from video row index to its encoded source content tensor (latent space).
        Each value has shape ``[1, C_v, 1, Hl, Wl]`` — a single temporal row.
    per_row_noise:
        Mapping from video row index to its fixed noise tensor
        (from :func:`draw_row_noise`).
    audio_row_original:
        Mapping from audio tick index to its encoded source audio content tensor.
        Each value has shape ``[1, C_a, 2, 1]`` — a single audio tick.
    audio_row_noise:
        Mapping from audio tick index to its fixed noise tensor.
    audio_scale_factor:
        Factor for converting raw audio latent values to sampler internal scale
        (see :func:`audio_internal_scale`).
    latent_shapes:
        Component shapes returned by ``comfy.utils.pack_latents`` when packing the
        input nested latent.  Closure-captured from ``_run_sampler`` so the wrapper
        does not depend on sampler-timing assignment to ``apply_model.__self__``.
    target_rows:
        Total number of latent video rows (the ``target_rows`` from the sampler).
        Used with ``audio_ticks`` to compute the canonical tick range per video row
        via :func:`~comfyui_h3_blended_inject.constants.audio_tick_range`.
    audio_ticks:
        Total number of audio ticks in the AV latent (the ``audio_ticks`` from the
        sampler).  Used with ``target_rows`` to compute canonical per-row tick ranges.
    default_shift_video:
        Fallback video sigma shift used to compute ``sigma_audio`` when
        ``transformer_options`` does not supply ``"minimax_h3_sigma_shift_video"``.
        Pass the DiT constructor default read from ``model.diffusion_model.sigma_shift_video``
        (typically 12.0 for H3).
    default_shift_audio:
        Fallback audio sigma shift used when ``transformer_options`` does not supply
        ``"minimax_h3_sigma_shift_audio"``.  Pass the DiT constructor default read from
        ``model.diffusion_model.sigma_shift_audio`` (typically 3.0 for H3).

    Returns
    -------
    Callable[[Callable, dict[str, Any]], Any]
        A ``(apply_model, args_dict) -> prediction`` callable suitable for assignment to
        ``model_options["model_function_wrapper"]``.  ``args_dict`` contains keys:
        ``"input"``, ``"timestep"``, ``"c"``, ``"cond_or_uncond"``.

    Notes
    -----
    The wrapper receives the raw k-diffusion sigma in [0, 1] as ``args_dict["timestep"]``; the
    ×1000 model-timestep conversion happens later inside ``_apply_model``.

    Sigma shifts are sourced the same way the real DiT does it (``model.py:533-534``):
    ``args_dict["c"]["transformer_options"]`` is checked first for
    ``"minimax_h3_sigma_shift_video"`` and ``"minimax_h3_sigma_shift_audio"`` (written by the
    ``MiniMax H3 Sigma Shift`` node); ``default_shift_video``/``default_shift_audio`` are used
    as fallback.

    Cond-batching: ``args_dict["input"]`` batch dim may be > 1 when cond/uncond are
    concatenated by ``calc_cond_batch``.  Held-row writes use ``[:, :, t:t+1, :, :]``
    (broadcasting), so they apply identically to every guidance branch.
    ``latent_shapes`` captures batch=1 shapes; ``unpack_latents`` uses the runtime
    batch from ``flat``, so it unpacks any batch correctly.
    """
    from comfyui_h3_blended_inject import constants as _constants

    # Pre-compute: audio tick -> denoise mapping.
    # Use audio_tick_range per row so the mapping is consistent with _run_sampler
    # and derive_mask (canonical range, not "next scheduled row" boundary).
    sorted_schedule = sorted(schedule, key=lambda s: s.row_idx)

    tick_denoise: dict[int, float] = {}
    for row_s in sorted_schedule:
        for tick in _constants.audio_tick_range(row_s.row_idx, target_rows, audio_ticks):
            tick_denoise[tick] = row_s.denoise

    def wrapper(
        apply_model: Callable[..., Any],
        args_dict: dict[str, Any],
    ) -> Any:
        import comfy.utils

        # Step 1: Recover sigma from timestep (raw k-diffusion sigma in [0, 1]).
        sigma_video = float(args_dict["timestep"].flatten()[0].item())
        # Source sigma shifts from transformer_options the same way the DiT does
        # (model.py:533-534), falling back to the DiT constructor defaults.
        topts = args_dict.get("c", {}).get("transformer_options", {}) or {}
        shift_v = float(topts.get("minimax_h3_sigma_shift_video", default_shift_video))
        shift_a = float(topts.get("minimax_h3_sigma_shift_audio", default_shift_audio))
        sigma_audio = _constants.time_shift_sigma(sigma_video, shift_v, shift_a)

        # Step 2: Unpack the packed AV latent into video and audio streams.
        # latent_shapes is closure-captured; unpack_latents keys element counts off
        # prod(shape[1:]) and takes batch from the runtime flat tensor.
        # unpack_latents returns a list of component tensors.
        packed_input = args_dict["input"]
        unpacked = comfy.utils.unpack_latents(packed_input, latent_shapes)
        video, audio = unpacked[0], unpacked[1]
        # video: [B, C_v, T, Hl, Wl]   audio: [B, C_a, 2, audio_t]

        # Work on fresh copies — never mutate sampler-owned tensors.
        video_edit = video.clone()
        audio_edit = audio.clone()

        # Step 3: Write hold_value into held VIDEO rows (temporal dim 2).
        # per_row_original[row_idx] has shape [1, C_v, 1, Hl, Wl]; broadcasts across B.
        for row_s in sorted_schedule:
            row_idx = row_s.row_idx
            if row_idx in per_row_original and is_held(sigma_video, row_s.denoise):
                hv = hold_value(per_row_original[row_idx], per_row_noise[row_idx], sigma_video)
                video_edit[:, :, row_idx : row_idx + 1, :, :] = hv

        # Step 4: Write held AUDIO ticks (audio_t dim 3).
        # audio_row_original[tick] has shape [1, C_a, 2, 1]; broadcasts across B.
        for tick, d in tick_denoise.items():
            if tick in audio_row_original and is_held(sigma_audio, d):
                held_audio = hold_value(
                    audio_row_original[tick], audio_row_noise[tick], sigma_audio
                )
                audio_edit[:, :, :, tick : tick + 1] = audio_internal_scale(
                    held_audio, sigma_audio, audio_scale_factor
                )

        # Step 5: Repack edited streams and call apply_model.
        # pack_latents takes an iterable and returns (flat, shapes); take index [0] for flat.
        packed_edit = comfy.utils.pack_latents([video_edit, audio_edit])[0]
        raw_prediction = apply_model(packed_edit, args_dict["timestep"], **args_dict["c"])

        # Step 6: Unpack prediction and overwrite held rows.
        unpacked_pred = comfy.utils.unpack_latents(raw_prediction, latent_shapes)
        pred_video, pred_audio = unpacked_pred[0], unpacked_pred[1]
        pred_video_edit = pred_video.clone()
        pred_audio_edit = pred_audio.clone()

        # Overwrite held video rows with their original content.
        for row_s in sorted_schedule:
            row_idx = row_s.row_idx
            if row_idx in per_row_original and is_held(sigma_video, row_s.denoise):
                pred_video_edit[:, :, row_idx : row_idx + 1, :, :] = per_row_original[row_idx]

        # Overwrite held audio ticks with original audio at internal scale.
        for tick, d in tick_denoise.items():
            if tick in audio_row_original and is_held(sigma_audio, d):
                pred_audio_edit[:, :, :, tick : tick + 1] = audio_internal_scale(
                    audio_row_original[tick], sigma_audio, audio_scale_factor
                )

        # Step 7: Repack and return the edited prediction.
        return comfy.utils.pack_latents([pred_video_edit, pred_audio_edit])[0]

    return wrapper
