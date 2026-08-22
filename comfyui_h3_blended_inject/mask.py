"""Derived nested AV noise mask construction.

The derived mask encodes *only* the exact ``d = 0`` spans (fully-preserved rows via the
trained preservation path) and ``audio_mode = "keep"`` audio ticks.  Fractional-denoise
rows are handled entirely by hold-and-release and are set to 1 (generate) in this mask.

Mask convention:
- 0 = preserve (the H3 mask engine holds this row at near-clean conditioning)
- 1 = generate (normal diffusion)

There is exactly one mask author per sampler call.  If the incoming latent already carries a
``noise_mask``, it is replaced with a :func:`warnings.warn` and the derived mask takes over.
Composing with foreign mask nodes is off the supported path.

``torch`` is imported at module top level; this module is not expected to import without
``torch`` present.  ``comfy`` is not imported here.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import torch

from comfyui_h3_blended_inject.constants import audio_tick_range
from comfyui_h3_blended_inject.schedule import RowSchedule


def _default_nested_factory(
    video_mask: torch.Tensor,
    audio_mask: torch.Tensor,
) -> Any:
    """Wrap (video_mask, audio_mask) in a real comfy NestedTensor.

    Imported lazily so this module remains importable without ComfyUI present.
    Tests pass a fake factory via the ``nested_factory`` parameter to avoid this import.
    """
    from comfy.nested_tensor import NestedTensor  # noqa: PLC0415

    return NestedTensor((video_mask, audio_mask))


def derive_mask(
    schedule: list[RowSchedule],
    video_rows: int,
    audio_ticks: int,
    video_component_shape: tuple[int, ...] | None = None,
    audio_component_shape: tuple[int, ...] | None = None,
    nested_factory: Callable[[torch.Tensor, torch.Tensor], Any] | None = None,
) -> Any:
    """Build the nested AV noise mask from the merged per-row schedule.

    For the **video stream**: rows with ``denoise == 0.0`` (exact d=0) are set to 0
    (preserve).  All other rows — including fractional-denoise rows handled by
    hold-and-release — are 1 (generate).  Rows absent from ``schedule`` are 1.

    For the **audio stream**: ticks whose corresponding row has ``audio_frozen == True``
    are set to 0 (preserve).  All other ticks are 1 (generate).

    **Non-nested path** (``video_component_shape`` is ``None``):
        Returns a ``dict`` with keys ``"video_mask"`` and ``"audio_mask"``, each a
        float32 tensor of shape ``[1, <count>]``.  This is the backward-compatible form
        used by CPU tests and the plain-tensor latent path.

    **Nested path** (``video_component_shape`` is provided):
        Returns a ``NestedTensor((video_mask, audio_mask))`` where each mask has the FULL
        component shape (``[B, C, T, Hl, Wl]`` for video; ``[B, C, 2, audio_t]`` for
        audio).  Per-row 0/1 values are expanded across ALL channels and spatial dims so
        that CFGGuider can unpack and repack each component via ``pack_latents``.
        ``nested_factory`` defaults to importing ``comfy.nested_tensor.NestedTensor``;
        tests supply a fake to avoid the comfy dependency.

    Parameters
    ----------
    schedule:
        Per-row schedule from :func:`~comfyui_h3_blended_inject.schedule.merge_schedule`.
    video_rows:
        Total number of video rows.
    audio_ticks:
        Total number of audio ticks.
    video_component_shape:
        Full shape of the video latent component, e.g. ``(B, 24, T, Hl, Wl)``.
        When ``None``, the non-nested (dict) path is used.
    audio_component_shape:
        Full shape of the audio latent component, e.g. ``(B, 32, 2, audio_t)``.
        Required when ``video_component_shape`` is provided.
    nested_factory:
        Callable ``(video_mask, audio_mask) -> NestedTensor``.  Defaults to importing
        ``comfy.nested_tensor.NestedTensor`` lazily.  Supply a fake in tests.

    Returns
    -------
    dict[str, torch.Tensor] | NestedTensor
        Non-nested: dict with ``"video_mask"`` and ``"audio_mask"`` keys.
        Nested: NestedTensor wrapping the two full-shape mask components.
    """
    # --- Derive per-row and per-tick boolean vectors (shared between both paths) ---
    # video_zeros[i] = True  iff schedule has a row at index i with denoise==0.0
    video_zeros: list[bool] = [False] * video_rows
    audio_zeros: list[bool] = [False] * audio_ticks

    for r in schedule:
        if r.denoise == 0.0:
            if 0 <= r.row_idx < video_rows:
                video_zeros[r.row_idx] = True
        if r.audio_frozen:
            for tick in audio_tick_range(r.row_idx, video_rows, audio_ticks):
                if 0 <= tick < audio_ticks:
                    audio_zeros[tick] = True

    if video_component_shape is None:
        # --- Non-nested (dict) path — backward-compatible ---
        video_mask = torch.ones(1, video_rows, dtype=torch.float32)
        audio_mask = torch.ones(1, audio_ticks, dtype=torch.float32)
        for i, zero in enumerate(video_zeros):
            if zero:
                video_mask[0, i] = 0.0
        for j, zero in enumerate(audio_zeros):
            if zero:
                audio_mask[0, j] = 0.0
        return {"video_mask": video_mask, "audio_mask": audio_mask}

    # --- Nested path: expand to full component shapes ---
    assert audio_component_shape is not None, (
        "audio_component_shape must be provided when video_component_shape is given"
    )

    # video_component_shape: (B, C, T, Hl, Wl)  — T == video_rows
    # Build per-T-row scalar values, then expand to [B, C, T, Hl, Wl].
    # Start from all-ones, zero out preserved rows.
    video_mask_full = torch.ones(*video_component_shape, dtype=torch.float32)
    for i, zero in enumerate(video_zeros):
        if zero:
            # Zero out slice at temporal dim 2 (index i) across all B, C, Hl, Wl.
            video_mask_full[:, :, i, :, :] = 0.0

    # audio_component_shape: (B, C, 2, audio_t)  — audio_t == audio_ticks
    audio_mask_full = torch.ones(*audio_component_shape, dtype=torch.float32)
    for j, zero in enumerate(audio_zeros):
        if zero:
            # Zero out slice at audio_t dim 3 (index j) across all B, C, and the size-2 dim.
            audio_mask_full[:, :, :, j] = 0.0

    factory = nested_factory if nested_factory is not None else _default_nested_factory
    return factory(video_mask_full, audio_mask_full)


def apply_derived_mask(
    latent: dict[str, Any],
    schedule: list[RowSchedule],
    video_rows: int,
    audio_ticks: int,
    video_component_shape: tuple[int, ...] | None = None,
    audio_component_shape: tuple[int, ...] | None = None,
    nested_factory: Callable[[torch.Tensor, torch.Tensor], Any] | None = None,
) -> dict[str, Any]:
    """Derive and write the nested AV noise mask onto the latent dict.

    If the incoming ``latent`` already has a ``"noise_mask"`` key, a :func:`warnings.warn`
    is issued and the existing mask is replaced.  Composing with a foreign ``noise_mask``
    (e.g., from ``MiniMaxH3SetAVNoiseMask`` or ``DifferentialDiffusionAdvanced``) is off the
    supported path; the warning explicitly states this.

    Internally calls :func:`derive_mask` to build the mask and writes it back into a shallow
    copy of ``latent`` (the original dict is not mutated).

    Parameters
    ----------
    latent:
        ComfyUI latent dict (must have a ``"samples"`` key).
    schedule:
        Merged per-row schedule from
        :func:`~comfyui_h3_blended_inject.schedule.merge_schedule`.
    video_rows:
        Total number of video rows in the target latent.
    audio_ticks:
        Total number of audio ticks in the audio latent stream.
    video_component_shape:
        Forwarded to :func:`derive_mask`; triggers the nested path when provided.
    audio_component_shape:
        Forwarded to :func:`derive_mask`.
    nested_factory:
        Forwarded to :func:`derive_mask`.

    Returns
    -------
    dict[str, Any]
        Shallow copy of ``latent`` with ``"noise_mask"`` set to the derived mask.
        The mask is a plain dict (non-nested path) or a NestedTensor (nested path);
        see :func:`derive_mask` for the distinction.

    Warns
    -----
    UserWarning
        If ``latent`` already contains a ``"noise_mask"`` key.  Message explains that the
        existing mask is replaced and that composing with foreign mask nodes is unsupported.
    """
    if "noise_mask" in latent:
        warnings.warn(
            "H3InjectSampler: the input latent already carries a 'noise_mask'. "
            "The existing mask will be replaced by the derived AV mask. "
            "Composing with foreign mask nodes (e.g. MiniMaxH3SetAVNoiseMask, "
            "DifferentialDiffusionAdvanced) is not supported and will produce incorrect "
            "results. Remove any upstream mask node before using H3InjectSampler.",
            UserWarning,
            stacklevel=2,
        )
    mask = derive_mask(
        schedule,
        video_rows,
        audio_ticks,
        video_component_shape=video_component_shape,
        audio_component_shape=audio_component_shape,
        nested_factory=nested_factory,
    )
    new_latent = dict(latent)
    new_latent["noise_mask"] = mask
    return new_latent
