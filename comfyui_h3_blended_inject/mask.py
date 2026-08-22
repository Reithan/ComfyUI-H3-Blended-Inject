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
from typing import Any

import torch

from comfyui_h3_blended_inject.constants import video_row_to_audio_tick
from comfyui_h3_blended_inject.schedule import RowSchedule


def derive_mask(
    schedule: list[RowSchedule],
    video_rows: int,
    audio_ticks: int,
) -> dict[str, torch.Tensor]:
    """Build the nested AV noise mask from the merged per-row schedule.

    For the **video stream**: rows with ``denoise == 0.0`` (exact d=0, as determined by
    :func:`~comfyui_h3_blended_inject.envelope.is_row_exactly_zero`) are set to 0 (preserve).
    All other rows — including fractional-denoise rows handled by hold-and-release — are 1
    (generate).  Rows not present in ``schedule`` are 1 (generate, normal diffusion).

    For the **audio stream**: ticks whose corresponding row has ``audio_frozen == True`` are
    set to 0 (preserve).  All other ticks are 1 (generate).

    Parameters
    ----------
    schedule:
        Per-row schedule output of
        :func:`~comfyui_h3_blended_inject.schedule.merge_schedule`.
        Rows absent from the list are treated as ``d = 1.0``.
    video_rows:
        Total number of video rows in the target latent.
    audio_ticks:
        Total number of audio ticks in the audio latent stream.

    Returns
    -------
    dict[str, torch.Tensor]
        Nested AV mask structure with keys:
        - ``"video_mask"``: float32 tensor of shape [1, video_rows], values in {0.0, 1.0}.
        - ``"audio_mask"``: float32 tensor of shape [1, audio_ticks], values in {0.0, 1.0}.

        This dict is the value that gets stored under the ``"noise_mask"`` key of the
        ComfyUI latent dict via :func:`apply_derived_mask`.
    """
    video_mask = torch.ones(1, video_rows, dtype=torch.float32)
    audio_mask = torch.ones(1, audio_ticks, dtype=torch.float32)

    for r in schedule:
        if r.denoise == 0.0:
            video_mask[0, r.row_idx] = 0.0
        if r.audio_frozen:
            tick = video_row_to_audio_tick(r.row_idx)
            if tick < audio_ticks:
                audio_mask[0, tick] = 0.0

    return {"video_mask": video_mask, "audio_mask": audio_mask}


def apply_derived_mask(
    latent: dict[str, Any],
    schedule: list[RowSchedule],
    video_rows: int,
    audio_ticks: int,
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

    Returns
    -------
    dict[str, Any]
        Shallow copy of ``latent`` with ``"noise_mask"`` set to the derived mask dict.

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
    mask = derive_mask(schedule, video_rows, audio_ticks)
    new_latent = dict(latent)
    new_latent["noise_mask"] = mask
    return new_latent
