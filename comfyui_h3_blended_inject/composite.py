"""Latent compositing for the per-row img2img sampler.

Two operations on the *unpacked* AV latent components (video ``[1,C,T,Hl,Wl]``, audio
``[1,C,2,audio_t]``):

- :func:`build_clean_reference` builds the ``clean`` reference the initial-noise lerp blends
  toward — the target latent with ALL inject video/audio content composited in (every covered
  row/tick, not only exact-preserve rows).  Fractional rows therefore img2img *from* the
  inject content, and full-generation rows (``m == 1``) ignore the clean term entirely.
- :func:`post_composite_preserve` performs the binary exact-preserve overwrite *after*
  sampling: ``m == 0`` video rows and audio-preserve ticks are copied verbatim from the clean
  reference, guaranteeing exact preservation with no compounding ghost (the old ``noise_mask``
  re-pin, which compounded every step, is gone).

Both are pure tensor ops (no comfy dependency) and CPU-testable.  ``torch`` is required.
"""

from __future__ import annotations

import torch

from comfyui_h3_blended_inject.constants import (
    audio_tick_range,
    inject_audio_ticks_for_row,
    inject_row_map,
)
from comfyui_h3_blended_inject.schedule import RowSchedule


def build_clean_reference(
    video: torch.Tensor | None,
    audio: torch.Tensor | None,
    schedule: list[RowSchedule],
    target_rows: int,
    audio_ticks: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return clones of ``video``/``audio`` with all inject content composited in.

    For every scheduled row backed by an inject, the inject's video latent row is written at
    the target row (regardless of that row's denoise), and — unless the inject is in ``drop``
    audio mode — its audio latent ticks are written at the owning target ticks.  Rows/ticks
    not covered by any inject keep the target latent's values.

    Inject content is cast to the target component's device/dtype before writing.

    Parameters
    ----------
    video:
        Target video latent ``[1, C, T, Hl, Wl]`` (``T == target_rows``), or ``None``.
    audio:
        Target audio latent ``[1, C, 2, audio_t]`` (``audio_t == audio_ticks``), or ``None``.
    schedule:
        Per-row schedule from :func:`~comfyui_h3_blended_inject.schedule.merge_schedule`.
    target_rows:
        Total number of video rows.
    audio_ticks:
        Total number of audio ticks.

    Returns
    -------
    tuple[torch.Tensor | None, torch.Tensor | None]
        ``(clean_video, clean_audio)`` — clones with inject content, or ``None`` for a
        component that was passed as ``None``.
    """
    clean_video = video.clone() if video is not None else None
    clean_audio = audio.clone() if audio is not None else None

    for row_s in schedule:
        inj = row_s.inject
        if inj is None:
            continue

        if clean_video is not None and inj.video_latent is not None:
            n_clip_rows = int(inj.video_latent.shape[2])
            row_map = dict(inject_row_map(inj.inject_at, n_clip_rows, target_rows))
            if row_s.row_idx in row_map:
                clip_row = row_map[row_s.row_idx]
                clean_video[:, :, row_s.row_idx, :, :] = inj.video_latent[:, :, clip_row, :, :].to(
                    device=clean_video.device, dtype=clean_video.dtype
                )

        if clean_audio is not None and inj.audio_latent is not None and inj.audio_mode != "drop":
            n_clip_ticks = int(inj.audio_latent.shape[-1])
            for tick, clip_tick in inject_audio_ticks_for_row(
                row_s.row_idx, inj.inject_at, n_clip_ticks, target_rows, audio_ticks
            ):
                clean_audio[:, :, :, tick] = inj.audio_latent[:, :, :, clip_tick].to(
                    device=clean_audio.device, dtype=clean_audio.dtype
                )

    return clean_video, clean_audio


def post_composite_preserve(
    video: torch.Tensor | None,
    audio: torch.Tensor | None,
    clean_video: torch.Tensor | None,
    clean_audio: torch.Tensor | None,
    schedule: list[RowSchedule],
    target_rows: int,
    audio_ticks: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Overwrite exact-preserve rows/ticks in the sampled latent from the clean reference.

    After sampling, video rows with ``denoise == 0.0`` and audio ticks whose owning row has
    :attr:`~comfyui_h3_blended_inject.schedule.RowSchedule.audio_preserve` are replaced
    verbatim from the clean reference — a binary composite that guarantees exact preservation
    with no ghost.  All other rows/ticks keep their sampled values.

    Parameters
    ----------
    video, audio:
        Sampled latent components (``None`` to skip that stream).
    clean_video, clean_audio:
        The clean reference components from :func:`build_clean_reference`.
    schedule:
        Per-row schedule.
    target_rows:
        Total number of video rows.
    audio_ticks:
        Total number of audio ticks.

    Returns
    -------
    tuple[torch.Tensor | None, torch.Tensor | None]
        ``(out_video, out_audio)`` — clones of the sampled components with preserve
        rows/ticks overwritten, or ``None`` where the input was ``None``.
    """
    out_video = video.clone() if video is not None else None
    out_audio = audio.clone() if audio is not None else None

    for row_s in schedule:
        if (
            out_video is not None
            and clean_video is not None
            and row_s.denoise == 0.0
            and 0 <= row_s.row_idx < target_rows
        ):
            out_video[:, :, row_s.row_idx, :, :] = clean_video[:, :, row_s.row_idx, :, :]

        if out_audio is not None and clean_audio is not None and row_s.audio_preserve:
            for tick in audio_tick_range(row_s.row_idx, target_rows, audio_ticks):
                if 0 <= tick < audio_ticks:
                    out_audio[:, :, :, tick] = clean_audio[:, :, :, tick]

    return out_video, out_audio
