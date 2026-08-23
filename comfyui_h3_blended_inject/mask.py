"""Per-row fractional denoise mask construction for the per-row img2img sampler.

The primary export is :func:`derive_fractional_mask`, which builds a nested AV mask
encoding each video row's fractional denoise value (``m_r``) and each audio tick's
corresponding ``audio_denoise``.  This mask is consumed by the conditioning wrapper
via ``model._denoise_mask_values``; ``noise_mask`` is set to ``None`` so no H3
compositing occurs.

Mask convention:
- ``0`` = exact preserve (row's ``m_r`` is 0.0, audio tick's ``audio_denoise`` is 0.0)
- ``1`` = full generation (row absent from schedule, or ``m_r`` / ``audio_denoise`` is 1.0)
- fractional = compressed per-row timestep schedule

``torch`` is imported at module top level; this module is not expected to import without
``torch`` present.  ``comfy`` is not imported here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from comfyui_h3_blended_inject.grid import audio_tick_range
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


def derive_fractional_mask(
    schedule: list[RowSchedule],
    video_rows: int,
    audio_ticks: int,
    video_component_shape: tuple[int, ...] | None = None,
    audio_component_shape: tuple[int, ...] | None = None,
    nested_factory: Callable[[torch.Tensor, torch.Tensor], Any] | None = None,
) -> Any:
    """Build the per-row *fractional* denoise mask for the per-row img2img sampler.

    Each row carries its full fractional schedule so the DiT compresses per-row timesteps:

    - **video**: row ``i`` = ``r.denoise`` (its ``m_r``) for scheduled rows; ``1.0`` for
      rows absent from ``schedule`` (full generation).
    - **audio**: tick ``j`` = the owning row's :attr:`RowSchedule.audio_denoise`
      (keep → ``0.0``, fade → ``r.denoise``, drop/none → ``1.0``); ``1.0`` for absent ticks.

    The mask convention matches H3's native ``denoise_mask`` semantics: ``1`` = full
    generation, ``0`` = exact preserve, fractional = compressed per-row schedule.  This mask
    is consumed by the conditioning wrapper (via ``model._denoise_mask_values``), *not* set as
    a ``noise_mask`` — the sampler receives ``noise_mask=None`` so no compositing occurs.

    **Non-nested path** (``video_component_shape`` is ``None``): returns
    ``{"video_mask": float32[1, video_rows], "audio_mask": float32[1, audio_ticks]}``.

    **Nested path** (``video_component_shape`` provided): returns
    ``nested_factory(video_mask_full, audio_mask_full)`` with each mask expanded to the full
    component shape, per-row/per-tick values broadcast across all channels and spatial dims.

    Parameters
    ----------
    schedule:
        Per-row schedule from :func:`~comfyui_h3_blended_inject.schedule.merge_schedule`.
    video_rows:
        Total number of video rows.
    audio_ticks:
        Total number of audio ticks.
    video_component_shape:
        Full video latent shape ``(B, C, T, Hl, Wl)``; triggers the nested path.
    audio_component_shape:
        Full audio latent shape ``(B, C, 2, audio_t)``; required with the nested path.
    nested_factory:
        ``(video_mask, audio_mask) -> NestedTensor``; defaults to importing
        ``comfy.nested_tensor.NestedTensor`` lazily.  Supply a fake in tests.

    Returns
    -------
    dict[str, torch.Tensor] | NestedTensor
        Non-nested: dict with fractional ``"video_mask"``/``"audio_mask"``.
        Nested: NestedTensor wrapping the two full-shape fractional mask components.
    """
    # Per-row video m_r (default 1.0 = generate) and per-tick audio m_r.
    video_values: list[float] = [1.0] * video_rows
    audio_values: list[float] = [1.0] * audio_ticks

    for r in schedule:
        if 0 <= r.row_idx < video_rows:
            video_values[r.row_idx] = r.denoise
        a = r.audio_denoise
        for tick in audio_tick_range(r.row_idx, video_rows, audio_ticks):
            if 0 <= tick < audio_ticks:
                audio_values[tick] = a

    if video_component_shape is None:
        video_mask = torch.tensor(video_values, dtype=torch.float32).unsqueeze(0)
        audio_mask = torch.tensor(audio_values, dtype=torch.float32).unsqueeze(0)
        return {"video_mask": video_mask, "audio_mask": audio_mask}

    if audio_component_shape is None:
        raise ValueError(
            "audio_component_shape must be provided when video_component_shape is given"
        )

    # video_component_shape: (B, C, T, Hl, Wl) — T == video_rows.
    # Broadcast per-row values across all B, C, Hl, Wl dims.
    vt = torch.tensor(video_values, dtype=torch.float32)
    video_mask_full = vt.view(1, 1, -1, 1, 1).expand(*video_component_shape).clone()

    # audio_component_shape: (B, C, 2, audio_t) — audio_t == audio_ticks.
    # Broadcast per-tick values across all B, C, and the size-2 dim.
    at = torch.tensor(audio_values, dtype=torch.float32)
    audio_mask_full = at.view(1, 1, 1, -1).expand(*audio_component_shape).clone()

    factory = nested_factory if nested_factory is not None else _default_nested_factory
    return factory(video_mask_full, audio_mask_full)
