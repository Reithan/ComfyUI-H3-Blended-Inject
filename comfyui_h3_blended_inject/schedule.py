"""Inject data model and per-row schedule merge.

An :class:`Inject` describes one inject configuration added to the chain by ``H3AddInject``.
:func:`merge_schedule` collapses a list of injects into a per-row schedule using last-in-wins
at row granularity: a later inject in the list overwrites earlier injects on every row it claims,
both schedule value and content.  No blending between overlapping injects occurs; the boundary
is a hard edge.

This module is pure Python / stdlib and must import without ``comfy`` or ``torch`` present.
Image and audio handles are typed as ``Any`` to avoid runtime torch/comfy dependencies here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from comfyui_h3_blended_inject.envelope import evaluate_envelope

# Type alias for the list of injects that flows between H3AddInject nodes and into
# H3InjectSampler.  Defined here for use by callers; the ComfyUI type *string* is defined
# separately in nodes.py as INJECT_LIST = "INJECT_LIST".
InjectList = list["Inject"]


@dataclass(eq=False)
class Inject:
    """All parameters needed to schedule and apply one inject into the target latent.

    Fields are stored as provided by the node; sanitization and envelope evaluation happen
    downstream in :mod:`~comfyui_h3_blended_inject.sanitize` and
    :mod:`~comfyui_h3_blended_inject.envelope`.

    Attributes
    ----------
    inject_at:
        Latent FRAME index in the target latent where this inject begins.  Must be a
        multiple of 17 frames after snapping
        (see :func:`~comfyui_h3_blended_inject.sanitize.snap_inject_at`).
        The fade indices (start_fade_in, start_keyframes, end_keyframes, end_fade_out) are
        **clip frame indices** — positions within this inject's own source content.
        Clip frame ``k`` maps to latent frame ``inject_at + k``.
    start_fade_in:
        Source clip frame index where the fade-in begins.  Denoise = 1.0 here.
    start_keyframes:
        Source clip frame index where the hold at ``min_denoise`` begins.
    end_keyframes:
        Source clip frame index where the hold at ``min_denoise`` ends (inclusive).
    end_fade_out:
        Source clip frame index where the fade-out ends.  Denoise = 1.0 here.
    min_denoise:
        Denoise floor during the hold region, in [0.0, 1.0].  For still-inject (degenerate
        envelope), this is the single frame's denoise value.
    interpolation_type:
        Curve applied to both fade-in and fade-out regions.  One of: "ease_in", "ease_out",
        "ease_in_out", "linear", "none".
    audio_mode:
        How the audio stream is handled for this inject.  One of:
        - "match": audio envelope follows the video denoise schedule.
        - "drop": no audio inject; audio rows from this inject are left as generation.
        - "frozen": audio inject at d=0 via the derived noise mask (exact preservation).
    images:
        IMAGE tensor batch ([batch, H, W, C] float32) or None if no video/image inject.
    audio:
        AUDIO dict ({"waveform": Tensor, "sample_rate": int}) or None if no audio inject.
    resolution:
        (width, height) in pixels.  Must be a multiple of 32 and match the target latent.
    source_length:
        Number of source frames in the inject content (used to validate envelope indices).
    """

    inject_at: int
    start_fade_in: int
    start_keyframes: int
    end_keyframes: int
    end_fade_out: int
    min_denoise: float
    interpolation_type: str
    audio_mode: str
    images: Any | None
    audio: Any | None
    resolution: tuple[int, int]
    source_length: int


@dataclass
class RowSchedule:
    """The resolved schedule entry for one target latent video row.

    Produced by :func:`merge_schedule`; consumed by :mod:`~comfyui_h3_blended_inject.mask`
    and :mod:`~comfyui_h3_blended_inject.hold_release`.

    Attributes
    ----------
    row_idx:
        Zero-based index of this row in the target latent.
    denoise:
        Resolved denoise value for this row in [0.0, 1.0].  0.0 = exact preserve (routes to
        derived mask); 1.0 = fully generated; fractional = hold-and-release.
    inject:
        The winning :class:`Inject` for this row, or ``None`` if no inject covers it.
    audio_frozen:
        True iff the winning inject has ``audio_mode == "frozen"``.  When True, the derived
        mask sets the audio ticks corresponding to this row to 0 (exact preserve).
    """

    row_idx: int
    denoise: float
    inject: Inject | None
    audio_frozen: bool = field(default=False)


def merge_schedule(
    inject_list: InjectList,
    target_rows: int,
) -> list[RowSchedule]:
    """Merge a list of injects into a flat per-row schedule using last-in-wins semantics.

    Each inject in ``inject_list`` is evaluated to produce per-row denoise values (via
    :func:`~comfyui_h3_blended_inject.envelope.evaluate_envelope`).  When two injects claim
    the same target row, the later inject in the list (higher index) wins: its denoise value
    and content overwrite the earlier inject entirely.  There is no blending at overlap
    boundaries; the earlier inject's envelope ends with a hard edge at the first row the later
    inject claims.

    Rows not claimed by any inject are represented with ``denoise = 1.0`` and ``inject = None``
    (normal generation; they are not included in the returned list to keep output sparse).

    Parameters
    ----------
    inject_list:
        Ordered list of :class:`Inject` instances.  Append order matches
        ``H3AddInject`` chain order; later entries win on overlap.
    target_rows:
        Total number of rows in the target latent.

    Returns
    -------
    list[RowSchedule]
        One :class:`RowSchedule` per target row *that has at least one inject claiming it*,
        sorted by ``row_idx`` ascending.  Rows with no inject are omitted.

    Notes
    -----
    The caller (``H3InjectSampler``) is responsible for iterating over uncovered rows (no
    entry in the result) and treating them as ``d = 1.0`` / pure generation.
    """
    row_map: dict[int, tuple[Inject, float]] = {}
    for inj in inject_list:
        for row_idx, d in evaluate_envelope(
            inj.start_fade_in,
            inj.start_keyframes,
            inj.end_keyframes,
            inj.end_fade_out,
            inj.min_denoise,
            inj.interpolation_type,
            inj.source_length,
            target_rows,
            inj.inject_at,
        ):
            if 0 <= row_idx < target_rows:
                row_map[row_idx] = (inj, d)  # last writer wins
    return [
        RowSchedule(
            row_idx=row_idx,
            denoise=d,
            inject=inj,
            audio_frozen=(inj.audio_mode == "frozen"),
        )
        for row_idx, (inj, d) in sorted(row_map.items())
    ]
