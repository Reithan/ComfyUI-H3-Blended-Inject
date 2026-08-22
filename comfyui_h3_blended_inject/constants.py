"""Grid constants and helper stubs for the MiniMax H3 1/4/4/4/4 latent grid.

The H3 latent grid encodes video as rows where the first row covers 1 source frame and every
subsequent row covers 4 source frames, giving a 17-frame repeating unit (1 + 4*4).  Audio is
encoded at its own tick rate derived from AUDIO_HZ.

Constants that are plain literals are assigned directly; any value requiring computation is a
stub function (``raise NotImplementedError``).
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Plain literal constants — direct assignments allowed per task contract
# ---------------------------------------------------------------------------

#: Frames covered per latent row across the five temporal groups.
#: Row 0 covers 1 frame; rows 1-4 each cover 4 frames (total 17 per token group).
FRAME_PER_TOKEN: tuple[int, ...] = (1, 4, 4, 4, 4)

#: Native video frame rate assumed by all inject nodes.  No fps conversion is performed.
FPS: int = 24

#: Audio tick rate in ticks-per-second used by the H3 audio latent stream.
AUDIO_HZ: float = 40.0

# ---------------------------------------------------------------------------
# Timestep constants sourced from comfy/ldm/minimax/model.py
# TODO: verify exact values against ComfyUI source before shipping
# ---------------------------------------------------------------------------

#: Conditioning timestep pinned for visual-stream preserved rows (near-clean label).
#: From ``VISUAL_COND_TIMESTEP`` in ``comfy/ldm/minimax/model.py``.
VISUAL_COND_TIMESTEP: float = 0.999  # sourced from comfy/ldm/minimax/model.py

#: Conditioning timestep pinned for audio-stream preserved ticks (near-clean label).
#: From ``AUDIO_COND_TIMESTEP`` in ``comfy/ldm/minimax/model.py``.
AUDIO_COND_TIMESTEP: float = 1.0  # sourced from comfy/ldm/minimax/model.py


# ---------------------------------------------------------------------------
# Grid helpers — computation required; bodies are stubs
# ---------------------------------------------------------------------------


def row_frame_count(row_idx: int) -> int:
    """Return the number of source frames covered by latent row ``row_idx``.

    Row 0 covers 1 frame (the first element of FRAME_PER_TOKEN); all subsequent rows
    cover 4 frames (the repeating elements).

    Parameters
    ----------
    row_idx:
        Zero-based latent row index.  Must be non-negative.

    Returns
    -------
    int
        1 for row 0, 4 for all other rows.

    Raises
    ------
    ValueError
        If ``row_idx`` is negative.
    """
    if row_idx < 0:
        raise ValueError(f"row_idx must be non-negative, got {row_idx}")
    return FRAME_PER_TOKEN[0] if row_idx == 0 else FRAME_PER_TOKEN[1]


def frame_to_row(frame_idx: int) -> int:
    """Return the latent row index that contains source frame ``frame_idx``.

    Row 0 holds frame 0; frames 1-4 are in row 1; frames 5-8 in row 2; etc.

    Parameters
    ----------
    frame_idx:
        Zero-based source frame index.  Must be non-negative.

    Returns
    -------
    int
        Zero-based latent row index containing the frame.

    Raises
    ------
    ValueError
        If ``frame_idx`` is negative.
    """
    if frame_idx < 0:
        raise ValueError(f"frame_idx must be non-negative, got {frame_idx}")
    if frame_idx == 0:
        return 0
    return 1 + (frame_idx - 1) // 4


def total_rows(n_frames: int) -> int:
    """Return the total number of latent rows for a clip with ``n_frames`` source frames.

    Parameters
    ----------
    n_frames:
        Total number of source frames.  Must be at least 1.

    Returns
    -------
    int
        Number of latent rows required.

    Raises
    ------
    ValueError
        If ``n_frames`` is less than 1.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be at least 1, got {n_frames}")
    return 1 + math.ceil((n_frames - 1) / 4)


def row_center_times(row_idx: int) -> tuple[float, ...]:
    """Return the source-frame center times (fractional frame units) covered by ``row_idx``.

    Uses the 1/4/4/4/4 FRAME_PER_TOKEN structure to determine which source frames the row
    spans, then returns one center time per source frame in that span.

    Row 0 covers exactly 1 frame; rows 1+ each cover 4 frames.  The center time of each
    frame within a row is computed from the row's absolute start frame position.

    Parameters
    ----------
    row_idx:
        Zero-based index of the latent row.  Must be non-negative.

    Returns
    -------
    tuple[float, ...]
        One fractional frame-time value per source frame covered by the row.
        Row 0 returns a 1-tuple; rows 1+ return 4-tuples.

    Raises
    ------
    ValueError
        If ``row_idx`` is negative.
    """
    if row_idx < 0:
        raise ValueError(f"row_idx must be non-negative, got {row_idx}")
    if row_idx == 0:
        # Row 0 covers frame 0; its center is at 0.5.
        return (0.5,)
    # Rows 1+ each cover 4 frames.  Row r starts at frame 1 + (r - 1) * 4.
    start_frame = 1 + (row_idx - 1) * 4
    return tuple(float(start_frame + i) + 0.5 for i in range(4))


def video_row_to_audio_tick(row_idx: int) -> int:
    """Return the audio tick index aligned to the start of latent video row ``row_idx``.

    Audio ticks are derived from the video row's absolute start frame position and AUDIO_HZ.
    The audio tick is the tick whose center time is closest to the row's start frame time
    (at FPS).

    Parameters
    ----------
    row_idx:
        Zero-based latent video row index.

    Returns
    -------
    int
        Zero-based audio tick index aligned to the row start.

    Raises
    ------
    ValueError
        If ``row_idx`` is negative.
    """
    if row_idx < 0:
        raise ValueError(f"row_idx must be non-negative, got {row_idx}")
    start_frame = 0 if row_idx == 0 else 1 + (row_idx - 1) * 4
    return round(start_frame * AUDIO_HZ / FPS)


def audio_tick_range(row_idx: int, total_rows: int, audio_ticks: int) -> range:
    """Return the range of audio tick indices owned by video row ``row_idx``.

    Row ``r`` owns the half-open interval ``[start, end)`` where:

    - ``start = video_row_to_audio_tick(row_idx)``, clamped to ``[0, audio_ticks)``.
    - ``end = video_row_to_audio_tick(row_idx + 1)`` for all rows except the last.
    - ``end = audio_ticks`` for the final row (``row_idx == total_rows - 1``).
    - ``end`` is further clamped to ``audio_ticks``.

    Adjacent rows' ranges are contiguous (no gap, no overlap), and together they tile
    ``[0, audio_ticks)`` exactly when ``audio_ticks == audio_ticks_for_rows(total_rows)``.

    Parameters
    ----------
    row_idx:
        Zero-based latent video row index.  Must be non-negative.
    total_rows:
        Total number of latent video rows (the ``target_rows`` from the sampler).
    audio_ticks:
        Total number of audio ticks (the ``audio_ticks`` from the sampler).

    Returns
    -------
    range
        Half-open integer range ``[start, end)`` of audio tick indices for this row.
        May be empty if ``row_idx`` is beyond the audio extent.

    Raises
    ------
    ValueError
        If ``row_idx`` is negative.
    """
    if row_idx < 0:
        raise ValueError(f"row_idx must be non-negative, got {row_idx}")
    start = max(0, video_row_to_audio_tick(row_idx))
    if row_idx >= total_rows - 1:
        end = audio_ticks
    else:
        end = video_row_to_audio_tick(row_idx + 1)
    start = min(start, audio_ticks)
    end = min(end, audio_ticks)
    return range(start, end)


def audio_ticks_for_rows(n_rows: int) -> int:
    """Return the number of audio ticks in the audio latent for a clip with ``n_rows`` video rows.

    Parameters
    ----------
    n_rows:
        Total number of latent video rows.

    Returns
    -------
    int
        Number of audio ticks.
    """
    if n_rows == 0:
        return 0
    # Total source frames for n_rows equals the start frame of the hypothetical next row,
    # which is the same formula used by video_row_to_audio_tick(n_rows).
    start_frame = 1 + (n_rows - 1) * 4
    return round(start_frame * AUDIO_HZ / FPS)


def time_shift_sigma(sigma: float) -> float:
    """Return the shifted audio sigma for a given video sigma.

    Mirrors ``time_shift_sigma`` from ``comfy/ldm/minimax/model.py``.  Audio rows release
    against this shifted sigma, not the raw video sigma, to keep audio and video fades
    temporally aligned.

    Parameters
    ----------
    sigma:
        Current video sigma value (scalar, in [0, 1] space).

    Returns
    -------
    float
        Shifted sigma value appropriate for the audio stream.
    """
    # Two-step warp from comfy/ldm/minimax/model.py: invert the video shift (12.0) to
    # recover the base grid, then re-apply the audio shift (3.0).
    # This module returns the raw warp value; ComfyUI applies `1.0 - warp` at the model
    # boundary as the audio conditioning timestep.  The contract here is f(0)=0, f(1)=1.
    from_shift = 12.0  # sigma_shift_video
    to_shift = 3.0  # sigma_shift_audio
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return float(to_shift * base / (1.0 + (to_shift - 1.0) * base))
