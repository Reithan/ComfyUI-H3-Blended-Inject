"""Grid constants and helpers for the MiniMax H3 latent temporal grid.

The H3 VAE encodes video in independent 17-frame chunks (``CLIP_LENGTH = 17``).  Each chunk
produces 5 latent rows (``TOKENS_PER_CHUNK``): local row 0 covers 1 source frame, local rows
1-4 each cover 4 source frames (``FRAME_PER_TOKEN = (1, 4, 4, 4, 4)``).  After concatenating
all chunks the last ``TOKEN_DROP = 3`` rows are dropped, so a clip of ``n_frames`` source
frames yields ``5 * ceil(n_frames / 17) - 3`` latent rows.  Valid clip lengths (those whose
last chunk produces exactly the dropped rows) are ``17k + 5``: 5, 22, 39, 56, …

The per-17-frame chunk reset means:
- Row 0 covers frame 0;  local row 0 of chunk 1 covers frame 17;  etc.
- Global row ``r`` belongs to chunk ``r // 5``; its local position within that chunk is
  ``r % 5``.

Audio is encoded at its own tick rate derived from ``AUDIO_HZ``.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Plain literal constants — direct assignments allowed per task contract
# ---------------------------------------------------------------------------

#: Frames covered per latent row across the five temporal groups in one chunk.
#: Local row 0 covers 1 frame; local rows 1-4 each cover 4 frames (total 17 per chunk).
FRAME_PER_TOKEN: tuple[int, ...] = (1, 4, 4, 4, 4)

#: Native video frame rate assumed by all inject nodes.  No fps conversion is performed.
FPS: int = 24

#: Audio tick rate in ticks-per-second used by the H3 audio latent stream.
AUDIO_HZ: float = 40.0

# ---------------------------------------------------------------------------
# VAE temporal-grid parameters — sourced from comfy/ldm/minimax/vae.py
# ---------------------------------------------------------------------------

#: Number of source frames encoded per VAE chunk (``clip_length`` in MiniMaxVAE).
CLIP_LENGTH: int = 17

#: Number of global tail rows dropped after concatenating all chunks (``token_drop`` in
#: MiniMaxVAE).  For a valid clip length these correspond to the padding rows of the last
#: chunk.
TOKEN_DROP: int = 3

#: Temporal downsampling ratio: ``prod(time_down)`` from MiniMaxVAE defaults.
#: Each chunk's ``CLIP_LENGTH`` source frames produce ``ceil(CLIP_LENGTH / VAE_RATIO_T) = 5``
#: latent rows.
VAE_RATIO_T: int = 4

#: Latent rows produced by one 17-frame chunk before the global token_drop.
#: Equal to ``ceil(CLIP_LENGTH / VAE_RATIO_T)``.
TOKENS_PER_CHUNK: int = 5  # ceil(17 / 4)

# ---------------------------------------------------------------------------
# Timestep constants — confirmed from comfy/ldm/minimax/model.py `_forward` 553-626
# (video pin ~589, audio pin ~601-609).
# See wiki: .claude/docs/per-row-img2img/native-h3-mechanism/dit-forward.md
# ---------------------------------------------------------------------------

#: Conditioning timestep pinned for visual-stream preserved rows (near-clean label).
#: Confirmed: ``t_pin_v = max(t_v, VISUAL_COND_TIMESTEP)`` at ``_forward`` ~589 in
#: ``comfy/ldm/minimax/model.py``.
VISUAL_COND_TIMESTEP: float = 0.999

#: Conditioning timestep pinned for audio-stream preserved ticks (near-clean label).
#: Confirmed: audio row-label clamp at ``_forward`` ~601-609 in
#: ``comfy/ldm/minimax/model.py``.
AUDIO_COND_TIMESTEP: float = 1.0


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def row_start_frame(row_idx: int) -> int:
    """Return the absolute source-frame index of the first frame covered by ``row_idx``.

    The latent grid resets every ``CLIP_LENGTH`` (17) frames.  For global row ``r``:
    - ``chunk = r // TOKENS_PER_CHUNK``
    - ``local_row = r % TOKENS_PER_CHUNK``
    - start = ``CLIP_LENGTH * chunk`` when ``local_row == 0`` (chunk boundary);
      ``CLIP_LENGTH * chunk + 1 + (local_row - 1) * VAE_RATIO_T`` otherwise.

    This helper is also used to compute the total source-frame count covered by ``n_rows``
    latent rows: ``row_start_frame(n_rows)`` equals the frame index at which a hypothetical
    next row would begin, which is the total source-frame extent for valid ``n_rows`` values.

    Parameters
    ----------
    row_idx:
        Zero-based latent row index.  May equal ``total_rows`` (used by
        :func:`audio_ticks_for_rows`); must be non-negative.

    Returns
    -------
    int
        Absolute source-frame index of the first frame in the row.
    """
    chunk = row_idx // TOKENS_PER_CHUNK
    local_row = row_idx % TOKENS_PER_CHUNK
    if local_row == 0:
        return CLIP_LENGTH * chunk
    return CLIP_LENGTH * chunk + 1 + (local_row - 1) * VAE_RATIO_T


def row_frame_count(row_idx: int) -> int:
    """Return the number of source frames covered by latent row ``row_idx``.

    The count resets with the per-17-frame chunk structure: the first local row of every
    chunk (local row 0, i.e., global rows 0, 5, 10, 15, …) covers 1 source frame; all
    other local rows cover ``VAE_RATIO_T`` (4) source frames.

    Parameters
    ----------
    row_idx:
        Zero-based latent row index.  Must be non-negative.

    Returns
    -------
    int
        1 for chunk-boundary rows (``row_idx % TOKENS_PER_CHUNK == 0``);
        ``VAE_RATIO_T`` (4) for all other rows.

    Raises
    ------
    ValueError
        If ``row_idx`` is negative.
    """
    if row_idx < 0:
        raise ValueError(f"row_idx must be non-negative, got {row_idx}")
    return FRAME_PER_TOKEN[0] if row_idx % TOKENS_PER_CHUNK == 0 else FRAME_PER_TOKEN[1]


def frame_to_row(frame_idx: int) -> int:
    """Return the latent row index that contains source frame ``frame_idx``.

    The mapping resets every ``CLIP_LENGTH`` (17) frames.  For global frame ``f``:
    - ``chunk = f // CLIP_LENGTH``
    - ``local = f % CLIP_LENGTH``
    - ``local_row = 0`` when ``local == 0`` (first frame of chunk);
      ``1 + (local - 1) // VAE_RATIO_T`` otherwise.
    - result = ``TOKENS_PER_CHUNK * chunk + local_row``

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
    chunk = frame_idx // CLIP_LENGTH
    local = frame_idx % CLIP_LENGTH
    local_row = 0 if local == 0 else 1 + (local - 1) // VAE_RATIO_T
    return TOKENS_PER_CHUNK * chunk + local_row


def total_rows(n_frames: int) -> int:
    """Return the total number of latent rows for a clip with ``n_frames`` source frames.

    **Special case — F=1:** The H3 VAE special-cases a single input frame
    (``x.shape[2] == 1``) and returns exactly **1** latent row, bypassing the
    17-frame chunked path and ``token_drop``.  So ``total_rows(1) == 1``.

    For ``n_frames >= 2`` each 17-frame chunk encodes to ``TOKENS_PER_CHUNK`` (5) rows;
    after concatenating all chunks the last ``TOKEN_DROP`` (3) rows are dropped.  The
    formula is: ``5 * ceil(n_frames / 17) - 3``.

    Valid clip lengths are ``{1} ∪ {17k + 5}`` (1, 5, 22, 39, 56, …) — the F=1
    single-frame path and the multi-frame paths whose dropped rows correspond exactly
    to padding-derived rows of the final chunk.

    Parameters
    ----------
    n_frames:
        Total number of source frames.  Must be at least 1.

    Returns
    -------
    int
        Number of latent rows after the global token-drop.

    Raises
    ------
    ValueError
        If ``n_frames`` is less than 1.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be at least 1, got {n_frames}")
    if n_frames == 1:
        return 1
    return TOKENS_PER_CHUNK * math.ceil(n_frames / CLIP_LENGTH) - TOKEN_DROP


def row_center_times(row_idx: int) -> tuple[float, ...]:
    """Return the source-frame center times (fractional frame units) covered by ``row_idx``.

    Uses the per-17-chunk grid structure to determine which source frames the row spans,
    then returns one center time per source frame in that span.

    Chunk-boundary rows (``row_idx % TOKENS_PER_CHUNK == 0``, e.g. rows 0, 5, 10, …)
    cover exactly 1 frame and return a 1-tuple.  All other rows cover 4 frames and return
    a 4-tuple.

    Parameters
    ----------
    row_idx:
        Zero-based index of the latent row.  Must be non-negative.

    Returns
    -------
    tuple[float, ...]
        One fractional frame-time value per source frame covered by the row.

    Raises
    ------
    ValueError
        If ``row_idx`` is negative.
    """
    if row_idx < 0:
        raise ValueError(f"row_idx must be non-negative, got {row_idx}")
    start = row_start_frame(row_idx)
    count = row_frame_count(row_idx)
    return tuple(float(start + i) + 0.5 for i in range(count))


def video_row_to_audio_tick(row_idx: int) -> int:
    """Return the audio tick index aligned to the start of latent video row ``row_idx``.

    The audio tick is derived from the row's absolute start frame via
    ``round(row_start_frame(row_idx) * AUDIO_HZ / FPS)``.

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
    return round(row_start_frame(row_idx) * AUDIO_HZ / FPS)


def audio_tick_range(row_idx: int, n_rows: int, audio_ticks: int) -> range:
    """Return the range of audio tick indices owned by video row ``row_idx``.

    Row ``r`` owns the half-open interval ``[start, end)`` where:

    - ``start = video_row_to_audio_tick(row_idx)``, clamped to ``[0, audio_ticks)``.
    - ``end = video_row_to_audio_tick(row_idx + 1)`` for all rows except the last.
    - ``end = audio_ticks`` for the final row (``row_idx == n_rows - 1``).
    - ``end`` is further clamped to ``audio_ticks``.

    Adjacent rows' ranges are contiguous (no gap, no overlap), and together they tile
    ``[0, audio_ticks)`` exactly when ``audio_ticks == audio_ticks_for_rows(n_rows)``.

    Parameters
    ----------
    row_idx:
        Zero-based latent video row index.  Must be non-negative.
    n_rows:
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
    if row_idx >= n_rows - 1:
        end = audio_ticks
    else:
        end = video_row_to_audio_tick(row_idx + 1)
    start = min(start, audio_ticks)
    end = min(end, audio_ticks)
    return range(start, end)


def audio_ticks_for_rows(n_rows: int) -> int:
    """Return the number of audio ticks in the audio latent for a clip with ``n_rows`` video rows.

    Uses ``row_start_frame(n_rows)`` to compute the total source-frame extent for the given
    number of rows, then converts to audio ticks via
    ``round(row_start_frame(n_rows) * AUDIO_HZ / FPS)``.

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
    return round(row_start_frame(n_rows) * AUDIO_HZ / FPS)


def inject_row_map(inject_at: int, n_clip_rows: int, target_rows: int) -> list[tuple[int, int]]:
    """Return ``(target_row, clip_row)`` pairs for the valid clip↔target row overlap.

    For each clip row index in ``[0, n_clip_rows)``, the corresponding target row is
    ``frame_to_row(inject_at) + clip_row``.  Only pairs where both indices are in-bounds
    are returned — clip rows that map to target rows outside ``[0, target_rows)`` are
    silently dropped.

    This helper encapsulates the clip↔target mapping used by both the hold-and-release
    loop and the d==0 composite in ``_run_sampler``.

    Because ``inject_at`` is always a multiple of 17 (enforced by
    :func:`~comfyui_h3_blended_inject.sanitize.snap_inject_at`), the result of
    ``frame_to_row(inject_at)`` is always a chunk-boundary row (``5 * (inject_at // 17)``).
    Clip row 0 therefore aligns exactly with the chunk boundary — no sub-row offset.

    Parameters
    ----------
    inject_at:
        Latent FRAME index in the target latent where the inject begins (multiple of 17).
    n_clip_rows:
        Number of rows in the inject clip's video latent (``video_latent.shape[2]``).
    target_rows:
        Total number of rows in the target latent.

    Returns
    -------
    list[tuple[int, int]]
        ``(target_row, clip_row)`` pairs in ascending ``clip_row`` order, with
        ``0 <= clip_row < n_clip_rows`` and ``0 <= target_row < target_rows``.
    """
    inject_at_row = frame_to_row(inject_at)
    return [
        (inject_at_row + clip_row, clip_row)
        for clip_row in range(n_clip_rows)
        if 0 <= inject_at_row + clip_row < target_rows
    ]


def inject_audio_ticks_for_row(
    row_idx: int,
    inject_at: int,
    n_clip_ticks: int,
    target_rows: int,
    audio_ticks: int,
) -> list[tuple[int, int]]:
    """Return ``(target_tick, clip_tick)`` pairs for the given target video row.

    Uses :func:`audio_tick_range` to enumerate the target ticks owned by ``row_idx``,
    then maps each tick to its clip offset via
    ``inject_start_tick = video_row_to_audio_tick(frame_to_row(inject_at))``.  Only pairs
    where ``clip_tick`` falls in ``[0, n_clip_ticks)`` are returned.

    This mirrors the per-tick mapping in the hold-and-release audio loop inside
    ``_run_sampler``, encapsulated here so both that loop and the d==0 composite
    use identical bounds logic.

    Parameters
    ----------
    row_idx:
        Zero-based target video row index.
    inject_at:
        Latent FRAME index in the target latent where the inject begins (multiple of 17).
    n_clip_ticks:
        Number of audio ticks in the inject clip's audio latent
        (``audio_latent.shape[-1]``).
    target_rows:
        Total number of rows in the target latent.
    audio_ticks:
        Total number of audio ticks in the target audio latent.

    Returns
    -------
    list[tuple[int, int]]
        ``(target_tick, clip_tick)`` pairs where ``0 <= clip_tick < n_clip_ticks``.
    """
    inject_at_row = frame_to_row(inject_at)
    inject_start_tick = video_row_to_audio_tick(inject_at_row)
    result = []
    for tick in audio_tick_range(row_idx, target_rows, audio_ticks):
        clip_tick = tick - inject_start_tick
        if 0 <= clip_tick < n_clip_ticks:
            result.append((tick, clip_tick))
    return result
