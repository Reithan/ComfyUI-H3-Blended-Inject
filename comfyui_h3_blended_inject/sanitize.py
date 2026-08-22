"""Validation and normalisation helpers for inject node inputs.

All functions that detect non-fatal mismatches issue ``warnings.warn`` with a descriptive
message (warn-and-continue).  Functions that detect fatal mismatches raise ``ValueError``.

Severity contract:
- **Warns** (continues): ``inject_at`` snap, audio-tick position rounding, AV length
  trim/silence-pad, foreign ``noise_mask`` replacement (that one lives in mask.py).
- **Raises** ``ValueError``: image resolution mismatch, envelope index ordering violation.

This module is pure Python / stdlib and must import without ``comfy`` or ``torch`` present.
Audio and image tensors are typed as ``Any`` to avoid runtime dependencies.
"""

from __future__ import annotations

from typing import Any


def snap_inject_at(inject_at: int) -> int:
    """Snap ``inject_at`` down to the nearest multiple of 17 and warn if a snap occurred.

    ``inject_at`` must align to the 17-frame row-group boundary so the inject starts at a
    valid token boundary in the target latent.  Values that are not already multiples of 17
    are silently rounded **down** (floor snap) and a :func:`warnings.warn` is issued with
    the original and snapped values.

    Parameters
    ----------
    inject_at:
        Requested start row index in the target latent.  Must be non-negative.

    Returns
    -------
    int
        Snapped value: ``inject_at - (inject_at % 17)``.

    Warns
    -----
    UserWarning
        If ``inject_at`` is not already a multiple of 17.  Message includes the original
        and snapped values.

    Raises
    ------
    ValueError
        If ``inject_at`` is negative.
    """
    raise NotImplementedError("snap_inject_at: floor-snap to 17n, warn on change")


def snap_inject_at_audio_tick(inject_at: int) -> int:
    """Apply the audio-tick position rule to a (possibly already snapped) ``inject_at``.

    Positions that are multiples of 51 (``51n``) are exact audio tick boundaries and require
    no rounding.  All other multiples of 17 (``17n`` but not ``51n``) land between audio tick
    boundaries (10.2 ticks offset due to ``17 * 0.6 = 10.2``); they are rounded to the
    nearest audio tick and a :func:`warnings.warn` is issued with the millisecond error
    (up to approximately 12.5 ms).

    This is a *position rule* distinct from the length rule.  Valid co-termination clip
    lengths (39, 90, 141, 192) are unaffected by this rule.

    Parameters
    ----------
    inject_at:
        A ``17n``-snapped inject_at value (output of :func:`snap_inject_at`).

    Returns
    -------
    int
        The same ``inject_at`` value unchanged; this function is called for its side-effect
        (the warning).  The rounded audio tick is embedded in the warning message only.

    Warns
    -----
    UserWarning
        If ``inject_at`` is not a multiple of 51.  Message includes the millisecond error
        between the requested position and the nearest audio tick (max ~12.5 ms).
    """
    raise NotImplementedError("snap_inject_at_audio_tick: warn if inject_at % 51 != 0")


def check_resolution(
    images: Any,
    target_width: int,
    target_height: int,
) -> None:
    """Validate that ``images`` resolution is a multiple of 32 and matches the target latent.

    Two conditions are checked in order:

    1. Width and height of ``images`` must each be a multiple of 32.
    2. Width and height must exactly match ``target_width`` and ``target_height``.

    Rescaling is deliberately not performed: a rescaled inject silently changes what "original"
    means for the hold-and-release mechanism, producing wrong results without any visible error.

    Parameters
    ----------
    images:
        IMAGE tensor of shape [batch, H, W, C].  Width = ``images.shape[2]``,
        height = ``images.shape[1]``.
    target_width:
        Expected pixel width (must be a multiple of 32).
    target_height:
        Expected pixel height (must be a multiple of 32).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the image width or height is not a multiple of 32, or if the image dimensions
        do not exactly match ``(target_width, target_height)``.  The message includes the
        actual vs expected dimensions.
    """
    raise NotImplementedError("check_resolution: validate multiple-of-32 and exact match")


def sanitize_audio(
    audio: Any,
    target_sample_rate: int,
    video_duration_frames: int,
    fps: int,
) -> Any:
    """Resample audio to ``target_sample_rate``, then trim or silence-pad to video duration.

    Processing order is intentional: resample *first*, compare lengths *after*.  Comparing
    before resampling would give incorrect duration mismatches when the input sample rate
    differs from the target.

    If the resampled audio is longer than the video duration, it is trimmed (trailing samples
    discarded) and a :func:`warnings.warn` is issued.  If it is shorter, it is silence-padded
    (zero samples appended) and a :func:`warnings.warn` is issued.  If lengths match exactly
    after resampling, no warning is issued.

    Parameters
    ----------
    audio:
        AUDIO dict ``{"waveform": Tensor [C, samples], "sample_rate": int}``.
    target_sample_rate:
        Target audio sample rate in Hz (the rate the H3 audio latent expects).
    video_duration_frames:
        Duration of the video inject content in source frames.
    fps:
        Frames per second of the source video (should be 24 for H3).

    Returns
    -------
    Any
        AUDIO dict with ``"waveform"`` resampled and length-adjusted, and
        ``"sample_rate"`` updated to ``target_sample_rate``.

    Warns
    -----
    UserWarning
        If the resampled audio length does not match the video duration.  Message includes
        the direction (trimmed or padded) and the length difference in samples and seconds.

    Raises
    ------
    TypeError
        If ``audio`` is not a dict with the expected keys.
    """
    raise NotImplementedError("sanitize_audio: resample then trim/pad, warn on mismatch")


def validate_envelope_indices(
    start_fade_in: int,
    start_keyframes: int,
    end_keyframes: int,
    end_fade_out: int,
    source_length: int,
    target_rows: int,
    inject_at_row: int,
) -> None:
    """Validate envelope index ordering and bounds; raise with offending values on violation.

    Ordering constraint (all must hold simultaneously):
    ``start_fade_in <= start_keyframes <= end_keyframes <= end_fade_out``

    Bounds constraints:
    - All indices must be >= 0.
    - ``end_fade_out`` must be < ``source_length``.
    - The row span of the envelope (from ``inject_at_row`` to the row containing
      ``inject_at_row + end_fade_out``) must fit within ``target_rows``.

    Parameters
    ----------
    start_fade_in:
        Source frame index where fade-in begins.
    start_keyframes:
        Source frame index where hold begins.
    end_keyframes:
        Source frame index where hold ends.
    end_fade_out:
        Source frame index where fade-out ends.
    source_length:
        Total number of source frames in the inject content.
    target_rows:
        Total number of rows in the target latent.
    inject_at_row:
        Row index in the target latent where the inject starts.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any ordering or bounds constraint is violated.  The message includes all four
        index values and the specific constraint that failed.
    """
    raise NotImplementedError(
        "validate_envelope_indices: check ordering start_fade_in<=start_keyframes<=end_keyframes"
        "<=end_fade_out and bounds vs source_length / target_rows"
    )
