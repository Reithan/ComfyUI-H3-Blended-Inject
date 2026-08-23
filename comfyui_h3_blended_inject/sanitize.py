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

import warnings
from typing import Any

from comfyui_h3_blended_inject.constants import frame_to_row

# H3 audio constants (matches nodes.py from the motion-context reference).
_FPS: int = 24
_AUDIO_HZ: float = 40.0


def snap_inject_at(inject_at: int) -> int:
    """Snap ``inject_at`` down to the nearest multiple of 17 frames and warn if a snap occurred.

    ``inject_at`` is a **latent FRAME index** indicating where the inject begins in the
    target latent.  It must align to the 17-frame row-group boundary so the inject starts at
    a valid token boundary.  Values that are not already multiples of 17 are rounded **down**
    (floor snap) and a :func:`warnings.warn` is issued with the original and snapped values.

    Parameters
    ----------
    inject_at:
        Requested latent FRAME index in the target latent.  Must be non-negative.

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
    if inject_at < 0:
        raise ValueError(f"inject_at must be non-negative, got {inject_at}")

    remainder = inject_at % 17
    if remainder == 0:
        return inject_at

    snapped = inject_at - remainder
    warnings.warn(
        f"inject_at={inject_at} is not a multiple of 17; snapped down to {snapped}.",
        UserWarning,
        stacklevel=2,
    )
    return snapped


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
    if inject_at % 51 == 0:
        return inject_at

    # Compute the ms error between inject_at and the nearest audio tick.
    position_s = inject_at / _FPS
    position_ticks = position_s * _AUDIO_HZ
    nearest_tick = round(position_ticks)
    nearest_tick_s = nearest_tick / _AUDIO_HZ
    error_ms = abs(position_s - nearest_tick_s) * 1000.0

    warnings.warn(
        f"inject_at={inject_at} is not a multiple of 51; audio insert start lands "
        f"{error_ms:.2f} ms from the nearest audio tick (tick {nearest_tick}). "
        "Use a multiple of 51 for exact audio-tick alignment.",
        UserWarning,
        stacklevel=2,
    )
    return inject_at


def snap_length_down(source_length: int) -> int:
    """Snap injected content length down to the largest valid H3 clip length (17n+5).

    Valid H3 clip lengths are ``17n + 5`` for n >= 0: 5, 22, 39, 56, 73, 90, 107, …
    A length is valid iff ``source_length >= 5 and (source_length - 5) % 17 == 0``.
    Already-valid lengths are returned unchanged with no warning.

    The snap-down formula is ``5 + 17 * ((source_length - 5) // 17)``.

    Note: this snaps only to the video-grid lattice (17n+5), NOT to the stricter
    joint audio+video lattice (51n+39 = 39, 90, 141, …).  A non-audio-aligned valid
    length mismatches audio by at most ~17 ms confined to the last 1–2 audio ticks
    (tail-local, no global desync, no hard assert in the model), so the joint lattice
    is only precision-optimal, not required.

    Parameters
    ----------
    source_length:
        Total number of source frames in the injected content.  Must be >= 5
        (the minimum valid H3 clip length).

    Returns
    -------
    int
        Snapped length: ``5 + 17 * ((source_length - 5) // 17)``.
        Equal to ``source_length`` when already valid.

    Warns
    -----
    UserWarning
        If ``source_length`` is not already a valid ``17n+5`` length.  Message includes
        the original length, the snapped length, and the number of frames discarded.

    Raises
    ------
    ValueError
        If ``source_length < 5`` (the minimum valid H3 clip length is 5 frames).
    """
    if source_length < 5:
        raise ValueError(
            f"source_length={source_length} is below the minimum valid H3 clip length of "
            "5 frames. Valid clip lengths are 17n+5: 5, 22, 39, 56, 73, 90, …"
        )
    snapped = 5 + 17 * ((source_length - 5) // 17)
    if snapped != source_length:
        discarded = source_length - snapped
        warnings.warn(
            f"Inject content length {source_length} is not a valid H3 clip length (17n+5); "
            f"snapping down to {snapped} (discarding {discarded} trailing frame(s)).",
            UserWarning,
            stacklevel=2,
        )
    return snapped


def warn_audio_tail_alignment(
    snapped_length: int,
    audio_mode: str,
    end_keyframes: int,
    end_fade_out: int,
    has_audio: bool,
) -> None:
    """Warn when the snapped clip length may expose a tail audio desync.

    H3 audio latents are aligned to the joint AV lattice (``51n+39``: 39, 90, 141, 192, …).
    Lengths that are valid video-grid lengths (``17n+5``) but not audio-sync-aligned produce
    up to ~17 ms of tail-local audio error confined to the last 1–2 audio ticks.  A fade-out
    ramp that reaches the clip tail masks this error; ``keep`` mode or an un-faded tail exposes
    it.

    Parameters
    ----------
    snapped_length:
        Post-trim clip length (already snapped to 17n+5 by :func:`snap_length_down`).
    audio_mode:
        One of ``"fade"``, ``"drop"``, or ``"keep"``.
    end_keyframes:
        EXCLUSIVE end of the hold region.  A fade-out ramp exists when
        ``end_fade_out > end_keyframes``.
    end_fade_out:
        EXCLUSIVE upper bound of the envelope.  The ramp reaches the clip tail when
        ``end_fade_out == snapped_length``.
    has_audio:
        Whether injected audio is present.

    Warns
    -----
    UserWarning
        Emitted when all three conditions hold:

        1. ``snapped_length`` is not audio-sync-aligned:
           ``ceil(snapped_length / 17) % 3 != 0``.
        2. ``has_audio`` is ``True`` and ``audio_mode != "drop"``.
        3. The tail is NOT faded through — either ``audio_mode == "keep"`` or
           ``audio_mode == "fade"`` without a fade-out ramp reaching the clip tail
           (``not (end_fade_out > end_keyframes and end_fade_out == snapped_length)``).
    """
    import math as _math

    # Condition 1: not audio-sync-aligned (ceil(n/17) % 3 != 0).
    if _math.ceil(snapped_length / 17) % 3 == 0:
        return

    # Condition 2: injected audio is present and not dropped.
    if not has_audio or audio_mode == "drop":
        return

    # Condition 3: tail is NOT faded through.
    is_faded_through = end_fade_out > end_keyframes and end_fade_out == snapped_length
    if is_faded_through:
        return

    # Compute nearest audio-sync-aligned lengths (51k+39).
    if snapped_length >= 39:
        k = (snapped_length - 39) // 51
        nearest_below: int | None = 39 + 51 * k
        nearest_above = 39 + 51 * (k + 1)
        aligned_str = f"{nearest_below} (below) and {nearest_above} (above)"
    else:
        nearest_above = 39
        aligned_str = (
            f"{nearest_above} (nearest above; no audio-aligned valid length shorter than 39)"
        )

    warnings.warn(
        f"Inject content length {snapped_length} is not audio-sync-aligned (not 51k+39). "
        f"The audio tail may be off by up to ~17 ms (last 1-2 audio ticks). "
        f"audio_mode={audio_mode!r} with an unfaded clip end exposes this error. "
        f"Nearest audio-sync-aligned lengths: {aligned_str}. "
        "To suppress: snap the content length to an audio-aligned value, or add a "
        "fade-out ramp reaching the clip end (end_fade_out == snapped content length).",
        UserWarning,
        stacklevel=2,
    )


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
    height = images.shape[1]
    width = images.shape[2]

    if width % 32 != 0 or height % 32 != 0:
        raise ValueError(
            f"Image dimensions must be multiples of 32; got width={width}, height={height}."
        )

    if width != target_width or height != target_height:
        raise ValueError(
            f"Image dimensions {width}x{height} do not match target "
            f"{target_width}x{target_height}. Rescaling is not supported."
        )


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
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise TypeError(
            "audio must be a dict with 'waveform' and 'sample_rate' keys; "
            f"got {type(audio).__name__}."
        )

    import torch

    waveform = audio["waveform"]
    orig_sr = audio["sample_rate"]

    # Resample first, then compare lengths.
    if orig_sr != target_sample_rate:
        new_len = round(waveform.shape[-1] * target_sample_rate / orig_sr)
        # interpolate expects (N, C, L); waveform is (C, L).
        x = waveform.unsqueeze(0).float()
        resampled = torch.nn.functional.interpolate(
            x, size=new_len, mode="linear", align_corners=False
        )
        waveform = resampled.squeeze(0)
    else:
        waveform = waveform.clone()

    # Target length in samples after resampling.
    target_samples = round(video_duration_frames / fps * target_sample_rate)
    current_samples = waveform.shape[-1]

    if current_samples > target_samples:
        diff = current_samples - target_samples
        diff_s = diff / target_sample_rate
        warnings.warn(
            f"Audio is {diff} samples ({diff_s:.4f} s) longer than the video duration; "
            "trimming trailing samples.",
            UserWarning,
            stacklevel=2,
        )
        waveform = waveform[..., :target_samples]
    elif current_samples < target_samples:
        diff = target_samples - current_samples
        diff_s = diff / target_sample_rate
        warnings.warn(
            f"Audio is {diff} samples ({diff_s:.4f} s) shorter than the video duration; "
            "silence-padding.",
            UserWarning,
            stacklevel=2,
        )
        pad_shape = list(waveform.shape)
        pad_shape[-1] = diff
        silence = torch.zeros(pad_shape, dtype=waveform.dtype, device=waveform.device)
        waveform = torch.cat([waveform, silence], dim=-1)

    return {"waveform": waveform, "sample_rate": target_sample_rate}


def validate_envelope_indices(
    start_fade_in: int,
    start_keyframes: int,
    end_keyframes: int,
    end_fade_out: int,
    source_length: int,
    target_rows: int,
    inject_at: int,
) -> None:
    """Validate envelope index ordering and bounds; raise with offending values on violation.

    Ordering constraint (all must hold simultaneously):
    ``start_fade_in <= start_keyframes <= end_keyframes <= end_fade_out``

    Bounds constraints:
    - All indices must be >= 0.
    - ``end_fade_out`` must be <= ``source_length`` (exclusive upper bound: frame back at 1.0).
    - The last latent frame the envelope touches is ``inject_at + end_fade_out``; the row
      containing that latent frame (``frame_to_row(inject_at + end_fade_out)``) must be
      ``< target_rows``.

    Parameters
    ----------
    start_fade_in:
        Clip frame index where fade-in begins.
    start_keyframes:
        Clip frame index where hold begins.
    end_keyframes:
        Clip frame index where hold ends.
    end_fade_out:
        Clip frame index where fade-out ends.
    source_length:
        Total number of source frames in the inject content.
    target_rows:
        Total number of rows in the target latent.
    inject_at:
        Latent FRAME index in the target latent where the inject starts.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any ordering or bounds constraint is violated.  The message includes all four
        index values and the specific constraint that failed.
    """
    indices_str = (
        f"(start_fade_in={start_fade_in}, start_keyframes={start_keyframes}, "
        f"end_keyframes={end_keyframes}, end_fade_out={end_fade_out})"
    )

    # Check non-negative bounds first.
    if start_fade_in < 0:
        raise ValueError(f"start_fade_in={start_fade_in} must be >= 0. {indices_str}")
    if start_keyframes < 0:
        raise ValueError(f"start_keyframes={start_keyframes} must be >= 0. {indices_str}")
    if end_keyframes < 0:
        raise ValueError(f"end_keyframes={end_keyframes} must be >= 0. {indices_str}")
    if end_fade_out < 0:
        raise ValueError(f"end_fade_out={end_fade_out} must be >= 0. {indices_str}")

    # Check ordering: start_fade_in <= start_keyframes <= end_keyframes <= end_fade_out.
    if start_fade_in > start_keyframes:
        raise ValueError(
            f"Ordering violated: start_fade_in={start_fade_in} > "
            f"start_keyframes={start_keyframes}. {indices_str}"
        )
    if start_keyframes > end_keyframes:
        raise ValueError(
            f"Ordering violated: start_keyframes={start_keyframes} > "
            f"end_keyframes={end_keyframes}. {indices_str}"
        )
    if end_keyframes > end_fade_out:
        raise ValueError(
            f"Ordering violated: end_keyframes={end_keyframes} > "
            f"end_fade_out={end_fade_out}. {indices_str}"
        )

    # Check end_fade_out <= source_length (half-open model: efo is exclusive upper bound).
    if end_fade_out > source_length:
        raise ValueError(
            f"end_fade_out={end_fade_out} must be <= source_length={source_length}. {indices_str}"
        )

    # Check that the last row the envelope touches fits within target_rows.
    # Convert the last latent frame to a row index (frame→row, not mixed units).
    last_row = frame_to_row(inject_at + end_fade_out)
    if last_row >= target_rows:
        raise ValueError(
            f"Envelope span exceeds target: frame_to_row(inject_at={inject_at} + "
            f"end_fade_out={end_fade_out}) = last_row={last_row} >= "
            f"target_rows={target_rows}. {indices_str}"
        )
