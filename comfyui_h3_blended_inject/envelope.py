"""Interpolation enum and envelope evaluation for per-row denoise schedules.

Envelope semantics: denoise ramps from 1.0 at ``start_fade_in`` down to ``min_denoise`` at
``start_keyframes`` along the chosen interpolation curve, holds through ``end_keyframes``, then
ramps back to 1.0 at ``end_fade_out``.  Values outside the envelope are not returned; callers
are responsible for treating absent rows as ``d = 1.0``.

Row d-values are evaluated at each latent row's *true center time* on the 1/4/4/4/4 grid (via
:func:`~comfyui_h3_blended_inject.constants.row_center_times`), not at a uniform per-row grid
point.  Only the fade indices (start_fade_in, start_keyframes, end_keyframes, end_fade_out) are
evaluated in continuous frame time; no snapping is applied.

This module is pure Python / stdlib and must import without ``comfy`` or ``torch`` present.
"""

from __future__ import annotations

from enum import Enum

from comfyui_h3_blended_inject.constants import frame_to_row, row_center_times


class InterpolationType(str, Enum):
    """Supported interpolation curves for envelope fade regions.

    All variants are lowercased strings so they round-trip cleanly with ComfyUI's combo widget
    values without a mapping step.
    """

    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    LINEAR = "linear"
    NONE = "none"


# Convenience set of valid string values — usable without importing the Enum.
INTERPOLATION_TYPES: tuple[str, ...] = tuple(m.value for m in InterpolationType)


def evaluate_curve(t: float, interpolation_type: str) -> float:
    """Evaluate an interpolation curve at normalised position ``t`` in [0, 1].

    The result is the *blend weight* in [0, 1].  The caller maps this weight onto the actual
    denoise range [min_denoise, 1.0] outside this function.

    Curve definitions:
    - ``ease_in``   — quadratic ramp: ``t ** 2``
    - ``ease_out``  — quadratic ramp: ``1 - (1 - t) ** 2``
    - ``ease_in_out`` — cubic Hermite S-curve: ``3t² - 2t³``
    - ``linear``    — identity: ``t``
    - ``none``      — step function: 0 for t < 1.0, 1 for t == 1.0

    Parameters
    ----------
    t:
        Normalised position in [0.0, 1.0].  0.0 is the start of the fade region
        (full denoise = 1.0) and 1.0 is the end (denoise reaches ``min_denoise``).
    interpolation_type:
        One of the :data:`INTERPOLATION_TYPES` strings.

    Returns
    -------
    float
        Blend weight in [0.0, 1.0].

    Raises
    ------
    ValueError
        If ``t`` is outside [0.0, 1.0] or ``interpolation_type`` is not a recognised value.
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"t must be in [0.0, 1.0], got {t!r}")
    if interpolation_type not in INTERPOLATION_TYPES:
        raise ValueError(f"Unknown interpolation_type: {interpolation_type!r}")
    if interpolation_type == "ease_in":
        return t**2
    if interpolation_type == "ease_out":
        return 1.0 - (1.0 - t) ** 2
    if interpolation_type == "ease_in_out":
        return 3.0 * t**2 - 2.0 * t**3
    if interpolation_type == "linear":
        return t
    # "none" — step: 0 for t < 1.0, 1 for t == 1.0
    return 0.0 if t < 1.0 else 1.0


def _denoise_at_frame_time(
    t: float,
    start_fade_in: int,
    start_keyframes: int,
    end_keyframes: int,
    end_fade_out: int,
    min_denoise: float,
    interpolation_type: str,
) -> float:
    """Return the denoise value at continuous source frame time ``t``.

    Returns 1.0 for times outside the half-open envelope [start_fade_in - 1, end_fade_out].
    The 1.0 anchor lives at start_fade_in - 1; start_fade_in is the first frame below 1.0.
    """
    anchor = start_fade_in - 1
    if t < anchor or t > end_fade_out:
        return 1.0

    if t <= start_keyframes:
        # Fade-in region: 1.0 at anchor (sfi-1) → min_denoise at start_keyframes.
        # Denominator is always >= 1 since start_keyframes >= start_fade_in > anchor.
        fade_t = (t - anchor) / (start_keyframes - anchor)
        w = evaluate_curve(fade_t, interpolation_type)
        return 1.0 - w * (1.0 - min_denoise)

    hold_end = end_keyframes - 1
    if t <= hold_end:
        # Hold region: [start_keyframes, end_keyframes).  Last held frame = end_keyframes - 1.
        return min_denoise

    # Fade-out region: min_denoise at end_keyframes - 1 → 1.0 at end_fade_out.
    # Denominator = end_fade_out - (end_keyframes - 1) >= 1 always.
    fade_t = (t - hold_end) / (end_fade_out - hold_end)
    w = evaluate_curve(fade_t, interpolation_type)
    return min_denoise + w * (1.0 - min_denoise)


def evaluate_envelope(
    start_fade_in: int,
    start_keyframes: int,
    end_keyframes: int,
    end_fade_out: int,
    min_denoise: float,
    interpolation_type: str,
    source_length: int,
    target_rows: int,
    inject_at: int,
) -> list[tuple[int, float]]:
    """Compute one ``(absolute_latent_row_idx, denoise)`` pair per row covered by this envelope.

    ALL fade indices (start_fade_in, start_keyframes, end_keyframes, end_fade_out) are
    **clip frame indices** — positions within the injected clip's own content.  ``inject_at``
    is a **latent frame index** (FRAME-space position in the target latent where the clip
    begins).  Clip frame ``k`` corresponds to latent frame ``inject_at + k``.

    Row denoise values are evaluated in continuous clip-frame time at each latent row's true
    center times on the 1/4/4/4/4 grid (via
    :func:`~comfyui_h3_blended_inject.constants.row_center_times`), converted to clip-frame
    time, then averaged across the row.

    A row is included in the result **only if** at least one of its clip-frame centers falls
    within ``[start_fade_in - 1, end_fade_out]`` (the anchor-inclusive range).  Rows whose
    every center is outside this range evaluate to 1.0 (pure generation) and are **omitted**
    — they must not claim a row under last-in-wins semantics.

    A row's final denoise is exactly 0.0 iff ``min_denoise == 0.0`` and **every** clip-frame
    center for that row lies within the hold region ``[start_keyframes, end_keyframes - 1]``
    (see :func:`is_row_exactly_zero`).

    Still-inject degenerate envelope: when ``start_fade_in == start_keyframes == end_keyframes
    == end_fade_out``, the result is a single ``(row, min_denoise)`` entry at
    ``frame_to_row(inject_at + start_fade_in)``, or ``[]`` if that row is out of bounds.

    Parameters
    ----------
    start_fade_in:
        First clip frame below 1.0.  The 1.0 anchor lives at ``start_fade_in - 1``
        (one virtual frame before this).
    start_keyframes:
        Clip frame index where hold at ``min_denoise`` begins (inclusive).
    end_keyframes:
        EXCLUSIVE: first fade-out frame.  Last held frame is ``end_keyframes - 1``.
    end_fade_out:
        EXCLUSIVE upper bound (half-open model): denoise returns to 1.0 here.
        The last content frame is ``end_fade_out - 1``.
    min_denoise:
        Denoise floor during the hold region.  In [0.0, 1.0].
    interpolation_type:
        Curve applied to both fade-in and fade-out regions.  One of :data:`INTERPOLATION_TYPES`.
    source_length:
        Total number of source frames in the inject content.
    target_rows:
        Total number of latent rows in the target (output) latent.
    inject_at:
        Latent FRAME index in the target latent where this inject begins.  Must be a
        multiple of 17 frames, already snapped by
        :func:`~comfyui_h3_blended_inject.sanitize.snap_inject_at`.

    Returns
    -------
    list[tuple[int, float]]
        ``(absolute_latent_row_idx, denoise)`` pairs for each target row covered by the
        envelope, sorted by row index ascending.  Rows outside the envelope or out of
        ``[0, target_rows)`` are omitted.

    Raises
    ------
    ValueError
        If envelope indices violate ordering constraints
        (``start_fade_in <= start_keyframes <= end_keyframes <= end_fade_out``).
    """
    if not (start_fade_in <= start_keyframes <= end_keyframes <= end_fade_out):
        raise ValueError(
            f"Envelope indices must satisfy start_fade_in <= start_keyframes <= "
            f"end_keyframes <= end_fade_out, got "
            f"{start_fade_in=}, {start_keyframes=}, {end_keyframes=}, {end_fade_out=}"
        )

    # Degenerate still-inject: all four indices equal → single row at min_denoise.
    if start_fade_in == start_keyframes == end_keyframes == end_fade_out:
        r = frame_to_row(inject_at + start_fade_in)
        if r >= target_rows:
            return []
        return [(r, min_denoise)]

    # Compute the latent-frame span the envelope touches.
    # Anchor is at start_fade_in - 1; the first included row may start before sfi.
    first_latent = inject_at + start_fade_in - 1
    last_latent = inject_at + end_fade_out

    result: list[tuple[int, float]] = []
    row_start = frame_to_row(first_latent) if first_latent >= 0 else 0
    for r in range(row_start, frame_to_row(last_latent) + 1):
        if r >= target_rows:
            break
        centers_latent = row_center_times(r)
        clip_centers = [c - inject_at for c in centers_latent]
        # Inclusion rule: include if any clip center falls in the anchor-inclusive range.
        if not any((start_fade_in - 1) <= cc <= end_fade_out for cc in clip_centers):
            continue
        values = [
            _denoise_at_frame_time(
                cc,
                start_fade_in,
                start_keyframes,
                end_keyframes,
                end_fade_out,
                min_denoise,
                interpolation_type,
            )
            for cc in clip_centers
        ]
        denoise = sum(values) / len(values)
        result.append((r, denoise))

    return result


def is_row_exactly_zero(
    row_idx: int,
    start_fade_in: int,
    start_keyframes: int,
    end_keyframes: int,
    end_fade_out: int,
    min_denoise: float,
    inject_at: int,
) -> bool:
    """Return True only if the row's averaged denoise is exactly 0.0.

    A row qualifies as ``d = 0`` (exact preserve, routed via the derived noise mask)
    iff ``min_denoise == 0.0`` *and* **every** clip-frame center of the row lies within
    the hold region ``[start_keyframes, end_keyframes - 1]`` (exclusive upper bound).

    This is consistent with :func:`evaluate_envelope`: the averaged denoise for a row is
    exactly 0.0 iff this function returns True for the same inputs.

    Parameters
    ----------
    row_idx:
        Absolute target latent row index to test.
    start_fade_in:
        Clip frame index where fade-in begins.
    start_keyframes:
        Clip frame index where hold at ``min_denoise`` begins.
    end_keyframes:
        EXCLUSIVE: first fade-out frame.  Last held frame is ``end_keyframes - 1``.
    end_fade_out:
        EXCLUSIVE upper bound; denoise returns to 1.0 here.
    min_denoise:
        Envelope floor during the hold region.
    inject_at:
        Latent FRAME index in the target latent where this inject begins.

    Returns
    -------
    bool
        True iff ``min_denoise == 0.0`` and all clip-frame centers of ``row_idx`` fall
        within ``[start_keyframes, end_keyframes - 1]``.
    """
    if min_denoise != 0.0:
        return False
    clip_centers = [c - inject_at for c in row_center_times(row_idx)]
    return all(start_keyframes <= cc <= end_keyframes - 1 for cc in clip_centers)


def classify_row_region(
    row_idx: int,
    inject_at: int,
    start_fade_in: int,
    start_keyframes: int,
    end_keyframes: int,
    end_fade_out: int,
    min_denoise: float,
    crossfade: bool = False,
) -> str:
    """Classify a scheduled row as ``'preserve'``, ``'hold'``, ``'fade'``, or ``'free'``.

    Uses integer clip-frame membership against the winning inject's half-open markers to
    determine the region.  Does NOT compare the float denoise value ``d`` — floating-point
    averaging in :func:`evaluate_envelope` can make a true hold row appear non-exact.

    Classification uses integer clip-frame membership, not the float ``d``.

    Regions
    -------
    ``'preserve'``:
        ALL clip-frame centers lie in the keyframe span ``[start_keyframes, end_keyframes)``
        AND ``min_denoise == 0.0``.  Routes to the derived noise mask + composite; the
        wrapper must not touch these rows.
    ``'hold'``:
        ALL clip-frame centers lie in ``[start_keyframes, end_keyframes)`` AND
        ``min_denoise > 0``.  Wrapper applies binary hold-and-release (is_held gate).
        Fade-ramp rows (centers in the fade-in or fade-out span) also return ``'hold'``
        by default (``crossfade=False``): with per-row release thresholds (task #29)
        staggered release across the ramp *is* the fade.
    ``'fade'``:
        At least one clip-frame center lies in the fade-in ramp
        ``[start_fade_in, start_keyframes)`` or the fade-out ramp
        ``[end_keyframes, end_fade_out)``, AND ``crossfade=True`` is passed explicitly.
        Wrapper blends prediction permanently: ``(1 - d) * original + d * model_pred``.
        This is a parked legacy path — it is NOT the default.
    ``'free'``:
        None of the above.  Wrapper does nothing.

    Degenerate still-inject (all four markers equal): classify solely by ``min_denoise``
    (``'preserve'`` if 0.0, ``'hold'`` otherwise).

    Parameters
    ----------
    row_idx:
        Absolute target latent row index to classify.
    inject_at:
        Latent FRAME index where the inject begins (multiple of 17 after snapping).
    start_fade_in:
        Clip frame index where fade-in begins (first frame below 1.0).
    start_keyframes:
        Clip frame index where hold at ``min_denoise`` begins (inclusive).
    end_keyframes:
        EXCLUSIVE: first fade-out clip frame.  Last held clip frame is
        ``end_keyframes - 1``.
    end_fade_out:
        EXCLUSIVE upper bound; denoise returns to 1.0 here.
    min_denoise:
        Envelope floor during the hold region.
    crossfade:
        When ``False`` (default), fade-ramp rows are classified as ``'hold'`` and use
        the ordinary fractional hold-and-release path — staggered per-row release is the
        fade.  When ``True``, ramp rows classify as ``'fade'`` and activate the legacy
        persistent prediction-blend path.

    Returns
    -------
    str
        One of ``'preserve'``, ``'hold'``, ``'fade'``, ``'free'``.
    """
    # Degenerate still-inject: no fade regions exist; classify by min_denoise only.
    if start_fade_in == start_keyframes == end_keyframes == end_fade_out:
        return "preserve" if min_denoise == 0.0 else "hold"

    clip_centers = [c - inject_at for c in row_center_times(row_idx)]

    # 'hold' / 'preserve': ALL centers lie in the keyframe span [skf, ekf).
    in_hold = all(start_keyframes <= cc < end_keyframes for cc in clip_centers)
    if in_hold:
        return "preserve" if min_denoise == 0.0 else "hold"

    # 'fade' or 'hold': at least one center in the fade-in ramp [sfi, skf) or
    #                   fade-out ramp [ekf, efo).
    # Default (crossfade=False): ramp rows use hold-and-release (staggered release IS
    # the fade).  Legacy prediction-blend path only active when crossfade=True.
    in_fade = any(
        (start_fade_in <= cc < start_keyframes) or (end_keyframes <= cc < end_fade_out)
        for cc in clip_centers
    )
    if in_fade:
        return "fade" if crossfade else "hold"

    return "free"


def still_inject_denoise(min_denoise: float) -> list[float]:
    """Return the single-element denoise list for a still (single-image) inject.

    A still inject is a degenerate envelope where all four fade indices are equal.
    The resulting schedule is one row with ``d = min_denoise``.

    Parameters
    ----------
    min_denoise:
        Denoise value for the single injected frame.  In [0.0, 1.0].

    Returns
    -------
    list[float]
        A one-element list ``[min_denoise]``.
    """
    return [min_denoise]
