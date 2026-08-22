"""Property-based tests for comfyui_h3_blended_inject.constants.

Behavior contract is taken from the module docstrings. All stub functions raise
NotImplementedError, so every test that calls a stub will fail until the bodies
are implemented. That is expected and correct at this stage.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from comfyui_h3_blended_inject.constants import (
    AUDIO_HZ,
    FPS,
    FRAME_PER_TOKEN,
    audio_ticks_for_rows,
    frame_to_row,
    row_center_times,
    row_frame_count,
    time_shift_sigma,
    total_rows,
    video_row_to_audio_tick,
)

# ---------------------------------------------------------------------------
# Helper — derives start frame for a row without calling the stub functions
# ---------------------------------------------------------------------------


def _row_start_frame(row_idx: int) -> int:
    """Start source-frame index for row_idx, computed from FRAME_PER_TOKEN directly."""
    if row_idx == 0:
        return 0
    return 1 + (row_idx - 1) * 4


# ---------------------------------------------------------------------------
# Plain literal constants — these assertions pass without any stub
# ---------------------------------------------------------------------------


def test_frame_per_token():
    assert FRAME_PER_TOKEN == (1, 4, 4, 4, 4)


def test_fps():
    assert FPS == 24


def test_audio_hz():
    assert AUDIO_HZ == 40.0


# ---------------------------------------------------------------------------
# row_frame_count
# ---------------------------------------------------------------------------


def test_row_frame_count_row_zero():
    assert row_frame_count(0) == 1


@given(st.integers(min_value=1, max_value=1_000))
def test_row_frame_count_positive_rows(row_idx):
    assert row_frame_count(row_idx) == 4


@given(st.integers(max_value=-1))
def test_row_frame_count_negative_raises(row_idx):
    with pytest.raises(ValueError):
        row_frame_count(row_idx)


# ---------------------------------------------------------------------------
# frame_to_row
# ---------------------------------------------------------------------------


def test_frame_to_row_frame_zero():
    assert frame_to_row(0) == 0


@given(st.integers(min_value=1, max_value=10_000))
def test_frame_to_row_formula(frame_idx):
    expected = 1 + (frame_idx - 1) // 4
    assert frame_to_row(frame_idx) == expected


@given(st.integers(max_value=-1))
def test_frame_to_row_negative_raises(frame_idx):
    with pytest.raises(ValueError):
        frame_to_row(frame_idx)


@given(st.integers(min_value=0, max_value=10_000))
def test_frame_to_row_cross_check_span(frame_idx):
    """Every frame maps to a row, and the frame falls within that row's frame span."""
    row = frame_to_row(frame_idx)
    start = _row_start_frame(row)
    count = row_frame_count(row)
    assert start <= frame_idx < start + count


# ---------------------------------------------------------------------------
# total_rows
# ---------------------------------------------------------------------------


@given(st.integers(min_value=1, max_value=10_000))
def test_total_rows_formula(n_frames):
    expected = 1 + math.ceil((n_frames - 1) / 4)
    assert total_rows(n_frames) == expected


@given(st.integers(max_value=0))
def test_total_rows_less_than_one_raises(n_frames):
    with pytest.raises(ValueError):
        total_rows(n_frames)


@given(st.integers(min_value=1, max_value=10_000))
def test_total_rows_consistent_with_frame_to_row(n_frames):
    """total_rows(n) == frame_to_row(n - 1) + 1 for all n >= 1."""
    assert total_rows(n_frames) == frame_to_row(n_frames - 1) + 1


# ---------------------------------------------------------------------------
# row_center_times
# ---------------------------------------------------------------------------


@given(st.integers(min_value=0, max_value=1_000))
def test_row_center_times_length(row_idx):
    """Returns a tuple with one entry per source frame in the row."""
    centers = row_center_times(row_idx)
    assert isinstance(centers, tuple)
    assert len(centers) == row_frame_count(row_idx)


@given(st.integers(min_value=0, max_value=1_000))
def test_row_center_times_strictly_increasing(row_idx):
    centers = row_center_times(row_idx)
    for a, b in zip(centers, centers[1:], strict=False):
        assert a < b


@given(st.integers(max_value=-1))
def test_row_center_times_negative_raises(row_idx):
    with pytest.raises(ValueError):
        row_center_times(row_idx)


@given(st.integers(min_value=0, max_value=1_000))
def test_row_center_times_within_frame_span(row_idx):
    """Every center time lies strictly within the row's absolute source-frame span."""
    centers = row_center_times(row_idx)
    start = _row_start_frame(row_idx)
    count = row_frame_count(row_idx)
    end = start + count  # exclusive upper bound (in frame units)
    for c in centers:
        assert start <= c < end


# ---------------------------------------------------------------------------
# video_row_to_audio_tick
# ---------------------------------------------------------------------------


@given(st.integers(min_value=0, max_value=1_000))
def test_video_row_to_audio_tick_non_negative(row_idx):
    tick = video_row_to_audio_tick(row_idx)
    assert isinstance(tick, int)
    assert tick >= 0


@given(st.integers(min_value=0, max_value=999))
def test_video_row_to_audio_tick_non_decreasing(row_idx):
    """Later rows map to equal-or-later audio ticks."""
    assert video_row_to_audio_tick(row_idx) <= video_row_to_audio_tick(row_idx + 1)


@given(st.integers(max_value=-1))
def test_video_row_to_audio_tick_negative_raises(row_idx):
    with pytest.raises(ValueError):
        video_row_to_audio_tick(row_idx)


# ---------------------------------------------------------------------------
# audio_ticks_for_rows
# ---------------------------------------------------------------------------


@given(st.integers(min_value=0, max_value=1_000))
def test_audio_ticks_for_rows_non_negative(n_rows):
    result = audio_ticks_for_rows(n_rows)
    assert isinstance(result, int)
    assert result >= 0


@given(st.integers(min_value=0, max_value=999))
def test_audio_ticks_for_rows_non_decreasing(n_rows):
    """More rows never yield fewer audio ticks."""
    assert audio_ticks_for_rows(n_rows) <= audio_ticks_for_rows(n_rows + 1)


# ---------------------------------------------------------------------------
# time_shift_sigma
#
# NOTE: Exact-formula assertions are intentionally omitted here.
# The implementation body will be sourced from comfy/ldm/minimax/model.py.
# Only safe, formula-agnostic invariants that any valid flow-matching
# time-shift function must satisfy are tested below.
# ---------------------------------------------------------------------------


def test_time_shift_sigma_endpoint_zero():
    assert time_shift_sigma(0.0) == 0.0


def test_time_shift_sigma_endpoint_one():
    assert time_shift_sigma(1.0) == 1.0


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_time_shift_sigma_returns_float(sigma):
    result = time_shift_sigma(sigma)
    assert isinstance(result, float)


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_time_shift_sigma_result_in_unit_interval(sigma):
    result = time_shift_sigma(sigma)
    assert 0.0 <= result <= 1.0


@given(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_time_shift_sigma_monotone_non_decreasing(sigma_a, sigma_b):
    """time_shift_sigma is monotone non-decreasing over [0, 1]."""
    lo, hi = min(sigma_a, sigma_b), max(sigma_a, sigma_b)
    assert time_shift_sigma(lo) <= time_shift_sigma(hi)


class TestAudioTickRange:
    """audio_tick_range: canonical tick range per video row, tiling [0, audio_ticks) exactly."""

    def test_row_zero_starts_at_zero(self) -> None:
        from comfyui_h3_blended_inject.constants import audio_tick_range

        r = audio_tick_range(0, 5, 20)
        assert r.start == 0

    def test_final_row_extends_to_audio_ticks(self) -> None:
        from comfyui_h3_blended_inject.constants import audio_tick_range

        audio_ticks_total = 20
        n_rows = 5
        r = audio_tick_range(n_rows - 1, n_rows, audio_ticks_total)
        assert r.stop == audio_ticks_total

    def test_adjacent_rows_contiguous(self) -> None:
        from comfyui_h3_blended_inject.constants import audio_tick_range

        n_rows = 5
        audio_ticks_total = audio_ticks_for_rows(n_rows)
        for row_idx in range(n_rows - 1):
            r_curr = audio_tick_range(row_idx, n_rows, audio_ticks_total)
            r_next = audio_tick_range(row_idx + 1, n_rows, audio_ticks_total)
            assert r_curr.stop == r_next.start, (
                f"Rows {row_idx} and {row_idx + 1} not contiguous: {r_curr} vs {r_next}"
            )

    def test_tiles_exactly_no_overlap_no_gap(self) -> None:
        from comfyui_h3_blended_inject.constants import audio_tick_range

        n_rows = 5
        audio_ticks_total = audio_ticks_for_rows(n_rows)
        ticks_seen: set[int] = set()
        for row_idx in range(n_rows):
            for t in audio_tick_range(row_idx, n_rows, audio_ticks_total):
                assert t not in ticks_seen, f"Tick {t} covered by row {row_idx} twice"
                ticks_seen.add(t)
        assert ticks_seen == set(range(audio_ticks_total)), (
            f"Ranges do not tile [0, {audio_ticks_total}): covered={sorted(ticks_seen)}"
        )

    def test_negative_row_idx_raises(self) -> None:
        from comfyui_h3_blended_inject.constants import audio_tick_range

        with pytest.raises(ValueError):
            audio_tick_range(-1, 5, 20)
