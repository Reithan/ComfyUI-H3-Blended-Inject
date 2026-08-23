"""Property-based and regression tests for comfyui_h3_blended_inject.grid.

Behavior contract is taken from the module docstrings and the verified grid table for the
per-17-frame chunk reset model (MiniMaxVAE: clip_length=17, token_drop=3, vae_ratio_t=4).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from comfyui_h3_blended_inject.grid import (
    AUDIO_HZ,
    CLIP_LENGTH,
    FPS,
    FRAME_PER_TOKEN,
    TOKEN_DROP,
    TOKENS_PER_CHUNK,
    VAE_RATIO_T,
    audio_ticks_for_rows,
    frame_to_row,
    row_center_times,
    row_frame_count,
    row_start_frame,
    total_rows,
    video_row_to_audio_tick,
)
from comfyui_h3_blended_inject.sampler import time_shift_sigma

# ---------------------------------------------------------------------------
# Helper — derives start frame for a row using the new per-17-chunk formula
# ---------------------------------------------------------------------------


def _row_start_frame(row_idx: int) -> int:
    """Start source-frame index for row_idx, per the per-17-chunk grid.

    Mirrors the formula in row_start_frame() without importing it, so that
    cross-check tests don't trivially compare a function to itself.
    """
    chunk = row_idx // TOKENS_PER_CHUNK
    local_row = row_idx % TOKENS_PER_CHUNK
    if local_row == 0:
        return CLIP_LENGTH * chunk
    return CLIP_LENGTH * chunk + 1 + (local_row - 1) * VAE_RATIO_T


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


@given(st.integers(min_value=0, max_value=1_000))
def test_row_frame_count_matches_local_row(row_idx):
    """Chunk-boundary rows (local_row==0) → 1 frame; all others → VAE_RATIO_T (4) frames."""
    local_row = row_idx % TOKENS_PER_CHUNK
    expected = 1 if local_row == 0 else VAE_RATIO_T
    assert row_frame_count(row_idx) == expected


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
    """frame_to_row uses the per-17-chunk reset formula."""
    chunk = frame_idx // CLIP_LENGTH
    local = frame_idx % CLIP_LENGTH
    local_row = 0 if local == 0 else 1 + (local - 1) // VAE_RATIO_T
    expected = TOKENS_PER_CHUNK * chunk + local_row
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


def test_total_rows_single_frame_returns_1():
    """H3 VAE single-frame path: n_frames=1 returns exactly 1 latent row.

    The H3 VAE special-cases a single input frame and bypasses the 17-frame chunked
    path, producing 1 latent row instead of the formula's 5*ceil(1/17)-3=2.
    FAIL-THEN-PASS: Before the special case, total_rows(1) returns 2; after it returns 1.
    """
    assert total_rows(1) == 1


def test_total_rows_n2_is_2():
    """Regression guard: n_frames=2 still gives 2 rows (formula path unchanged)."""
    assert total_rows(2) == 2


def test_total_rows_n17_is_2():
    """Regression guard: n_frames=17 still gives 2 rows (formula path unchanged)."""
    assert total_rows(17) == 2


def test_total_rows_n22_is_7():
    """Regression guard: n_frames=22 still gives 7 rows (formula path unchanged)."""
    assert total_rows(22) == 7


@given(st.integers(min_value=2, max_value=10_000))
def test_total_rows_formula(n_frames):
    """total_rows uses the per-17-chunk formula for n_frames >= 2.

    The formula is TOKENS_PER_CHUNK*ceil(n/CLIP_LENGTH)-TOKEN_DROP.
    n_frames=1 is a special case (H3 VAE single-frame path) handled separately.
    """
    expected = TOKENS_PER_CHUNK * math.ceil(n_frames / CLIP_LENGTH) - TOKEN_DROP
    assert total_rows(n_frames) == expected


@given(st.integers(max_value=0))
def test_total_rows_less_than_one_raises(n_frames):
    with pytest.raises(ValueError):
        total_rows(n_frames)


@given(st.integers(min_value=0, max_value=200))
def test_total_rows_valid_clip_lengths(k):
    """Valid clip lengths 17k+5 yield exactly TOKENS_PER_CHUNK*(k+1) - TOKEN_DROP rows."""
    # 17*0+5=5, 17*1+5=22, 17*2+5=39, ...
    n_frames = CLIP_LENGTH * k + 1 + VAE_RATIO_T  # 17k + 5
    expected = TOKENS_PER_CHUNK * (k + 1) - TOKEN_DROP
    assert total_rows(n_frames) == expected


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


def test_time_shift_sigma_non_default_from_shift_changes_output():
    """Changing from_shift from the default 12.0 must change the output at a midpoint.

    This test FAILS if time_shift_sigma ignores its from_shift argument (e.g. still
    uses a hardcoded 12.0 internally).
    """
    mid_default = time_shift_sigma(0.5)  # from_shift=12.0, to_shift=3.0
    mid_custom = time_shift_sigma(0.5, from_shift=8.0, to_shift=3.0)
    assert mid_default != mid_custom, (
        "from_shift=8.0 must produce a different result from the default from_shift=12.0"
    )


def test_time_shift_sigma_endpoints_hold_for_non_default_shifts():
    """f(0)=0 and f(1)=1 for a non-default shift pair.

    The endpoint contract must hold regardless of the chosen shift values.
    This test FAILS if the parametrization changes the formula structure in a way
    that breaks the endpoint invariant.
    """
    assert time_shift_sigma(0.0, from_shift=8.0, to_shift=2.0) == 0.0
    assert time_shift_sigma(1.0, from_shift=8.0, to_shift=2.0) == 1.0


class TestAudioTickRange:
    """audio_tick_range: canonical tick range per video row, tiling [0, audio_ticks) exactly."""

    def test_row_zero_starts_at_zero(self) -> None:
        from comfyui_h3_blended_inject.grid import audio_tick_range

        r = audio_tick_range(0, 5, 20)
        assert r.start == 0

    def test_final_row_extends_to_audio_ticks(self) -> None:
        from comfyui_h3_blended_inject.grid import audio_tick_range

        audio_ticks_total = 20
        n_rows = 5
        r = audio_tick_range(n_rows - 1, n_rows, audio_ticks_total)
        assert r.stop == audio_ticks_total

    def test_adjacent_rows_contiguous(self) -> None:
        from comfyui_h3_blended_inject.grid import audio_tick_range

        n_rows = 5
        audio_ticks_total = audio_ticks_for_rows(n_rows)
        for row_idx in range(n_rows - 1):
            r_curr = audio_tick_range(row_idx, n_rows, audio_ticks_total)
            r_next = audio_tick_range(row_idx + 1, n_rows, audio_ticks_total)
            assert r_curr.stop == r_next.start, (
                f"Rows {row_idx} and {row_idx + 1} not contiguous: {r_curr} vs {r_next}"
            )

    def test_tiles_exactly_no_overlap_no_gap(self) -> None:
        from comfyui_h3_blended_inject.grid import audio_tick_range

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
        from comfyui_h3_blended_inject.grid import audio_tick_range

        with pytest.raises(ValueError):
            audio_tick_range(-1, 5, 20)


# ---------------------------------------------------------------------------
# Per-17-chunk grid regression tests
#
# These values were verified against the real MiniMaxVAE source at
# comfy/ldm/minimax/vae.py (clip_length=17, token_drop=3, time_down→vae_ratio_t=4).
# The BOLD entries below are the ones the OLD uniform-4 code got wrong.
# Every test in this class MUST FAIL on the old implementation and PASS after.
# ---------------------------------------------------------------------------


class TestPerChunkGridRegressions:
    """Pinned table for frame_to_row, total_rows, row_frame_count, row_center_times."""

    # -- row_start_frame ---------------------------------------------------------

    @pytest.mark.parametrize(
        "row_idx, expected_start",
        [
            (0, 0),
            (1, 1),
            (4, 13),
            (5, 17),  # chunk boundary — OLD: 1+(5-1)*4=17 (same by coincidence)
            (6, 18),  # OLD: 1+(6-1)*4=21 — WRONG
            (7, 22),  # OLD: 1+(7-1)*4=25 — WRONG
            (10, 34),  # OLD: 1+(10-1)*4=37 — WRONG
            (11, 35),  # OLD: 1+(11-1)*4=41 — WRONG
        ],
    )
    def test_row_start_frame(self, row_idx, expected_start):
        """row_start_frame resets correctly at chunk boundaries."""
        assert row_start_frame(row_idx) == expected_start

    # -- frame_to_row ------------------------------------------------------------

    @pytest.mark.parametrize(
        "frame_idx, expected_row",
        [
            (0, 0),
            (1, 1),
            (4, 1),
            (5, 2),
            (13, 4),
            (16, 4),
            (17, 5),  # OLD: 1+(17-1)//4=5 — same by coincidence
            (18, 6),  # OLD: 1+(18-1)//4=5 — WRONG
            (21, 6),  # OLD: 1+(21-1)//4=6 — WRONG (gives 6 actually) hmm
            (22, 7),  # OLD: 1+(22-1)//4=6 — WRONG
            (26, 8),  # OLD: 1+(26-1)//4=7 — WRONG
            (30, 9),  # OLD: 1+(30-1)//4=8 — WRONG
            (34, 10),  # OLD: 1+(34-1)//4=9 — WRONG
            (35, 11),  # OLD: 1+(35-1)//4=9 — WRONG
            (38, 11),  # OLD: 1+(38-1)//4=10 — WRONG
        ],
    )
    def test_frame_to_row_verified_table(self, frame_idx, expected_row):
        """Pinned frame→row mapping from the verified per-17-chunk grid table."""
        assert frame_to_row(frame_idx) == expected_row

    # -- total_rows --------------------------------------------------------------

    @pytest.mark.parametrize(
        "n_frames, expected_rows",
        [
            (5, 2),
            (22, 7),
            (39, 12),  # OLD: 1+ceil(38/4)=10 — WRONG
            (56, 17),  # OLD: 1+ceil(55/4)=15 — WRONG
        ],
    )
    def test_total_rows_verified_table(self, n_frames, expected_rows):
        """Pinned frame-count→row-count mapping from the verified grid table."""
        assert total_rows(n_frames) == expected_rows

    # -- row_frame_count ---------------------------------------------------------

    @pytest.mark.parametrize(
        "row_idx, expected_count",
        [
            (0, 1),
            (1, 4),
            (4, 4),
            (5, 1),  # chunk boundary — OLD: 4 — WRONG
            (10, 1),  # chunk boundary — OLD: 4 — WRONG
            (11, 4),  # non-boundary — OLD: 4 — same, but checks context
        ],
    )
    def test_row_frame_count_verified_table(self, row_idx, expected_count):
        """Pinned row→frame-count mapping including chunk-boundary resets."""
        assert row_frame_count(row_idx) == expected_count

    # -- row_center_times --------------------------------------------------------

    @pytest.mark.parametrize(
        "row_idx, expected_centers",
        [
            (0, (0.5,)),
            (1, (1.5, 2.5, 3.5, 4.5)),
            (5, (17.5,)),  # OLD: 4-tuple starting here — WRONG count
            (6, (18.5, 19.5, 20.5, 21.5)),  # OLD: (21.5,22.5,23.5,24.5) — WRONG
            (11, (35.5, 36.5, 37.5, 38.5)),  # OLD: (41.5,42.5,43.5,44.5) — WRONG
        ],
    )
    def test_row_center_times_verified_table(self, row_idx, expected_centers):
        """Pinned row→center-times mapping including chunk-boundary resets."""
        assert row_center_times(row_idx) == pytest.approx(expected_centers)


# ---------------------------------------------------------------------------
# inject_row_map — clip↔target row correspondence
# ---------------------------------------------------------------------------


class TestInjectRowMap:
    """inject_row_map: clip↔target row correspondence for hold-and-release + composite."""

    def test_inject_at_0_identity(self):
        """inject_at=0 → clip row j maps to target row j (frame_to_row(0)=0)."""
        from comfyui_h3_blended_inject.grid import inject_row_map

        result = inject_row_map(0, 5, 10)
        assert result == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

    def test_inject_at_17_target_rows_start_at_5(self):
        """inject_at=17 → target rows start at 5 (frame_to_row(17)=5)."""
        from comfyui_h3_blended_inject.grid import inject_row_map

        result = inject_row_map(17, 3, 20)
        assert result == [(5, 0), (6, 1), (7, 2)]

    def test_inject_at_34_target_rows_start_at_10(self):
        """inject_at=34 → target rows start at 10 (frame_to_row(34)=10)."""
        from comfyui_h3_blended_inject.grid import inject_row_map

        result = inject_row_map(34, 3, 20)
        assert result == [(10, 0), (11, 1), (12, 2)]

    def test_out_of_range_clip_rows_dropped(self):
        """Clip rows whose target_row >= target_rows are dropped."""
        from comfyui_h3_blended_inject.grid import inject_row_map

        # inject_at=0, 10 clip rows, but only 3 target rows → clips 3-9 are out
        result = inject_row_map(0, 10, 3)
        assert result == [(0, 0), (1, 1), (2, 2)]

    def test_target_rows_beyond_limit_dropped(self):
        """inject_at=17 with partial overlap: only rows within target_rows kept."""
        from comfyui_h3_blended_inject.grid import inject_row_map

        # inject_at_row=5; target_rows=7 → rows 5,6 fit; row 7 is out
        result = inject_row_map(17, 20, 7)
        assert result == [(5, 0), (6, 1)]

    def test_no_overlap_returns_empty(self):
        """inject_at=17 with target_rows=5 → inject_at_row=5 is out of [0,5)."""
        from comfyui_h3_blended_inject.grid import inject_row_map

        result = inject_row_map(17, 3, 5)
        assert result == []

    def test_zero_clip_rows_returns_empty(self):
        """n_clip_rows=0 → no iterations → empty list."""
        from comfyui_h3_blended_inject.grid import inject_row_map

        assert inject_row_map(0, 0, 10) == []


# ---------------------------------------------------------------------------
# inject_audio_ticks_for_row — per-row audio tick mapping
# ---------------------------------------------------------------------------


class TestInjectAudioTicksForRow:
    """inject_audio_ticks_for_row: per-row audio tick mapping for hold-and-release + composite."""

    def test_inject_at_0_row_0_maps_to_self(self):
        """inject_at=0, row 0: inject_start_tick=0; target_tick == clip_tick for each tick."""
        from comfyui_h3_blended_inject.grid import (
            audio_tick_range,
            audio_ticks_for_rows,
            inject_audio_ticks_for_row,
        )

        n_rows = 5
        n_ticks = audio_ticks_for_rows(n_rows)
        row_ticks = list(audio_tick_range(0, n_rows, n_ticks))
        result = inject_audio_ticks_for_row(0, 0, n_ticks, n_rows, n_ticks)
        # inject_start_tick=0; clip_tick = tick - 0 = tick
        assert result == [(t, t) for t in row_ticks]

    def test_inject_at_17_row_5_clip_tick_starts_at_0(self):
        """inject_at=17, row 5: inject_start_tick=video_row_to_audio_tick(5); clips start at 0."""
        from comfyui_h3_blended_inject.grid import (
            audio_tick_range,
            audio_ticks_for_rows,
            inject_audio_ticks_for_row,
            video_row_to_audio_tick,
        )

        n_rows = 10
        n_ticks = audio_ticks_for_rows(n_rows)
        inject_at = 17  # frame_to_row(17) = 5
        inject_start_tick = video_row_to_audio_tick(5)
        row_ticks = list(audio_tick_range(5, n_rows, n_ticks))
        result = inject_audio_ticks_for_row(5, inject_at, n_ticks, n_rows, n_ticks)
        assert result == [(t, t - inject_start_tick) for t in row_ticks]

    def test_out_of_range_clip_ticks_dropped(self):
        """Ticks whose clip_tick >= n_clip_ticks are dropped."""
        from comfyui_h3_blended_inject.grid import (
            audio_ticks_for_rows,
            inject_audio_ticks_for_row,
        )

        n_rows = 10
        n_ticks = audio_ticks_for_rows(n_rows)
        # Only allow 2 clip ticks from inject_at=0 (inject_start_tick=0)
        result = inject_audio_ticks_for_row(0, 0, 2, n_rows, n_ticks)
        assert all(0 <= clip_tick < 2 for _, clip_tick in result)
        assert len(result) == 2  # row 0 owns exactly 2 ticks (range(0, 2))
