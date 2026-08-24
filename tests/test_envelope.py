"""Property-based tests for comfyui_h3_blended_inject.envelope.

Behavior contract is taken from the module and function docstrings, and from the
"H3 Add Inject" and "Test plan" sections of .claude/plans/plan.md.

After the frame-space coordinate rework, evaluate_envelope returns
list[tuple[int, float]] where each pair is (absolute_latent_row_idx, denoise).
All fade indices are CLIP frame indices; inject_at is a LATENT FRAME index.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from comfyui_h3_blended_inject.constants import frame_to_row, row_center_times
from comfyui_h3_blended_inject.envelope import (
    INTERPOLATION_TYPES,
    InterpolationType,
    _denoise_at_frame_time,
    classify_row_region,
    evaluate_curve,
    evaluate_envelope,
    is_row_exactly_zero,
    still_inject_denoise,
)

# ---------------------------------------------------------------------------
# INTERPOLATION_TYPES tuple and InterpolationType enum
# ---------------------------------------------------------------------------


def test_interpolation_types_value():
    assert INTERPOLATION_TYPES == ("ease_in", "ease_out", "ease_in_out", "linear", "none")


def test_interpolation_types_is_tuple():
    assert isinstance(INTERPOLATION_TYPES, tuple)


def test_interpolation_type_enum_values_match_tuple():
    """InterpolationType enum values are exactly the strings in INTERPOLATION_TYPES."""
    assert set(m.value for m in InterpolationType) == set(INTERPOLATION_TYPES)
    assert len(list(InterpolationType)) == len(INTERPOLATION_TYPES)


def test_interpolation_type_enum_string_members():
    assert InterpolationType.EASE_IN == "ease_in"
    assert InterpolationType.EASE_OUT == "ease_out"
    assert InterpolationType.EASE_IN_OUT == "ease_in_out"
    assert InterpolationType.LINEAR == "linear"
    assert InterpolationType.NONE == "none"


# ---------------------------------------------------------------------------
# evaluate_curve — endpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("itype", list(INTERPOLATION_TYPES))
def test_evaluate_curve_endpoint_zero(itype):
    """evaluate_curve(0.0, type) == 0.0 for all types (including 'none', since 0.0 < 1.0)."""
    assert evaluate_curve(0.0, itype) == 0.0


@pytest.mark.parametrize("itype", list(INTERPOLATION_TYPES))
def test_evaluate_curve_endpoint_one(itype):
    """evaluate_curve(1.0, type) == 1.0 for all types."""
    assert evaluate_curve(1.0, itype) == 1.0


# ---------------------------------------------------------------------------
# evaluate_curve — 'none' step behavior
# ---------------------------------------------------------------------------


@given(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False).filter(
        lambda t: t < 1.0
    )
)
def test_evaluate_curve_none_zero_for_t_lt_one(t):
    """'none' step function returns 0.0 for all t in [0.0, 1.0)."""
    assert evaluate_curve(t, "none") == 0.0


# ---------------------------------------------------------------------------
# evaluate_curve — hand-checked midpoint values
# ---------------------------------------------------------------------------


def test_evaluate_curve_ease_in_midpoint():
    """ease_in: t**2 at t=0.5 is 0.25."""
    assert evaluate_curve(0.5, "ease_in") == pytest.approx(0.25)


def test_evaluate_curve_ease_out_midpoint():
    """ease_out: 1-(1-t)**2 at t=0.5 is 0.75."""
    assert evaluate_curve(0.5, "ease_out") == pytest.approx(0.75)


def test_evaluate_curve_ease_in_out_midpoint():
    """ease_in_out: 3t^2-2t^3 at t=0.5 is 3*0.25-2*0.125 = 0.75-0.25 = 0.5."""
    assert evaluate_curve(0.5, "ease_in_out") == pytest.approx(0.5)


def test_evaluate_curve_linear_midpoint():
    """linear: t at t=0.5 is 0.5."""
    assert evaluate_curve(0.5, "linear") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# evaluate_curve — linear identity, monotonicity, bounds
# ---------------------------------------------------------------------------


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_evaluate_curve_linear_is_identity(t):
    assert evaluate_curve(t, "linear") == pytest.approx(t)


@given(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.sampled_from(list(INTERPOLATION_TYPES)),
)
def test_evaluate_curve_monotone_non_decreasing(t_a, t_b, itype):
    """evaluate_curve is monotone non-decreasing for every interpolation type."""
    lo, hi = min(t_a, t_b), max(t_a, t_b)
    # Allow a tiny epsilon for floating-point rounding in the math operations.
    assert evaluate_curve(lo, itype) <= evaluate_curve(hi, itype) + 1e-12


@given(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.sampled_from(list(INTERPOLATION_TYPES)),
)
def test_evaluate_curve_result_in_unit_interval(t, itype):
    result = evaluate_curve(t, itype)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# evaluate_curve — ValueError for out-of-range t and unknown type
# ---------------------------------------------------------------------------


@given(st.floats(max_value=-0.1, allow_nan=False, allow_infinity=False))
def test_evaluate_curve_t_below_zero_raises(t):
    with pytest.raises(ValueError):
        evaluate_curve(t, "linear")


@given(st.floats(min_value=1.1, allow_nan=False, allow_infinity=False))
def test_evaluate_curve_t_above_one_raises(t):
    with pytest.raises(ValueError):
        evaluate_curve(t, "linear")


def test_evaluate_curve_unknown_type_raises():
    with pytest.raises(ValueError):
        evaluate_curve(0.5, "not_a_valid_interpolation_type")


# ---------------------------------------------------------------------------
# evaluate_envelope — return type and structure
# ---------------------------------------------------------------------------


def test_evaluate_envelope_returns_list_of_tuples():
    """evaluate_envelope returns list[tuple[int, float]]."""
    result = evaluate_envelope(
        start_fade_in=0,
        start_keyframes=4,
        end_keyframes=8,
        end_fade_out=16,
        min_denoise=0.3,
        interpolation_type="linear",
        source_length=17,
        target_rows=5,
        inject_at=0,
    )
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, tuple)
        assert len(item) == 2
        row_idx, denoise = item
        assert isinstance(row_idx, int)
        assert isinstance(denoise, float)


def test_evaluate_envelope_rows_are_absolute_latent_indices():
    """Row indices in the result are absolute latent-row indices (not local/clip offsets)."""
    # inject_at=34 (2*17 frames), start_fade_in=0, end_fade_out=8
    # anchor = inject_at + sfi - 1 = 33; frame_to_row(33) = 9 (chunk=1, local=16, local_row=4)
    result = evaluate_envelope(
        start_fade_in=0,
        start_keyframes=4,
        end_keyframes=6,
        end_fade_out=8,
        min_denoise=0.3,
        interpolation_type="linear",
        source_length=20,
        target_rows=64,
        inject_at=34,
    )
    assert len(result) > 0
    # All row indices must be >= frame_to_row(anchor=33) = 9
    for row_idx, _ in result:
        assert row_idx >= 9, f"row {row_idx} is below anchor frame 33 (row 9)"


def test_evaluate_envelope_rows_sorted_ascending_and_unique():
    """Row indices are sorted ascending with no duplicates."""
    result = evaluate_envelope(
        start_fade_in=0,
        start_keyframes=4,
        end_keyframes=20,
        end_fade_out=30,
        min_denoise=0.2,
        interpolation_type="ease_in_out",
        source_length=40,
        target_rows=20,
        inject_at=0,
    )
    row_idxs = [r for r, _ in result]
    assert row_idxs == sorted(row_idxs), "rows must be sorted ascending"
    assert len(set(row_idxs)) == len(row_idxs), "row indices must be unique"


def test_evaluate_envelope_values_in_range():
    """All denoise values lie in [min_denoise, 1.0]."""
    min_denoise = 0.3
    result = evaluate_envelope(
        start_fade_in=0,
        start_keyframes=4,
        end_keyframes=8,
        end_fade_out=16,
        min_denoise=min_denoise,
        interpolation_type="linear",
        source_length=17,
        target_rows=5,
        inject_at=0,
    )
    for _, d in result:
        assert min_denoise - 1e-12 <= d <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# evaluate_envelope — ordering constraint raises ValueError
# ---------------------------------------------------------------------------


def test_evaluate_envelope_ordering_violation_fade_in_after_keyframes_raises():
    """start_fade_in > start_keyframes must raise ValueError."""
    with pytest.raises(ValueError):
        evaluate_envelope(
            start_fade_in=5,
            start_keyframes=2,
            end_keyframes=8,
            end_fade_out=12,
            min_denoise=0.3,
            interpolation_type="linear",
            source_length=13,
            target_rows=10,
            inject_at=0,
        )


def test_evaluate_envelope_ordering_violation_keyframes_after_fade_out_raises():
    """end_keyframes > end_fade_out must raise ValueError."""
    with pytest.raises(ValueError):
        evaluate_envelope(
            start_fade_in=0,
            start_keyframes=4,
            end_keyframes=12,
            end_fade_out=8,
            min_denoise=0.3,
            interpolation_type="linear",
            source_length=13,
            target_rows=10,
            inject_at=0,
        )


def test_evaluate_envelope_ordering_violation_start_after_end_keyframes_raises():
    """start_keyframes > end_keyframes must raise ValueError."""
    with pytest.raises(ValueError):
        evaluate_envelope(
            start_fade_in=0,
            start_keyframes=8,
            end_keyframes=4,
            end_fade_out=12,
            min_denoise=0.3,
            interpolation_type="linear",
            source_length=13,
            target_rows=10,
            inject_at=0,
        )


# ---------------------------------------------------------------------------
# evaluate_envelope — Hypothesis bounds invariant
# ---------------------------------------------------------------------------


@given(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.sampled_from(list(INTERPOLATION_TYPES)),
)
def test_evaluate_envelope_values_bounded(min_denoise, itype):
    """All returned denoise values lie in [min_denoise, 1.0] for any valid envelope."""
    result = evaluate_envelope(
        start_fade_in=0,
        start_keyframes=4,
        end_keyframes=12,
        end_fade_out=16,
        min_denoise=min_denoise,
        interpolation_type=itype,
        source_length=17,
        target_rows=5,
        inject_at=0,
    )
    for _, d in result:
        # Small epsilon for floating-point accumulation in curve evaluation.
        assert min_denoise - 1e-12 <= d <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# Reference implementation for property tests
# ---------------------------------------------------------------------------


def _reference_evaluate_envelope(
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
    """Reference implementation of the frame-space envelope algorithm.

    Implements the spec from scratch for comparison against evaluate_envelope.
    Uses the frame_to_row + clip-center-averaging algorithm described in the plan.
    """
    if not (start_fade_in <= start_keyframes <= end_keyframes <= end_fade_out):
        raise ValueError("Ordering violated")

    if start_fade_in == start_keyframes == end_keyframes == end_fade_out:
        r = frame_to_row(inject_at + start_fade_in)
        if r >= target_rows:
            return []
        return [(r, float(min_denoise))]

    # Half-open anchor: the 1.0 point is at start_fade_in - 1 (one frame before sfi).
    first_latent = inject_at + start_fade_in - 1
    last_latent = inject_at + end_fade_out
    result = []
    row_start = frame_to_row(first_latent) if first_latent >= 0 else 0
    for r in range(row_start, frame_to_row(last_latent) + 1):
        if r >= target_rows:
            break
        centers_latent = row_center_times(r)
        clip_centers = [c - inject_at for c in centers_latent]
        # New inclusion rule: center >= anchor (sfi - 1).
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
        result.append((r, sum(values) / len(values)))
    return result


# Hypothesis strategy for valid (sfi, skf, ekf, efo) tuples in sorted order.
@st.composite
def _sorted_clip_indices(draw: st.DrawFn, max_val: int = 80) -> tuple[int, int, int, int]:
    vals = draw(
        st.lists(st.integers(min_value=0, max_value=max_val), min_size=4, max_size=4).map(sorted)
    )
    return tuple(vals)  # type: ignore[return-value]


@given(
    clip_indices=_sorted_clip_indices(max_val=60),
    inject_at=st.integers(min_value=0, max_value=4).map(lambda n: n * 17),
    min_denoise=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    itype=st.sampled_from(list(INTERPOLATION_TYPES)),
    target_rows=st.integers(min_value=5, max_value=40),
)
@settings(max_examples=100)
def test_evaluate_envelope_matches_reference_implementation(
    clip_indices, inject_at, min_denoise, itype, target_rows
):
    """evaluate_envelope output matches an independent reference for random valid inputs."""
    sfi, skf, ekf, efo = clip_indices
    # source_length must exceed end_fade_out
    source_length = efo + 2

    result = evaluate_envelope(
        sfi, skf, ekf, efo, min_denoise, itype, source_length, target_rows, inject_at
    )
    reference = _reference_evaluate_envelope(
        sfi, skf, ekf, efo, min_denoise, itype, source_length, target_rows, inject_at
    )

    assert len(result) == len(reference), (
        f"length mismatch: got {len(result)}, want {len(reference)}"
    )
    for (r_row, r_d), (e_row, e_d) in zip(result, reference, strict=True):
        assert r_row == e_row, f"row mismatch: got {r_row}, want {e_row}"
        assert abs(r_d - e_d) < 1e-12, f"denoise mismatch at row {r_row}: got {r_d}, want {e_d}"


@given(
    clip_indices=_sorted_clip_indices(max_val=60),
    inject_at=st.integers(min_value=0, max_value=4).map(lambda n: n * 17),
    min_denoise=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    itype=st.sampled_from(list(INTERPOLATION_TYPES)),
    target_rows=st.integers(min_value=5, max_value=40),
)
@settings(max_examples=80)
def test_evaluate_envelope_rows_sorted_unique_bounded(
    clip_indices, inject_at, min_denoise, itype, target_rows
):
    """Rows are sorted ascending, unique, and in [0, target_rows)."""
    sfi, skf, ekf, efo = clip_indices
    result = evaluate_envelope(
        sfi, skf, ekf, efo, min_denoise, itype, efo + 2, target_rows, inject_at
    )
    rows = [r for r, _ in result]
    assert rows == sorted(rows), "rows not sorted"
    assert len(set(rows)) == len(rows), "duplicate rows"
    for r in rows:
        assert 0 <= r < target_rows, f"row {r} out of bounds [0, {target_rows})"


# ---------------------------------------------------------------------------
# evaluate_envelope — hold-region invariant (concrete)
#
# With inject_at=0 and start_keyframes=5, end_keyframes=9:
# Row 2 covers latent frames 5-8, clip centers (5.5,6.5,7.5,8.5), all in [5,9].
# ---------------------------------------------------------------------------


def test_evaluate_envelope_hold_region_row_equals_min_denoise():
    """Row whose centers are fully inside the hold region gets exactly min_denoise."""
    min_denoise = 0.25
    result = evaluate_envelope(
        start_fade_in=0,
        start_keyframes=5,  # hold starts at clip frame 5
        end_keyframes=10,  # exclusive: last hold frame = ekf-1=9 (row 2 centers 5.5-8.5 all <= 9)
        end_fade_out=17,
        min_denoise=min_denoise,
        interpolation_type="linear",
        source_length=18,
        target_rows=5,
        inject_at=0,
    )
    # Row 2 centers (inject_at=0): latent row 2 = clip row 2 = centers (5.5,6.5,7.5,8.5).
    # All are in [5, ekf-1=9], so row 2 gets exactly min_denoise.
    row_map = {r: d for r, d in result}
    assert 2 in row_map, "row 2 should be included"
    assert row_map[2] == pytest.approx(min_denoise)


def test_evaluate_envelope_wide_hold_all_rows_equal_min_denoise():
    """No fade-in or fade-out: every row in the result equals min_denoise."""
    # start_fade_in == start_keyframes and end_keyframes == end_fade_out → pure hold.
    min_denoise = 0.4
    result = evaluate_envelope(
        start_fade_in=0,
        start_keyframes=0,
        end_keyframes=100,
        end_fade_out=100,
        min_denoise=min_denoise,
        interpolation_type="linear",
        source_length=101,
        target_rows=5,
        inject_at=0,
    )
    assert len(result) > 0
    for _, d in result:
        assert d == pytest.approx(min_denoise)


# ---------------------------------------------------------------------------
# evaluate_envelope — hold region with min_denoise==0 yields exactly 0.0
# ---------------------------------------------------------------------------


def test_evaluate_envelope_hold_rows_with_zero_min_denoise_yield_zero():
    """Rows fully inside the hold region with min_denoise=0 must produce denoise exactly 0.0.

    Row 3 covers clip centers (9.5,10.5,11.5,12.5) with inject_at=0.
    Hold region [9, ekf-1=13] contains all four centers.
    """
    result = evaluate_envelope(
        start_fade_in=0,
        start_keyframes=9,
        end_keyframes=14,  # exclusive: last hold frame = ekf-1=13 (centers 9.5-12.5 all <= 13)
        end_fade_out=20,
        min_denoise=0.0,
        interpolation_type="linear",
        source_length=25,
        target_rows=10,
        inject_at=0,
    )
    row_map = {r: d for r, d in result}
    assert 3 in row_map, "row 3 should be included"
    assert row_map[3] == 0.0, f"expected exactly 0.0, got {row_map[3]}"


# ---------------------------------------------------------------------------
# evaluate_envelope — degenerate still-inject envelope
# ---------------------------------------------------------------------------


def test_evaluate_envelope_still_inject_degenerate():
    """All four fade indices equal: result is a single (row, min_denoise) entry."""
    min_denoise = 0.7
    inject_at = 0
    start_fade_in = 5
    result = evaluate_envelope(
        start_fade_in=start_fade_in,
        start_keyframes=5,
        end_keyframes=5,
        end_fade_out=5,
        min_denoise=min_denoise,
        interpolation_type="linear",
        source_length=10,
        target_rows=10,
        inject_at=inject_at,
    )
    expected_row = frame_to_row(inject_at + start_fade_in)
    assert result == [(expected_row, min_denoise)]


def test_evaluate_envelope_still_inject_out_of_bounds_returns_empty():
    """Degenerate still-inject whose row is >= target_rows returns []."""
    result = evaluate_envelope(
        start_fade_in=100,
        start_keyframes=100,
        end_keyframes=100,
        end_fade_out=100,
        min_denoise=0.5,
        interpolation_type="linear",
        source_length=200,
        target_rows=5,  # row for frame 100 = chunk=5,local=15,local_row=4 → row 29; 29 >= 5 → empty
        inject_at=0,
    )
    assert result == []


# ---------------------------------------------------------------------------
# is_row_exactly_zero — new frame-space contract
# ---------------------------------------------------------------------------


def test_is_row_exactly_zero_true_when_min_denoise_zero_and_fully_in_hold():
    """min_denoise==0 and all clip centers in hold [skf, ekf-1] → True.

    Row 1 with inject_at=0 has clip centers (1.5, 2.5, 3.5, 4.5).
    Hold [1, ekf-1=5] contains all four centers (4.5 <= ekf-1=5).
    """
    assert (
        is_row_exactly_zero(
            row_idx=1,
            start_fade_in=0,
            start_keyframes=1,
            end_keyframes=6,  # exclusive: ekf-1=5 is the last hold frame; 4.5 <= 5
            end_fade_out=7,
            min_denoise=0.0,
            inject_at=0,
        )
        is True
    )


def test_is_row_exactly_zero_false_when_min_denoise_nonzero():
    """min_denoise > 0 yields False even when all clip centers are in the hold region."""
    assert (
        is_row_exactly_zero(
            row_idx=1,
            start_fade_in=0,
            start_keyframes=1,
            end_keyframes=5,
            end_fade_out=6,
            min_denoise=0.1,
            inject_at=0,
        )
        is False
    )


def test_is_row_exactly_zero_false_when_row_only_partially_in_hold():
    """Row partially outside hold region → False even with min_denoise==0.

    Row 1 with inject_at=0 has clip centers (1.5,2.5,3.5,4.5).
    Hold starts at 2, so 1.5 < 2 is outside.
    """
    assert (
        is_row_exactly_zero(
            row_idx=1,
            start_fade_in=0,
            start_keyframes=2,
            end_keyframes=5,
            end_fade_out=6,
            min_denoise=0.0,
            inject_at=0,
        )
        is False
    )


def test_is_row_exactly_zero_false_when_row_partially_above_hold():
    """Row partially above hold region end → False even with min_denoise==0.

    Row 1 with inject_at=0 has clip centers (1.5,2.5,3.5,4.5).
    Hold ends at 4, so 4.5 > 4 is outside.
    """
    assert (
        is_row_exactly_zero(
            row_idx=1,
            start_fade_in=0,
            start_keyframes=1,
            end_keyframes=4,
            end_fade_out=6,
            min_denoise=0.0,
            inject_at=0,
        )
        is False
    )


@given(st.floats(min_value=1e-9, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_is_row_exactly_zero_false_for_any_positive_min_denoise(min_denoise):
    """min_denoise > 0 always yields False, regardless of frame coverage."""
    assert (
        is_row_exactly_zero(
            row_idx=1,
            start_fade_in=0,
            start_keyframes=1,
            end_keyframes=5,
            end_fade_out=6,
            min_denoise=min_denoise,
            inject_at=0,
        )
        is False
    )


def test_is_row_exactly_zero_consistent_with_evaluate_envelope_zero():
    """is_row_exactly_zero(r) is True iff evaluate_envelope yields denoise exactly 0.0 for r.

    Row 3 with inject_at=0: clip centers (9.5,10.5,11.5,12.5).
    Hold [9, ekf-1=13] contains all centers. min_denoise=0 → both is_row_exactly_zero=True
    and averaged denoise=0.0.
    """
    inject_at = 0
    row_idx = 3
    start_keyframes = 9
    end_keyframes = 14  # exclusive: ekf-1=13 is the last hold frame; 12.5 <= 13

    zero_check = is_row_exactly_zero(
        row_idx=row_idx,
        start_fade_in=0,
        start_keyframes=start_keyframes,
        end_keyframes=end_keyframes,
        end_fade_out=20,
        min_denoise=0.0,
        inject_at=inject_at,
    )
    env = evaluate_envelope(
        start_fade_in=0,
        start_keyframes=start_keyframes,
        end_keyframes=end_keyframes,
        end_fade_out=20,
        min_denoise=0.0,
        interpolation_type="linear",
        source_length=25,
        target_rows=10,
        inject_at=inject_at,
    )
    row_map = {r: d for r, d in env}
    assert row_idx in row_map
    assert zero_check is True
    assert row_map[row_idx] == 0.0


def test_is_row_exactly_zero_consistent_with_evaluate_envelope_nonzero():
    """is_row_exactly_zero(r) is False iff evaluate_envelope yields denoise > 0.0 for r.

    Row 1 with inject_at=0: clip centers (1.5,2.5,3.5,4.5).
    Hold [2, 4] — 1.5 < 2 and 4.5 > 4, so not all centers in hold.
    is_row_exactly_zero should be False and denoise should be > 0.0.
    """
    inject_at = 0
    row_idx = 1

    zero_check = is_row_exactly_zero(
        row_idx=row_idx,
        start_fade_in=0,
        start_keyframes=2,
        end_keyframes=4,
        end_fade_out=8,
        min_denoise=0.0,
        inject_at=inject_at,
    )
    env = evaluate_envelope(
        start_fade_in=0,
        start_keyframes=2,
        end_keyframes=4,
        end_fade_out=8,
        min_denoise=0.0,
        interpolation_type="linear",
        source_length=10,
        target_rows=10,
        inject_at=inject_at,
    )
    row_map = {r: d for r, d in env}
    assert row_idx in row_map
    assert zero_check is False
    assert row_map[row_idx] > 0.0


@given(
    inject_at=st.integers(min_value=0, max_value=4).map(lambda n: n * 17),
    row_offset=st.integers(min_value=0, max_value=10),
    skf=st.integers(min_value=0, max_value=30),
    ekf_gap=st.integers(min_value=0, max_value=20),
    efo_gap=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
def test_is_row_exactly_zero_consistent_with_averaged_denoise(
    inject_at, row_offset, skf, ekf_gap, efo_gap
):
    """is_row_exactly_zero(r) matches (averaged denoise==0.0) from evaluate_envelope."""
    row_idx = frame_to_row(inject_at) + row_offset
    ekf = skf + ekf_gap
    efo = ekf + efo_gap
    sfi = 0
    min_denoise = 0.0
    source_length = inject_at + efo + 5
    target_rows = row_idx + 5

    env = evaluate_envelope(
        sfi, skf, ekf, efo, min_denoise, "linear", source_length, target_rows, inject_at
    )
    row_map = {r: d for r, d in env}

    zero_check = is_row_exactly_zero(
        row_idx=row_idx,
        start_fade_in=sfi,
        start_keyframes=skf,
        end_keyframes=ekf,
        end_fade_out=efo,
        min_denoise=min_denoise,
        inject_at=inject_at,
    )

    if row_idx in row_map:
        # If envelope covers the row, is_row_exactly_zero must match (denoise==0.0)
        assert zero_check == (row_map[row_idx] == 0.0), (
            f"is_row_exactly_zero={zero_check} but denoise={row_map[row_idx]} "
            f"at row {row_idx} (inject_at={inject_at}, skf={skf}, ekf={ekf})"
        )


# ---------------------------------------------------------------------------
# still_inject_denoise
# ---------------------------------------------------------------------------


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_still_inject_denoise_returns_single_element_list(min_denoise):
    """still_inject_denoise(m) == [m] for any m in [0, 1]."""
    result = still_inject_denoise(min_denoise)
    assert result == [min_denoise]


# ---------------------------------------------------------------------------
# Half-open anchor regression tests — FAIL before the anchor fix, PASS after
# ---------------------------------------------------------------------------


class TestHalfOpenEnvelopeAnchorRegression:
    """Regression tests for the half-open [start_fade_in, end_fade_out) anchor model.

    start_fade_in is the FIRST frame below 1.0; the 1.0 anchor lives at
    start_fade_in - 1.  end_fade_out is the EXCLUSIVE upper bound (denoise
    returns to 1.0 here).

    Every test in this class FAILS with the old closed-interval model and
    PASSES after the anchor fix.
    """

    def test_denoise_at_start_fade_in_is_below_one_not_one(self):
        """At t = start_fade_in exactly, the new anchor shifts the ramp so denoise < 1.0.

        OLD: anchor at sfi → fade_t = (sfi - sfi) / (skf - sfi) = 0 → denoise = 1.0
        NEW: anchor at sfi-1 → fade_t = 1 / (skf - sfi + 1) > 0 → denoise < 1.0

        FAILS before fix: old returns exactly 1.0.
        """
        sfi, skf, ekf, efo = 3, 7, 10, 14
        d = _denoise_at_frame_time(
            float(sfi), sfi, skf, ekf, efo, min_denoise=0.3, interpolation_type="linear"
        )
        assert d < 1.0, f"Expected denoise < 1.0 at t=sfi under new anchor, got {d}"

    def test_evaluate_envelope_pulls_in_row_with_center_before_sfi(self):
        """Row whose center falls in [sfi-1, sfi) is excluded by old model but included by new.

        With sfi=1, inject_at=0: row 0 has a single center at 0.5.
        OLD: row_start = frame_to_row(inject_at + sfi) = 1 → row 0 absent.
        NEW: anchor=0; row_start = frame_to_row(0) = 0; inclusion 0 <= 0.5 → row 0 present.

        FAILS before fix: 0 not in result.
        """
        result = evaluate_envelope(
            start_fade_in=1,
            start_keyframes=5,
            end_keyframes=9,
            end_fade_out=13,
            min_denoise=0.3,
            interpolation_type="linear",
            source_length=20,
            target_rows=10,
            inject_at=0,
        )
        row_map = {r: d for r, d in result}
        assert 0 in row_map, "Row 0 (center 0.5) should be claimed under new anchor (sfi-1=0)"
        assert row_map[0] < 1.0, f"Row 0 denoise should be < 1.0, got {row_map[0]}"

    def test_one_frame_fade_in_ramps_not_instant_min_at_anchor_row(self):
        """With sfi=skf (zero-length fade-in under old model), the anchor row gets a ramp value.

        OLD: row 0 center 0.5 < sfi=1 → excluded entirely.
        NEW: anchor=0; 0 <= 0.5 → included; fade_t = 0.5 → denoise strictly between 1.0 and min.

        FAILS before fix: row 0 absent from result.
        """
        min_d = 0.3
        result = evaluate_envelope(
            start_fade_in=1,
            start_keyframes=1,
            end_keyframes=5,
            end_fade_out=9,
            min_denoise=min_d,
            interpolation_type="linear",
            source_length=15,
            target_rows=10,
            inject_at=0,
        )
        row_map = {r: d for r, d in result}
        assert 0 in row_map, "Row 0 (center 0.5) should be included under new anchor"
        assert row_map[0] > min_d, f"Row 0 should be above min_denoise={min_d}, got {row_map[0]}"
        assert row_map[0] < 1.0, f"Row 0 should be below 1.0, got {row_map[0]}"

    def test_whole_clip_sfi_zero_efo_source_length_all_rows_below_one(self):
        """sfi=0, efo=source_length (valid after COMMIT 1) → all rows claimed with denoise < 1.0.

        The anchor at -1 ensures even row 0 (center 0.5) is in the fade-in ramp.
        This test is a sanity confirmation; it may also pass with the old model for sfi=0.
        """
        result = evaluate_envelope(
            start_fade_in=0,
            start_keyframes=3,
            end_keyframes=7,
            end_fade_out=13,
            min_denoise=0.3,
            interpolation_type="linear",
            source_length=13,
            target_rows=10,
            inject_at=0,
        )
        assert len(result) > 0, "Should have at least one row"
        assert all(d < 1.0 for _, d in result), "All rows should have denoise < 1.0"


# ---------------------------------------------------------------------------
# end_keyframes exclusive regression tests — FAIL before COMMIT 3, PASS after
# ---------------------------------------------------------------------------


class TestExclusiveEndKeyframesRegression:
    """Regression tests for the exclusive end_keyframes boundary (half-open hold region).

    Hold region = [start_keyframes, end_keyframes).  end_keyframes is the FIRST fade-out
    frame; the last held frame is end_keyframes - 1.

    Every test in this class FAILS with the old inclusive end_keyframes and PASSES after
    the COMMIT 3 fix.
    """

    def test_denoise_at_end_keyframes_is_above_min(self):
        """At t = end_keyframes exactly, denoise must be > min_denoise (fade-out has started).

        OLD: hold branch `t <= ekf` fires → min_denoise.  FAILS assertion `> min_denoise`.
        NEW: hold branch `t <= ekf-1` does not fire → fade-out → denoise > min_denoise.
        """
        sfi, skf, ekf, efo = 0, 3, 8, 12
        min_d = 0.3
        d_at_ekf_minus_1 = _denoise_at_frame_time(
            float(ekf - 1), sfi, skf, ekf, efo, min_d, "linear"
        )
        d_at_ekf = _denoise_at_frame_time(float(ekf), sfi, skf, ekf, efo, min_d, "linear")
        assert d_at_ekf_minus_1 == pytest.approx(min_d), (
            f"t=ekf-1 should be exactly min_denoise={min_d}, got {d_at_ekf_minus_1}"
        )
        assert d_at_ekf > min_d, (
            f"t=ekf should be above min_denoise={min_d} (fade-out started), got {d_at_ekf}"
        )

    def test_mirror_symmetry_under_exclusive_endpoints(self):
        """sfi=0, skf=2, ekf=15, efo=17 (linear, min_denoise=0.0): the ramp is symmetric.

        Under the half-open model:
          anchor = -1, last-hold = ekf-1 = 14, efo-anchor = 18
          t=0 -> fade_t=1/3 -> 2/3     t=16 -> fade_t=2/3 -> 2/3  (equal)
          t=1 -> fade_t=2/3 -> 1/3     t=15 -> fade_t=1/3 -> 1/3  (equal)
          t=2..14 -> 0.0

        OLD (inclusive ekf=15): t=15 in hold (15<=15) → 0.0 ≠ t=1 → 0.5 (sfi-anchor-less model).
        FAILS before: frame 1 ≠ frame 15.
        """
        sfi, skf, ekf, efo = 0, 2, 15, 17

        def d(t: float) -> float:
            return _denoise_at_frame_time(t, sfi, skf, ekf, efo, 0.0, "linear")

        assert d(0) == pytest.approx(d(16)), f"frame 0 ({d(0)}) != frame 16 ({d(16)})"
        assert d(1) == pytest.approx(d(15)), f"frame 1 ({d(1)}) != frame 15 ({d(15)})"
        for t in range(2, 15):
            assert d(float(t)) == pytest.approx(0.0), f"frame {t} should be 0.0, got {d(float(t))}"

    def test_is_row_exactly_zero_false_when_max_center_equals_end_keyframes_minus_half(self):
        """A row whose highest center is ekf - 0.5 is NOT exactly zero under the new model.

        Row 1, skf=1, ekf=5: centers (1.5, 2.5, 3.5, 4.5).  Highest center = 4.5 = ekf - 0.5.
        OLD: all <= ekf=5 → is_row_exactly_zero=True (with min_denoise=0).  FAILS assertion.
        NEW: 4.5 > ekf-1=4 → not all in hold → is_row_exactly_zero=False.  PASSES.
        """
        result = is_row_exactly_zero(
            row_idx=1,
            start_fade_in=0,
            start_keyframes=1,
            end_keyframes=5,
            end_fade_out=8,
            min_denoise=0.0,
            inject_at=0,
        )
        assert result is False, "center 4.5 = ekf-0.5 falls outside hold [skf=1, ekf-1=4]"


# ---------------------------------------------------------------------------
# classify_row_region — region classifier tests
# FAIL before classify_row_region is added to envelope.py; PASS after.
# ---------------------------------------------------------------------------


class TestClassifyRowRegion:
    """classify_row_region classifies a scheduled row by integer clip-frame membership.

    Grid reference (inject_at=0):
      Row 0: centers (0.5,)
      Row 1: centers (1.5, 2.5, 3.5, 4.5)
      Row 2: centers (5.5, 6.5, 7.5, 8.5)
      Row 3: centers (9.5, 10.5, 11.5, 12.5)
      Row 4: centers (13.5, 14.5, 15.5, 16.5)

    For inject_at=17: Row 5 centers (17.5,) → clip_center = 17.5 - 17 = 0.5; etc.
    """

    # --- preserve (min_denoise=0, all centers in [skf, ekf)) ---

    def test_preserve_when_min_denoise_zero_and_all_centers_in_hold(self) -> None:
        """Row 2 centers (5.5–8.5) all in [5, 10) with min_denoise=0 → 'preserve'."""
        result = classify_row_region(
            row_idx=2,
            inject_at=0,
            start_fade_in=0,
            start_keyframes=5,
            end_keyframes=10,
            end_fade_out=17,
            min_denoise=0.0,
        )
        assert result == "preserve", f"Expected 'preserve', got {result!r}"

    # --- hold (min_denoise>0, all centers in [skf, ekf)) ---

    def test_hold_when_min_denoise_positive_and_all_centers_in_hold(self) -> None:
        """Row 2 centers (5.5–8.5) all in [5, 10) with min_denoise=0.3 → 'hold'."""
        result = classify_row_region(
            row_idx=2,
            inject_at=0,
            start_fade_in=0,
            start_keyframes=5,
            end_keyframes=10,
            end_fade_out=17,
            min_denoise=0.3,
        )
        assert result == "hold", f"Expected 'hold', got {result!r}"

    # --- fade-in ramp ---
    # Updated (E1): ramp rows now return 'hold' by default (crossfade=False).
    # The parked prediction-blend path is only active when crossfade=True.

    def test_fade_in_ramp_row_classified_as_hold_by_default(self) -> None:
        """Row 1 centers (1.5–4.5) all in fade-in span [1, 5) → 'hold' (default, crossfade=False).

        E1 change: ramp rows use fractional hold-and-release by default; staggered
        per-row release IS the fade. The old assertion was 'fade'; updated to 'hold'.
        """
        result = classify_row_region(
            row_idx=1,
            inject_at=0,
            start_fade_in=1,
            start_keyframes=5,
            end_keyframes=10,
            end_fade_out=17,
            min_denoise=0.3,
        )
        assert result == "hold", f"Expected 'hold' (fade-in ramp, default), got {result!r}"

    def test_fade_in_ramp_row_classified_as_fade_crossfade_true(self) -> None:
        """Row 1 centers (1.5–4.5) in fade-in span [1, 5) → 'fade' when crossfade=True.

        Companion to test_fade_in_ramp_row_classified_as_hold_by_default; verifies the
        parked prediction-blend path is reachable via the crossfade flag.
        """
        result = classify_row_region(
            row_idx=1,
            inject_at=0,
            start_fade_in=1,
            start_keyframes=5,
            end_keyframes=10,
            end_fade_out=17,
            min_denoise=0.3,
            crossfade=True,
        )
        assert result == "fade", f"Expected 'fade' (fade-in ramp, crossfade=True), got {result!r}"

    def test_fade_in_ramp_row_min_denoise_zero_classified_as_hold_by_default(self) -> None:
        """Fade-in ramp row with min_denoise=0 → 'hold' by default (not 'preserve', not 'fade').

        E1 change: ramp rows with min_denoise=0 used to return 'fade'; now return 'hold'
        by default because classification uses integer clip-frame membership (ramp span),
        not min_denoise. min_denoise=0 only yields 'preserve' inside the hold span.
        """
        result = classify_row_region(
            row_idx=1,
            inject_at=0,
            start_fade_in=1,
            start_keyframes=5,
            end_keyframes=10,
            end_fade_out=17,
            min_denoise=0.0,
        )
        assert result == "hold", f"Expected 'hold' (fade-in ramp, d=0, default), got {result!r}"

    def test_fade_in_ramp_row_min_denoise_zero_classified_as_fade_crossfade_true(self) -> None:
        """Fade-in ramp row with min_denoise=0 → 'fade' when crossfade=True (not 'preserve').

        Companion to test_fade_in_ramp_row_min_denoise_zero_classified_as_hold_by_default.
        """
        result = classify_row_region(
            row_idx=1,
            inject_at=0,
            start_fade_in=1,
            start_keyframes=5,
            end_keyframes=10,
            end_fade_out=17,
            min_denoise=0.0,
            crossfade=True,
        )
        assert result == "fade", (
            f"Expected 'fade' (fade-in ramp, d=0, crossfade=True), got {result!r}"
        )

    # --- fade-out ramp ---

    def test_fade_out_ramp_row_classified_as_hold_by_default(self) -> None:
        """Row 3 centers (9.5–12.5) in fade-out span [9, 14) → 'hold' (default, crossfade=False).

        E1 change: ramp rows now return 'hold' by default; old assertion was 'fade'.
        """
        result = classify_row_region(
            row_idx=3,
            inject_at=0,
            start_fade_in=0,
            start_keyframes=5,
            end_keyframes=9,
            end_fade_out=14,
            min_denoise=0.3,
        )
        assert result == "hold", f"Expected 'hold' (fade-out ramp, default), got {result!r}"

    def test_fade_out_ramp_row_classified_as_fade_crossfade_true(self) -> None:
        """Row 3 centers (9.5–12.5) in fade-out span [9, 14) → 'fade' when crossfade=True.

        Companion to test_fade_out_ramp_row_classified_as_hold_by_default.
        """
        result = classify_row_region(
            row_idx=3,
            inject_at=0,
            start_fade_in=0,
            start_keyframes=5,
            end_keyframes=9,
            end_fade_out=14,
            min_denoise=0.3,
            crossfade=True,
        )
        assert result == "fade", f"Expected 'fade' (fade-out ramp, crossfade=True), got {result!r}"

    # --- free (row outside envelope) ---

    def test_free_when_row_outside_envelope(self) -> None:
        """Row 0 center (0.5) below fade-in start (5) and not in hold → 'free'."""
        result = classify_row_region(
            row_idx=0,
            inject_at=0,
            start_fade_in=5,
            start_keyframes=6,
            end_keyframes=10,
            end_fade_out=14,
            min_denoise=0.3,
        )
        assert result == "free", f"Expected 'free' (row outside envelope), got {result!r}"

    # --- boundary: row spanning hold+fade ---
    # Updated (E1): these rows are now 'hold' by default; 'fade' only when crossfade=True.

    def test_boundary_row_spanning_hold_and_fade_in_classified_as_hold_by_default(self) -> None:
        """Row 1 centers (1.5–4.5): 1.5 and 2.5 in fade-in [1,3), 3.5 and 4.5 in hold [3,10).
        Not ALL in hold → ramp row. Default (crossfade=False) → 'hold'.

        E1 change: boundary ramp rows now return 'hold' by default; old assertion was 'fade'.
        """
        result = classify_row_region(
            row_idx=1,
            inject_at=0,
            start_fade_in=1,
            start_keyframes=3,  # skf=3; centers 1.5,2.5 in [1,3) → ramp
            end_keyframes=10,
            end_fade_out=17,
            min_denoise=0.3,
        )
        assert result == "hold", (
            f"Expected 'hold' (spans hold+fade-in boundary, default), got {result!r}"
        )

    def test_boundary_row_spanning_hold_and_fade_in_classified_as_fade_crossfade_true(self) -> None:
        """Row 1 spanning hold+fade-in boundary → 'fade' when crossfade=True.

        Companion to test_boundary_row_spanning_hold_and_fade_in_classified_as_hold_by_default.
        """
        result = classify_row_region(
            row_idx=1,
            inject_at=0,
            start_fade_in=1,
            start_keyframes=3,
            end_keyframes=10,
            end_fade_out=17,
            min_denoise=0.3,
            crossfade=True,
        )
        assert result == "fade", (
            f"Expected 'fade' (spans hold+fade-in, crossfade=True), got {result!r}"
        )

    def test_boundary_row_spanning_hold_and_fade_out_classified_as_hold_by_default(self) -> None:
        """Row 2 centers (5.5–8.5): some in hold [5,8), some in fade-out [8,14).
        Not ALL in hold → ramp row. Default (crossfade=False) → 'hold'.

        E1 change: boundary ramp rows now return 'hold' by default; old assertion was 'fade'.
        """
        result = classify_row_region(
            row_idx=2,
            inject_at=0,
            start_fade_in=0,
            start_keyframes=5,
            end_keyframes=8,  # ekf=8; centers 8.5 in [8,14) → ramp
            end_fade_out=14,
            min_denoise=0.3,
        )
        assert result == "hold", (
            f"Expected 'hold' (spans hold+fade-out boundary, default), got {result!r}"
        )

    def test_boundary_row_spanning_hold_and_fade_out_classified_as_fade_crossfade_true(
        self,
    ) -> None:
        """Row 2 spanning hold+fade-out boundary → 'fade' when crossfade=True.

        Companion to test_boundary_row_spanning_hold_and_fade_out_classified_as_hold_by_default.
        """
        result = classify_row_region(
            row_idx=2,
            inject_at=0,
            start_fade_in=0,
            start_keyframes=5,
            end_keyframes=8,
            end_fade_out=14,
            min_denoise=0.3,
            crossfade=True,
        )
        assert result == "fade", (
            f"Expected 'fade' (spans hold+fade-out, crossfade=True), got {result!r}"
        )

    # --- inject_at > 0 offset ---

    def test_inject_at_offset_hold_row(self) -> None:
        """With inject_at=17, row 5 center (17.5) → clip_center=0.5 in [0, 5) hold → 'hold'."""
        result = classify_row_region(
            row_idx=5,
            inject_at=17,
            start_fade_in=0,
            start_keyframes=0,
            end_keyframes=5,
            end_fade_out=10,
            min_denoise=0.4,
        )
        assert result == "hold", f"Expected 'hold' with inject_at=17 offset, got {result!r}"

    def test_inject_at_offset_preserve_row(self) -> None:
        """With inject_at=17, row 5 center clip=0.5 in [0, 5) hold + min_denoise=0 → 'preserve'."""
        result = classify_row_region(
            row_idx=5,
            inject_at=17,
            start_fade_in=0,
            start_keyframes=0,
            end_keyframes=5,
            end_fade_out=10,
            min_denoise=0.0,
        )
        assert result == "preserve", f"Expected 'preserve' with inject_at=17 offset, got {result!r}"

    def test_inject_at_offset_ramp_row_hold_by_default(self) -> None:
        """Row 6 (inject_at=17) clip centers in fade-in [1,5) → 'hold' by default (crossfade=False).

        E1 change: ramp rows return 'hold' by default; old assertion was 'fade'.
        """
        result = classify_row_region(
            row_idx=6,
            inject_at=17,
            start_fade_in=1,
            start_keyframes=5,
            end_keyframes=10,
            end_fade_out=17,
            min_denoise=0.3,
        )
        assert result == "hold", f"Expected 'hold' (ramp, inject_at=17, default), got {result!r}"

    def test_inject_at_offset_ramp_row_fade_crossfade_true(self) -> None:
        """Row 6 (inject_at=17) clip centers in fade-in [1,5) → 'fade' when crossfade=True.

        Companion to test_inject_at_offset_ramp_row_hold_by_default.
        """
        result = classify_row_region(
            row_idx=6,
            inject_at=17,
            start_fade_in=1,
            start_keyframes=5,
            end_keyframes=10,
            end_fade_out=17,
            min_denoise=0.3,
            crossfade=True,
        )
        assert result == "fade", (
            f"Expected 'fade' (ramp, inject_at=17, crossfade=True), got {result!r}"
        )

    # --- degenerate still-inject (all four markers equal) ---

    def test_degenerate_still_inject_min_denoise_zero_is_preserve(self) -> None:
        """Degenerate envelope (all four equal) with min_denoise=0.0 → 'preserve'."""
        result = classify_row_region(
            row_idx=2,
            inject_at=0,
            start_fade_in=5,
            start_keyframes=5,
            end_keyframes=5,
            end_fade_out=5,
            min_denoise=0.0,
        )
        assert result == "preserve", f"Expected 'preserve' for degenerate d=0, got {result!r}"

    def test_degenerate_still_inject_min_denoise_positive_is_hold(self) -> None:
        """Degenerate envelope (all four equal) with min_denoise>0 → 'hold'."""
        result = classify_row_region(
            row_idx=2,
            inject_at=0,
            start_fade_in=5,
            start_keyframes=5,
            end_keyframes=5,
            end_fade_out=5,
            min_denoise=0.5,
        )
        assert result == "hold", f"Expected 'hold' for degenerate d>0, got {result!r}"

    # --- Hypothesis: consistency with evaluate_envelope hold-region rows ---

    @given(
        inject_at=st.integers(min_value=0, max_value=4).map(lambda n: n * 17),
        ekf_gap=st.integers(min_value=1, max_value=20),
        min_denoise=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=80)
    def test_classify_row_region_pure_hold_all_centers_inside(
        self, inject_at: int, ekf_gap: int, min_denoise: float
    ) -> None:
        """When ALL clip-frame centers of a row lie in [skf, ekf), classify_row_region
        returns 'hold' (if min_denoise>0) or 'preserve' (if min_denoise==0).

        This tests the classifier's core contract directly, without relying on
        evaluate_envelope (which can include boundary rows with centers partially
        outside the hold span due to the anchor-zone effect).

        Uses skf=0 so the hold region starts at clip frame 0 and the anchor is at -1,
        ensuring all non-negative clip centers are fully inside [0, ekf).
        """
        skf = 0  # hold starts at clip frame 0; anchor at -1
        ekf = ekf_gap  # exclusive end
        efo = ekf  # no fade-out
        sfi = skf  # no fade-in

        # Find a row whose ALL clip centers are in [0, ekf) after subtracting inject_at.
        # Row at frame_to_row(inject_at) has center inject_at + 0.5; clip center = 0.5.
        # All rows up to (but not including) the first row whose centers exceed ekf qualify.
        base_row = frame_to_row(inject_at)
        expected_region = "preserve" if min_denoise == 0.0 else "hold"

        for row_idx_offset in range(10):
            row_idx = base_row + row_idx_offset
            clip_centers = [c - inject_at for c in row_center_times(row_idx)]
            if all(skf <= cc < ekf for cc in clip_centers):
                region = classify_row_region(row_idx, inject_at, sfi, skf, ekf, efo, min_denoise)
                assert region == expected_region, (
                    f"row {row_idx} (inject_at={inject_at}, skf={skf}, ekf={ekf}, "
                    f"clip_centers={[round(c, 2) for c in clip_centers]}): "
                    f"all centers in [{skf},{ekf}) → expected {expected_region!r}, "
                    f"got {region!r}"
                )
