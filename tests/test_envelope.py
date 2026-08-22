"""Property-based tests for comfyui_h3_blended_inject.envelope.

Behavior contract is taken from the module and function docstrings, and from the
"H3 Add Inject" and "Test plan" sections of .claude/plans/plan.md.

All stub functions raise NotImplementedError, so every test that calls a stub
will fail until the bodies are implemented. That is expected and correct at this
stage. Tests for the working enum / constant assignments will pass immediately.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from comfyui_h3_blended_inject.envelope import (
    INTERPOLATION_TYPES,
    InterpolationType,
    evaluate_curve,
    evaluate_envelope,
    is_row_exactly_zero,
    still_inject_denoise,
)

# ---------------------------------------------------------------------------
# INTERPOLATION_TYPES tuple and InterpolationType enum
# These are direct assignments in the module — not stubs — so they pass now.
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
# evaluate_envelope — basic structural checks
# ---------------------------------------------------------------------------


def test_evaluate_envelope_returns_list():
    result = evaluate_envelope(
        start_fade_in=0,
        start_keyframes=4,
        end_keyframes=8,
        end_fade_out=16,
        min_denoise=0.3,
        interpolation_type="linear",
        source_length=17,
        target_rows=5,
        inject_at_row=0,
    )
    assert isinstance(result, list)


def test_evaluate_envelope_values_in_range():
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
        inject_at_row=0,
    )
    for d in result:
        assert min_denoise <= d <= 1.0


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
            inject_at_row=0,
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
            inject_at_row=0,
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
            inject_at_row=0,
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
        inject_at_row=0,
    )
    for d in result:
        # Small epsilon for floating-point accumulation in curve evaluation.
        assert min_denoise - 1e-12 <= d <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# evaluate_envelope — hold-region invariant (concrete)
#
# Row 2 covers source frames 5–8 (center times 5.5, 6.5, 7.5, 8.5).
# With start_keyframes=5 and end_keyframes=9, all centers are within [5, 9].
# The entire row falls in the hold region and must equal min_denoise.
# ---------------------------------------------------------------------------


def test_evaluate_envelope_hold_region_row_equals_min_denoise():
    """Row whose centers are fully inside the hold region gets exactly min_denoise."""
    # target_rows=5 covers rows 0–4 with inject_at_row=0.
    # Envelope span [0, 17] contains all five rows, so result has 5 entries in row order.
    min_denoise = 0.25
    result = evaluate_envelope(
        start_fade_in=0,
        start_keyframes=5,  # hold starts at frame 5
        end_keyframes=9,  # hold ends at frame 9 (row 2 centers 5.5–8.5 fully within [5,9])
        end_fade_out=17,
        min_denoise=min_denoise,
        interpolation_type="linear",
        source_length=18,
        target_rows=5,
        inject_at_row=0,
    )
    # Row 2 is the third covered row (result[2]).
    assert len(result) >= 3
    assert result[2] == pytest.approx(min_denoise)


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
        inject_at_row=0,
    )
    assert len(result) > 0
    for d in result:
        assert d == pytest.approx(min_denoise)


# ---------------------------------------------------------------------------
# evaluate_envelope — degenerate still-inject envelope
# ---------------------------------------------------------------------------


def test_evaluate_envelope_still_inject_degenerate():
    """All four fade indices equal: result is a single entry equal to min_denoise."""
    min_denoise = 0.7
    result = evaluate_envelope(
        start_fade_in=5,
        start_keyframes=5,
        end_keyframes=5,
        end_fade_out=5,
        min_denoise=min_denoise,
        interpolation_type="linear",
        source_length=10,
        target_rows=10,
        inject_at_row=0,
    )
    assert result == [min_denoise]


# ---------------------------------------------------------------------------
# is_row_exactly_zero
# ---------------------------------------------------------------------------


def test_is_row_exactly_zero_true_when_min_denoise_zero_and_fully_in_hold():
    """min_denoise==0 and all frames in hold region [start_kf, end_kf] → True.

    Row 1 covers source frames 1–4. Hold region [1, 4] contains all four frames.
    """
    assert (
        is_row_exactly_zero(
            row_idx=1,
            start_fade_in=0,
            start_keyframes=1,
            end_keyframes=4,
            end_fade_out=5,
            min_denoise=0.0,
            inject_at_row=0,
        )
        is True
    )


def test_is_row_exactly_zero_false_when_min_denoise_nonzero():
    """min_denoise > 0 yields False even when all frames are in the hold region."""
    assert (
        is_row_exactly_zero(
            row_idx=1,
            start_fade_in=0,
            start_keyframes=1,
            end_keyframes=4,
            end_fade_out=5,
            min_denoise=0.1,
            inject_at_row=0,
        )
        is False
    )


def test_is_row_exactly_zero_false_when_row_only_partially_in_hold():
    """Row partially outside hold region → False even with min_denoise==0.

    Row 1 covers frames 1–4. Hold starts at frame 2, so frame 1 is outside.
    """
    assert (
        is_row_exactly_zero(
            row_idx=1,
            start_fade_in=0,
            start_keyframes=2,
            end_keyframes=4,
            end_fade_out=5,
            min_denoise=0.0,
            inject_at_row=0,
        )
        is False
    )


def test_is_row_exactly_zero_false_when_row_partially_above_hold():
    """Row partially above hold region end → False even with min_denoise==0.

    Row 1 covers frames 1–4. Hold ends at frame 3, so frame 4 is outside.
    """
    assert (
        is_row_exactly_zero(
            row_idx=1,
            start_fade_in=0,
            start_keyframes=1,
            end_keyframes=3,
            end_fade_out=5,
            min_denoise=0.0,
            inject_at_row=0,
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
            end_keyframes=4,
            end_fade_out=5,
            min_denoise=min_denoise,
            inject_at_row=0,
        )
        is False
    )


# ---------------------------------------------------------------------------
# still_inject_denoise
# ---------------------------------------------------------------------------


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_still_inject_denoise_returns_single_element_list(min_denoise):
    """still_inject_denoise(m) == [m] for any m in [0, 1]."""
    result = still_inject_denoise(min_denoise)
    assert result == [min_denoise]
