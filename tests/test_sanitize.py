"""Tests for comfyui_h3_blended_inject.sanitize.

All functions currently raise NotImplementedError; every test here is expected
to fail (error) until the implementations land.  The test bodies encode the
documented contract so that a correct implementation makes them pass without
modification.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from comfyui_h3_blended_inject.sanitize import (
    check_resolution,
    sanitize_audio,
    snap_inject_at,
    snap_length_down,
    validate_envelope_indices,
    warn_audio_tail_alignment,
    warn_audio_tick_alignment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(h: int, w: int, batch: int = 1, channels: int = 3) -> SimpleNamespace:
    """Minimal IMAGE-shaped object exposing .shape = (batch, H, W, C)."""
    return SimpleNamespace(shape=(batch, h, w, channels))


def _make_audio(waveform: torch.Tensor, sample_rate: int) -> dict:
    return {"waveform": waveform, "sample_rate": sample_rate}


def _expected_samples(frames: int, fps: int, sample_rate: int) -> int:
    """Reference formula: video_duration_frames / fps * target_sample_rate."""
    return round(frames / fps * sample_rate)


# ---------------------------------------------------------------------------
# snap_inject_at
# ---------------------------------------------------------------------------


class TestSnapInjectAt:
    """snap_inject_at(x) -> x - (x % 17); warn on change; raise ValueError for x < 0."""

    # -- ValueError for negative inputs -----------------------------------------

    def test_negative_one_raises(self):
        with pytest.raises(ValueError):
            snap_inject_at(-1)

    def test_negative_multiple_of_17_raises(self):
        with pytest.raises(ValueError):
            snap_inject_at(-17)

    # -- Already-aligned (17n): return unchanged, no UserWarning ----------------

    def test_zero_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = snap_inject_at(0)
        assert result == 0
        assert not any(issubclass(x.category, UserWarning) for x in w)

    def test_17_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = snap_inject_at(17)
        assert result == 17
        assert not any(issubclass(x.category, UserWarning) for x in w)

    def test_34_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = snap_inject_at(34)
        assert result == 34
        assert not any(issubclass(x.category, UserWarning) for x in w)

    def test_51_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = snap_inject_at(51)
        assert result == 51
        assert not any(issubclass(x.category, UserWarning) for x in w)

    def test_170_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = snap_inject_at(170)
        assert result == 170
        assert not any(issubclass(x.category, UserWarning) for x in w)

    # -- Non-aligned: floor-snap and emit UserWarning ---------------------------

    def test_1_snaps_to_0_warns(self):
        with pytest.warns(UserWarning):
            result = snap_inject_at(1)
        assert result == 0

    def test_16_snaps_to_0_warns(self):
        with pytest.warns(UserWarning):
            result = snap_inject_at(16)
        assert result == 0

    def test_18_snaps_to_17_warns(self):
        with pytest.warns(UserWarning):
            result = snap_inject_at(18)
        assert result == 17

    def test_20_snaps_to_17_warns(self):
        with pytest.warns(UserWarning):
            result = snap_inject_at(20)
        assert result == 17

    def test_35_snaps_to_34_warns(self):
        with pytest.warns(UserWarning):
            result = snap_inject_at(35)
        assert result == 34

    def test_52_snaps_to_51_warns(self):
        with pytest.warns(UserWarning):
            result = snap_inject_at(52)
        assert result == 51

    # -- Properties: result is a multiple of 17, <= x, and x - result < 17 -----

    @given(x=st.integers(min_value=0, max_value=10_000))
    def test_result_is_multiple_of_17(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = snap_inject_at(x)
        assert result % 17 == 0

    @given(x=st.integers(min_value=0, max_value=10_000))
    def test_result_le_x(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = snap_inject_at(x)
        assert result <= x

    @given(x=st.integers(min_value=0, max_value=10_000))
    def test_remainder_lt_17(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = snap_inject_at(x)
        assert x - result < 17

    @given(n=st.integers(min_value=0, max_value=588))  # 17 * 588 < 10_000
    def test_aligned_multiples_never_warn(self, n):
        x = n * 17
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = snap_inject_at(x)
        assert result == x
        assert not any(issubclass(uw.category, UserWarning) for uw in w)


# ---------------------------------------------------------------------------
# warn_audio_tick_alignment
# ---------------------------------------------------------------------------


class TestWarnAudioTickAlignment:
    """warn_audio_tick_alignment: pure side-effect; warns with ms error for 17n-not-51n."""

    # -- Multiples of 51 (exact audio tick): no warning -------------------------

    def test_zero_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_audio_tick_alignment(0)
        assert not any(issubclass(x.category, UserWarning) for x in w)

    def test_51_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_audio_tick_alignment(51)
        assert not any(issubclass(x.category, UserWarning) for x in w)

    def test_102_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_audio_tick_alignment(102)
        assert not any(issubclass(x.category, UserWarning) for x in w)

    def test_153_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_audio_tick_alignment(153)
        assert not any(issubclass(x.category, UserWarning) for x in w)

    # -- 17n but not 51n: warn with ms error ------------------------------------

    def test_17_warns_with_ms(self):
        with pytest.warns(UserWarning) as record:
            warn_audio_tick_alignment(17)
        assert "ms" in str(record[0].message).lower()

    def test_34_warns_with_ms(self):
        with pytest.warns(UserWarning) as record:
            warn_audio_tick_alignment(34)
        assert "ms" in str(record[0].message).lower()

    def test_68_warns_with_ms(self):
        with pytest.warns(UserWarning) as record:
            warn_audio_tick_alignment(68)
        assert "ms" in str(record[0].message).lower()

    def test_85_warns_with_ms(self):
        with pytest.warns(UserWarning) as record:
            warn_audio_tick_alignment(85)
        assert "ms" in str(record[0].message).lower()

    # -- Property: all 51n values produce no warning ----------------------------

    @given(n=st.integers(min_value=0, max_value=196))  # 51 * 196 < 10_000
    def test_51n_never_warns(self, n):
        x = n * 51
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_audio_tick_alignment(x)
        assert not any(issubclass(uw.category, UserWarning) for uw in w)

    # -- Property: 17n-not-51n always warns -------------------------------------

    @given(n=st.integers(min_value=1, max_value=588))
    def test_17n_not_51n_always_warns(self, n):
        if n % 3 == 0:
            # This is a multiple of 51; covered by the no-warn property above.
            return
        x = n * 17
        with pytest.warns(UserWarning):
            warn_audio_tick_alignment(x)


# ---------------------------------------------------------------------------
# check_resolution
# ---------------------------------------------------------------------------


class TestCheckResolution:
    """check_resolution: None when W==tw, H==th, both multiples of 32; ValueError otherwise."""

    # -- Valid cases ------------------------------------------------------------

    def test_valid_square(self):
        img = _make_image(h=512, w=512)
        assert check_resolution(img, target_width=512, target_height=512) is None

    def test_valid_rectangular(self):
        img = _make_image(h=288, w=512)
        assert check_resolution(img, target_width=512, target_height=288) is None

    def test_valid_minimal_32x32(self):
        img = _make_image(h=32, w=32)
        assert check_resolution(img, target_width=32, target_height=32) is None

    # -- Width not a multiple of 32 raises ValueError ---------------------------

    def test_width_481_raises(self):
        img = _make_image(h=512, w=481)
        with pytest.raises(ValueError):
            check_resolution(img, target_width=481, target_height=512)

    def test_width_1_raises(self):
        img = _make_image(h=512, w=1)
        with pytest.raises(ValueError):
            check_resolution(img, target_width=1, target_height=512)

    # -- Height not a multiple of 32 raises ValueError --------------------------

    def test_height_481_raises(self):
        img = _make_image(h=481, w=512)
        with pytest.raises(ValueError):
            check_resolution(img, target_width=512, target_height=481)

    # -- Dimension mismatch raises ValueError -----------------------------------

    def test_width_mismatch_raises(self):
        img = _make_image(h=512, w=512)
        with pytest.raises(ValueError):
            check_resolution(img, target_width=640, target_height=512)

    def test_height_mismatch_raises(self):
        img = _make_image(h=512, w=512)
        with pytest.raises(ValueError):
            check_resolution(img, target_width=512, target_height=640)

    # -- Property: any (w_factor*32, h_factor*32) matching target passes --------

    @given(
        w_factor=st.integers(min_value=1, max_value=32),
        h_factor=st.integers(min_value=1, max_value=24),
    )
    def test_valid_multiples_of_32_pass(self, w_factor, h_factor):
        w = w_factor * 32
        h = h_factor * 32
        img = _make_image(h=h, w=w)
        assert check_resolution(img, target_width=w, target_height=h) is None

    # -- Property: any non-multiple-of-32 raises --------------------------------

    @given(
        base_factor=st.integers(min_value=1, max_value=16),
        offset=st.integers(min_value=1, max_value=31),
    )
    def test_nonmultiple_of_32_width_raises(self, base_factor, offset):
        w = base_factor * 32 + offset
        h = base_factor * 32
        img = _make_image(h=h, w=w)
        with pytest.raises(ValueError):
            check_resolution(img, target_width=w, target_height=h)

    # -- Property: matching multiples-of-32 but differing from target raise -----

    @given(
        a=st.integers(min_value=1, max_value=16),
        b=st.integers(min_value=1, max_value=16),
    )
    def test_dimension_mismatch_raises(self, a, b):
        if a == b:
            return
        w, h = a * 32, a * 32
        tw, th = b * 32, b * 32
        img = _make_image(h=h, w=w)
        with pytest.raises(ValueError):
            check_resolution(img, target_width=tw, target_height=th)


# ---------------------------------------------------------------------------
# sanitize_audio
# ---------------------------------------------------------------------------


class TestSanitizeAudio:
    """sanitize_audio: resample -> trim/pad; warn on mismatch; TypeError on bad input."""

    # -- TypeError on malformed input -------------------------------------------

    def test_non_dict_raises_typeerror(self):
        with pytest.raises(TypeError):
            sanitize_audio("not a dict", target_sample_rate=16000, video_duration_frames=24, fps=24)

    def test_none_raises_typeerror(self):
        with pytest.raises(TypeError):
            sanitize_audio(None, target_sample_rate=16000, video_duration_frames=24, fps=24)

    def test_missing_waveform_key_raises_typeerror(self):
        with pytest.raises(TypeError):
            sanitize_audio(
                {"sample_rate": 16000}, target_sample_rate=16000, video_duration_frames=24, fps=24
            )

    def test_missing_sample_rate_key_raises_typeerror(self):
        with pytest.raises(TypeError):
            sanitize_audio(
                {"waveform": torch.zeros(1, 16000)},
                target_sample_rate=16000,
                video_duration_frames=24,
                fps=24,
            )

    # -- Exact length match at target_sr: no warning, correct output ------------

    def test_exact_match_no_warning(self):
        """Audio at target_sr with exact video-duration samples produces no warning."""
        target_sr = 16000
        fps = 24
        frames = 24  # 1 second at 24 fps -> 16000 samples at 16000 Hz
        expected = _expected_samples(frames, fps, target_sr)
        audio = _make_audio(torch.zeros(1, expected), target_sr)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = sanitize_audio(audio, target_sr, frames, fps)

        assert not any(issubclass(x.category, UserWarning) for x in w)
        assert result["sample_rate"] == target_sr
        assert result["waveform"].shape[-1] == expected

    # -- Longer audio: trim + UserWarning ---------------------------------------

    def test_too_long_warns_and_trims(self):
        target_sr = 16000
        fps = 24
        frames = 24
        expected = _expected_samples(frames, fps, target_sr)
        # 2x the expected duration
        audio = _make_audio(torch.zeros(1, expected * 2), target_sr)

        with pytest.warns(UserWarning):
            result = sanitize_audio(audio, target_sr, frames, fps)

        assert result["sample_rate"] == target_sr
        assert result["waveform"].shape[-1] == expected

    # -- Shorter audio: zero-pad + UserWarning ----------------------------------

    def test_too_short_warns_and_pads(self):
        target_sr = 16000
        fps = 24
        frames = 48  # 2 seconds
        expected = _expected_samples(frames, fps, target_sr)  # 32000
        # Only 25% of expected duration
        short_len = expected // 4
        audio = _make_audio(torch.zeros(1, short_len), target_sr)

        with pytest.warns(UserWarning):
            result = sanitize_audio(audio, target_sr, frames, fps)

        assert result["sample_rate"] == target_sr
        assert result["waveform"].shape[-1] == expected
        # Padded region (after original content) must be silent
        assert (result["waveform"][..., short_len:] == 0.0).all()

    # -- Resample-before-compare: correct duration after resample -> no mismatch warning

    def test_resample_before_compare_no_spurious_warning(self):
        """Input at a different sr whose duration matches after resampling must not warn."""
        target_sr = 16000
        fps = 24
        frames = 24  # 1 second
        expected = _expected_samples(frames, fps, target_sr)  # 16000

        # 8000 Hz input for 1 second = 8000 samples; after resampling to 16000 Hz -> 16000
        input_sr = 8000
        input_samples = _expected_samples(frames, fps, input_sr)  # 8000
        audio = _make_audio(torch.zeros(1, input_samples), input_sr)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = sanitize_audio(audio, target_sr, frames, fps)

        assert not any(issubclass(x.category, UserWarning) for x in w), (
            "Audio matching video duration after resampling must not emit a mismatch warning"
        )
        assert result["sample_rate"] == target_sr
        assert result["waveform"].shape[-1] == expected

    # -- Output sample_rate is always target_sr ---------------------------------

    def test_output_sample_rate_updated(self):
        target_sr = 22050
        fps = 24
        frames = 24
        expected = _expected_samples(frames, fps, target_sr)
        audio = _make_audio(torch.zeros(1, expected), target_sr)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sanitize_audio(audio, target_sr, frames, fps)

        assert result["sample_rate"] == target_sr


# ---------------------------------------------------------------------------
# validate_envelope_indices
# ---------------------------------------------------------------------------


class TestValidateEnvelopeIndices:
    """validate_envelope_indices: None on valid ordering/bounds; ValueError with offending
    values on any violation."""

    # -- Valid cases return None ------------------------------------------------

    def test_basic_valid(self):
        assert validate_envelope_indices(0, 5, 10, 15, 20, 30, 0) is None

    def test_degenerate_still_inject(self):
        # All equal - valid degenerate envelope
        assert validate_envelope_indices(5, 5, 5, 5, 10, 30, 0) is None

    def test_valid_with_inject_at_frame_offset(self):
        # inject_at=2 (FRAME), end_fade_out=5: last row = frame_to_row(2+5)=frame_to_row(7)=2 < 30
        assert validate_envelope_indices(0, 2, 4, 5, 10, 30, 2) is None

    def test_valid_end_fade_out_just_inside_source(self):
        # end_fade_out = source_length - 1 is valid
        assert validate_envelope_indices(0, 1, 2, 9, 10, 30, 0) is None

    # -- Ordering violations raise ValueError with offending values in message --

    def test_start_fade_in_gt_start_kf_raises(self):
        with pytest.raises(ValueError, match=r"\d") as exc_info:
            validate_envelope_indices(10, 5, 8, 15, 20, 30, 0)
        msg = str(exc_info.value)
        assert "10" in msg or "5" in msg

    def test_start_kf_gt_end_kf_raises(self):
        with pytest.raises(ValueError, match=r"\d"):
            validate_envelope_indices(0, 10, 5, 15, 20, 30, 0)

    def test_end_kf_gt_end_fade_out_raises(self):
        with pytest.raises(ValueError, match=r"\d"):
            validate_envelope_indices(0, 5, 15, 10, 20, 30, 0)

    # -- Bounds violations raise ValueError -------------------------------------

    def test_negative_start_fade_in_raises(self):
        with pytest.raises(ValueError):
            validate_envelope_indices(-1, 5, 10, 15, 20, 30, 0)

    def test_negative_start_kf_raises(self):
        with pytest.raises(ValueError):
            validate_envelope_indices(0, -1, 5, 10, 20, 30, 0)

    def test_end_fade_out_equals_source_length_allowed(self):
        # Under the half-open [sfi, efo) model, efo == source_length is valid.
        # FAILS before the guardrail fix (old code raises), PASSES after.
        assert validate_envelope_indices(0, 5, 10, 20, 20, 30, 0) is None

    def test_end_fade_out_gt_source_length_raises(self):
        with pytest.raises(ValueError):
            validate_envelope_indices(0, 5, 10, 25, 20, 30, 0)

    def test_row_span_exceeds_target_rows_raises(self):
        # inject_at=0 (FRAME), end_fade_out=20, target_rows=5.
        # frame_to_row(20) = chunk=1,local=3,local_row=1 → row 6; 6 >= target_rows=5 → raises.
        with pytest.raises(ValueError):
            validate_envelope_indices(0, 5, 10, 20, 25, 5, 0)

    def test_row_span_just_fits_passes(self):
        # inject_at=0, end_fade_out=16, target_rows=5.
        # frame_to_row(0 + 16) = 1 + 15//4 = 4 < 5 → passes.
        assert validate_envelope_indices(0, 5, 10, 16, 20, 5, 0) is None

    # -- Error message includes the values that violated the constraint ---------

    def test_bounds_violation_message_includes_values(self):
        # efo=25 > source_length=20: must still raise under the >= fix.
        with pytest.raises(ValueError) as exc_info:
            validate_envelope_indices(0, 5, 10, 25, 20, 30, 0)
        msg = str(exc_info.value)
        # The offending end_fade_out or source_length should appear
        assert "25" in msg or "20" in msg

    def test_ordering_violation_message_includes_values(self):
        with pytest.raises(ValueError) as exc_info:
            validate_envelope_indices(0, 10, 5, 15, 20, 30, 0)
        msg = str(exc_info.value)
        assert "10" in msg or "5" in msg

    # -- Property: sorted valid tuples always pass ------------------------------

    @given(
        a=st.integers(min_value=0, max_value=10),
        b=st.integers(min_value=0, max_value=10),
        c=st.integers(min_value=0, max_value=10),
        d=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=200)
    def test_sorted_indices_pass(self, a, b, c, d):
        sfi, skf, ekf, efo = sorted([a, b, c, d])
        source_length = efo + 2  # ensures end_fade_out < source_length
        target_rows = 50
        inject_at_row = 0
        assert (
            validate_envelope_indices(sfi, skf, ekf, efo, source_length, target_rows, inject_at_row)
            is None
        )

    # -- Property: strictly reversed start_kf > end_kf always raises -----------

    @given(
        skf=st.integers(min_value=2, max_value=20),
        gap=st.integers(min_value=1, max_value=5),
    )
    def test_start_kf_gt_end_kf_always_raises(self, skf, gap):
        ekf = skf - gap  # ekf < skf: ordering violation
        efo = skf + 5
        source_length = efo + 2
        with pytest.raises(ValueError):
            validate_envelope_indices(0, skf, ekf, efo, source_length, 50, 0)


# ---------------------------------------------------------------------------
# snap_length_down
# ---------------------------------------------------------------------------


class TestSnapLengthDown:
    """snap_length_down(n) → largest 17k+5 <= n; warn on snap; ValueError when n < 5."""

    # -- Snap-down cases: warn and return snapped value -------------------------

    def test_snap_100_to_90(self):
        """100 → 90: (100-5)//17=5, 5*17+5=90."""
        with pytest.warns(UserWarning):
            result = snap_length_down(100)
        assert result == 90

    def test_snap_45_to_39(self):
        """45 → 39: (45-5)//17=2, 2*17+5=39."""
        with pytest.warns(UserWarning):
            result = snap_length_down(45)
        assert result == 39

    def test_snap_20_to_5(self):
        """20 → 5: (20-5)//17=0, 0*17+5=5."""
        with pytest.warns(UserWarning):
            result = snap_length_down(20)
        assert result == 5

    def test_snap_6_to_5(self):
        """6 → 5: the smallest non-trivial snap."""
        with pytest.warns(UserWarning):
            result = snap_length_down(6)
        assert result == 5

    def test_snap_21_to_5(self):
        """21 → 5 (not 22): 22 > 21 so we can't reach 22 from 21 by snapping down."""
        with pytest.warns(UserWarning):
            result = snap_length_down(21)
        assert result == 5

    def test_warning_message_mentions_original_and_snapped(self):
        """Warning message should include both the original length and the snapped length."""
        with pytest.warns(UserWarning) as record:
            snap_length_down(100)
        msg = str(record[0].message)
        assert "100" in msg
        assert "90" in msg

    # -- No-op when already valid (17n+5): no warning --------------------------

    def test_no_op_5(self):
        """5 = 5+17*0: minimum valid length; no warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = snap_length_down(5)
        assert result == 5
        snap_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert not snap_warns

    def test_no_op_22(self):
        """22 = 5+17*1: valid; no warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = snap_length_down(22)
        assert result == 22
        assert not [x for x in w if issubclass(x.category, UserWarning)]

    def test_no_op_39(self):
        """39 = 5+17*2: valid; no warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = snap_length_down(39)
        assert result == 39
        assert not [x for x in w if issubclass(x.category, UserWarning)]

    def test_no_op_90(self):
        """90 = 5+17*5: valid; no warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = snap_length_down(90)
        assert result == 90
        assert not [x for x in w if issubclass(x.category, UserWarning)]

    def test_no_op_107(self):
        """107 = 5+17*6: valid; no warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = snap_length_down(107)
        assert result == 107
        assert not [x for x in w if issubclass(x.category, UserWarning)]

    # -- F=1 special case: single frame returns 1 with no warning ---------------

    def test_single_frame_returns_1_no_warning(self):
        """source_length=1 → return 1 unchanged; no UserWarning emitted.

        Valid set is {1} ∪ {17n+5}; the H3 VAE single-frame path produces exactly
        1 latent row, so length 1 is a first-class valid inject length.
        FAIL-THEN-PASS: Before this change, snap_length_down(1) raises ValueError
        (caught by the old 'source_length < 5' guard); after it returns 1.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = snap_length_down(1)
        assert result == 1
        snap_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert not snap_warns, "No warning expected for single-frame (keyframe) length"

    # -- Lengths 2, 3, 4 are explicitly invalid --------------------------------

    def test_error_2_raises(self):
        """source_length=2 → ValueError: not 1 frame and not >= 5 (17n+5).

        FAIL-THEN-PASS: Before this change the error message doesn't mention '1 frame';
        the test matches the new guidance message.
        """
        with pytest.raises(ValueError, match="1 frame"):
            snap_length_down(2)

    def test_error_3_raises(self):
        """source_length=3 → ValueError with guidance about valid lengths."""
        with pytest.raises(ValueError, match="1 frame"):
            snap_length_down(3)

    def test_error_4_raises_with_guidance(self):
        """source_length=4 → ValueError with guidance about valid lengths.

        FAIL-THEN-PASS: The new error message mentions '1 frame'; the old message does not.
        """
        with pytest.raises(ValueError, match="1 frame"):
            snap_length_down(4)

    # -- ValueError when source_length < 1 -------------------------------------

    def test_error_4(self):
        """4 is now in the 'invalid 2-4' set → still ValueError."""
        with pytest.raises(ValueError):
            snap_length_down(4)

    def test_error_0(self):
        """0 < 1 → ValueError."""
        with pytest.raises(ValueError):
            snap_length_down(0)

    def test_error_negative(self):
        """-1 < 1 → ValueError."""
        with pytest.raises(ValueError):
            snap_length_down(-1)

    def test_error_message_mentions_5(self):
        """Error message for lengths 2-4 should mention the minimum valid clip length (5)."""
        with pytest.raises(ValueError, match="5"):
            snap_length_down(3)

    # -- Properties: snapped result is always a valid 17n+5 length -------------

    @given(n=st.integers(min_value=5, max_value=10_000))
    def test_result_is_valid_17n5(self, n):
        """snap_length_down(n) always produces a valid 17k+5 length."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = snap_length_down(n)
        assert result >= 5
        assert (result - 5) % 17 == 0

    @given(n=st.integers(min_value=5, max_value=10_000))
    def test_result_le_n(self, n):
        """Snapped result is always <= n."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = snap_length_down(n)
        assert result <= n

    @given(n=st.integers(min_value=5, max_value=10_000))
    def test_gap_lt_17(self, n):
        """Gap between n and result is always < 17."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = snap_length_down(n)
        assert n - result < 17

    @given(k=st.integers(min_value=0, max_value=588))
    def test_valid_lengths_are_no_op(self, k):
        """Already-valid lengths 17k+5 are returned unchanged without a warning."""
        n = 5 + 17 * k
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = snap_length_down(n)
        assert result == n
        assert not [x for x in w if issubclass(x.category, UserWarning)]


# ---------------------------------------------------------------------------
# warn_audio_tail_alignment
# ---------------------------------------------------------------------------


class TestWarnAudioTailAlignment:
    """warn_audio_tail_alignment emits UserWarning on exposed non-audio-aligned tail."""

    # -- Warn cases ------------------------------------------------------------

    def test_warn_keep_mode_non_aligned(self):
        """keep-mode + non-aligned length (56 = 5+17*3) → UserWarning.

        ceil(56/17)=4, 4%3=1 → not audio-aligned.
        FAIL-THEN-PASS: Without the emit logic, no warning is produced.
        """
        with pytest.warns(UserWarning, match="audio-sync-aligned"):
            warn_audio_tail_alignment(
                snapped_length=56,
                audio_mode="keep",
                end_keyframes=55,
                end_fade_out=55,
                has_audio=True,
            )

    def test_warn_fade_unfaded_end_non_aligned(self):
        """fade-mode + ramp doesn't reach tail → UserWarning.

        end_fade_out=55 < snapped_length=56 → not faded through.
        """
        with pytest.warns(UserWarning):
            warn_audio_tail_alignment(
                snapped_length=56,
                audio_mode="fade",
                end_keyframes=50,
                end_fade_out=55,  # < 56: ramp doesn't reach clip tail
                has_audio=True,
            )

    def test_warn_fade_no_ramp_non_aligned(self):
        """fade-mode + no fade-out ramp (ekf == efo) + non-aligned → UserWarning."""
        with pytest.warns(UserWarning):
            warn_audio_tail_alignment(
                snapped_length=56,
                audio_mode="fade",
                end_keyframes=55,
                end_fade_out=55,  # no ramp: efo == ekf
                has_audio=True,
            )

    def test_warn_22_keep_mode(self):
        """Length 22 (non-aligned: ceil(22/17)=2, 2%3=2) + keep + audio → warns."""
        with pytest.warns(UserWarning):
            warn_audio_tail_alignment(
                snapped_length=22,
                audio_mode="keep",
                end_keyframes=21,
                end_fade_out=21,
                has_audio=True,
            )

    def test_warn_keep_mode_with_video_fadeout_reaching_tail(self):
        """Regression: keep-mode + video fade-out reaching tail → MUST warn.

        Bug (pre-fix): is_faded_through checked ALL modes; end_fade_out=56 ==
        snapped_length=56 AND end_fade_out > end_keyframes → suppressed warning
        even in keep mode.  Fix: gate is_faded_through on audio_mode == "fade".

        FAIL-THEN-PASS: Against the unfixed code this test fails with
        "Failed: DID NOT WARN".  After the one-line fix it passes.
        """
        with pytest.warns(UserWarning, match="audio-sync-aligned"):
            warn_audio_tail_alignment(
                snapped_length=56,
                audio_mode="keep",
                end_keyframes=50,
                end_fade_out=56,  # video fade-out reaches tail — must NOT suppress keep-mode
                has_audio=True,
            )

    # -- Warning message content -----------------------------------------------

    def test_warn_message_mentions_nearest_aligned_lengths(self):
        """Warning message includes the nearest audio-aligned lengths."""
        with pytest.warns(UserWarning) as record:
            warn_audio_tail_alignment(
                snapped_length=56,  # nearest below=39, nearest above=90
                audio_mode="keep",
                end_keyframes=55,
                end_fade_out=55,
                has_audio=True,
            )
        msg = str(record[0].message)
        assert "39" in msg and "90" in msg

    def test_warn_message_mentions_tail_error_ms(self):
        """Warning message quantifies the tail error in ms."""
        with pytest.warns(UserWarning) as record:
            warn_audio_tail_alignment(
                snapped_length=56,
                audio_mode="keep",
                end_keyframes=55,
                end_fade_out=55,
                has_audio=True,
            )
        msg = str(record[0].message)
        assert "ms" in msg

    # -- No-warn cases ---------------------------------------------------------

    def test_no_warn_audio_aligned_39(self):
        """Length 39 (audio-aligned: ceil(39/17)=3, 3%3=0) → no warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_audio_tail_alignment(
                snapped_length=39,
                audio_mode="keep",
                end_keyframes=38,
                end_fade_out=38,
                has_audio=True,
            )
        tail_warns = [x for x in w if "audio-sync-aligned" in str(x.message)]
        assert not tail_warns

    def test_no_warn_audio_aligned_90(self):
        """Length 90 (audio-aligned: ceil(90/17)=6, 6%3=0) → no warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_audio_tail_alignment(
                snapped_length=90,
                audio_mode="keep",
                end_keyframes=89,
                end_fade_out=89,
                has_audio=True,
            )
        assert not [x for x in w if "audio-sync-aligned" in str(x.message)]

    def test_no_warn_drop_mode(self):
        """drop-mode → no warning even if non-aligned and has_audio=True."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_audio_tail_alignment(
                snapped_length=56,
                audio_mode="drop",
                end_keyframes=55,
                end_fade_out=55,
                has_audio=True,
            )
        assert not [x for x in w if issubclass(x.category, UserWarning)]

    def test_no_warn_no_audio(self):
        """has_audio=False → no warning (nothing to misalign)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_audio_tail_alignment(
                snapped_length=56,
                audio_mode="keep",
                end_keyframes=55,
                end_fade_out=55,
                has_audio=False,
            )
        assert not [x for x in w if issubclass(x.category, UserWarning)]

    def test_no_warn_faded_through(self):
        """fade-mode with ramp reaching clip tail → no warning (masked)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_audio_tail_alignment(
                snapped_length=56,
                audio_mode="fade",
                end_keyframes=50,
                end_fade_out=56,  # == snapped_length: faded through → no warn
                has_audio=True,
            )
        assert not [x for x in w if issubclass(x.category, UserWarning)]

    # -- Properties ------------------------------------------------------------

    @given(k=st.integers(min_value=0, max_value=50))
    def test_audio_aligned_lengths_never_warn(self, k):
        """51k+39 lengths always suppress the warning (condition 1 fails)."""
        n = 39 + 51 * k
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_audio_tail_alignment(
                snapped_length=n,
                audio_mode="keep",
                end_keyframes=n - 1,
                end_fade_out=n - 1,
                has_audio=True,
            )
        assert not [x for x in w if issubclass(x.category, UserWarning)]


# ---------------------------------------------------------------------------
# Regression: sanitize._FPS / _AUDIO_HZ track constants, not hardcoded literals
# ---------------------------------------------------------------------------


class TestSanitizeConstantsTrackModule:
    """Verify that sanitize uses FPS/AUDIO_HZ from constants, not redeclared literals.

    These tests would fail if someone re-hardcoded drifted values in sanitize.py.
    """

    def test_fps_tracks_constants(self):
        import comfyui_h3_blended_inject.sanitize as _sanitize
        from comfyui_h3_blended_inject import grid

        assert _sanitize._FPS is grid.FPS or _sanitize._FPS == grid.FPS, (
            f"sanitize._FPS={_sanitize._FPS!r} does not match grid.FPS={grid.FPS!r}. "
            "Import FPS from grid instead of redeclaring."
        )

    def test_audio_hz_tracks_constants(self):
        import comfyui_h3_blended_inject.sanitize as _sanitize
        from comfyui_h3_blended_inject import grid

        assert _sanitize._AUDIO_HZ == grid.AUDIO_HZ, (
            f"sanitize._AUDIO_HZ={_sanitize._AUDIO_HZ!r} does not match "
            f"grid.AUDIO_HZ={grid.AUDIO_HZ!r}. "
            "Import AUDIO_HZ from grid instead of redeclaring."
        )

    def test_fps_value_unchanged(self):
        """Catch drift: if grid.FPS changes, tests should surface it explicitly."""
        from comfyui_h3_blended_inject import grid

        assert grid.FPS == 24, (
            f"grid.FPS changed to {grid.FPS!r}; update sanitize tests if intentional."
        )

    def test_audio_hz_value_unchanged(self):
        """Catch drift: if grid.AUDIO_HZ changes, tests should surface it explicitly."""
        from comfyui_h3_blended_inject import grid

        assert grid.AUDIO_HZ == 40.0, (
            f"grid.AUDIO_HZ changed to {grid.AUDIO_HZ!r}; update sanitize tests if intentional."
        )
