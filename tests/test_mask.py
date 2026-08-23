"""Tests for comfyui_h3_blended_inject.mask.

Contract under test (from mask.py docstrings):

derive_mask(schedule, video_rows, audio_ticks):
  - Returns {"video_mask": float32[1, video_rows], "audio_mask": float32[1, audio_ticks]}
  - Values in {0.0, 1.0} only
  - video_mask is 0.0 EXACTLY where denoise==0.0; 1.0 everywhere else
    (fractional-denoise rows and absent rows both get 1.0)
  - audio_mask is 0.0 EXACTLY on ticks whose corresponding row has audio_frozen==True; 1.0 elsewhere

apply_derived_mask(latent, schedule, video_rows, audio_ticks):
  - Returns a shallow copy of latent with "noise_mask" set to the derived mask
  - Original latent dict is not mutated
  - Emits UserWarning if latent already has "noise_mask"; replaces it
  - Emits no warning if latent has no "noise_mask"

All tests are written against the final implemented behavior. They currently fail because
derive_mask and apply_derived_mask both raise NotImplementedError.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from comfyui_h3_blended_inject.constants import audio_tick_range, video_row_to_audio_tick
from comfyui_h3_blended_inject.mask import apply_derived_mask, derive_mask
from comfyui_h3_blended_inject.schedule import Inject, RowSchedule

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def row(
    row_idx: int,
    denoise: float = 1.0,
    audio_frozen: bool = False,
) -> RowSchedule:
    """Build a minimal RowSchedule with no inject content."""
    return RowSchedule(row_idx=row_idx, denoise=denoise, inject=None, audio_frozen=audio_frozen)


def latent_dict(has_noise_mask: bool = False) -> dict:
    """Build a minimal ComfyUI latent dict."""
    d: dict = {"samples": torch.zeros(1, 4, 10, 10)}
    if has_noise_mask:
        d["noise_mask"] = {
            "video_mask": torch.ones(1, 5),
            "audio_mask": torch.ones(1, 8),
        }
    return d


# ---------------------------------------------------------------------------
# derive_mask: return shape and dtype
# ---------------------------------------------------------------------------


class TestDeriveMaskShape:
    """derive_mask output dict must have correct keys, shapes, and dtypes."""

    def test_returns_dict_with_video_mask_key(self):
        result = derive_mask([row(0, denoise=0.0)], video_rows=5, audio_ticks=8)
        assert "video_mask" in result

    def test_returns_dict_with_audio_mask_key(self):
        result = derive_mask([row(0, denoise=0.0)], video_rows=5, audio_ticks=8)
        assert "audio_mask" in result

    def test_video_mask_shape(self):
        result = derive_mask([row(0, denoise=0.0)], video_rows=7, audio_ticks=8)
        assert result["video_mask"].shape == (1, 7)

    def test_audio_mask_shape(self):
        result = derive_mask([row(0, denoise=0.0)], video_rows=5, audio_ticks=12)
        assert result["audio_mask"].shape == (1, 12)

    def test_video_mask_dtype_float32(self):
        result = derive_mask([row(0, denoise=0.0)], video_rows=5, audio_ticks=8)
        assert result["video_mask"].dtype == torch.float32

    def test_audio_mask_dtype_float32(self):
        result = derive_mask([row(0, denoise=0.0)], video_rows=5, audio_ticks=8)
        assert result["audio_mask"].dtype == torch.float32


# ---------------------------------------------------------------------------
# derive_mask: video_mask value semantics
# ---------------------------------------------------------------------------


class TestDeriveMaskVideoValues:
    """video_mask: 0 on denoise==0 rows, 1 everywhere else."""

    def test_only_binary_values(self):
        schedule = [
            row(0, denoise=0.0),
            row(1, denoise=0.5),
            row(2, denoise=1.0),
        ]
        mask = derive_mask(schedule, video_rows=5, audio_ticks=8)["video_mask"]
        assert torch.all((mask == 0.0) | (mask == 1.0))

    def test_zero_on_denoise_zero_row(self):
        result = derive_mask([row(2, denoise=0.0)], video_rows=5, audio_ticks=8)
        assert result["video_mask"][0, 2].item() == 0.0

    def test_one_on_fractional_denoise_row(self):
        # Fractional denoise is handled by hold-and-release, not the mask
        result = derive_mask([row(1, denoise=0.5)], video_rows=5, audio_ticks=8)
        assert result["video_mask"][0, 1].item() == 1.0

    def test_one_on_full_denoise_row(self):
        result = derive_mask([row(3, denoise=1.0)], video_rows=5, audio_ticks=8)
        assert result["video_mask"][0, 3].item() == 1.0

    def test_one_on_absent_rows(self):
        # Rows not present in schedule default to generate (d=1)
        schedule = [row(0, denoise=0.0)]
        result = derive_mask(schedule, video_rows=5, audio_ticks=8)
        for absent_row in range(1, 5):
            assert result["video_mask"][0, absent_row].item() == 1.0

    def test_empty_schedule_all_ones(self):
        result = derive_mask([], video_rows=5, audio_ticks=8)
        assert torch.all(result["video_mask"] == 1.0)

    def test_mixed_schedule_zero_set_exact(self):
        # Mix: denoise=0 (x2), fractional, full, absent row
        schedule = [
            row(0, denoise=0.0),  # preserve
            row(1, denoise=0.3),  # hold-and-release -> 1
            row(2, denoise=0.0),  # preserve
            row(3, denoise=1.0),  # generate -> 1
            # row 4 absent -> 1
        ]
        video_mask = derive_mask(schedule, video_rows=5, audio_ticks=8)["video_mask"][0]
        assert video_mask[0].item() == 0.0
        assert video_mask[1].item() == 1.0
        assert video_mask[2].item() == 0.0
        assert video_mask[3].item() == 1.0
        assert video_mask[4].item() == 1.0

    def test_very_small_fractional_denoise_is_one(self):
        # Even nearly-zero (but not exactly zero) denoise is 1 in the mask
        result = derive_mask([row(0, denoise=1e-9)], video_rows=5, audio_ticks=8)
        assert result["video_mask"][0, 0].item() == 1.0

    def test_all_rows_denoise_zero(self):
        # When every row has denoise==0, every position should be 0
        schedule = [row(i, denoise=0.0) for i in range(5)]
        result = derive_mask(schedule, video_rows=5, audio_ticks=8)
        assert torch.all(result["video_mask"] == 0.0)


# ---------------------------------------------------------------------------
# derive_mask: audio_mask value semantics
# ---------------------------------------------------------------------------


class TestDeriveMaskAudioValues:
    """audio_mask: 0 on ticks from audio_frozen rows, 1 elsewhere."""

    def test_only_binary_values(self):
        schedule = [row(0, denoise=1.0, audio_frozen=True)]
        mask = derive_mask(schedule, video_rows=5, audio_ticks=8)["audio_mask"]
        assert torch.all((mask == 0.0) | (mask == 1.0))

    def test_zero_on_frozen_audio_tick(self):
        # The tick corresponding to a frozen row must be 0
        r_idx = 0
        schedule = [row(r_idx, denoise=1.0, audio_frozen=True)]
        result = derive_mask(schedule, video_rows=5, audio_ticks=8)
        tick = video_row_to_audio_tick(r_idx)
        assert result["audio_mask"][0, tick].item() == 0.0

    def test_one_on_non_frozen_tick(self):
        # audio_frozen=False: tick is generate (1)
        r_idx = 0
        schedule = [row(r_idx, denoise=0.0, audio_frozen=False)]
        result = derive_mask(schedule, video_rows=5, audio_ticks=8)
        tick = video_row_to_audio_tick(r_idx)
        assert result["audio_mask"][0, tick].item() == 1.0

    def test_empty_schedule_all_ones(self):
        result = derive_mask([], video_rows=5, audio_ticks=8)
        assert torch.all(result["audio_mask"] == 1.0)

    def test_frozen_flag_independent_of_denoise(self):
        # audio_frozen applies even when denoise is fractional (not a d=0 row)
        # The audio tick still gets 0 when audio_frozen=True
        r_idx = 0
        schedule = [row(r_idx, denoise=0.7, audio_frozen=True)]
        result = derive_mask(schedule, video_rows=5, audio_ticks=8)
        tick = video_row_to_audio_tick(r_idx)
        assert result["audio_mask"][0, tick].item() == 0.0

    def test_frozen_false_on_absent_row_ticks_remain_one(self):
        # Rows absent from schedule are not frozen; their ticks stay 1
        # Only row 0 in schedule with audio_frozen=False; row 1 absent
        schedule = [row(0, denoise=0.0, audio_frozen=False)]
        result = derive_mask(schedule, video_rows=5, audio_ticks=16)
        tick_1 = video_row_to_audio_tick(1)
        # tick_1 has no schedule entry, so it must be 1
        assert result["audio_mask"][0, tick_1].item() == 1.0


# ---------------------------------------------------------------------------
# Hypothesis: property tests
# ---------------------------------------------------------------------------


@st.composite
def row_schedule_list(draw, max_row_idx: int = 19):
    """Generate a list of RowSchedule with unique row_idx values."""
    available = list(range(max_row_idx + 1))
    n = draw(st.integers(min_value=0, max_value=len(available)))
    chosen = draw(
        st.lists(
            st.integers(min_value=0, max_value=max_row_idx),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    rows = []
    for idx in chosen:
        denoise = draw(
            st.one_of(
                st.just(0.0),
                st.floats(min_value=1e-6, max_value=1.0 - 1e-6, allow_nan=False),
                st.just(1.0),
            )
        )
        audio_frozen = draw(st.booleans())
        rows.append(row(idx, denoise=denoise, audio_frozen=audio_frozen))
    return rows


@given(schedule=row_schedule_list(max_row_idx=19))
@settings(max_examples=60)
def test_property_video_mask_zero_set_equals_zero_denoise_rows(schedule):
    """video_mask zero positions == exactly {r.row_idx for r in schedule if r.denoise == 0.0}."""
    video_rows = 20
    audio_ticks = 32
    result = derive_mask(schedule, video_rows=video_rows, audio_ticks=audio_ticks)
    video_mask = result["video_mask"][0]

    expected_zeros = {r.row_idx for r in schedule if r.denoise == 0.0}
    actual_zeros = {i for i in range(video_rows) if video_mask[i].item() == 0.0}
    assert actual_zeros == expected_zeros


@given(schedule=row_schedule_list(max_row_idx=19))
@settings(max_examples=60)
def test_property_video_mask_all_nonzero_positions_are_one(schedule):
    """Every non-preserved position in video_mask is exactly 1.0 (no fractional values)."""
    video_rows = 20
    audio_ticks = 32
    result = derive_mask(schedule, video_rows=video_rows, audio_ticks=audio_ticks)
    video_mask = result["video_mask"][0]

    one_positions = video_mask[video_mask != 0.0]
    assert torch.all(one_positions == 1.0)


@given(schedule=row_schedule_list(max_row_idx=9))
@settings(max_examples=60)
def test_property_audio_mask_zero_set_equals_frozen_ticks(schedule):
    """audio_mask zero positions == full canonical tick range for audio_frozen rows."""
    video_rows = 10
    audio_ticks = 64
    result = derive_mask(schedule, video_rows=video_rows, audio_ticks=audio_ticks)
    audio_mask = result["audio_mask"][0]

    expected_zero_ticks: set[int] = set()
    for r in schedule:
        if r.audio_frozen:
            start_tick = video_row_to_audio_tick(r.row_idx)
            if r.row_idx < video_rows - 1:
                end_tick = min(video_row_to_audio_tick(r.row_idx + 1), audio_ticks)
            else:
                end_tick = audio_ticks
            expected_zero_ticks.update(range(max(0, start_tick), end_tick))
    actual_zero_ticks = {i for i in range(audio_ticks) if audio_mask[i].item() == 0.0}
    assert actual_zero_ticks == expected_zero_ticks


@given(schedule=row_schedule_list(max_row_idx=19))
@settings(max_examples=60)
def test_property_audio_mask_all_nonzero_positions_are_one(schedule):
    """Every non-frozen tick in audio_mask is exactly 1.0 (no fractional values)."""
    video_rows = 20
    audio_ticks = 32
    result = derive_mask(schedule, video_rows=video_rows, audio_ticks=audio_ticks)
    audio_mask = result["audio_mask"][0]

    one_positions = audio_mask[audio_mask != 0.0]
    assert torch.all(one_positions == 1.0)


# ---------------------------------------------------------------------------
# apply_derived_mask: return value and shallow copy semantics
# ---------------------------------------------------------------------------


class TestApplyDerivedMaskReturnValue:
    """apply_derived_mask must return a shallow copy with noise_mask set."""

    def test_returns_dict(self):
        result = apply_derived_mask(latent_dict(), [], video_rows=5, audio_ticks=8)
        assert isinstance(result, dict)

    def test_result_has_noise_mask_key(self):
        result = apply_derived_mask(latent_dict(), [], video_rows=5, audio_ticks=8)
        assert "noise_mask" in result

    def test_noise_mask_contains_video_mask(self):
        schedule = [row(0, denoise=0.0)]
        result = apply_derived_mask(latent_dict(), schedule, video_rows=5, audio_ticks=8)
        assert "video_mask" in result["noise_mask"]

    def test_noise_mask_contains_audio_mask(self):
        schedule = [row(0, denoise=0.0)]
        result = apply_derived_mask(latent_dict(), schedule, video_rows=5, audio_ticks=8)
        assert "audio_mask" in result["noise_mask"]

    def test_original_latent_not_mutated_no_noise_mask_key_added(self):
        latent = latent_dict(has_noise_mask=False)
        apply_derived_mask(latent, [], video_rows=5, audio_ticks=8)
        assert "noise_mask" not in latent

    def test_original_latent_samples_not_replaced(self):
        latent = latent_dict()
        original_id = id(latent["samples"])
        apply_derived_mask(latent, [], video_rows=5, audio_ticks=8)
        assert id(latent["samples"]) == original_id

    def test_shallow_copy_shares_samples_tensor(self):
        # The returned dict is a shallow copy: samples tensor object is identical
        latent = latent_dict()
        result = apply_derived_mask(latent, [], video_rows=5, audio_ticks=8)
        assert result["samples"] is latent["samples"]

    def test_result_is_not_original_dict(self):
        latent = latent_dict()
        result = apply_derived_mask(latent, [], video_rows=5, audio_ticks=8)
        assert result is not latent


# ---------------------------------------------------------------------------
# apply_derived_mask: warning behavior
# ---------------------------------------------------------------------------


class TestApplyDerivedMaskWarnings:
    """apply_derived_mask warns iff the input latent already has a noise_mask."""

    def test_warns_userwarning_when_existing_noise_mask(self):
        latent = latent_dict(has_noise_mask=True)
        with pytest.warns(UserWarning):
            apply_derived_mask(latent, [], video_rows=5, audio_ticks=8)

    def test_warning_message_mentions_noise_mask(self):
        latent = latent_dict(has_noise_mask=True)
        with pytest.warns(UserWarning, match="noise_mask"):
            apply_derived_mask(latent, [], video_rows=5, audio_ticks=8)

    def test_warning_message_mentions_replacement(self):
        latent = latent_dict(has_noise_mask=True)
        with pytest.warns(UserWarning, match="replaced"):
            apply_derived_mask(latent, [], video_rows=5, audio_ticks=8)

    def test_no_warning_without_existing_noise_mask(self):
        latent = latent_dict(has_noise_mask=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            apply_derived_mask(latent, [], video_rows=5, audio_ticks=8)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_existing_noise_mask_is_replaced_not_preserved(self):
        # The original foreign noise_mask must not appear in the result
        foreign_mask = {"video_mask": torch.ones(1, 5), "audio_mask": torch.ones(1, 8)}
        latent = {"samples": torch.zeros(1, 4, 10, 10), "noise_mask": foreign_mask}
        with pytest.warns(UserWarning):
            result = apply_derived_mask(latent, [], video_rows=5, audio_ticks=8)
        assert result["noise_mask"] is not foreign_mask


# ---------------------------------------------------------------------------
# Nested NestedTensor mask path (new in NestedTensor fix)
# ---------------------------------------------------------------------------
#
# When video_component_shape and audio_component_shape are provided to derive_mask,
# the mask must be a NestedTensor (via factory) with FULL component shapes —
# video [B,C_v,T,Hl,Wl] and audio [B,C_a,2,audio_t] — with 0/1 values expanded
# across all channels and spatial dims.  Tests use a fake factory that returns a
# simple namespace so comfy is not imported.


class FakeNestedTensor:
    """Minimal stand-in for comfy.nested_tensor.NestedTensor."""

    def __init__(self, video_mask: torch.Tensor, audio_mask: torch.Tensor) -> None:
        self.tensors = [video_mask, audio_mask]


def _fake_nested_factory(video_mask: torch.Tensor, audio_mask: torch.Tensor) -> FakeNestedTensor:
    return FakeNestedTensor(video_mask, audio_mask)


# Shared component shapes for the nested-mask tests.
# Video: B=1, C=24, T=5, Hl=4, Wl=4  —  audio: B=1, C=32, 2, audio_t=7
_V_SHAPE = (1, 24, 5, 4, 4)
_A_SHAPE = (1, 32, 2, 7)
_VIDEO_ROWS = _V_SHAPE[2]  # 5
_AUDIO_TICKS = _A_SHAPE[3]  # 7


class TestDeriveMaskNestedPath:
    """derive_mask: with component shapes + factory → NestedTensor with full shapes."""

    def _run(self, schedule=None):
        if schedule is None:
            schedule = []
        return derive_mask(
            schedule,
            video_rows=_VIDEO_ROWS,
            audio_ticks=_AUDIO_TICKS,
            video_component_shape=_V_SHAPE,
            audio_component_shape=_A_SHAPE,
            nested_factory=_fake_nested_factory,
        )

    def test_returns_nested_tensor_not_dict(self):
        """With component shapes, result must not be a plain dict."""
        result = self._run()
        assert not isinstance(result, dict), (
            "derive_mask must return a NestedTensor (not a dict) when component shapes given"
        )

    def test_video_mask_full_shape(self):
        """Video mask must have the full component shape [B,C,T,Hl,Wl]."""
        result = self._run()
        assert result.tensors[0].shape == torch.Size(_V_SHAPE)

    def test_audio_mask_full_shape(self):
        """Audio mask must have the full component shape [B,C,2,audio_t]."""
        result = self._run()
        assert result.tensors[1].shape == torch.Size(_A_SHAPE)

    def test_video_mask_all_ones_empty_schedule(self):
        """Empty schedule → all video mask entries are 1 (generate)."""
        result = self._run(schedule=[])
        assert torch.all(result.tensors[0] == 1.0)

    def test_audio_mask_all_ones_empty_schedule(self):
        """Empty schedule → all audio mask entries are 1 (generate)."""
        result = self._run(schedule=[])
        assert torch.all(result.tensors[1] == 1.0)

    def test_video_mask_zero_row_expanded_across_channels_and_spatial(self):
        """A d=0 row → ALL channels and spatial positions at that T slice are 0."""
        target_row = 2  # within the 5-row range
        schedule = [row(target_row, denoise=0.0)]
        result = self._run(schedule)
        # tensors[0] shape: [1, 24, 5, 4, 4]; T dim is dim 2
        t_slice = result.tensors[0][:, :, target_row, :, :]  # [1, 24, 4, 4]
        assert torch.all(t_slice == 0.0), (
            f"Row {target_row} d=0: all channels/spatial at T={target_row} must be 0"
        )

    def test_video_mask_other_rows_are_one(self):
        """Non-zero-denoise rows must have 1 in all video mask positions."""
        schedule = [row(2, denoise=0.0)]  # only row 2 is zeroed
        result = self._run(schedule)
        vm = result.tensors[0]
        for t in range(_VIDEO_ROWS):
            if t != 2:
                assert torch.all(vm[:, :, t, :, :] == 1.0), f"Row {t} should be 1 in video mask"

    def test_audio_mask_frozen_tick_expanded_across_channels(self):
        """A frozen-audio row → ALL channels at that audio tick are 0."""
        target_row = 0  # maps to audio tick 0
        schedule = [row(target_row, denoise=1.0, audio_frozen=True)]
        from comfyui_h3_blended_inject.constants import video_row_to_audio_tick

        tick = video_row_to_audio_tick(target_row)
        result = self._run(schedule)
        # tensors[1] shape: [1, 32, 2, 7]; audio_t dim is dim 3
        if tick < _AUDIO_TICKS:
            a_slice = result.tensors[1][:, :, :, tick]  # [1, 32, 2]
            assert torch.all(a_slice == 0.0), (
                f"Frozen tick {tick}: all channels at audio_t={tick} must be 0"
            )

    def test_video_mask_binary_values_only(self):
        """All video mask values must be exactly 0.0 or 1.0 (no fractional)."""
        schedule = [row(i, denoise=d) for i, d in enumerate([0.0, 0.5, 1.0, 0.0, 0.3])]
        result = self._run(schedule)
        vm = result.tensors[0]
        assert torch.all((vm == 0.0) | (vm == 1.0))

    def test_audio_mask_binary_values_only(self):
        """All audio mask values must be exactly 0.0 or 1.0."""
        schedule = [row(0, denoise=1.0, audio_frozen=True)]
        result = self._run(schedule)
        am = result.tensors[1]
        assert torch.all((am == 0.0) | (am == 1.0))


class TestApplyDerivedMaskNestedPath:
    """apply_derived_mask: warning still fires; nested mask stored under noise_mask."""

    def test_warning_still_fires_with_component_shapes(self):
        """Foreign noise_mask warning still fires on the nested path."""
        foreign_mask = {"video_mask": torch.ones(1, 5), "audio_mask": torch.ones(1, 8)}
        latent = {"samples": torch.zeros(1, 4, 10, 10), "noise_mask": foreign_mask}
        with pytest.warns(UserWarning, match="noise_mask"):
            apply_derived_mask(
                latent,
                [],
                video_rows=_VIDEO_ROWS,
                audio_ticks=_AUDIO_TICKS,
                video_component_shape=_V_SHAPE,
                audio_component_shape=_A_SHAPE,
                nested_factory=_fake_nested_factory,
            )

    def test_noise_mask_is_fake_nested_tensor_with_component_shapes(self):
        """noise_mask must be a FakeNestedTensor when component shapes are provided."""
        latent = {"samples": torch.zeros(1, 4, 10, 10)}
        result = apply_derived_mask(
            latent,
            [],
            video_rows=_VIDEO_ROWS,
            audio_ticks=_AUDIO_TICKS,
            video_component_shape=_V_SHAPE,
            audio_component_shape=_A_SHAPE,
            nested_factory=_fake_nested_factory,
        )
        assert isinstance(result["noise_mask"], FakeNestedTensor)

    def test_no_component_shapes_returns_dict_as_before(self):
        """Without component shapes, noise_mask is the old dict form (backward compat)."""
        latent = {"samples": torch.zeros(1, 4, 10, 10)}
        result = apply_derived_mask(latent, [], video_rows=5, audio_ticks=8)
        assert isinstance(result["noise_mask"], dict)


class TestDeriveMaskAudioFullTickRange:
    """derive_mask: frozen row zeros ALL ticks in its canonical range, not just the start tick.

    Row 1 with video_rows=5, audio_ticks=7:
      video_row_to_audio_tick(1) = 2, video_row_to_audio_tick(2) = 8 -> clamped to 7.
      Full range = ticks 2, 3, 4, 5, 6 (5 ticks).
    Old implementation only zeroed tick 2.  These tests FAIL before the fix.
    """

    def test_dict_path_frozen_mid_row_zeros_full_range(self) -> None:
        """Dict path: ALL ticks in row 1's range must be 0 after fix.

        Fails before fix: only tick 2 is zeroed; ticks 3-6 remain 1.
        """
        schedule = [RowSchedule(row_idx=1, denoise=1.0, inject=None, audio_frozen=True)]
        result = derive_mask(schedule, video_rows=5, audio_ticks=7)
        audio_mask = result["audio_mask"][0]
        zero_count = int((audio_mask == 0.0).sum().item())
        assert zero_count > 1, (
            f"Row 1 owns 5 audio ticks; expected > 1 zero in dict path, got {zero_count}"
        )
        for tick in range(2, 7):
            assert audio_mask[tick].item() == 0.0, (
                f"Dict path: tick {tick} should be 0 (row 1 owns range [2, 7)), "
                f"got {audio_mask[tick].item()}"
            )

    def test_nested_path_frozen_mid_row_zeros_full_range(self) -> None:
        """Nested path: ALL ticks in row 1's range [2, 7) must be 0 across all channels.

        Fails before fix: only tick 2 is zeroed.
        """
        schedule = [RowSchedule(row_idx=1, denoise=1.0, inject=None, audio_frozen=True)]
        result = derive_mask(
            schedule,
            video_rows=_VIDEO_ROWS,
            audio_ticks=_AUDIO_TICKS,
            video_component_shape=_V_SHAPE,
            audio_component_shape=_A_SHAPE,
            nested_factory=_fake_nested_factory,
        )
        am = result.tensors[1]  # [1, 32, 2, 7]
        for tick in range(2, 7):
            assert torch.all(am[:, :, :, tick] == 0.0), (
                f"Nested path: tick {tick} should be 0 (row 1 owns [2, 7))"
            )


# ---------------------------------------------------------------------------
# Regression: fade-mode d==0 audio preserve (the bug case)
# ---------------------------------------------------------------------------


def _make_fade_inject(audio_mode: str = "fade") -> Inject:
    """Build a minimal Inject with the given audio_mode for test use."""
    return Inject(
        inject_at=0,
        start_fade_in=0,
        start_keyframes=0,
        end_keyframes=17,
        end_fade_out=39,
        min_denoise=0.0,
        interpolation_type="linear",
        audio_mode=audio_mode,
        images=None,
        audio=None,
        resolution=(0, 0),
        source_length=39,
    )


class TestFadeModeAudioPreserveInMask:
    """Regression for the bug where fade-mode d==0 audio was left as generate (mask=1).

    Before the fix, audio_frozen was used as the gate in derive_mask.  audio_frozen is
    False for fade-mode, so audio ticks were never zeroed even on preserve rows.

    After the fix, audio_preserve is used: it returns True for fade-mode rows with
    denoise==0.0, causing the mask to zero those ticks just like video.

    Pre-fix failure:
        AssertionError: audio tick <N> for fade d=0 row must be 0.0 (preserve),
        got 1.0 — the tick was not zeroed because audio_frozen was False.
    """

    _VIDEO_ROWS = 5
    _AUDIO_TICKS = 8

    def _make_preserve_rs(self, audio_mode: str = "fade") -> RowSchedule:
        return RowSchedule(
            row_idx=0,
            denoise=0.0,
            inject=_make_fade_inject(audio_mode=audio_mode),
            audio_frozen=(audio_mode == "keep"),
            region="preserve",
        )

    def test_fade_d0_preserve_row_zeros_audio_ticks(self) -> None:
        """Fade-mode d=0 row must set all its owned audio ticks to 0 in the mask.

        This is the primary regression: before the fix audio_mask stayed 1.0 for
        fade-mode rows regardless of denoise value.
        """
        rs = self._make_preserve_rs(audio_mode="fade")
        result = derive_mask([rs], video_rows=self._VIDEO_ROWS, audio_ticks=self._AUDIO_TICKS)
        audio_mask = result["audio_mask"][0]
        owned_ticks = list(audio_tick_range(rs.row_idx, self._VIDEO_ROWS, self._AUDIO_TICKS))
        assert owned_ticks, "row 0 must own at least one audio tick"
        for tick in owned_ticks:
            assert audio_mask[tick].item() == 0.0, (
                f"audio tick {tick} for fade d=0 row must be 0.0 (preserve), "
                f"got {audio_mask[tick].item()}"
            )

    def test_fade_d0_video_also_zeroed(self) -> None:
        """Sanity: the video mask for the same d=0 row is still 0 (unchanged by fix)."""
        rs = self._make_preserve_rs(audio_mode="fade")
        result = derive_mask([rs], video_rows=self._VIDEO_ROWS, audio_ticks=self._AUDIO_TICKS)
        assert result["video_mask"][0, rs.row_idx].item() == 0.0

    def test_fade_non_d0_row_does_not_freeze_audio(self) -> None:
        """Guard against over-freezing: fade-mode rows with denoise != 0 must stay 1.

        Only d=0 preserve rows trigger audio preservation in fade mode.
        """
        rs = RowSchedule(
            row_idx=0,
            denoise=0.7,
            inject=_make_fade_inject(audio_mode="fade"),
            audio_frozen=False,
            region="fade",
        )
        result = derive_mask([rs], video_rows=self._VIDEO_ROWS, audio_ticks=self._AUDIO_TICKS)
        audio_mask = result["audio_mask"][0]
        owned_ticks = list(audio_tick_range(rs.row_idx, self._VIDEO_ROWS, self._AUDIO_TICKS))
        for tick in owned_ticks:
            assert audio_mask[tick].item() == 1.0, (
                f"audio tick {tick} for fade d=0.7 row must be 1.0 (generate), "
                f"got {audio_mask[tick].item()}"
            )
