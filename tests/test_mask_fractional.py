"""Tests for comfyui_h3_blended_inject.mask.derive_fractional_mask.

The per-row img2img sampler feeds the DiT a *fractional* denoise mask: each video row
carries its own ``m_r`` (= RowSchedule.denoise) and each audio tick carries its owning
row's ``audio_denoise``.  Unlike derive_mask (which is binary 0/1 for the exact-preserve +
composite path), this mask carries the full per-row schedule so the DiT compresses each
row's timestep schedule.

Value semantics:
  - video[row] = r.denoise for scheduled rows; 1.0 for absent rows (full generation)
  - audio[tick] = r.audio_denoise for the owning scheduled row; 1.0 for absent ticks
    (keep → 0.0, fade → r.denoise, drop/none → 1.0)

Both the non-nested (dict) and nested (full-component-shape NestedTensor) paths mirror
derive_mask's structure.
"""

from __future__ import annotations

import torch

from comfyui_h3_blended_inject.grid import audio_tick_range, video_row_to_audio_tick
from comfyui_h3_blended_inject.mask import derive_fractional_mask
from comfyui_h3_blended_inject.schedule import Inject, RowSchedule


def row(row_idx: int, denoise: float = 1.0, audio_frozen: bool = False) -> RowSchedule:
    """Minimal RowSchedule with no inject content."""
    return RowSchedule(row_idx=row_idx, denoise=denoise, inject=None, audio_frozen=audio_frozen)


def fade_row(row_idx: int, denoise: float, audio_mode: str = "fade") -> RowSchedule:
    """RowSchedule backed by an inject with the given audio_mode."""
    inj = Inject(
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
    return RowSchedule(
        row_idx=row_idx,
        denoise=denoise,
        inject=inj,
        audio_frozen=(audio_mode == "keep"),
    )


# ---------------------------------------------------------------------------
# Non-nested (dict) path
# ---------------------------------------------------------------------------


class TestFractionalMaskDictShape:
    def test_keys_and_shapes(self) -> None:
        result = derive_fractional_mask([row(0, 0.5)], video_rows=5, audio_ticks=8)
        assert result["video_mask"].shape == (1, 5)
        assert result["audio_mask"].shape == (1, 8)

    def test_dtype_float32(self) -> None:
        result = derive_fractional_mask([row(0, 0.5)], video_rows=5, audio_ticks=8)
        assert result["video_mask"].dtype == torch.float32
        assert result["audio_mask"].dtype == torch.float32


class TestFractionalMaskVideoValues:
    def test_carries_fractional_denoise(self) -> None:
        """Video mask stores the row's exact fractional m_r (not binarized)."""
        result = derive_fractional_mask([row(1, 0.35)], video_rows=5, audio_ticks=8)
        assert abs(result["video_mask"][0, 1].item() - 0.35) < 1e-6

    def test_zero_row_stays_zero(self) -> None:
        result = derive_fractional_mask([row(2, 0.0)], video_rows=5, audio_ticks=8)
        assert result["video_mask"][0, 2].item() == 0.0

    def test_absent_rows_are_one(self) -> None:
        result = derive_fractional_mask([row(0, 0.0)], video_rows=5, audio_ticks=8)
        for i in range(1, 5):
            assert result["video_mask"][0, i].item() == 1.0

    def test_empty_schedule_all_ones(self) -> None:
        result = derive_fractional_mask([], video_rows=5, audio_ticks=8)
        assert torch.all(result["video_mask"] == 1.0)

    def test_mixed_schedule(self) -> None:
        schedule = [row(0, 0.0), row(1, 0.3), row(2, 0.7), row(3, 1.0)]
        vm = derive_fractional_mask(schedule, video_rows=5, audio_ticks=8)["video_mask"][0]
        assert vm[0].item() == 0.0
        assert abs(vm[1].item() - 0.3) < 1e-6
        assert abs(vm[2].item() - 0.7) < 1e-6
        assert vm[3].item() == 1.0
        assert vm[4].item() == 1.0  # absent


class TestFractionalMaskAudioValues:
    def test_keep_mode_tick_is_zero(self) -> None:
        schedule = [fade_row(0, 0.6, audio_mode="keep")]
        result = derive_fractional_mask(schedule, video_rows=5, audio_ticks=8)
        tick = video_row_to_audio_tick(0)
        assert result["audio_mask"][0, tick].item() == 0.0

    def test_fade_mode_tick_follows_denoise(self) -> None:
        schedule = [fade_row(0, 0.4, audio_mode="fade")]
        result = derive_fractional_mask(schedule, video_rows=5, audio_ticks=8)
        for tick in audio_tick_range(0, 5, 8):
            assert abs(result["audio_mask"][0, tick].item() - 0.4) < 1e-6

    def test_drop_mode_tick_is_one(self) -> None:
        schedule = [fade_row(0, 0.0, audio_mode="drop")]
        result = derive_fractional_mask(schedule, video_rows=5, audio_ticks=8)
        for tick in audio_tick_range(0, 5, 8):
            assert result["audio_mask"][0, tick].item() == 1.0

    def test_absent_ticks_are_one(self) -> None:
        schedule = [fade_row(0, 0.4, audio_mode="fade")]
        result = derive_fractional_mask(schedule, video_rows=5, audio_ticks=16)
        owned = set(audio_tick_range(0, 5, 16))
        for tick in range(16):
            if tick not in owned:
                assert result["audio_mask"][0, tick].item() == 1.0


# ---------------------------------------------------------------------------
# Nested (full-component-shape) path
# ---------------------------------------------------------------------------


class FakeNestedTensor:
    def __init__(self, video_mask: torch.Tensor, audio_mask: torch.Tensor) -> None:
        self.tensors = [video_mask, audio_mask]


def _fake_factory(video_mask: torch.Tensor, audio_mask: torch.Tensor) -> FakeNestedTensor:
    return FakeNestedTensor(video_mask, audio_mask)


_V_SHAPE = (1, 24, 5, 4, 4)
_A_SHAPE = (1, 32, 2, 7)


class TestFractionalMaskNestedPath:
    def _run(self, schedule: list[RowSchedule]) -> FakeNestedTensor:
        return derive_fractional_mask(
            schedule,
            video_rows=_V_SHAPE[2],
            audio_ticks=_A_SHAPE[3],
            video_component_shape=_V_SHAPE,
            audio_component_shape=_A_SHAPE,
            nested_factory=_fake_factory,
        )

    def test_returns_nested_not_dict(self) -> None:
        result = self._run([])
        assert not isinstance(result, dict)

    def test_full_shapes(self) -> None:
        result = self._run([])
        assert result.tensors[0].shape == torch.Size(_V_SHAPE)
        assert result.tensors[1].shape == torch.Size(_A_SHAPE)

    def test_empty_schedule_all_ones(self) -> None:
        result = self._run([])
        assert torch.all(result.tensors[0] == 1.0)
        assert torch.all(result.tensors[1] == 1.0)

    def test_fractional_row_expanded_across_channels_and_spatial(self) -> None:
        result = self._run([row(2, 0.35)])
        t_slice = result.tensors[0][:, :, 2, :, :]
        assert torch.allclose(t_slice, torch.full_like(t_slice, 0.35))

    def test_other_rows_are_one(self) -> None:
        result = self._run([row(2, 0.35)])
        vm = result.tensors[0]
        for t in range(_V_SHAPE[2]):
            if t != 2:
                assert torch.all(vm[:, :, t, :, :] == 1.0)

    def test_fade_audio_tick_expanded(self) -> None:
        result = self._run([fade_row(0, 0.4, audio_mode="fade")])
        for tick in audio_tick_range(0, _V_SHAPE[2], _A_SHAPE[3]):
            a_slice = result.tensors[1][:, :, :, tick]
            assert torch.allclose(a_slice, torch.full_like(a_slice, 0.4))


# ---------------------------------------------------------------------------
# Regression: audio_component_shape required on nested path (Task #55)
# ---------------------------------------------------------------------------


class TestMissingAudioComponentShapeRaisesValueError:
    """derive_fractional_mask must raise ValueError (not assert) when
    video_component_shape is given but audio_component_shape is omitted.

    Before the fix, this was guarded by ``assert audio_component_shape is not None``,
    which is stripped silently under ``python -O`` — producing undefined behavior instead
    of a clear error.  The fix uses ``raise ValueError``, which is never stripped.

    Fail-then-pass: with the old assert code a ``pytest.raises(ValueError)`` block
    catches nothing (assert raises AssertionError, not ValueError) and the test fails.
    With the fix it passes.
    """

    def test_raises_value_error_when_audio_shape_missing(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="audio_component_shape"):
            derive_fractional_mask(
                [],
                video_rows=_V_SHAPE[2],
                audio_ticks=_A_SHAPE[3],
                video_component_shape=_V_SHAPE,
                # audio_component_shape intentionally omitted
            )
