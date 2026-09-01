"""Tests for comfyui_h3_blended_inject.observer_split — CPU-testable per-call helpers.

The pure helpers ``observer_call_update``, ``_fractional_rows``, and ``_call_plan`` are exercised
here.  The comfy-coupled functions (install_observer_split, _attention_with_cached_kv,
_make_block_patch) are GPU-only (# pragma: no cover) and require a live ComfyUI + H3 model.
"""

from __future__ import annotations

import torch

from comfyui_h3_blended_inject.observer_split import (
    _call_plan,
    _fractional_rows,
    observer_call_update,
)


class TestObserverCallUpdate:
    """The clean-K/V mechanism carries no per-call labels — the capture forward publishes the
    m labels through ``pooled_current`` — so this helper only bumps the per-forward token that
    invalidates the cached splice-position plan between the two forwards of a step.
    """

    def test_sets_call_and_token(self) -> None:
        obs: dict = {}
        observer_call_update(obs)
        assert obs["_token"] == 1
        assert obs["call"] == {"token": 1}

    def test_token_increments_across_calls(self) -> None:
        obs: dict = {}
        observer_call_update(obs)
        first = obs["_token"]
        assert obs["call"]["token"] == first
        observer_call_update(obs)
        assert obs["_token"] == first + 1
        assert obs["call"]["token"] == first + 1

    def test_fresh_call_dict_each_forward(self) -> None:
        # A new call dict per forward means a stale ``plan`` cannot leak across forwards.
        obs: dict = {}
        observer_call_update(obs)
        obs["call"]["plan"] = "stale"
        observer_call_update(obs)
        assert "plan" not in obs["call"]


def _seg(a: int, b: int, tag: int) -> tuple[int, int, torch.Tensor]:
    """A ``mod_segments`` entry (a, b, row) whose modality tag = int(row[0] % 3)."""
    return (a, b, torch.tensor([float(tag), 0.0, 0.0]))


class TestCallPlan:
    """``_call_plan`` maps fractional-row indices to GLOBAL token positions across the video/audio
    ``mod_segments`` and caches the result on the per-forward ``call`` dict.
    """

    def test_cached_plan_returned_verbatim(self) -> None:
        cached = torch.tensor([7, 9])
        call = {"plan": cached}
        assert _call_plan({}, call, []) is cached

    def test_video_only_positions_offset_by_segment_start(self) -> None:
        state = {"video": {"n": 4, "pos": torch.tensor([1, 3])}}
        call: dict = {}
        pos = _call_plan(state, call, [_seg(0, 4, 0)])
        assert pos is not None
        assert pos.tolist() == [1, 3]
        # Result is cached on the call dict.
        assert call["plan"] is pos

    def test_video_and_audio_positions_concatenated(self) -> None:
        state = {
            "video": {"n": 4, "pos": torch.tensor([1])},
            "audio": {"n": 4, "pos": torch.tensor([2])},
        }
        pos = _call_plan(state, {}, [_seg(0, 4, 0), _seg(4, 8, 2)])
        assert pos is not None
        assert pos.tolist() == [1, 6]  # video 1+0, audio 2+4

    def test_non_tensor_rows_skipped(self) -> None:
        state = {"video": {"n": 4, "pos": torch.tensor([1])}}
        segs = [(0, 4, "not-a-tensor"), _seg(0, 4, 0)]
        pos = _call_plan(state, {}, segs)
        assert pos is not None
        assert pos.tolist() == [1]

    def test_absent_stream_segment_skipped(self) -> None:
        # No audio stream armed; a tag-2 segment contributes nothing.
        state = {"video": {"n": 4, "pos": torch.tensor([1])}}
        pos = _call_plan(state, {}, [_seg(0, 4, 0), _seg(4, 8, 2)])
        assert pos is not None
        assert pos.tolist() == [1]

    def test_layout_mismatch_segment_skipped(self) -> None:
        # Segment width (b - a) != stream["n"] → skipped rather than mis-spliced.
        state = {"video": {"n": 4, "pos": torch.tensor([1])}}
        pos = _call_plan(state, {}, [_seg(0, 5, 0)])
        assert pos is None

    def test_no_matching_segments_returns_none_and_caches(self) -> None:
        state = {"video": {"n": 4, "pos": torch.tensor([1])}}
        call: dict = {}
        assert _call_plan(state, call, []) is None
        assert call["plan"] is None


class TestFractionalRows:
    def test_all_integer_returns_none(self) -> None:
        assert _fractional_rows(torch.tensor([0.0, 1.0, 0.0, 1.0])) is None

    def test_empty_ish_no_fractional_returns_none(self) -> None:
        assert _fractional_rows(torch.zeros(5)) is None

    def test_picks_only_strictly_fractional_rows(self) -> None:
        rows = torch.tensor([0.0, 0.5, 1.0, 0.25])
        out = _fractional_rows(rows)
        assert out is not None
        assert out["n"] == 4
        assert out["pos"].tolist() == [1, 3]
        assert torch.allclose(out["m"], torch.tensor([0.5, 0.25]))

    def test_boundary_just_above_low_tol_included(self) -> None:
        # m just above 1e-6 is included; exactly 0.0 excluded.
        rows = torch.tensor([0.0, 2e-6])
        out = _fractional_rows(rows)
        assert out is not None
        assert out["pos"].tolist() == [1]

    def test_boundary_just_below_high_tol_included(self) -> None:
        # m just below 1 - 1e-3 included; exactly 1.0 excluded.
        rows = torch.tensor([1.0, 1.0 - 2e-3])
        out = _fractional_rows(rows)
        assert out is not None
        assert out["pos"].tolist() == [1]

    def test_boundary_at_high_tol_excluded(self) -> None:
        # m >= 1 - 1e-3 is treated as full-denoise (excluded).
        rows = torch.tensor([1.0 - 1e-4])
        assert _fractional_rows(rows) is None

    def test_exact_zero_and_one_excluded(self) -> None:
        rows = torch.tensor([0.0, 1.0])
        assert _fractional_rows(rows) is None
