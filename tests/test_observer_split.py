"""Tests for comfyui_h3_blended_inject.observer_split — CPU-testable per-call helpers.

Only ``observer_call_update`` and ``_fractional_rows`` are exercised here; everything below
them (install_observer_split, _call_plan, _attention_with_observer_kv, _make_block_patch) is
GPU-only (# pragma: no cover) and requires a live ComfyUI + H3 model.
"""

from __future__ import annotations

import torch

from comfyui_h3_blended_inject.observer_split import (
    AUDIO_COND_TIMESTEP,
    VISUAL_COND_TIMESTEP,
    _fractional_rows,
    observer_call_update,
)
from comfyui_h3_blended_inject.sampler import time_shift_sigma


class TestConstants:
    def test_cond_timestep_pins(self) -> None:
        assert VISUAL_COND_TIMESTEP == 0.999
        assert AUDIO_COND_TIMESTEP == 1.0


class TestObserverCallUpdate:
    def test_video_only(self) -> None:
        m = torch.tensor([0.25, 0.5, 0.75])
        obs = {"video": {"m": m}, "shift_v": 12.0, "shift_a": 3.0}
        sigma_v = 0.4
        observer_call_update(obs, sigma_v)
        t_pin = max(1.0 - sigma_v, VISUAL_COND_TIMESTEP)
        expected = (1.0 - m * sigma_v).clamp(max=t_pin)
        assert torch.allclose(obs["call"]["t_obs_v"], expected)
        assert "t_obs_a" not in obs["call"]

    def test_audio_only_uses_shifted_sigma(self) -> None:
        m = torch.tensor([0.3, 0.6])
        obs = {"audio": {"m": m}, "shift_v": 12.0, "shift_a": 3.0}
        sigma_v = 0.4
        observer_call_update(obs, sigma_v)
        sigma_a = time_shift_sigma(sigma_v, 12.0, 3.0)
        t_pin = max(1.0 - sigma_a, AUDIO_COND_TIMESTEP)
        expected = (1.0 - m * sigma_a).clamp(max=t_pin)
        assert torch.allclose(obs["call"]["t_obs_a"], expected)
        assert "t_obs_v" not in obs["call"]

    def test_both_streams(self) -> None:
        mv = torch.tensor([0.5])
        ma = torch.tensor([0.5])
        obs = {"video": {"m": mv}, "audio": {"m": ma}, "shift_v": 12.0, "shift_a": 3.0}
        sigma_v = 0.5
        observer_call_update(obs, sigma_v)
        assert "t_obs_v" in obs["call"]
        assert "t_obs_a" in obs["call"]
        # Audio runs the shifted sigma → different label than video at the same m.
        assert not torch.allclose(obs["call"]["t_obs_v"], obs["call"]["t_obs_a"])

    def test_default_shifts_when_absent(self) -> None:
        """shift_v/shift_a default to 12/3 when not provided in obs."""
        m = torch.tensor([0.5])
        obs = {"audio": {"m": m}}
        observer_call_update(obs, 0.4)
        sigma_a = time_shift_sigma(0.4, 12.0, 3.0)
        expected = (1.0 - m * sigma_a).clamp(max=max(1.0 - sigma_a, AUDIO_COND_TIMESTEP))
        assert torch.allclose(obs["call"]["t_obs_a"], expected)

    def test_token_increments_across_calls(self) -> None:
        obs = {"video": {"m": torch.tensor([0.5])}, "shift_v": 12.0, "shift_a": 3.0}
        observer_call_update(obs, 0.5)
        first = obs["_token"]
        assert obs["call"]["token"] == first
        observer_call_update(obs, 0.4)
        assert obs["_token"] == first + 1
        assert obs["call"]["token"] == first + 1

    def test_no_streams_still_sets_call_and_token(self) -> None:
        obs: dict = {}
        observer_call_update(obs, 0.5)
        assert obs["call"]["token"] == 1
        assert "t_obs_v" not in obs["call"]
        assert "t_obs_a" not in obs["call"]


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
