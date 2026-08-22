"""Hypothesis-based tests for comfyui_h3_blended_inject.hold_release.

Contract sources:
  - hold_release.py docstrings (each function's documented return value / invariants)
  - plan.md "Mechanism: hold-and-release", "Audio specifics", "Integration point",
    and the "Hold math" test bullet

These tests FAIL now because every hold_release function raises NotImplementedError.
They are written as if implemented so they PASS once all stubs are filled in.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from comfyui_h3_blended_inject import constants
from comfyui_h3_blended_inject.hold_release import (
    audio_internal_scale,
    build_model_function_wrapper,
    draw_row_noise,
    hold_value,
    is_held,
)
from comfyui_h3_blended_inject.schedule import RowSchedule

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

sigma_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# hold_value
# ---------------------------------------------------------------------------


class TestHoldValue:
    """hold_value(original, noise, sigma) == (1 - sigma) * original + sigma * noise."""

    @given(
        rows=st.integers(min_value=1, max_value=4),
        cols=st.integers(min_value=1, max_value=8),
        sigma=sigma_st,
    )
    def test_formula_matches_hand_computation(self, rows: int, cols: int, sigma: float) -> None:
        """Result equals the hand-computed linear blend across all shapes and sigma values."""
        torch.manual_seed(7)
        original = torch.randn(rows, cols)
        noise = torch.randn(rows, cols)
        expected = (1.0 - sigma) * original + sigma * noise
        result = hold_value(original, noise, sigma)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_endpoint_sigma_zero_returns_original(self) -> None:
        """At sigma=0 the hold value equals the original (no noise contribution)."""
        original = torch.randn(3, 5)
        noise = torch.randn(3, 5)
        assert torch.allclose(hold_value(original, noise, 0.0), original, atol=1e-7)

    def test_endpoint_sigma_one_returns_noise(self) -> None:
        """At sigma=1 the hold value equals the noise (no original contribution)."""
        original = torch.randn(3, 5)
        noise = torch.randn(3, 5)
        assert torch.allclose(hold_value(original, noise, 1.0), noise, atol=1e-7)

    def test_output_shape_matches_input(self) -> None:
        """Output shape is identical to original / noise shape."""
        shape = (2, 4, 3)
        original = torch.randn(*shape)
        noise = torch.randn(*shape)
        assert hold_value(original, noise, 0.5).shape == torch.Size(shape)


# ---------------------------------------------------------------------------
# is_held
# ---------------------------------------------------------------------------


class TestIsHeld:
    """is_held(sigma, d) == (sigma > d); strict inequality."""

    @given(sigma=sigma_st, d=sigma_st)
    def test_true_when_sigma_strictly_exceeds_d(self, sigma: float, d: float) -> None:
        assume(sigma > d)
        assert is_held(sigma, d) is True

    @given(sigma=sigma_st, d=sigma_st)
    def test_false_when_sigma_strictly_below_d(self, sigma: float, d: float) -> None:
        assume(sigma < d)
        assert is_held(sigma, d) is False

    @given(sigma=sigma_st)
    def test_false_at_exact_boundary(self, sigma: float) -> None:
        """sigma == d is NOT held (strict inequality; equality means released)."""
        assert is_held(sigma, sigma) is False

    def test_concrete_held(self) -> None:
        assert is_held(0.8, 0.5) is True

    def test_concrete_released(self) -> None:
        assert is_held(0.3, 0.5) is False


# ---------------------------------------------------------------------------
# draw_row_noise
# ---------------------------------------------------------------------------


class TestDrawRowNoise:
    """draw_row_noise: deterministic, correct shape, dtype-aware, CPU-default."""

    def test_returns_correct_shape(self) -> None:
        shape = (3, 4, 5)
        assert draw_row_noise(0, shape).shape == torch.Size(shape)

    def test_deterministic_same_seed(self) -> None:
        """Same seed produces identical tensors across calls."""
        shape = (4, 6)
        assert torch.equal(draw_row_noise(42, shape), draw_row_noise(42, shape))

    def test_different_seeds_produce_different_tensors(self) -> None:
        """Different seeds produce different tensors (probability ~1 for non-trivial shape)."""
        shape = (4, 8)
        assert not torch.allclose(draw_row_noise(1, shape), draw_row_noise(2, shape))

    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_hypothesis_same_seed_equal(self, seed: int) -> None:
        shape = (3, 5)
        assert torch.equal(draw_row_noise(seed, shape), draw_row_noise(seed, shape))

    def test_respects_float32_dtype(self) -> None:
        assert draw_row_noise(0, (3, 4), dtype=torch.float32).dtype == torch.float32

    def test_respects_float64_dtype(self) -> None:
        assert draw_row_noise(0, (3, 4), dtype=torch.float64).dtype == torch.float64

    def test_default_dtype_is_float32(self) -> None:
        assert draw_row_noise(0, (2, 3)).dtype == torch.float32

    def test_device_is_cpu_by_default(self) -> None:
        assert draw_row_noise(0, (2, 3)).device.type == "cpu"


# ---------------------------------------------------------------------------
# audio_internal_scale
# ---------------------------------------------------------------------------


class TestAudioInternalScale:
    """audio_internal_scale: rescales raw audio latent to sampler-internal scale.

    The docstring specifies the rescaling uses audio_scale_factor but does not pin the
    exact formula.  Tests assert the invariants the docstring DOES state:
    linearity in value, shape preservation, finite output.
    """

    def test_linearity_in_value(self) -> None:
        """Doubling the input value doubles the output (linear in value)."""
        sigma = 0.5
        asf = 2.0
        v1 = torch.ones(2, 3)
        v2 = 2.0 * v1
        r1 = audio_internal_scale(v1, sigma, asf)
        r2 = audio_internal_scale(v2, sigma, asf)
        assert torch.allclose(r2, 2.0 * r1, atol=1e-6), (
            "audio_internal_scale must be linear in value"
        )

    def test_output_shape_matches_input(self) -> None:
        value = torch.randn(3, 5)
        assert audio_internal_scale(value, 0.4, 1.5).shape == value.shape

    @given(
        sigma=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
        audio_scale_factor=st.floats(
            min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_hypothesis_output_is_finite(self, sigma: float, audio_scale_factor: float) -> None:
        value = torch.ones(2, 3)
        assert torch.isfinite(audio_internal_scale(value, sigma, audio_scale_factor)).all()


# ---------------------------------------------------------------------------
# Fake comfy module for wrapper tests
# ---------------------------------------------------------------------------
#
# Fake packed layout (simple, documented):
#   packed tensor shape = (N_VIDEO + N_AUDIO, FEAT)
#   rows [0, N_VIDEO)            = video rows
#   rows [N_VIDEO, N_VIDEO+N_AUDIO) = audio ticks
#
#   unpack_latents(packed, latent_shapes) -> (video, audio)
#     Splits at _N_VIDEO; ignores latent_shapes (test constants are fixed).
#   pack_latents(video, audio) -> torch.cat([video, audio], dim=0)
#
# Goal: exercise the wrapper's held-row edit/overwrite logic, not H3's real AV packing.

_N_VIDEO: int = 3  # video rows used in wrapper tests
_N_AUDIO: int = 2  # audio ticks used in wrapper tests
_FEAT: int = 4  # feature dimension per row/tick


def _fake_unpack_latents(
    packed: torch.Tensor,
    latent_shapes: Any,  # ignored by the fake; split point is fixed
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (video, audio) tensors from the concatenated packed tensor."""
    return packed[:_N_VIDEO].clone(), packed[_N_VIDEO : _N_VIDEO + _N_AUDIO].clone()


def _fake_pack_latents(video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
    """Concatenate (video, audio) into the test's packed format."""
    return torch.cat([video, audio], dim=0)


@pytest.fixture()
def fake_comfy(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Patch sys.modules with a fake comfy.utils so the wrapper's lazy imports work."""
    fake_utils = types.ModuleType("comfy.utils")
    fake_utils.unpack_latents = _fake_unpack_latents  # type: ignore[attr-defined]
    fake_utils.pack_latents = _fake_pack_latents  # type: ignore[attr-defined]

    fake_comfy_mod = types.ModuleType("comfy")
    fake_comfy_mod.utils = fake_utils  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "comfy", fake_comfy_mod)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_utils)
    return fake_comfy_mod


# ---------------------------------------------------------------------------
# Shared fixtures for wrapper tests
# ---------------------------------------------------------------------------


def _make_wrapper_fixtures(
    denoise: float = 0.3,
) -> tuple[
    list[RowSchedule],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    torch.Tensor,
]:
    """Build schedule, per-row/tick tensors, and a packed input tensor.

    Video row i has original=fill(i+1) and noise=fill(0.1*(i+1)).
    Audio tick j has original=fill(j+10) and noise=fill(0.1*(j+10)).
    Packed input has video rows=fill(i+100) and audio ticks=fill(j+200).
    """
    schedule = [RowSchedule(row_idx=i, denoise=denoise, inject=None) for i in range(_N_VIDEO)]

    per_row_original = {i: torch.full((_FEAT,), float(i + 1)) for i in range(_N_VIDEO)}
    per_row_noise = {i: torch.full((_FEAT,), 0.1 * (i + 1)) for i in range(_N_VIDEO)}

    audio_row_original = {j: torch.full((_FEAT,), float(j + 10)) for j in range(_N_AUDIO)}
    audio_row_noise = {j: torch.full((_FEAT,), 0.1 * (j + 10)) for j in range(_N_AUDIO)}

    video_rows = torch.stack([torch.full((_FEAT,), float(i + 100)) for i in range(_N_VIDEO)])
    audio_rows = torch.stack([torch.full((_FEAT,), float(j + 200)) for j in range(_N_AUDIO)])
    packed_input = torch.cat([video_rows, audio_rows], dim=0)

    return (
        schedule,
        per_row_original,
        per_row_noise,
        audio_row_original,
        audio_row_noise,
        packed_input,
    )


def _args_dict(packed_input: torch.Tensor, sigma: float) -> dict[str, Any]:
    """Construct the args_dict the ComfyUI sampler passes to the wrapper.

    Sigma is recovered inside the wrapper via timestep / 1000 (per plan).
    """
    return {
        "input": packed_input,
        "timestep": torch.tensor([sigma * 1000.0]),
        "c": {},
        "cond_or_uncond": [0],
    }


# ---------------------------------------------------------------------------
# build_model_function_wrapper
# ---------------------------------------------------------------------------


class TestBuildModelFunctionWrapper:
    """Wrapper: held rows get hold_value; prediction rows get original; aliasing-safe."""

    def test_held_video_rows_written_with_hold_value_in_input(
        self, fake_comfy: types.ModuleType
    ) -> None:
        """Held video rows: apply_model receives hold_value(original, noise, sigma) per row."""
        denoise = 0.3
        sigma = 0.7  # sigma > denoise => all rows held
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
        ) = _make_wrapper_fixtures(denoise=denoise)

        received: list[torch.Tensor] = []

        def mock_apply_model(input_x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            received.append(input_x.detach().clone())
            return input_x  # return input as-is; prediction content irrelevant here

        wrapper = build_model_function_wrapper(
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            audio_scale_factor=1.0,
        )
        wrapper(mock_apply_model, _args_dict(packed_input, sigma))

        assert len(received) == 1, "apply_model must be called exactly once"
        video_in = received[0][:_N_VIDEO]

        for i, row_s in enumerate(schedule):
            if is_held(sigma, row_s.denoise):
                expected = hold_value(per_row_original[i], per_row_noise[i], sigma)
                assert torch.allclose(video_in[i], expected, atol=1e-6), (
                    f"Held video row {i}: apply_model input must equal hold_value at sigma={sigma}"
                )

    def test_non_held_video_rows_untouched_in_input(self, fake_comfy: types.ModuleType) -> None:
        """Non-held rows (sigma <= d): apply_model receives the unmodified input rows."""
        denoise = 0.9
        sigma = 0.5  # sigma < denoise => no rows held
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
        ) = _make_wrapper_fixtures(denoise=denoise)

        received: list[torch.Tensor] = []

        def mock_apply_model(input_x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            received.append(input_x.detach().clone())
            return input_x

        wrapper = build_model_function_wrapper(
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            audio_scale_factor=1.0,
        )
        wrapper(mock_apply_model, _args_dict(packed_input, sigma))

        video_in = received[0][:_N_VIDEO]
        original_video = packed_input[:_N_VIDEO]

        for i in range(_N_VIDEO):
            assert torch.allclose(video_in[i], original_video[i], atol=1e-7), (
                f"Non-held video row {i} must be unchanged in apply_model input"
            )

    def test_held_video_rows_prediction_overwritten_with_original(
        self, fake_comfy: types.ModuleType
    ) -> None:
        """Held video rows: wrapper overwrites returned prediction with per_row_original."""
        denoise = 0.3
        sigma = 0.7
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
        ) = _make_wrapper_fixtures(denoise=denoise)

        sentinel = torch.full((_N_VIDEO + _N_AUDIO, _FEAT), 999.0)

        wrapper = build_model_function_wrapper(
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            audio_scale_factor=1.0,
        )
        result = wrapper(
            lambda *a, **kw: sentinel.clone(),  # noqa: ARG005
            _args_dict(packed_input, sigma),
        )

        result_video = result[:_N_VIDEO]
        for i, row_s in enumerate(schedule):
            if is_held(sigma, row_s.denoise):
                assert torch.allclose(result_video[i], per_row_original[i], atol=1e-7), (
                    f"Held video row {i}: wrapper must overwrite prediction with per_row_original"
                )

    def test_audio_held_ticks_use_shifted_sigma_and_internal_scale(
        self, fake_comfy: types.ModuleType
    ) -> None:
        """Held audio ticks: apply_model receives hold_value(..., time_shift_sigma(sigma)) at
        audio_internal_scale, and the prediction rows are overwritten with audio_original
        at internal scale.

        Uses constants.time_shift_sigma and hold_release.audio_internal_scale directly so
        the test tracks the real implementations (both currently unimplemented -> fails now).
        """
        denoise = 0.2
        sigma = 0.8
        audio_scale = 1.5
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
        ) = _make_wrapper_fixtures(denoise=denoise)

        received: list[torch.Tensor] = []
        sentinel = torch.full((_N_VIDEO + _N_AUDIO, _FEAT), 999.0)

        def mock_apply_model(input_x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            received.append(input_x.detach().clone())
            return sentinel.clone()

        wrapper = build_model_function_wrapper(
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            audio_scale_factor=audio_scale,
        )
        result = wrapper(mock_apply_model, _args_dict(packed_input, sigma))

        sigma_audio = constants.time_shift_sigma(sigma)
        audio_in = received[0][_N_VIDEO : _N_VIDEO + _N_AUDIO]
        result_audio = result[_N_VIDEO : _N_VIDEO + _N_AUDIO]

        for j in range(_N_AUDIO):
            if is_held(sigma_audio, denoise):
                # Input: hold_value at shifted sigma, written at internal scale
                expected_in = audio_internal_scale(
                    hold_value(audio_orig[j], audio_noise[j], sigma_audio),
                    sigma_audio,
                    audio_scale,
                )
                assert torch.allclose(audio_in[j], expected_in, atol=1e-6), (
                    f"Audio tick {j}: apply_model input must be hold_value "
                    f"at shifted sigma={sigma_audio}, at internal scale"
                )
                # Prediction: overwritten with audio_original at internal scale
                expected_pred = audio_internal_scale(audio_orig[j], sigma_audio, audio_scale)
                assert torch.allclose(result_audio[j], expected_pred, atol=1e-7), (
                    f"Audio tick {j}: prediction must be overwritten with audio_original "
                    f"at internal scale"
                )

    def test_aliasing_safety_no_inplace_mutation_of_input(
        self, fake_comfy: types.ModuleType
    ) -> None:
        """Wrapper must not mutate the caller's packed input tensor in place."""
        denoise = 0.3
        sigma = 0.7
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
        ) = _make_wrapper_fixtures(denoise=denoise)

        snapshot = packed_input.clone()

        wrapper = build_model_function_wrapper(
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            audio_scale_factor=1.0,
        )
        wrapper(
            lambda *a, **kw: torch.zeros_like(packed_input),  # noqa: ARG005
            _args_dict(packed_input, sigma),
        )

        assert torch.equal(packed_input, snapshot), (
            "Wrapper must not mutate the caller's input tensor in place"
        )


# ---------------------------------------------------------------------------
# Trajectory identity — pure math on hold_value, no wrapper
# ---------------------------------------------------------------------------


class TestTrajectoryIdentity:
    """Euler step with denoised=original stays on the noised-original trajectory.

    From the plan (Integration point):
      x_next = x + (x - denoised) / sigma * (sigma_next - sigma)
      with x = hold_value(original, noise, sigma), denoised = original
      => x_next == hold_value(original, noise, sigma_next)

    This validates the mechanism's core claim.  Calls hold_value only — if hold_value is
    unimplemented the test fails with NotImplementedError, which is the expected state now.
    """

    @given(
        rows=st.integers(min_value=1, max_value=4),
        cols=st.integers(min_value=1, max_value=6),
        sigma=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        sigma_next=st.floats(min_value=0.0, max_value=0.99, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_euler_step_lands_on_trajectory(
        self, rows: int, cols: int, sigma: float, sigma_next: float
    ) -> None:
        """x + (x - original) / sigma * (sigma_next - sigma) == hold_value(..., sigma_next)."""
        assume(sigma > sigma_next)
        assume(sigma > 1e-7)  # guard against division by zero

        torch.manual_seed(0)
        original = torch.randn(rows, cols)
        noise = torch.randn(rows, cols)

        x = hold_value(original, noise, sigma)
        # Euler derivative: d = (x - denoised) / sigma; reported denoised = original
        x_next = x + (x - original) / sigma * (sigma_next - sigma)

        expected = hold_value(original, noise, sigma_next)
        assert torch.allclose(x_next, expected, atol=1e-5), (
            f"Euler step at sigma={sigma} -> sigma_next={sigma_next} must equal "
            f"hold_value(..., sigma_next)"
        )
