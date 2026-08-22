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
# Fake comfy.utils for wrapper tests
# ---------------------------------------------------------------------------
#
# Faithful implementation of comfy.utils.pack_latents / unpack_latents geometry:
#
#   pack_latents(iterable_of_tensors) -> (flat, latent_shapes)
#     Each component tensor is reshaped to [B,1,-1] and concatenated along dim=-1.
#     latent_shapes is the list of original component shapes (as tuples of ints).
#     flat shape: [B, 1, sum_of_prod_shape[1:]]
#
#   unpack_latents(flat, latent_shapes) -> list[Tensor]
#     Splits flat along the last dim using prod(shape[1:]) per component.
#     Batch dim B comes from the runtime flat tensor (not from latent_shapes).
#     Each chunk is reshaped to [B, *shape[1:]].
#
# This matches the real comfy.utils contract so wrapper index math can be verified
# without a live ComfyUI install.


def _fake_pack_latents(
    tensors_iterable: Any,
) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
    """Pack list-of-tensors → (flat [B,1,C], latent_shapes)."""
    tensors = list(tensors_iterable)
    B = tensors[0].shape[0]
    shapes: list[tuple[int, ...]] = [tuple(int(d) for d in t.shape) for t in tensors]
    reshaped = [t.reshape(B, 1, -1) for t in tensors]
    flat = torch.cat(reshaped, dim=-1)
    return flat, shapes


def _fake_unpack_latents(
    flat: torch.Tensor,
    latent_shapes: list[tuple[int, ...]],
) -> list[torch.Tensor]:
    """Unpack flat [B,1,C] → list of component tensors using latent_shapes."""
    B = flat.shape[0]
    results: list[torch.Tensor] = []
    offset = 0
    for shape in latent_shapes:
        n = 1
        for d in shape[1:]:
            n *= d
        chunk = flat[:, :, offset : offset + n]  # [B, 1, n]
        results.append(chunk.reshape(B, *shape[1:]))
        offset += n
    return results


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
#
# Video component shape: [B=1, C_v=24, T=3, Hl=2, Wl=2]
# Audio component shape: [B=1, C_a=32, 2, audio_t=2]
# (Small spatial dims to keep pack/unpack fast in tests.)

_B: int = 1
_C_V: int = 24
_T: int = 3  # video rows (was _N_VIDEO)
_HL: int = 2
_WL: int = 2
_C_A: int = 32
_AUDIO_T: int = 2  # audio ticks (was _N_AUDIO)

_VIDEO_SHAPE: tuple[int, ...] = (_B, _C_V, _T, _HL, _WL)
_AUDIO_SHAPE: tuple[int, ...] = (_B, _C_A, 2, _AUDIO_T)


def _make_wrapper_fixtures(
    denoise: float = 0.3,
    batch: int = 1,
) -> tuple[
    list[RowSchedule],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    torch.Tensor,
    list[tuple[int, ...]],
]:
    """Build schedule, per-row/tick tensors, packed input, and latent_shapes.

    Video row i:  original=fill(i+1),     noise=fill(0.1*(i+1)),   shape [1,C_v,1,Hl,Wl]
    Audio tick j: original=fill(j+10),    noise=fill(0.1*(j+10)),  shape [1,C_a,2,1]
    Full video:   row i has all elements = float(i+100),            shape [B,C_v,T,Hl,Wl]
    Full audio:   tick j has all elements = float(j+200),           shape [B,C_a,2,audio_t]

    The packed tensor is built via _fake_pack_latents so the geometry matches the
    real comfy.utils.pack_latents contract.  batch controls the batch dim B of the
    PACKED input (simulates cond-batch; per_row_original stays at B=1).
    """
    schedule = [RowSchedule(row_idx=i, denoise=denoise, inject=None) for i in range(_T)]

    # Per-row originals/noise: shape [1, C_v, 1, Hl, Wl] (single temporal row).
    per_row_original = {i: torch.full((1, _C_V, 1, _HL, _WL), float(i + 1)) for i in range(_T)}
    per_row_noise = {i: torch.full((1, _C_V, 1, _HL, _WL), 0.1 * (i + 1)) for i in range(_T)}

    # Per-tick originals/noise: shape [1, C_a, 2, 1] (single audio tick).
    audio_row_original = {j: torch.full((1, _C_A, 2, 1), float(j + 10)) for j in range(_AUDIO_T)}
    audio_row_noise = {j: torch.full((1, _C_A, 2, 1), 0.1 * (j + 10)) for j in range(_AUDIO_T)}

    # Full video tensor [B, C_v, T, Hl, Wl]; each row i filled with float(i+100).
    video = torch.zeros(batch, _C_V, _T, _HL, _WL)
    for i in range(_T):
        video[:, :, i, :, :] = float(i + 100)

    # Full audio tensor [B, C_a, 2, audio_t]; each tick j filled with float(j+200).
    audio = torch.zeros(batch, _C_A, 2, _AUDIO_T)
    for j in range(_AUDIO_T):
        audio[:, :, :, j] = float(j + 200)

    # Pack → latent_shapes uses shape[0]=1 (from original tensors, not `batch`).
    # latent_shapes is derived from the canonical B=1 shapes so unpack works for any B.
    _, latent_shapes = _fake_pack_latents(
        [
            torch.zeros(1, _C_V, _T, _HL, _WL),
            torch.zeros(1, _C_A, 2, _AUDIO_T),
        ]
    )
    packed_input, _ = _fake_pack_latents([video, audio])

    return (
        schedule,
        per_row_original,
        per_row_noise,
        audio_row_original,
        audio_row_noise,
        packed_input,
        latent_shapes,
    )


def _args_dict(packed_input: torch.Tensor, sigma: float) -> dict[str, Any]:
    """Construct the args_dict the ComfyUI sampler passes to the wrapper."""
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
    """Wrapper: held rows get hold_value; prediction rows get original; aliasing-safe.

    All tests use the faithful fake pack/unpack and proper 5D video + 4D audio tensors.
    The wrapper must index the temporal dim (dim 2) of video and dim 3 of audio, NOT
    the batch dim.
    """

    def test_held_video_rows_written_with_hold_value_in_input(
        self, fake_comfy: types.ModuleType
    ) -> None:
        """Held video rows: apply_model receives hold_value(original, noise, sigma) per row.

        This test MUST fail before the wrapper fix (wrong indexing / pack API) and
        pass after.
        """
        denoise = 0.3
        sigma = 0.7  # sigma > denoise => all rows held
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
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
            latent_shapes=latent_shapes,
        )
        wrapper(mock_apply_model, _args_dict(packed_input, sigma))

        assert len(received) == 1, "apply_model must be called exactly once"
        # Unpack to inspect the video component at the temporal dim.
        unpacked_in = _fake_unpack_latents(received[0], latent_shapes)
        video_in = unpacked_in[0]  # [B, C_v, T, Hl, Wl]

        for i, row_s in enumerate(schedule):
            if is_held(sigma, row_s.denoise):
                expected = hold_value(per_row_original[i], per_row_noise[i], sigma)
                actual = video_in[:, :, i : i + 1, :, :]  # [B, C_v, 1, Hl, Wl]
                assert torch.allclose(actual, expected.expand_as(actual), atol=1e-6), (
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
            latent_shapes,
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
            latent_shapes=latent_shapes,
        )
        wrapper(mock_apply_model, _args_dict(packed_input, sigma))

        unpacked_in = _fake_unpack_latents(received[0], latent_shapes)
        video_in = unpacked_in[0]
        unpacked_orig = _fake_unpack_latents(packed_input, latent_shapes)
        original_video = unpacked_orig[0]

        for i in range(_T):
            assert torch.allclose(
                video_in[:, :, i, :, :], original_video[:, :, i, :, :], atol=1e-7
            ), f"Non-held video row {i} must be unchanged in apply_model input"

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
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise)

        # Sentinel prediction packed to the same flat shape as packed_input.
        flat_sentinel, _ = _fake_pack_latents(
            [
                torch.full((1, _C_V, _T, _HL, _WL), 999.0),
                torch.full((1, _C_A, 2, _AUDIO_T), 999.0),
            ]
        )

        wrapper = build_model_function_wrapper(
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            audio_scale_factor=1.0,
            latent_shapes=latent_shapes,
        )
        result = wrapper(
            lambda *a, **kw: flat_sentinel.clone(),  # noqa: ARG005
            _args_dict(packed_input, sigma),
        )

        unpacked_result = _fake_unpack_latents(result, latent_shapes)
        result_video = unpacked_result[0]  # [B, C_v, T, Hl, Wl]

        for i, row_s in enumerate(schedule):
            if is_held(sigma, row_s.denoise):
                expected = per_row_original[i]  # [1, C_v, 1, Hl, Wl]
                actual = result_video[:, :, i : i + 1, :, :]
                assert torch.allclose(actual, expected.expand_as(actual), atol=1e-7), (
                    f"Held video row {i}: wrapper must overwrite prediction with per_row_original"
                )

    def test_audio_held_ticks_use_shifted_sigma_and_internal_scale(
        self, fake_comfy: types.ModuleType
    ) -> None:
        """Held audio ticks: apply_model input is hold_value at shifted sigma + internal scale;
        prediction is overwritten with audio_original at internal scale.
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
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise)

        received: list[torch.Tensor] = []
        flat_sentinel, _ = _fake_pack_latents(
            [
                torch.full((1, _C_V, _T, _HL, _WL), 999.0),
                torch.full((1, _C_A, 2, _AUDIO_T), 999.0),
            ]
        )

        def mock_apply_model(input_x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            received.append(input_x.detach().clone())
            return flat_sentinel.clone()

        wrapper = build_model_function_wrapper(
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            audio_scale_factor=audio_scale,
            latent_shapes=latent_shapes,
        )
        result = wrapper(mock_apply_model, _args_dict(packed_input, sigma))

        sigma_audio = constants.time_shift_sigma(sigma)
        unpacked_in = _fake_unpack_latents(received[0], latent_shapes)
        audio_in = unpacked_in[1]  # [B, C_a, 2, audio_t]
        unpacked_result = _fake_unpack_latents(result, latent_shapes)
        result_audio = unpacked_result[1]

        for j in range(_AUDIO_T):
            if is_held(sigma_audio, denoise):
                expected_in = audio_internal_scale(
                    hold_value(audio_orig[j], audio_noise[j], sigma_audio),
                    sigma_audio,
                    audio_scale,
                )
                actual_in = audio_in[:, :, :, j : j + 1]  # [B, C_a, 2, 1]
                assert torch.allclose(actual_in, expected_in.expand_as(actual_in), atol=1e-6), (
                    f"Audio tick {j}: apply_model input must be hold_value at shifted sigma"
                )
                expected_pred = audio_internal_scale(audio_orig[j], sigma_audio, audio_scale)
                actual_pred = result_audio[:, :, :, j : j + 1]
                expanded_pred = expected_pred.expand_as(actual_pred)
                assert torch.allclose(actual_pred, expanded_pred, atol=1e-7), (
                    f"Audio tick {j}: prediction must be overwritten with "
                    f"audio_original at internal scale"
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
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise)

        snapshot = packed_input.clone()

        flat_zeros, _ = _fake_pack_latents(
            [
                torch.zeros(1, _C_V, _T, _HL, _WL),
                torch.zeros(1, _C_A, 2, _AUDIO_T),
            ]
        )

        wrapper = build_model_function_wrapper(
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            audio_scale_factor=1.0,
            latent_shapes=latent_shapes,
        )
        wrapper(
            lambda *a, **kw: flat_zeros.clone(),  # noqa: ARG005
            _args_dict(packed_input, sigma),
        )

        assert torch.equal(packed_input, snapshot), (
            "Wrapper must not mutate the caller's input tensor in place"
        )

    def test_temporal_dim_targeted_not_batch_dim(self, fake_comfy: types.ModuleType) -> None:
        """Regression: wrapper must write into temporal dim 2, NOT batch dim 0.

        Before the fix, video_edit[row_idx] indexed the batch dim, so row 0 of the
        batch received row 0's hold_value but other batch-0 rows were untouched.
        After the fix, video_edit[:,:,row_idx:row_idx+1,:,:] writes into the temporal
        slice for ALL batch elements.

        This test MUST fail before the wrapper indexing fix and pass after.
        """
        denoise = 0.3
        sigma = 0.7
        target_row = 1  # middle row to avoid boundary ambiguity
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
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
            latent_shapes=latent_shapes,
        )
        wrapper(mock_apply_model, _args_dict(packed_input, sigma))

        unpacked_in = _fake_unpack_latents(received[0], latent_shapes)
        video_in = unpacked_in[0]  # [1, C_v, T, Hl, Wl]

        # Confirm that the TEMPORAL slice at target_row was written, not a batch-dim slot.
        # The held value must be at [:, :, target_row, :, :] (temporal), not at [target_row, :].
        expected = hold_value(
            per_row_original[target_row], per_row_noise[target_row], sigma
        )  # [1, C_v, 1, Hl, Wl]
        actual = video_in[:, :, target_row : target_row + 1, :, :]  # [1, C_v, 1, Hl, Wl]
        assert torch.allclose(actual, expected, atol=1e-6), (
            f"Temporal row {target_row} must contain hold_value (temporal-dim targeting)"
        )

    def test_cond_batch_broadcast_both_batch_rows_receive_hold(
        self, fake_comfy: types.ModuleType
    ) -> None:
        """Cond-batch broadcast: when batch=2 (cond+uncond), held rows are written
        to ALL batch elements (not just batch[0]).

        This test MUST fail before the fix and pass after.
        """
        denoise = 0.3
        sigma = 0.7
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,  # shape [2, 1, C] because batch=2
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise, batch=2)

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
            latent_shapes=latent_shapes,
        )
        wrapper(mock_apply_model, _args_dict(packed_input, sigma))

        unpacked_in = _fake_unpack_latents(received[0], latent_shapes)
        video_in = unpacked_in[0]  # [2, C_v, T, Hl, Wl]

        for i, row_s in enumerate(schedule):
            if is_held(sigma, row_s.denoise):
                expected = hold_value(per_row_original[i], per_row_noise[i], sigma)
                for b in range(2):
                    actual = video_in[b : b + 1, :, i : i + 1, :, :]
                    assert torch.allclose(actual, expected, atol=1e-6), (
                        f"Batch element {b}, held video row {i}: must receive hold_value"
                    )

    def test_pack_unpack_round_trip_through_wrapper(self, fake_comfy: types.ModuleType) -> None:
        """Round-trip: packing then unpacking returns tensors equal to the originals.

        Verifies the fake pack/unpack geometry is consistent and that the wrapper's
        repack-for-apply_model produces a tensor that unpacks cleanly.
        """
        denoise = 0.9  # no rows held, so wrapper passes input through
        sigma = 0.5
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
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
            latent_shapes=latent_shapes,
        )
        result = wrapper(mock_apply_model, _args_dict(packed_input, sigma))

        # With no held rows (sigma < denoise), the packed result should equal packed_input.
        assert torch.allclose(result, packed_input, atol=1e-7), (
            "Non-held wrapper: result must equal input when no rows are held"
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
