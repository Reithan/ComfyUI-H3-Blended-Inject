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
    denoise_to_sigma_threshold,
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
    region: str = "hold",
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

    ``region`` sets the RowSchedule.region field for every row in the returned
    schedule.  Defaults to "hold" so existing hold-region tests work without
    modification; pass "fade" when constructing fixtures for fade-regime tests.
    """
    schedule = [
        RowSchedule(row_idx=i, denoise=denoise, inject=None, region=region) for i in range(_T)
    ]

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
    """Construct the args_dict the ComfyUI sampler passes to the wrapper.

    ComfyUI passes the raw k-diffusion sigma in [0, 1] as ``timestep``; the ×1000
    model-timestep conversion happens inside ``_apply_model``, after the wrapper has run.
    """
    return {
        "input": packed_input,
        "timestep": torch.tensor([sigma]),
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
        sigma = 0.9  # sigma > denoise_to_sigma_threshold(0.3, 12.0) ≈ 0.837 => all rows held
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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
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
        sigma = 0.9  # above denoise_to_sigma_threshold(0.3, 12.0) ≈ 0.837 => rows held
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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
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
        sigma = 0.9  # above denoise_to_sigma_threshold(0.3, 12.0) ≈ 0.837 → hold path active
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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
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
        sigma = 0.9  # above denoise_to_sigma_threshold(0.3, 12.0) ≈ 0.837 => row held
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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
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
        sigma = 0.9  # above denoise_to_sigma_threshold(0.3, 12.0) ≈ 0.837 => all rows held
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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )
        result = wrapper(mock_apply_model, _args_dict(packed_input, sigma))

        # With no held rows (sigma < denoise), the packed result should equal packed_input.
        assert torch.allclose(result, packed_input, atol=1e-7), (
            "Non-held wrapper: result must equal input when no rows are held"
        )

    def test_wrapper_accepts_target_rows_and_audio_ticks_params(
        self, fake_comfy: types.ModuleType
    ) -> None:
        """build_model_function_wrapper must accept target_rows and audio_ticks kwargs.

        FAILS before fix: TypeError (unexpected keyword argument).
        PASSES after: params accepted, full tick range gets tick_denoise entries.
        """
        denoise = 0.2
        sigma = 0.8
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise)

        wrapper = build_model_function_wrapper(
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            audio_scale_factor=1.0,
            latent_shapes=latent_shapes,
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )

        received: list[torch.Tensor] = []

        def mock_apply_model(input_x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            received.append(input_x.detach().clone())
            return input_x

        wrapper(mock_apply_model, _args_dict(packed_input, sigma))

        sigma_audio = constants.time_shift_sigma(sigma)
        unpacked_in = _fake_unpack_latents(received[0], latent_shapes)
        audio_in = unpacked_in[1]

        # All audio ticks in audio_orig should be held (full range check)
        for j in audio_orig:
            if is_held(sigma_audio, denoise):
                expected = audio_internal_scale(
                    hold_value(audio_orig[j], audio_noise[j], sigma_audio),
                    sigma_audio,
                    1.0,
                )
                actual = audio_in[:, :, :, j : j + 1]
                assert torch.allclose(actual, expected.expand_as(actual), atol=1e-6), (
                    f"Audio tick {j} must be held when target_rows/audio_ticks provided"
                )


# ---------------------------------------------------------------------------
# Regression: sigma recovery must NOT divide by 1000
# ---------------------------------------------------------------------------


class TestSigmaRecovery:
    """Regression: wrapper must recover the raw k-diffusion sigma, not sigma/1000.

    ComfyUI passes the raw sigma (e.g. 0.5) as args_dict["timestep"]; the ×1000
    model-timestep conversion happens inside _apply_model, after the wrapper runs.
    Before the fix, the wrapper divided by 1000 (sigma_video = timestep / 1000 ≈ 0.0005),
    making is_held always False for any fractional-denoise row — the hold-and-release
    mechanism silently never fired.

    FAIL before fix: wrapper computes sigma_video = 0.5/1000 = 0.0005 → is_held(0.0005,
    0.3) = False → no row held → held_video_row[:] equals the original unmodified
    sampler input, not hold_value.
    PASS after fix:  wrapper computes sigma_video = 0.5 → is_held(0.5, 0.3) = True →
    row is held → held_video_row[:] equals hold_value(original, noise, 0.5).
    """

    def test_fractional_denoise_row_is_held_with_raw_sigma(
        self, fake_comfy: types.ModuleType
    ) -> None:
        """Wrapper holds a fractional-denoise row when timestep carries raw sigma [0,1].

        Constructs args_dict with timestep=tensor([0.9]) — the raw sigma as ComfyUI
        actually sends it.  Asserts that the apply_model input has been overwritten with
        hold_value(original, noise, 0.9) for the held video row.

        sigma=0.9 is above denoise_to_sigma_threshold(0.3, 12.0) ≈ 0.837, so the row
        is held.  If the wrapper divided timestep by 1000 (pre-fix bug), sigma_video
        would be 0.0009, which is below 0.837, so the row would NOT be held.

        FAILS against pre-fix code (sigma_video = 0.9/1000 = 0.0009, not held).
        PASSES after fix (sigma_video = 0.9, held because 0.9 > threshold ≈ 0.837).
        """
        raw_sigma = 0.9
        denoise = 0.3  # threshold ≈ 0.837; 0.9 > 0.837 → row must be held

        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise)

        # Build args_dict with the raw sigma tensor — exactly what ComfyUI passes.
        raw_args = {
            "input": packed_input,
            "timestep": torch.tensor([raw_sigma]),  # raw k-diffusion sigma, NOT * 1000
            "c": {},
            "cond_or_uncond": [0],
        }

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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )
        wrapper(mock_apply_model, raw_args)

        assert len(received) == 1
        unpacked_in = _fake_unpack_latents(received[0], latent_shapes)
        video_in = unpacked_in[0]  # [B, C_v, T, Hl, Wl]

        # Every video row must be held (sigma=0.5 > denoise=0.3 for all rows).
        for i in range(_T):
            expected = hold_value(per_row_original[i], per_row_noise[i], raw_sigma)
            actual = video_in[:, :, i : i + 1, :, :]
            assert torch.allclose(actual, expected.expand_as(actual), atol=1e-6), (
                f"Video row {i}: wrapper must hold it when raw_sigma={raw_sigma} > "
                f"threshold≈0.837 (denoise={denoise}, shift=12). "
                f"Pre-fix: sigma_video=0.0009 (÷1000) → not held. "
                f"Post-fix: sigma_video=0.9 → held (0.9 > 0.837)."
            )

    def test_audio_sigma_uses_transformer_options_shift(self, fake_comfy: types.ModuleType) -> None:
        """Wrapper reads sigma shifts from transformer_options to compute sigma_audio.

        The audio sigma (sigma_audio = time_shift_sigma(sigma_video, shift_v, shift_a)) uses
        transformer_options shift_v / shift_a — this is UNCHANGED by E2.  A different shift_v
        produces a different sigma_audio, which shows up as a different hold_value written to
        the audio input when the row is held.

        Scenario: sigma_video=0.9, denoise=0.1.  The video row is always held (sigma_video=0.9
        > denoise_to_sigma_threshold(0.1, 12.0) ≈ 0.571; v_shift defaults to 12.0 when no
        model_sampling is available from the plain-function mock).  sigma_audio differs:
          - shift_v=8.0:  sigma_audio = time_shift_sigma(0.9, 8.0, 3.0) = 27/35 ≈ 0.771
          - shift_v=12.0: sigma_audio = time_shift_sigma(0.9, 12.0, 3.0) = 9/13 ≈ 0.692
        The audio hold_value (proportional to sigma_audio) differs, confirming transformer_options
        shift_v is used for sigma_audio — regardless of the E2 threshold change.

        FAILS against code that ignores transformer_options (hardcoded shift produces the same
        sigma_audio regardless of the override, so hold_value is identical in both cases).
        PASSES once the wrapper reads the override key for sigma_audio computation.

        NOTE: updated from the pre-E2 version (which tested hold/release gating at sigma_video=0.5,
        denoise=0.1).  Under E2 the audio threshold is the hybrid formula — at sigma_video=0.5 the
        video row is NOT held (sigma < v_thresh≈0.571), so the audio sigma write path is not
        reached.  The new scenario (sigma_video=0.9) exercises the same sigma_audio code path
        and isolates transformer_options influence from the threshold change.
        """
        sigma_video = 0.9
        denoise = 0.1
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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )

        # Run with transformer_options override (shift_v=8.0).
        args_with_override = {
            "input": packed_input,
            "timestep": torch.tensor([sigma_video]),
            "c": {"transformer_options": {"minimax_h3_sigma_shift_video": 8.0}},
            "cond_or_uncond": [0],
        }
        wrapper(mock_apply_model, args_with_override)

        assert len(received) == 1
        unpacked_in = _fake_unpack_latents(received[0], latent_shapes)
        audio_in_override = unpacked_in[1]  # [B, C_a, 2, audio_t]

        # Run with default shifts (no override, shift_v=12.0).
        received.clear()
        wrapper(mock_apply_model, _args_dict(packed_input, sigma_video))

        unpacked_no = _fake_unpack_latents(received[0], latent_shapes)
        audio_in_default = unpacked_no[1]

        # Tick 0 original fill = float(0+10) = 10.0; noise fill = float(0.1*(0+10)) = 1.0.
        # sigma_audio_override = time_shift_sigma(0.9, 8.0, 3.0) = 27/35 ≈ 0.77143
        # sigma_audio_default  = time_shift_sigma(0.9, 12.0, 3.0) = 9/13 ≈ 0.69231
        # hold_value(10, 1, sigma_a) = (1-sigma_a)*10 + sigma_a*1 = 10 - 9*sigma_a
        sigma_audio_override = constants.time_shift_sigma(sigma_video, 8.0, 3.0)
        sigma_audio_default = constants.time_shift_sigma(sigma_video, 12.0, 3.0)
        expected_override = 10.0 - 9.0 * sigma_audio_override  # ≈ 3.057
        expected_default = 10.0 - 9.0 * sigma_audio_default  # ≈ 3.769

        tick0_override = audio_in_override[:, :, :, 0].mean().item()
        tick0_default = audio_in_default[:, :, :, 0].mean().item()

        assert abs(tick0_override - expected_override) < 1e-4, (
            f"Tick 0 with shift_v=8.0: expected hold_value≈{expected_override:.4f} "
            f"(sigma_audio≈{sigma_audio_override:.4f}), got {tick0_override:.4f}. "
            "Wrapper must read transformer_options['minimax_h3_sigma_shift_video'] for sigma_audio."
        )
        assert abs(tick0_default - expected_default) < 1e-4, (
            f"Tick 0 with default shift_v=12.0: expected hold_value≈{expected_default:.4f} "
            f"(sigma_audio≈{sigma_audio_default:.4f}), got {tick0_default:.4f}."
        )
        assert abs(tick0_override - tick0_default) > 0.1, (
            f"Audio tick 0 value must differ between shift_v=8.0 ({tick0_override:.4f}) and "
            f"shift_v=12.0 ({tick0_default:.4f}): "
            "transformer_options shift_v must influence sigma_audio computation."
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


# ---------------------------------------------------------------------------
# Two-regime fade-row tests
# FAIL on the old wrapper (which applied hold logic to all rows); PASS after
# the region-aware wrapper is implemented.
# ---------------------------------------------------------------------------


class TestWrapperFadeRow:
    """'fade' region rows: prediction is ALWAYS blended; model INPUT is NEVER modified.

    The permanent prediction blend ``(1 - d) * original + d * model_pred`` is applied
    on every model call regardless of sigma — no is_held gate.  The model INPUT passes
    through unchanged (writing it would drain the sampler's noise budget).
    """

    def test_fade_row_prediction_blended_when_sigma_above_d(
        self, fake_comfy: types.ModuleType
    ) -> None:
        """sigma > d: fade prediction == (1-d)*original + d*model_pred.

        FAILS on old code (hold logic would overwrite prediction with original).
        PASSES after fade regime is implemented (blend is applied).
        """
        denoise = 0.4
        sigma = 0.7  # sigma > d
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise, region="fade")

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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )
        result = wrapper(
            lambda *a, **kw: flat_sentinel.clone(),  # noqa: ARG005
            _args_dict(packed_input, sigma),
        )

        unpacked_result = _fake_unpack_latents(result, latent_shapes)
        result_video = unpacked_result[0]  # [B, C_v, T, Hl, Wl]
        # model_pred is 999.0 per the sentinel; original is fill(i+1)
        model_pred_row_val = 999.0
        for i, row_s in enumerate(schedule):
            d = row_s.denoise
            orig_val = float(i + 1)
            expected_val = (1.0 - d) * orig_val + d * model_pred_row_val
            actual = result_video[:, :, i : i + 1, :, :].mean().item()
            assert abs(actual - expected_val) < 1e-4, (
                f"Fade row {i} (sigma={sigma}>d={d}): expected blend {expected_val:.4f}, "
                f"got {actual:.4f}"
            )

    def test_fade_row_prediction_blended_when_sigma_below_d(
        self, fake_comfy: types.ModuleType
    ) -> None:
        """sigma < d: fade prediction == (1-d)*original + d*model_pred — NOT released.

        This is the critical case: the old hold logic would release the row (no change)
        when sigma <= d, but the fade regime must keep blending at ALL sigmas.

        FAILS on old code (no blend applied when sigma < d).
        PASSES after fade regime (permanent blend regardless of sigma).
        """
        denoise = 0.4
        sigma = 0.2  # sigma < d — would be "released" by old hold logic
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise, region="fade")

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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )
        result = wrapper(
            lambda *a, **kw: flat_sentinel.clone(),  # noqa: ARG005
            _args_dict(packed_input, sigma),
        )

        unpacked_result = _fake_unpack_latents(result, latent_shapes)
        result_video = unpacked_result[0]
        model_pred_row_val = 999.0
        for i, row_s in enumerate(schedule):
            d = row_s.denoise
            orig_val = float(i + 1)
            expected_val = (1.0 - d) * orig_val + d * model_pred_row_val
            actual = result_video[:, :, i : i + 1, :, :].mean().item()
            assert abs(actual - expected_val) < 1e-4, (
                f"Fade row {i} (sigma={sigma}<d={d}): MUST still blend — "
                f"expected {expected_val:.4f}, got {actual:.4f}.  "
                f"Old hold logic would NOT blend (released), causing this to fail."
            )

    def test_fade_row_input_not_modified(self, fake_comfy: types.ModuleType) -> None:
        """Fade row: apply_model receives the UNMODIFIED sampler input.

        The hold regime writes hold_value to the input; the fade regime must NOT touch
        the input at all (writing it drains the sampler's noise budget).

        FAILS on old code (hold_value written to input at sigma > d).
        PASSES after fade regime (input untouched).
        """
        denoise = 0.4
        sigma = 0.7  # sigma > d → old code WOULD write hold_value to input
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise, region="fade")

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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )
        wrapper(mock_apply_model, _args_dict(packed_input, sigma))

        assert len(received) == 1
        unpacked_in = _fake_unpack_latents(received[0], latent_shapes)
        video_in = unpacked_in[0]
        unpacked_orig = _fake_unpack_latents(packed_input, latent_shapes)
        original_video = unpacked_orig[0]

        for i in range(_T):
            assert torch.allclose(
                video_in[:, :, i, :, :], original_video[:, :, i, :, :], atol=1e-7
            ), (
                f"Fade row {i}: apply_model input must be UNCHANGED — "
                f"fade regime must NOT write hold_value to the input."
            )

    @pytest.mark.parametrize("sigma", [0.1, 0.35, 0.4, 0.6, 0.9])
    def test_fade_row_prediction_blended_at_multiple_sigmas(
        self, sigma: float, fake_comfy: types.ModuleType
    ) -> None:
        """Fade prediction blend applies at every sigma, including sigma==d and sigma<d.

        Parameterized over sigmas spanning below, at, and above d=0.4 to confirm the
        blend has no sigma-dependent gate.
        """
        denoise = 0.4
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise, region="fade")

        flat_sentinel, _ = _fake_pack_latents(
            [
                torch.full((1, _C_V, _T, _HL, _WL), 888.0),
                torch.full((1, _C_A, 2, _AUDIO_T), 888.0),
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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )
        result = wrapper(
            lambda *a, **kw: flat_sentinel.clone(),  # noqa: ARG005
            _args_dict(packed_input, sigma),
        )

        unpacked_result = _fake_unpack_latents(result, latent_shapes)
        result_video = unpacked_result[0]
        model_pred_val = 888.0
        for i, row_s in enumerate(schedule):
            d = row_s.denoise
            orig_val = float(i + 1)
            expected_val = (1.0 - d) * orig_val + d * model_pred_val
            actual = result_video[:, :, i : i + 1, :, :].mean().item()
            assert abs(actual - expected_val) < 1e-4, (
                f"Fade row {i} at sigma={sigma}: expected {expected_val:.4f}, got {actual:.4f}"
            )


# ---------------------------------------------------------------------------
# Hold-row region: existing binary behavior is UNCHANGED
# ---------------------------------------------------------------------------


class TestWrapperHoldRowRegion:
    """Hold-row region: binary is-held gate is IDENTICAL to pre-change behavior.

    These tests confirm that the region='hold' path is equivalent to the old
    unconditional hold logic (which applied to all rows regardless of region).
    """

    def test_hold_row_input_written_when_sigma_above_d(self, fake_comfy: types.ModuleType) -> None:
        """Hold row at sigma > threshold: apply_model receives hold_value in the input."""
        denoise = 0.3
        sigma = 0.9  # above denoise_to_sigma_threshold(0.3, 12.0) ≈ 0.837 → held
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise, region="hold")

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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )
        wrapper(mock_apply_model, _args_dict(packed_input, sigma))

        unpacked_in = _fake_unpack_latents(received[0], latent_shapes)
        video_in = unpacked_in[0]
        for i in range(_T):
            expected = hold_value(per_row_original[i], per_row_noise[i], sigma)
            actual = video_in[:, :, i : i + 1, :, :]
            assert torch.allclose(actual, expected.expand_as(actual), atol=1e-6), (
                f"Hold row {i}: input must equal hold_value at sigma={sigma}"
            )

    def test_hold_row_input_not_written_when_sigma_below_d(
        self, fake_comfy: types.ModuleType
    ) -> None:
        """Hold row at sigma < d: apply_model receives unmodified input (released)."""
        denoise = 0.9
        sigma = 0.5  # sigma < denoise → released
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise, region="hold")

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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )
        wrapper(mock_apply_model, _args_dict(packed_input, sigma))

        unpacked_in = _fake_unpack_latents(received[0], latent_shapes)
        video_in = unpacked_in[0]
        unpacked_orig = _fake_unpack_latents(packed_input, latent_shapes)
        original_video = unpacked_orig[0]
        for i in range(_T):
            assert torch.allclose(
                video_in[:, :, i, :, :], original_video[:, :, i, :, :], atol=1e-7
            ), f"Hold row {i} released at sigma={sigma} < d={denoise}: input must be unchanged"

    def test_hold_row_prediction_overwritten_with_original_when_sigma_above_d(
        self, fake_comfy: types.ModuleType
    ) -> None:
        """Hold row at sigma > threshold: prediction row is overwritten with original."""
        denoise = 0.3
        sigma = 0.9  # above denoise_to_sigma_threshold(0.3, 12.0) ≈ 0.837 → held
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise, region="hold")

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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )
        result = wrapper(
            lambda *a, **kw: flat_sentinel.clone(),  # noqa: ARG005
            _args_dict(packed_input, sigma),
        )

        unpacked_result = _fake_unpack_latents(result, latent_shapes)
        result_video = unpacked_result[0]
        for i in range(_T):
            expected = per_row_original[i]
            actual = result_video[:, :, i : i + 1, :, :]
            assert torch.allclose(actual, expected.expand_as(actual), atol=1e-7), (
                f"Hold row {i} at sigma={sigma}: prediction must equal original"
            )

    def test_hold_row_prediction_not_overwritten_when_sigma_below_d(
        self, fake_comfy: types.ModuleType
    ) -> None:
        """Hold row released at sigma < d: prediction passes through unchanged."""
        denoise = 0.9
        sigma = 0.5
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise, region="hold")

        sentinel_val = 777.0
        flat_sentinel, _ = _fake_pack_latents(
            [
                torch.full((1, _C_V, _T, _HL, _WL), sentinel_val),
                torch.full((1, _C_A, 2, _AUDIO_T), sentinel_val),
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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )
        result = wrapper(
            lambda *a, **kw: flat_sentinel.clone(),  # noqa: ARG005
            _args_dict(packed_input, sigma),
        )

        unpacked_result = _fake_unpack_latents(result, latent_shapes)
        result_video = unpacked_result[0]
        for i in range(_T):
            actual = result_video[:, :, i, :, :].mean().item()
            assert abs(actual - sentinel_val) < 1e-6, (
                f"Hold row {i} released at sigma={sigma} < d={denoise}: "
                f"prediction must be unchanged sentinel {sentinel_val}"
            )


# ---------------------------------------------------------------------------
# Audio fade-tick tests
# ---------------------------------------------------------------------------


class TestWrapperAudioFadeTick:
    """Audio 'fade' ticks: prediction ALWAYS blended; INPUT never modified.

    Mirrors TestWrapperFadeRow for the audio stream.  The blend uses
    audio_internal_scale(original, sigma_audio, asf) for the original term.
    """

    def test_audio_fade_tick_prediction_blended(self, fake_comfy: types.ModuleType) -> None:
        """Audio fade tick: prediction = (1-d)*original*scale + d*model_pred_tick."""
        denoise = 0.4
        sigma = 0.7
        audio_scale = 2.0
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise, region="fade")

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
            audio_scale_factor=audio_scale,
            latent_shapes=latent_shapes,
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )
        result = wrapper(
            lambda *a, **kw: flat_sentinel.clone(),  # noqa: ARG005
            _args_dict(packed_input, sigma),
        )

        sigma_audio = constants.time_shift_sigma(sigma)
        unpacked_result = _fake_unpack_latents(result, latent_shapes)
        result_audio = unpacked_result[1]  # [B, C_a, 2, audio_t]

        model_pred_tick_val = 999.0
        for j in range(_AUDIO_T):
            orig_val = float(j + 10)
            original_scaled_val = orig_val * audio_scale  # audio_internal_scale
            expected_val = (1.0 - denoise) * original_scaled_val + denoise * model_pred_tick_val
            actual = result_audio[:, :, :, j : j + 1].mean().item()
            assert abs(actual - expected_val) < 1e-4, (
                f"Audio fade tick {j}: expected blend {expected_val:.4f}, got {actual:.4f}. "
                f"sigma_audio={sigma_audio:.4f}, scale={audio_scale}"
            )

    def test_audio_fade_tick_input_not_modified(self, fake_comfy: types.ModuleType) -> None:
        """Audio fade ticks: apply_model input is NOT modified (no hold_value write)."""
        denoise = 0.4
        sigma = 0.7
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise, region="fade")

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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )
        wrapper(mock_apply_model, _args_dict(packed_input, sigma))

        unpacked_in = _fake_unpack_latents(received[0], latent_shapes)
        audio_in = unpacked_in[1]
        unpacked_orig = _fake_unpack_latents(packed_input, latent_shapes)
        original_audio = unpacked_orig[1]

        for j in range(_AUDIO_T):
            assert torch.allclose(audio_in[:, :, :, j], original_audio[:, :, :, j], atol=1e-7), (
                f"Audio fade tick {j}: input must be UNCHANGED (no hold_value write for fade)"
            )

    def test_audio_fade_tick_prediction_blended_when_sigma_below_d(
        self, fake_comfy: types.ModuleType
    ) -> None:
        """Audio fade tick at sigma < d: prediction still blended (no is_held gate)."""
        denoise = 0.4
        sigma = 0.2  # sigma < d; audio sigma will also likely be < d
        audio_scale = 1.5
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise, region="fade")

        flat_sentinel, _ = _fake_pack_latents(
            [
                torch.full((1, _C_V, _T, _HL, _WL), 555.0),
                torch.full((1, _C_A, 2, _AUDIO_T), 555.0),
            ]
        )

        wrapper = build_model_function_wrapper(
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            audio_scale_factor=audio_scale,
            latent_shapes=latent_shapes,
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )
        result = wrapper(
            lambda *a, **kw: flat_sentinel.clone(),  # noqa: ARG005
            _args_dict(packed_input, sigma),
        )

        unpacked_result = _fake_unpack_latents(result, latent_shapes)
        result_audio = unpacked_result[1]

        model_pred_tick_val = 555.0
        for j in range(_AUDIO_T):
            orig_val = float(j + 10)
            original_scaled_val = orig_val * audio_scale
            expected_val = (1.0 - denoise) * original_scaled_val + denoise * model_pred_tick_val
            actual = result_audio[:, :, :, j : j + 1].mean().item()
            assert abs(actual - expected_val) < 1e-4, (
                f"Audio fade tick {j} at sigma={sigma} < d={denoise}: "
                f"MUST still blend — expected {expected_val:.4f}, got {actual:.4f}"
            )


# ---------------------------------------------------------------------------
# Fade/hold boundary continuity
# ---------------------------------------------------------------------------


class TestFadeHoldBoundary:
    """At min_denoise: the fade blend weight is (1-min_denoise); hold releases to zero.

    This confirms the two regimes are correctly separated and the boundary is
    predictable — not a regression, but a designed discontinuity.
    """

    def test_fade_blend_weight_at_min_denoise(self, fake_comfy: types.ModuleType) -> None:
        """Fade row with d==min_denoise: blend weight for original is (1-min_denoise)."""
        min_denoise = 0.3
        sigma = 0.1  # sigma < d; released under old hold logic
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=min_denoise, region="fade")

        flat_sentinel, _ = _fake_pack_latents(
            [
                torch.full((1, _C_V, _T, _HL, _WL), 888.0),
                torch.full((1, _C_A, 2, _AUDIO_T), 888.0),
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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )
        result = wrapper(
            lambda *a, **kw: flat_sentinel.clone(),  # noqa: ARG005
            _args_dict(packed_input, sigma),
        )

        unpacked_result = _fake_unpack_latents(result, latent_shapes)
        result_video = unpacked_result[0]
        for i in range(_T):
            orig_val = float(i + 1)
            model_val = 888.0
            # (1 - min_denoise) * original + min_denoise * model_pred
            expected = (1.0 - min_denoise) * orig_val + min_denoise * model_val
            actual = result_video[:, :, i : i + 1, :, :].mean().item()
            assert abs(actual - expected) < 1e-4, (
                f"Fade row {i} at d=min_denoise={min_denoise}: "
                f"blend weight must be (1-{min_denoise})={1 - min_denoise:.2f}. "
                f"Expected {expected:.4f}, got {actual:.4f}"
            )

    def test_hold_row_releases_when_sigma_below_d(self, fake_comfy: types.ModuleType) -> None:
        """Hold row at sigma < d: is_held is False, so prediction passes through unchanged.

        Uses sigma = 0.25 < d = 0.3 (strictly below, not equal) to avoid float32
        precision issues: torch.tensor([0.3]) encodes as 0.30000001... (float32),
        which would make is_held True at the equality boundary.  The semantic being
        tested — "hold releases when sigma no longer exceeds d" — is correctly captured
        by sigma < d.
        """
        min_denoise = 0.3
        sigma = 0.25  # clearly below d=0.3; is_held(0.25, 0.3) = False
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=min_denoise, region="hold")

        sentinel_val = 456.0
        flat_sentinel, _ = _fake_pack_latents(
            [
                torch.full((1, _C_V, _T, _HL, _WL), sentinel_val),
                torch.full((1, _C_A, 2, _AUDIO_T), sentinel_val),
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
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )
        result = wrapper(
            lambda *a, **kw: flat_sentinel.clone(),  # noqa: ARG005
            _args_dict(packed_input, sigma),
        )

        unpacked_result = _fake_unpack_latents(result, latent_shapes)
        result_video = unpacked_result[0]
        for i in range(_T):
            actual = result_video[:, :, i, :, :].mean().item()
            assert abs(actual - sentinel_val) < 1e-6, (
                f"Hold row {i} at sigma={sigma} < d={min_denoise}: "
                f"is_held is False → prediction must be sentinel {sentinel_val}, "
                f"got {actual:.4f}"
            )


# ---------------------------------------------------------------------------
# Device alignment regression tests
# ---------------------------------------------------------------------------


class TestWrapperDeviceAlignment:
    """Wrapper must align per-row/tick tensors to the working stream's device/dtype.

    _run_sampler builds the per_row_original / per_row_noise / audio_row_original /
    audio_row_noise dicts on CPU, then comfy.sample moves the latent to the sampling
    device (e.g. cuda) before invoking the wrapper.  Without alignment, every site that
    mixes a closure-captured CPU tensor with the runtime (cuda / meta) stream tensor
    raises RuntimeError: Expected all tensors to be on the same device.

    We reproduce the exact error class on a CPU-only box by putting the packed latent on
    torch.device("meta") while the per-row dicts stay on CPU — same device-class mismatch
    as the real cuda+cpu bug.  .to(device="meta") is a supported PyTorch operation;
    meta-device arithmetic and pack/unpack (cat/reshape/slice) all work correctly.
    """

    def test_fade_region_no_device_error(self, fake_comfy: types.ModuleType) -> None:
        """Fade-region wrapper must not raise when packed input is on meta (cpu vs meta mismatch).

        FAILS before fix: RuntimeError device mismatch at the fade blend arithmetic
        ((1-d)*per_row_original[cpu] + d*pred_row[meta] raises).
        PASSES after fix: per-row dicts aligned to meta before blend executes.
        """
        denoise = 0.4
        sigma = 0.6  # sigma > d → blend path active
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise, region="fade")

        # Move the packed working latent to meta — simulates the cuda device move that
        # comfy.sample performs before invoking the wrapper.  Per-row dicts stay on CPU.
        packed_meta = packed_input.to(device="meta")

        def meta_apply_model(input_x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            # Return a meta prediction of the correct packed shape.
            return torch.zeros(input_x.shape, device="meta", dtype=input_x.dtype)

        wrapper = build_model_function_wrapper(
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            audio_scale_factor=1.0,
            latent_shapes=latent_shapes,
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )

        # Must not raise after the fix; raises RuntimeError before the fix.
        result = wrapper(meta_apply_model, _args_dict(packed_meta, sigma))
        assert result.device.type == "meta", (
            f"Result must be on meta device (matches working stream), got {result.device}"
        )

    def test_hold_region_no_device_error(self, fake_comfy: types.ModuleType) -> None:
        """Hold-region wrapper must not raise when packed input is on meta (cpu vs meta mismatch).

        FAILS before fix: RuntimeError at the hold input-write (hold_value computed on
        CPU original/noise then assigned into a meta video_edit slice).
        PASSES after fix: per-row dicts aligned to meta before hold logic executes.
        """
        denoise = 0.3
        sigma = 0.9  # above threshold ≈ 0.837 → is_held True → hold input-write active
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise, region="hold")

        packed_meta = packed_input.to(device="meta")

        def meta_apply_model(input_x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            return torch.zeros(input_x.shape, device="meta", dtype=input_x.dtype)

        wrapper = build_model_function_wrapper(
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            audio_scale_factor=1.0,
            latent_shapes=latent_shapes,
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )

        result = wrapper(meta_apply_model, _args_dict(packed_meta, sigma))
        assert result.device.type == "meta", (
            f"Result must be on meta device (matches working stream), got {result.device}"
        )


# ---------------------------------------------------------------------------
# Regression tests for threshold-fix (task #29):
# map user d (schedule position) to shifted sigma threshold.
# ---------------------------------------------------------------------------
# All tests in this class are FAIL-THEN-PASS verified:
#   - FAIL on the pre-fix code where the wrapper compared sigma directly to raw d.
#   - PASS on the post-fix code where the wrapper uses denoise_to_sigma_threshold(d, shift).
# ---------------------------------------------------------------------------


class TestDenoiseThresholdFix:
    """Regression tests for the hold-release threshold fix (task #29).

    Pre-fix: ``is_held(sigma, d)`` compared raw sigma against raw user denoise ``d``.
    Post-fix: threshold = ``denoise_to_sigma_threshold(d, shift)`` — the shifted sigma at
    schedule position ``t=d`` — is used, so video and audio both release at ``t=d``.
    """

    # -----------------------------------------------------------------------
    # (a) Numeric value + is_held gating witness for d=0.6, shift=12
    # -----------------------------------------------------------------------

    def test_threshold_value_d06_shift12(self) -> None:
        """denoise_to_sigma_threshold(0.6, 12.0) == 12*0.6/(1+11*0.6) == 7.2/7.6 ≈ 0.9474.

        Pre-fix this would compare sigma against raw d=0.6; at sigma=0.8 the row was HELD
        (0.8 > 0.6).  Post-fix the threshold is ≈0.9474; at sigma=0.8 the row is RELEASED
        (0.8 < 0.9474) — so the frame gets denoised rather than frozen.

        FAILS pre-fix: denoise_to_sigma_threshold does not exist.
        PASSES post-fix: function exists and returns the correct value.
        """
        expected = 12.0 * 0.6 / (1.0 + 11.0 * 0.6)  # 7.2 / 7.6 ≈ 0.94736842...
        thresh = denoise_to_sigma_threshold(0.6, 12.0)
        assert abs(thresh - expected) < 1e-9, (
            f"denoise_to_sigma_threshold(0.6, 12.0): expected {expected:.9f}, got {thresh:.9f}"
        )

    def test_is_held_gating_witness_d06_shift12(self) -> None:
        """Behavior-change witness for d=0.6, shift=12.0 at sigma=0.8.

        Pre-fix: is_held(0.8, raw_d=0.6) = True  — row HELD (frozen, no denoising).
        Post-fix: is_held(0.8, thresh≈0.947) = False — row RELEASED (denoising allowed).

        This is the exact symptom reported: a single-frame inject at min_denoise=0.6
        came out identical to the input because the row was held for ~89% of the schedule.

        FAILS pre-fix: no denoise_to_sigma_threshold; raw-d comparison gives True for sigma=0.8.
        PASSES post-fix: threshold≈0.947 > 0.8 — row released at sigma=0.8.
        """
        d = 0.6
        sigma = 0.8
        thresh = denoise_to_sigma_threshold(d, 12.0)
        # Post-fix: sigma=0.8 is BELOW the threshold -> released
        assert not is_held(sigma, thresh), (
            f"Post-fix: is_held({sigma}, thresh={thresh:.4f}) must be False "
            f"(sigma < threshold -> row released, denoising allowed)"
        )
        # Verify the pre-fix behavior that caused the bug (direct d comparison)
        assert is_held(sigma, d), (
            f"Pre-fix witness: is_held({sigma}, raw_d={d}) = True "
            f"(row incorrectly held at sigma=0.8 under old direct-d comparison)"
        )

    # -----------------------------------------------------------------------
    # (b) AV sync: both thresholds map back to the same schedule position t=d
    # -----------------------------------------------------------------------

    def test_av_sync_thresholds_recover_same_schedule_position(self) -> None:
        """For any d, both video and audio thresholds correspond to the same t=d.

        The analytic inverse of time_snr_shift is t = sigma / (shift + sigma*(1-shift)).
        Applying this inverse to each stream's threshold must recover t=d exactly,
        proving both streams release at the same schedule position regardless of shift.

        This is a pure-math test (no wrapper or tensors) that confirms AV sync.

        FAILS pre-fix: denoise_to_sigma_threshold does not exist.
        PASSES post-fix: inverse maps both thresholds back to d within floating-point tolerance.
        """
        for d in [0.1, 0.3, 0.6, 0.9]:
            shift_v, shift_a = 12.0, 3.0
            thresh_v = denoise_to_sigma_threshold(d, shift_v)
            thresh_a = denoise_to_sigma_threshold(d, shift_a)
            # Analytic inverse: t = sigma / (shift + sigma*(1 - shift))
            t_from_v = thresh_v / (shift_v + thresh_v * (1.0 - shift_v))
            t_from_a = thresh_a / (shift_a + thresh_a * (1.0 - shift_a))
            assert abs(t_from_v - d) < 1e-9, f"d={d}: video inverse {t_from_v:.12f} != d={d:.12f}"
            assert abs(t_from_a - d) < 1e-9, f"d={d}: audio inverse {t_from_a:.12f} != d={d:.12f}"

    # -----------------------------------------------------------------------
    # (c) Boundary conditions
    # -----------------------------------------------------------------------

    def test_boundary_d0_returns_zero(self) -> None:
        """d=0 -> threshold=0 for all shifts (0-denoise row is trivially always released)."""
        for shift in [1.0, 3.0, 12.0]:
            assert denoise_to_sigma_threshold(0.0, shift) == 0.0, (
                f"shift={shift}: d=0.0 must return 0.0"
            )

    def test_boundary_d1_returns_one(self) -> None:
        """d=1 -> threshold=1.0 for all shifts (full-denoise row is held for entire schedule)."""
        for shift in [1.0, 3.0, 12.0]:
            thresh = denoise_to_sigma_threshold(1.0, shift)
            assert abs(thresh - 1.0) < 1e-9, f"shift={shift}: d=1.0 must return 1.0, got {thresh}"

    def test_shift_one_returns_d_unchanged(self) -> None:
        """shift=1.0 (unshifted schedule) returns d unchanged — identity guard."""
        for d in [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]:
            result = denoise_to_sigma_threshold(d, 1.0)
            assert result == d, f"shift=1.0: must return d={d} unchanged, got {result}"

    # -----------------------------------------------------------------------
    # (d) End-to-end wrapper behavior-change test (sigma=0.8, d=0.6)
    # -----------------------------------------------------------------------

    def test_wrapper_hold_row_released_at_sigma_08_d06(self, fake_comfy: types.ModuleType) -> None:
        """End-to-end wrapper: hold row with d=0.6 is RELEASED at sigma=0.8 post-fix.

        Behavior-change witness:
          Pre-fix: is_held(0.8, raw_d=0.6) = True -> hold_value written to input;
          prediction overwritten with original -> frame frozen (reported bug symptom).
          Post-fix: is_held(0.8, thresh=0.947) = False -> input untouched; prediction
          untouched -> frame denoised normally.

        FAILS pre-fix: video input contains hold_value (not the sampler input) and
        video prediction contains per_row_original (not the sentinel 999.0).
        PASSES post-fix: input unchanged; prediction passes through as sentinel 999.
        """
        denoise = 0.6
        sigma = 0.8  # witness: pre-fix held (0.8 > 0.6); post-fix released (0.8 < 0.947)
        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=denoise, region="hold")

        sentinel_val = 999.0
        flat_sentinel, _ = _fake_pack_latents(
            [
                torch.full((1, _C_V, _T, _HL, _WL), sentinel_val),
                torch.full((1, _C_A, 2, _AUDIO_T), sentinel_val),
            ]
        )

        received: list[torch.Tensor] = []

        def mock_apply_model(input_x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            received.append(input_x.detach().clone())
            return flat_sentinel.clone()

        wrapper = build_model_function_wrapper(
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            audio_scale_factor=1.0,
            latent_shapes=latent_shapes,
            target_rows=_T,
            audio_ticks=_AUDIO_T,
            # default_shift_video=12.0 -> threshold = 12*0.6/(1+11*0.6) = 7.2/7.6 ≈ 0.947
        )
        result = wrapper(mock_apply_model, _args_dict(packed_input, sigma))

        assert len(received) == 1

        # Post-fix: rows are RELEASED at sigma=0.8 (below threshold ≈ 0.947).
        # INPUT: wrapper must NOT write hold_value (input passes through unmodified).
        unpacked_in = _fake_unpack_latents(received[0], latent_shapes)
        video_in = unpacked_in[0]
        unpacked_orig = _fake_unpack_latents(packed_input, latent_shapes)
        original_video = unpacked_orig[0]

        for i in range(_T):
            assert torch.allclose(
                video_in[:, :, i, :, :], original_video[:, :, i, :, :], atol=1e-7
            ), (
                f"Post-fix: hold row {i} released at sigma=0.8 < threshold≈0.947 "
                f"(d=0.6, shift=12.0) — input must be UNCHANGED. "
                f"Pre-fix: is_held(0.8, raw_d=0.6)=True would have written hold_value."
            )

        # Post-fix: PREDICTION must also pass through unchanged (no overwrite).
        # The mock returns sentinel=999.0; post-fix this is left as-is.
        unpacked_result = _fake_unpack_latents(result, latent_shapes)
        result_video = unpacked_result[0]

        for i in range(_T):
            actual = result_video[:, :, i, :, :].mean().item()
            assert abs(actual - sentinel_val) < 1e-6, (
                f"Post-fix: hold row {i} released — prediction must be sentinel {sentinel_val}, "
                f"got {actual:.4f}. "
                f"Pre-fix: prediction would have been overwritten with per_row_original."
            )


# ---------------------------------------------------------------------------
# Mock apply_model with __self__.model_sampling.shift for knob-divergence tests
# ---------------------------------------------------------------------------


class _ApplyModelMock:
    """Mock apply_model callable exposing ``__self__.model_sampling.shift``.

    Simulates a ComfyUI bound method where ``getattr(am, '__self__')`` returns a
    BaseModel-like namespace whose ``model_sampling.shift`` is the given float.
    Each call appends the input tensor to ``received`` and returns it unchanged.
    """

    def __init__(self, model_sampling_shift: float) -> None:
        self.__self__ = types.SimpleNamespace(
            model_sampling=types.SimpleNamespace(shift=model_sampling_shift)
        )
        self.received: list[torch.Tensor] = []

    def __call__(self, input_x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self.received.append(input_x.detach().clone())
        return input_x


# ---------------------------------------------------------------------------
# TestKnobDivergence: model_sampling.shift ≠ transformer_options shift_v
# ---------------------------------------------------------------------------


class TestKnobDivergence:
    """model_sampling.shift ≠ transformer_options shift_v: thresholds track the right source.

    E2 (plan.v2): the video release threshold uses model_sampling.shift (the shift used to
    build the sampler's sigma schedule via ModelSamplingAV), not transformer_options shift_v.
    The audio threshold is the hybrid: time_shift_sigma(denoise_to_sigma_threshold(d, v_shift),
    shift_v, shift_a) — pushed through the same DiT chain as sigma_audio.

    At defaults (model_sampling.shift == transformer_options shift_v == 12.0) both formulas
    reduce to the pre-E2 values, so all existing tests stay green.  This class exercises
    the divergent case: model_sampling.shift=8.0, transformer_options shift_v=12.0.

    Shared boundary arithmetic:
      d = 0.5,  model_sampling.shift = 8.0,  topts shift_v = 12.0, shift_a = 3.0
      v_thresh_new = denoise_to_sigma_threshold(0.5, 8.0) = 8/9  ≈ 0.889  (model_sampling)
      v_thresh_old = denoise_to_sigma_threshold(0.5, 12.0) = 12/13 ≈ 0.923  (transformer_options)
      sigma_video  = 0.91  (between the two thresholds)
        new: HELD (0.91 > 8/9 ≈ 0.889)   old: NOT HELD (0.91 < 12/13 ≈ 0.923)

      sigma_audio  = time_shift_sigma(0.91, 12.0, 3.0) = 91/127 ≈ 0.717
      a_thresh_new = time_shift_sigma(8/9,  12.0, 3.0) = 2/3     ≈ 0.667
      a_thresh_old = denoise_to_sigma_threshold(0.5, 3.0)          = 0.75
        new: HELD (0.717 > 2/3)            old: NOT HELD (0.717 < 0.75)

    All tests verified fail-then-pass (pre-E2 code → FAIL; post-E2 code → PASS).
    """

    _MS_SHIFT: float = 8.0  # model_sampling.shift
    _TOPTS_SHIFT_V: float = 12.0  # transformer_options shift_v
    _TOPTS_SHIFT_A: float = 3.0  # transformer_options shift_a
    _D: float = 0.5
    _SIGMA_VIDEO: float = 0.91

    def _args_with_topts(self, packed_input: torch.Tensor) -> dict[str, Any]:
        """Build args_dict with explicit transformer_options shifts and the divergent sigma."""
        return {
            "input": packed_input,
            "timestep": torch.tensor([self._SIGMA_VIDEO]),
            "c": {
                "transformer_options": {
                    "minimax_h3_sigma_shift_video": self._TOPTS_SHIFT_V,
                    "minimax_h3_sigma_shift_audio": self._TOPTS_SHIFT_A,
                }
            },
            "cond_or_uncond": [0],
        }

    def test_video_threshold_tracks_model_sampling_shift(
        self, fake_comfy: types.ModuleType
    ) -> None:
        """Video row is held when model_sampling.shift=8.0 gives thresh≈0.889 < sigma_video=0.91.

        With transformer_options shift_v=12.0 the old threshold would be ≈0.923 > 0.91 → NOT held.
        The new code re-points the video threshold to model_sampling.shift=8.0 → HELD.

        Semantic assertion (not call expression): v_thresh == denoise_to_sigma_threshold(d, 8.0)
        and v_thresh != denoise_to_sigma_threshold(d, 12.0) for d=0.5.

        FAILS pre-E2: threshold = denoise_to_sigma_threshold(0.5, shift_v=12.0) ≈ 0.923;
        0.91 < 0.923 → NOT held → input passes through unmodified.
        PASSES post-E2: threshold = denoise_to_sigma_threshold(0.5, model_sampling.shift=8.0)
        ≈ 0.889; 0.91 > 0.889 → held → input = hold_value.
        """
        # Verify the boundary arithmetic this test relies on.
        v_thresh_new = denoise_to_sigma_threshold(self._D, self._MS_SHIFT)
        v_thresh_old = denoise_to_sigma_threshold(self._D, self._TOPTS_SHIFT_V)
        assert self._SIGMA_VIDEO > v_thresh_new, "sigma_video must exceed the model_sampling thresh"
        assert self._SIGMA_VIDEO < v_thresh_old, "sigma_video must be below the topts thresh"

        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=self._D, region="hold")

        mock_am = _ApplyModelMock(model_sampling_shift=self._MS_SHIFT)

        wrapper = build_model_function_wrapper(
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            audio_scale_factor=1.0,
            latent_shapes=latent_shapes,
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )
        wrapper(mock_am, self._args_with_topts(packed_input))

        assert len(mock_am.received) == 1
        unpacked_in = _fake_unpack_latents(mock_am.received[0], latent_shapes)
        video_in = unpacked_in[0]  # [B, C_v, T, Hl, Wl]

        # Post-E2: rows are HELD (model_sampling.shift=8.0 → v_thresh≈0.889 < sigma_video=0.91).
        # The input must equal hold_value(original, noise, sigma_video) for each held row.
        for i in range(_T):
            expected = hold_value(per_row_original[i], per_row_noise[i], self._SIGMA_VIDEO)
            actual = video_in[:, :, i : i + 1, :, :]
            assert torch.allclose(actual, expected.expand_as(actual), atol=1e-6), (
                f"Video row {i}: must be HELD (input == hold_value) at "
                f"sigma_video={self._SIGMA_VIDEO} using "
                f"model_sampling.shift={self._MS_SHIFT} (v_thresh_new≈{v_thresh_new:.4f}). "
                f"Pre-E2 uses transformer_options shift_v={self._TOPTS_SHIFT_V} "
                f"(v_thresh_old≈{v_thresh_old:.4f} > sigma_video) → NOT held."
            )

    def test_audio_threshold_uses_hybrid_formula(self, fake_comfy: types.ModuleType) -> None:
        """Audio tick is held via the hybrid threshold (not denoise_to_sigma_threshold(d, shift_a)).

        sigma_audio = time_shift_sigma(0.91, 12.0, 3.0) = 91/127 ≈ 0.717.
        a_thresh_new = time_shift_sigma(8/9, 12.0, 3.0) = 2/3 ≈ 0.667  (hybrid, new)
        a_thresh_old = denoise_to_sigma_threshold(0.5, 3.0) = 0.75       (pre-E2)

        Semantic: a_thresh == time_shift_sigma(denoise_to_sigma_threshold(d, ms_shift), sv, sa)
        and a_thresh != denoise_to_sigma_threshold(d, shift_a).

        FAILS pre-E2: a_thresh = 0.75; sigma_audio=0.717 < 0.75 → NOT held → input unchanged.
        PASSES post-E2: a_thresh = 2/3 ≈ 0.667; sigma_audio=0.717 > 0.667 → HELD → input modified.
        """
        sigma_audio = constants.time_shift_sigma(
            self._SIGMA_VIDEO, self._TOPTS_SHIFT_V, self._TOPTS_SHIFT_A
        )
        a_thresh_new = constants.time_shift_sigma(
            denoise_to_sigma_threshold(self._D, self._MS_SHIFT),
            self._TOPTS_SHIFT_V,
            self._TOPTS_SHIFT_A,
        )
        a_thresh_old = denoise_to_sigma_threshold(self._D, self._TOPTS_SHIFT_A)

        # Verify the boundary arithmetic this test relies on.
        assert sigma_audio > a_thresh_new, (
            f"sigma_audio={sigma_audio:.4f} must exceed new hybrid thresh={a_thresh_new:.4f}"
        )
        assert sigma_audio < a_thresh_old, (
            f"sigma_audio={sigma_audio:.4f} must be below old thresh={a_thresh_old:.4f}"
        )

        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=self._D, region="hold")

        mock_am = _ApplyModelMock(model_sampling_shift=self._MS_SHIFT)

        wrapper = build_model_function_wrapper(
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            audio_scale_factor=1.0,
            latent_shapes=latent_shapes,
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )
        wrapper(mock_am, self._args_with_topts(packed_input))

        assert len(mock_am.received) == 1
        unpacked_in = _fake_unpack_latents(mock_am.received[0], latent_shapes)
        audio_in = unpacked_in[1]  # [B, C_a, 2, audio_t]
        unpacked_orig = _fake_unpack_latents(packed_input, latent_shapes)
        original_audio = unpacked_orig[1]

        # Tick 0 (owned by row 0 with d=0.5, original fill=10.0, noise fill=1.0) must be held.
        # is_held(sigma_audio≈0.717, a_thresh_new≈0.667) = True
        tick0_original = original_audio[:, :, :, 0].mean().item()
        tick0_held = audio_in[:, :, :, 0].mean().item()

        # Input must differ from the background fill (hold_value ≠ original).
        assert abs(tick0_held - tick0_original) > 0.1, (
            f"Audio tick 0 must be HELD (sigma_audio≈{sigma_audio:.4f} > "
            f"a_thresh_new≈{a_thresh_new:.4f}). "
            f"Pre-E2: a_thresh_old={a_thresh_old:.3f} > sigma_audio → NOT held. "
            f"Got input value {tick0_held:.4f} vs original {tick0_original:.4f}."
        )

        # Verify the held value matches hold_value at the correct sigma_audio.
        # Tick 0: original=10.0, noise=1.0 → hold_value = (1-sigma_a)*10 + sigma_a*1
        expected_held = (1.0 - sigma_audio) * 10.0 + sigma_audio * 1.0
        assert abs(tick0_held - expected_held) < 1e-4, (
            f"Audio tick 0 hold_value mismatch: expected {expected_held:.4f} "
            f"(sigma_audio≈{sigma_audio:.4f}), got {tick0_held:.4f}."
        )

    def test_d_equals_one_row_never_held(self, fake_comfy: types.ModuleType) -> None:
        """d=1.0 → v_thresh=1.0 → is_held(sigma, 1.0)=False for all sigma ≤ 1.0 → no-op.

        Boundary condition: denoise_to_sigma_threshold(1.0, any_shift) = 1.0 by the formula
        shift*1/(1+(shift-1)*1) = shift/shift = 1.  Since the sampler's sigmas never exceed
        1.0, is_held is always False → wrapper is a strict no-op.  Holds for any shift value.
        """
        sigma_video = 0.999
        d = 1.0

        # Verify the boundary formula.
        v_thresh = denoise_to_sigma_threshold(d, self._MS_SHIFT)
        assert abs(v_thresh - 1.0) < 1e-9, (
            f"denoise_to_sigma_threshold(1.0, {self._MS_SHIFT}) must equal 1.0, got {v_thresh}"
        )
        assert not is_held(sigma_video, v_thresh), (
            f"is_held({sigma_video}, v_thresh=1.0) must be False (strict inequality)"
        )

        (
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            packed_input,
            latent_shapes,
        ) = _make_wrapper_fixtures(denoise=d, region="hold")

        snapshot = packed_input.clone()
        mock_am = _ApplyModelMock(model_sampling_shift=self._MS_SHIFT)

        wrapper = build_model_function_wrapper(
            schedule,
            per_row_original,
            per_row_noise,
            audio_orig,
            audio_noise,
            audio_scale_factor=1.0,
            latent_shapes=latent_shapes,
            target_rows=_T,
            audio_ticks=_AUDIO_T,
        )
        wrapper(
            mock_am,
            {
                "input": packed_input,
                "timestep": torch.tensor([sigma_video]),
                "c": {},
                "cond_or_uncond": [0],
            },
        )

        # The wrapper must not modify the caller's input tensor (aliasing-safe).
        assert torch.equal(packed_input, snapshot), (
            "d=1.0: wrapper must not mutate the input tensor (v_thresh=1.0, never held)"
        )

        # apply_model received the unmodified input (no hold_value written for any row).
        assert len(mock_am.received) == 1
        unpacked_in = _fake_unpack_latents(mock_am.received[0], latent_shapes)
        video_in = unpacked_in[0]
        unpacked_orig = _fake_unpack_latents(packed_input, latent_shapes)
        original_video = unpacked_orig[0]

        for i in range(_T):
            assert torch.allclose(
                video_in[:, :, i, :, :], original_video[:, :, i, :, :], atol=1e-7
            ), f"d=1.0: video row {i} must pass through to apply_model unmodified (never held)"
