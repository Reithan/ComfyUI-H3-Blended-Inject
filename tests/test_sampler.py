"""Tests for comfyui_h3_blended_inject.sampler — per-row img2img pure functions.

These cover the CPU-testable pieces of the per-row img2img redesign:
  - per_row_init_lerp: the initial-noise lerp x = m*x + (1-m)*clean
  - build_per_row_sampler_function: wraps a base k-diffusion sampler, applies the
    lerp to x on entry, optionally injects a per-row-scaling noise_sampler
  - sampler_accepts_noise_sampler: signature probe
  - make_per_row_noise_sampler: scales injected noise per-row by m

The model_function_wrapper (conditioning injection) is covered separately once the
_denoise_mask_values pooling contract is pinned.
"""

from __future__ import annotations

from typing import Any

import torch
from hypothesis import given
from hypothesis import strategies as st

from comfyui_h3_blended_inject.sampler import (
    build_conditioning_wrapper,
    build_per_row_sampler_function,
    make_per_row_noise_sampler,
    per_row_init_lerp,
    sampler_accepts_noise_sampler,
)

frac_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# per_row_init_lerp
# ---------------------------------------------------------------------------


class TestPerRowInitLerp:
    """x_out = m * x + (1 - m) * clean, per row."""

    @given(rows=st.integers(1, 6), cols=st.integers(1, 6), m=frac_st)
    def test_formula_scalar_m(self, rows: int, cols: int, m: float) -> None:
        """Matches the hand-computed blend for a uniform m."""
        torch.manual_seed(3)
        x = torch.randn(1, rows, cols)
        clean = torch.randn(1, rows, cols)
        m_t = torch.full((1, rows, 1), m)
        expected = m * x + (1.0 - m) * clean
        result = per_row_init_lerp(x, m_t, clean)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_endpoint_m_zero_returns_clean(self) -> None:
        """m=0 collapses every row to the clean reference."""
        x = torch.randn(1, 4, 3)
        clean = torch.randn(1, 4, 3)
        m_t = torch.zeros(1, 4, 1)
        assert torch.allclose(per_row_init_lerp(x, m_t, clean), clean, atol=1e-7)

    def test_endpoint_m_one_returns_x(self) -> None:
        """m=1 leaves the fully-noised x untouched."""
        x = torch.randn(1, 4, 3)
        clean = torch.randn(1, 4, 3)
        m_t = torch.ones(1, 4, 1)
        assert torch.allclose(per_row_init_lerp(x, m_t, clean), x, atol=1e-7)

    def test_per_row_mixed_m(self) -> None:
        """A per-row m vector applies independently to each row."""
        x = torch.ones(1, 3, 2)
        clean = torch.zeros(1, 3, 2)
        m_t = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 3, 1)
        result = per_row_init_lerp(x, m_t, clean)
        assert torch.allclose(result[0, 0], torch.zeros(2))
        assert torch.allclose(result[0, 1], torch.full((2,), 0.5))
        assert torch.allclose(result[0, 2], torch.ones(2))

    def test_aligns_m_and_clean_dtype_to_x(self) -> None:
        """m and clean are cast to x's dtype so mixed-precision inputs do not raise."""
        x = torch.randn(1, 2, 2, dtype=torch.float32)
        clean = torch.randn(1, 2, 2, dtype=torch.float64)
        m_t = torch.ones(1, 2, 1, dtype=torch.float64)
        result = per_row_init_lerp(x, m_t, clean)
        assert result.dtype == x.dtype


# ---------------------------------------------------------------------------
# sampler_accepts_noise_sampler
# ---------------------------------------------------------------------------


class TestSamplerAcceptsNoiseSampler:
    def test_true_when_param_present(self) -> None:
        def f(model, x, sigmas, extra_args=None, noise_sampler=None):  # noqa: ANN001, ANN202
            return x

        assert sampler_accepts_noise_sampler(f) is True

    def test_false_when_absent(self) -> None:
        def f(model, x, sigmas, extra_args=None, s_churn=0.0):  # noqa: ANN001, ANN202
            return x

        assert sampler_accepts_noise_sampler(f) is False

    def test_false_on_var_keyword_only(self) -> None:
        """A bare **kwargs does not count as explicit noise_sampler support."""

        def f(model, x, sigmas, **kwargs):  # noqa: ANN001, ANN202
            return x

        assert sampler_accepts_noise_sampler(f) is False


# ---------------------------------------------------------------------------
# make_per_row_noise_sampler
# ---------------------------------------------------------------------------


class TestMakePerRowNoiseSampler:
    def test_scales_base_output_by_m(self) -> None:
        """Injected noise is scaled per-row by m (so m*sigma_up*noise is correct)."""
        base = lambda s, sn: torch.ones(1, 3, 2)  # noqa: E731
        m_t = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 3, 1)
        ns = make_per_row_noise_sampler(base, m_t)
        out = ns(1.0, 0.5)
        assert torch.allclose(out[0, 0], torch.zeros(2))
        assert torch.allclose(out[0, 1], torch.full((2,), 0.5))
        assert torch.allclose(out[0, 2], torch.ones(2))


# ---------------------------------------------------------------------------
# build_per_row_sampler_function
# ---------------------------------------------------------------------------


class _RecordingBase:
    """A fake base sampler_function recording the x and kwargs it receives."""

    def __init__(self) -> None:
        self.received_x: Any = None
        self.received_kwargs: dict[str, Any] = {}
        self.sentinel = object()

    def __call__(
        self,
        model: Any,
        x: Any,
        sigmas: Any,
        extra_args: Any = None,
        callback: Any = None,
        disable: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.received_x = x
        self.received_kwargs = {
            "extra_args": extra_args,
            "callback": callback,
            "disable": disable,
            **kwargs,
        }
        return self.sentinel


class TestBuildPerRowSamplerFunction:
    def test_applies_lerp_before_base(self) -> None:
        """The base sampler receives the lerp'd x, not the raw noised x."""
        base = _RecordingBase()
        x = torch.ones(1, 3, 2)
        clean = torch.zeros(1, 3, 2)
        m_packed = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 3, 1)
        fn = build_per_row_sampler_function(base, m_packed, clean)
        fn(object(), x, torch.tensor([1.0, 0.0]))
        expected = m_packed * x + (1.0 - m_packed) * clean
        assert torch.allclose(base.received_x, expected, atol=1e-6)

    def test_returns_base_result_and_forwards_args(self) -> None:
        """Return value and extra_args/callback/disable are forwarded verbatim."""
        base = _RecordingBase()
        m_packed = torch.ones(1, 2, 1)
        fn = build_per_row_sampler_function(base, m_packed, torch.zeros(1, 2, 1))
        cb = object()
        ea = {"seed": 5}
        result = fn(
            object(),
            torch.randn(1, 2, 1),
            torch.tensor([1.0, 0.0]),
            extra_args=ea,
            callback=cb,
            disable=True,
        )
        assert result is base.sentinel
        assert base.received_kwargs["extra_args"] is ea
        assert base.received_kwargs["callback"] is cb
        assert base.received_kwargs["disable"] is True

    def test_no_noise_sampler_injected_when_flag_off(self) -> None:
        """With stochastic scaling off, no noise_sampler is added to kwargs."""
        base = _RecordingBase()
        m_packed = torch.ones(1, 2, 1)
        fn = build_per_row_sampler_function(base, m_packed, torch.zeros(1, 2, 1))
        fn(object(), torch.randn(1, 2, 1), torch.tensor([1.0, 0.0]))
        assert "noise_sampler" not in base.received_kwargs

    def test_injects_scaling_noise_sampler_when_enabled(self) -> None:
        """When enabled, a per-row-scaling noise_sampler is built from x and injected."""
        base = _RecordingBase()
        m_packed = torch.tensor([0.0, 1.0]).reshape(1, 2, 1)
        # factory mimics comfy default_noise_sampler(x): returns a callable (s, sn) -> ones_like(x)
        factory = lambda x, seed=None: lambda s, sn: torch.ones_like(x)  # noqa: E731
        fn = build_per_row_sampler_function(
            base,
            m_packed,
            torch.zeros(1, 2, 1),
            scale_stochastic_noise=True,
            noise_sampler_factory=factory,
        )
        fn(object(), torch.randn(1, 2, 1), torch.tensor([1.0, 0.0]), extra_args={"seed": 1})
        ns = base.received_kwargs.get("noise_sampler")
        assert ns is not None
        out = ns(1.0, 0.5)
        # row 0 scaled by 0, row 1 by 1
        assert torch.allclose(out[0, 0], torch.zeros(1))
        assert torch.allclose(out[0, 1], torch.ones(1))

    def test_does_not_override_existing_noise_sampler(self) -> None:
        """A caller-supplied noise_sampler is left untouched."""
        base = _RecordingBase()
        m_packed = torch.ones(1, 2, 1)
        supplied = object()
        factory = lambda x, seed=None: lambda s, sn: torch.ones_like(x)  # noqa: E731
        fn = build_per_row_sampler_function(
            base,
            m_packed,
            torch.zeros(1, 2, 1),
            scale_stochastic_noise=True,
            noise_sampler_factory=factory,
        )
        fn(object(), torch.randn(1, 2, 1), torch.tensor([1.0, 0.0]), noise_sampler=supplied)
        assert base.received_kwargs.get("noise_sampler") is supplied


# ---------------------------------------------------------------------------
# build_conditioning_wrapper
# ---------------------------------------------------------------------------


class _RecordingApplyModel:
    """Fake bound apply_model recording positional input/timestep and kwargs."""

    def __init__(self) -> None:
        self.input: Any = None
        self.timestep: Any = None
        self.kwargs: dict[str, Any] = {}
        self.sentinel = object()

    def __call__(self, input_: Any, timestep: Any, **kwargs: Any) -> Any:
        self.input = input_
        self.timestep = timestep
        self.kwargs = kwargs
        return self.sentinel


class TestBuildConditioningWrapper:
    def _args(self, input_: torch.Tensor, c: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "input": input_,
            "timestep": torch.tensor([0.5]),
            "c": {"transformer_options": {}} if c is None else c,
            "cond_or_uncond": [0],
        }

    def test_injects_pooled_conds_as_named_keys(self) -> None:
        """Each pooled cond is placed into c under its own key (not transformer_options)."""
        pooled = {"denoise_mask": torch.zeros(1, 1, 3), "audio_denoise_mask": torch.zeros(1, 1, 2)}
        wrapper = build_conditioning_wrapper(pooled)
        am = _RecordingApplyModel()
        wrapper(am, self._args(torch.randn(1, 4, 3)))
        assert "denoise_mask" in am.kwargs
        assert "audio_denoise_mask" in am.kwargs
        # not smuggled into transformer_options
        assert "denoise_mask" not in am.kwargs.get("transformer_options", {})

    def test_forwards_input_and_timestep_and_returns_result(self) -> None:
        pooled = {"denoise_mask": torch.zeros(1, 1, 3)}
        wrapper = build_conditioning_wrapper(pooled)
        am = _RecordingApplyModel()
        inp = torch.randn(1, 4, 3)
        args = self._args(inp)
        result = wrapper(am, args)
        assert result is am.sentinel
        assert am.input is inp
        assert am.timestep is args["timestep"]

    def test_preserves_existing_c_keys(self) -> None:
        pooled = {"denoise_mask": torch.zeros(1, 1, 3)}
        wrapper = build_conditioning_wrapper(pooled)
        am = _RecordingApplyModel()
        topts = {"foo": 1}
        wrapper(am, self._args(torch.randn(1, 4, 3), c={"transformer_options": topts}))
        assert am.kwargs["transformer_options"] is topts

    def test_does_not_mutate_original_c(self) -> None:
        pooled = {"denoise_mask": torch.zeros(1, 1, 3)}
        wrapper = build_conditioning_wrapper(pooled)
        am = _RecordingApplyModel()
        c = {"transformer_options": {}}
        wrapper(am, self._args(torch.randn(1, 4, 3), c=c))
        assert "denoise_mask" not in c

    def test_aligns_dtype_to_input(self) -> None:
        pooled = {"denoise_mask": torch.zeros(1, 1, 3, dtype=torch.float32)}
        wrapper = build_conditioning_wrapper(pooled)
        am = _RecordingApplyModel()
        wrapper(am, self._args(torch.randn(1, 4, 3, dtype=torch.float16)))
        assert am.kwargs["denoise_mask"].dtype == torch.float16

    def test_empty_pooled_forwards_unchanged(self) -> None:
        wrapper = build_conditioning_wrapper({})
        am = _RecordingApplyModel()
        wrapper(am, self._args(torch.randn(1, 4, 3)))
        assert "denoise_mask" not in am.kwargs
        assert "audio_denoise_mask" not in am.kwargs


class _ConstApplyModel:
    """Fake bound apply_model that returns a fixed denoised tensor regardless of input."""

    def __init__(self, denoised: torch.Tensor) -> None:
        self._denoised = denoised

    def __call__(self, input_: Any, timestep: Any, **kwargs: Any) -> torch.Tensor:
        return self._denoised


class TestConditioningWrapperDenoisedCorrection:
    """The wrapper must blend the denoised toward the input by the per-row fraction m.

    Regression for the "noise runs in reverse / low-denoise rows decode as grey static" bug:
    H3's process_timestep compresses only the network timestep, but calculate_denoised still
    divides by the outer sigma, so without ``corrected = m*denoised + (1-m)*input`` the sampler
    integrates every row over the full global interval and low-m rows blow up.
    """

    def _args(self, input_: torch.Tensor) -> dict[str, Any]:
        return {
            "input": input_,
            "timestep": torch.tensor([0.5]),
            "c": {"transformer_options": {}},
            "cond_or_uncond": [0],
        }

    def test_correction_blends_denoised_toward_input(self) -> None:
        inp = torch.randn(1, 4, 3)
        denoised = torch.randn(1, 4, 3)
        m = torch.full((1, 4, 3), 0.25)
        wrapper = build_conditioning_wrapper({}, m)
        out = wrapper(_ConstApplyModel(denoised), self._args(inp))
        expected = m * denoised + (1.0 - m) * inp
        assert torch.allclose(out, expected)

    def test_m_one_returns_denoised_unchanged(self) -> None:
        """m == 1 rows (full generation) pass the raw denoised straight through."""
        inp = torch.randn(1, 4, 3)
        denoised = torch.randn(1, 4, 3)
        m = torch.ones(1, 4, 3)
        wrapper = build_conditioning_wrapper({}, m)
        out = wrapper(_ConstApplyModel(denoised), self._args(inp))
        assert torch.allclose(out, denoised)

    def test_m_zero_freezes_row_at_input(self) -> None:
        """m == 0 rows (preserve) return the input, so the sampler's d = (x-out)/sigma == 0."""
        inp = torch.randn(1, 4, 3)
        denoised = torch.randn(1, 4, 3)
        m = torch.zeros(1, 4, 3)
        wrapper = build_conditioning_wrapper({}, m)
        out = wrapper(_ConstApplyModel(denoised), self._args(inp))
        assert torch.allclose(out, inp)

    def test_per_row_m_applies_independently(self) -> None:
        """Different m per row → each row blended by its own fraction."""
        inp = torch.zeros(1, 1, 3)
        denoised = torch.ones(1, 1, 3)
        m = torch.tensor([[[0.0, 0.5, 1.0]]])
        wrapper = build_conditioning_wrapper({}, m)
        out = wrapper(_ConstApplyModel(denoised), self._args(inp))
        # out = m*1 + (1-m)*0 = m
        assert torch.allclose(out, m)

    def test_correction_aligns_m_dtype_to_denoised(self) -> None:
        inp = torch.randn(1, 4, 3, dtype=torch.float16)
        denoised = torch.randn(1, 4, 3, dtype=torch.float16)
        m = torch.full((1, 4, 3), 0.5, dtype=torch.float32)
        wrapper = build_conditioning_wrapper({}, m)
        out = wrapper(_ConstApplyModel(denoised), self._args(inp))
        assert out.dtype == torch.float16
