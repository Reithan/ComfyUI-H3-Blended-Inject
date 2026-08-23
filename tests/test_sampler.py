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
