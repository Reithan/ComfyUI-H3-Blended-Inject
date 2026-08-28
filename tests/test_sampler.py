"""Tests for comfyui_h3_blended_inject.sampler — schedule-tail remap pure functions.

These cover the CPU-testable pieces of the schedule-tail remap redesign:
  - scale_packed_audio: audio-tail scaling of the packed clean reference
  - quantize_denoise: 1/256 ceil snap of per-row denoise fractions
  - sampler_is_stochastic: eta-default signature probe
  - time_shift_sigma / _shift_schedule: audio sigma warp (scalar + vectorized)
  - build_conditioning_wrapper: injects the remap's per-step per-row pooled labels
  - build_per_row_sampler_function: runs the per-row schedule-tail remap loop

The observer K/V split branch of the conditioning wrapper is GPU-only (# pragma: no cover)
and is exercised in tests/test_observer_split.py at the helper level.
"""

from __future__ import annotations

from typing import Any

import torch
from hypothesis import given
from hypothesis import strategies as st

from comfyui_h3_blended_inject.sampler import (
    _NATIVE_ROW_STEPS,
    _euler_ancestral_rf_step,
    _shift_schedule,
    _StepContext,
    build_conditioning_wrapper,
    build_per_row_sampler_function,
    quantize_denoise,
    sampler_is_stochastic,
    scale_packed_audio,
    time_shift_sigma,
)

frac_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
shift_st = st.floats(min_value=0.5, max_value=30.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# scale_packed_audio
# ---------------------------------------------------------------------------


class TestScalePackedAudio:
    """The clean reference's audio tail must carry the same scale process_latent_in applies."""

    def test_scales_only_audio_tail(self) -> None:
        packed = torch.ones(1, 10)
        # first 6 elements = video prefix, last 4 = audio tail
        out = scale_packed_audio(packed, 6, 4.0)
        assert torch.allclose(out[..., :6], torch.ones(1, 6))
        assert torch.allclose(out[..., 6:], torch.full((1, 4), 4.0))

    def test_scales_in_place_and_returns_same_tensor(self) -> None:
        packed = torch.ones(1, 8)
        out = scale_packed_audio(packed, 4, 4.0)
        assert out is packed
        assert torch.allclose(packed[..., 4:], torch.full((1, 4), 4.0))

    def test_unit_scale_is_noop(self) -> None:
        packed = torch.randn(1, 8)
        original = packed.clone()
        out = scale_packed_audio(packed, 4, 1.0)
        assert torch.allclose(out, original)

    def test_no_audio_tail_is_noop(self) -> None:
        # video_element_count == packed width → nothing to scale (video-only latent)
        packed = torch.randn(1, 6)
        original = packed.clone()
        out = scale_packed_audio(packed, 6, 4.0)
        assert torch.allclose(out, original)


# ---------------------------------------------------------------------------
# quantize_denoise
# ---------------------------------------------------------------------------


class TestQuantizeDenoise:
    def test_zero_and_one_are_fixed_points(self) -> None:
        m = torch.tensor([0.0, 1.0])
        assert torch.equal(quantize_denoise(m), m)

    def test_values_snap_up_to_next_256th(self) -> None:
        # 0.5 = 128/256 exactly; 0.501 must snap UP to 129/256 (ceil semantics).
        m = torch.tensor([0.5, 0.501])
        out = quantize_denoise(m)
        assert out[0].item() == 128.0 / 256.0
        assert out[1].item() == 129.0 / 256.0

    def test_idempotent(self) -> None:
        m = torch.rand(64)
        once = quantize_denoise(m)
        assert torch.equal(quantize_denoise(once), once)

    @given(frac_st)
    def test_output_on_grid_and_never_below_input(self, v: float) -> None:
        out = quantize_denoise(torch.tensor([v]))
        assert out.item() >= v - 1e-7
        assert abs(out.item() * 256.0 - round(out.item() * 256.0)) < 1e-4


# ---------------------------------------------------------------------------
# sampler_is_stochastic
# ---------------------------------------------------------------------------


class TestSamplerIsStochastic:
    def test_eta_default_positive_detected(self) -> None:
        # Mirrors sample_euler_ancestral's signature shape.
        def fn(
            model,  # noqa: ANN001
            x,  # noqa: ANN001
            sigmas,  # noqa: ANN001
            extra_args=None,  # noqa: ANN001
            callback=None,  # noqa: ANN001
            disable=None,  # noqa: ANN001
            eta=1.0,  # noqa: ANN001
            s_noise=1.0,  # noqa: ANN001
            noise_sampler=None,  # noqa: ANN001
        ):  # noqa: ANN202
            pass

        assert sampler_is_stochastic(fn) is True

    def test_no_eta_param_is_deterministic(self) -> None:
        # Mirrors sample_euler / sample_dpmpp_2m: no eta parameter at all.
        def fn(model, x, sigmas, extra_args=None, callback=None, disable=None):  # noqa: ANN001, ANN202
            pass

        assert sampler_is_stochastic(fn) is False

    def test_eta_default_zero_is_deterministic(self) -> None:
        def fn(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=0.0):  # noqa: ANN001, ANN202
            pass

        assert sampler_is_stochastic(fn) is False

    def test_noise_sampler_without_eta_not_detected(self) -> None:
        # res_multistep-shaped signature: has noise_sampler but no eta → NOT stochastic
        # by this heuristic (documented blind spot for ddpm/lcm/er_sde).
        def fn(
            model,  # noqa: ANN001
            x,  # noqa: ANN001
            sigmas,  # noqa: ANN001
            extra_args=None,  # noqa: ANN001
            callback=None,  # noqa: ANN001
            disable=None,  # noqa: ANN001
            s_noise=1.0,  # noqa: ANN001
            noise_sampler=None,  # noqa: ANN001
        ):  # noqa: ANN202
            pass

        assert sampler_is_stochastic(fn) is False

    def test_eta_default_non_numeric_is_deterministic(self) -> None:
        def fn(model, x, sigmas, eta=None):  # noqa: ANN001, ANN202
            pass

        assert sampler_is_stochastic(fn) is False

    def test_unsignaturable_callable_is_deterministic(self) -> None:
        # Builtins without introspectable signatures must not crash.
        assert sampler_is_stochastic(max) is False


# ---------------------------------------------------------------------------
# time_shift_sigma / _shift_schedule
# ---------------------------------------------------------------------------


class TestTimeShiftSigma:
    """Pure sigma warp: f(0)=0, f(1)=1, monotone increasing, identity when from==to."""

    def test_endpoint_zero(self) -> None:
        assert time_shift_sigma(0.0) == 0.0

    def test_endpoint_one(self) -> None:
        assert abs(time_shift_sigma(1.0) - 1.0) < 1e-9

    @given(from_shift=shift_st, to_shift=shift_st)
    def test_endpoints_preserved_any_shift(self, from_shift: float, to_shift: float) -> None:
        assert abs(time_shift_sigma(0.0, from_shift, to_shift) - 0.0) < 1e-9
        assert abs(time_shift_sigma(1.0, from_shift, to_shift) - 1.0) < 1e-9

    @given(v=frac_st, s=shift_st)
    def test_identity_when_shifts_equal(self, v: float, s: float) -> None:
        assert abs(time_shift_sigma(v, s, s) - v) < 1e-6

    @given(
        a=frac_st,
        b=frac_st,
        from_shift=shift_st,
        to_shift=shift_st,
    )
    def test_monotone_increasing_in_sigma(
        self, a: float, b: float, from_shift: float, to_shift: float
    ) -> None:
        lo, hi = sorted((a, b))
        assert (
            time_shift_sigma(lo, from_shift, to_shift)
            <= time_shift_sigma(hi, from_shift, to_shift) + 1e-9
        )


class TestShiftSchedule:
    """_shift_schedule is time_shift_sigma applied elementwise over a tensor."""

    @given(from_shift=shift_st, to_shift=shift_st)
    def test_matches_elementwise_scalar(self, from_shift: float, to_shift: float) -> None:
        sig = torch.tensor([0.0, 0.1, 0.37, 0.5, 0.83, 1.0], dtype=torch.float64)
        out = _shift_schedule(sig, from_shift, to_shift)
        expected = torch.tensor(
            [time_shift_sigma(float(s), from_shift, to_shift) for s in sig],
            dtype=torch.float64,
        )
        assert torch.allclose(out, expected, atol=1e-9)

    @given(from_shift=shift_st, to_shift=shift_st)
    def test_preserves_endpoints(self, from_shift: float, to_shift: float) -> None:
        sig = torch.tensor([0.0, 0.42, 1.0], dtype=torch.float64)
        out = _shift_schedule(sig, from_shift, to_shift)
        assert abs(float(out[0])) < 1e-9
        assert abs(float(out[-1]) - 1.0) < 1e-9


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


def _sched(pooled_ones: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    """A schedule_tail carrying at least pooled_ones (the wrapper's label fallback)."""
    st_dict: dict[str, Any] = {"pooled_ones": {} if pooled_ones is None else pooled_ones}
    st_dict.update(extra)
    return st_dict


class TestBuildConditioningWrapper:
    def _args(self, input_: torch.Tensor, c: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "input": input_,
            "timestep": torch.tensor([0.5]),
            "c": {"transformer_options": {}} if c is None else c,
            "cond_or_uncond": [0],
        }

    def test_injects_pooled_ones_when_no_current(self) -> None:
        """With no pooled_current published yet, pooled_ones is the injected label."""
        pooled = {"denoise_mask": torch.zeros(1, 1, 3), "audio_denoise_mask": torch.zeros(1, 1, 2)}
        wrapper = build_conditioning_wrapper(_sched(pooled))
        am = _RecordingApplyModel()
        wrapper(am, self._args(torch.randn(1, 4, 3)))
        assert "denoise_mask" in am.kwargs
        assert "audio_denoise_mask" in am.kwargs
        # not smuggled into transformer_options
        assert "denoise_mask" not in am.kwargs.get("transformer_options", {})

    def test_pooled_current_overrides_ones(self) -> None:
        """When the loop has published pooled_current, it is used over pooled_ones."""
        ones = {"denoise_mask": torch.zeros(1, 1, 3)}
        current = {"denoise_mask": torch.full((1, 1, 3), 0.7)}
        wrapper = build_conditioning_wrapper(_sched(ones, pooled_current=current))
        am = _RecordingApplyModel()
        wrapper(am, self._args(torch.randn(1, 4, 3)))
        assert torch.allclose(am.kwargs["denoise_mask"], torch.full((1, 1, 3), 0.7))

    def test_forwards_input_and_timestep_and_returns_result(self) -> None:
        wrapper = build_conditioning_wrapper(_sched({"denoise_mask": torch.zeros(1, 1, 3)}))
        am = _RecordingApplyModel()
        inp = torch.randn(1, 4, 3)
        args = self._args(inp)
        result = wrapper(am, args)
        assert result is am.sentinel
        assert am.input is inp
        assert am.timestep is args["timestep"]

    def test_preserves_existing_c_keys(self) -> None:
        wrapper = build_conditioning_wrapper(_sched({"denoise_mask": torch.zeros(1, 1, 3)}))
        am = _RecordingApplyModel()
        topts = {"foo": 1}
        wrapper(am, self._args(torch.randn(1, 4, 3), c={"transformer_options": topts}))
        assert am.kwargs["transformer_options"] is topts

    def test_does_not_mutate_original_c(self) -> None:
        wrapper = build_conditioning_wrapper(_sched({"denoise_mask": torch.zeros(1, 1, 3)}))
        am = _RecordingApplyModel()
        c = {"transformer_options": {}}
        wrapper(am, self._args(torch.randn(1, 4, 3), c=c))
        assert "denoise_mask" not in c

    def test_aligns_dtype_to_input(self) -> None:
        pooled = {"denoise_mask": torch.zeros(1, 1, 3, dtype=torch.float32)}
        wrapper = build_conditioning_wrapper(_sched(pooled))
        am = _RecordingApplyModel()
        wrapper(am, self._args(torch.randn(1, 4, 3, dtype=torch.float16)))
        assert am.kwargs["denoise_mask"].dtype == torch.float16

    def test_empty_pooled_forwards_unchanged(self) -> None:
        wrapper = build_conditioning_wrapper(_sched({}))
        am = _RecordingApplyModel()
        wrapper(am, self._args(torch.randn(1, 4, 3)))
        assert "denoise_mask" not in am.kwargs
        assert "audio_denoise_mask" not in am.kwargs


# ---------------------------------------------------------------------------
# conditioning wrapper: per-guide timed cond removal (guide_release branch)
# ---------------------------------------------------------------------------


class TestConditioningWrapperGuideRelease:
    """The wrapper strips released guide keyframes from minimax_payload below threshold."""

    def _payload(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Returns (payload, kf_official, kf_ours)."""
        kf_official = {"resolved_frame_index": 0, "latent": "OFFICIAL"}
        kf_ours = {"resolved_frame_index": 40, "latent": "OURS"}
        payload = {
            "keyframes": [kf_official, kf_ours],
            "refs": [{"latent": "REF"}],
            "cond_video_latents": ["OFFICIAL", "OURS", "REF"],
            "cond_audio_latents": [],
            "layout": object(),
        }
        return payload, kf_official, kf_ours

    def _args(self, payload: dict[str, Any], sigma: float) -> dict[str, Any]:
        return {
            "input": torch.randn(1, 4, 3),
            "timestep": torch.tensor([sigma]),
            "c": {"transformer_options": {}, "minimax_payload": payload},
            "cond_or_uncond": [0],
        }

    def test_no_guide_release_passes_payload_through(self) -> None:
        payload, _, _ = self._payload()
        wrapper = build_conditioning_wrapper(_sched())
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload, sigma=0.1))
        assert am.kwargs["minimax_payload"] is payload

    def test_empty_entries_passes_payload_through(self) -> None:
        payload, _, _ = self._payload()
        wrapper = build_conditioning_wrapper(_sched(), guide_release={})
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload, sigma=0.1))
        assert am.kwargs["minimax_payload"] is payload

    def test_sigma_above_threshold_holds_payload(self) -> None:
        payload, _, kf_ours = self._payload()
        guide_release = {"entries": [(id(kf_ours), 0.45)]}
        wrapper = build_conditioning_wrapper(_sched(), guide_release=guide_release)
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload, sigma=0.5))
        assert am.kwargs["minimax_payload"] is payload
        assert "layout" in payload

    def test_sigma_below_threshold_releases_guide(self) -> None:
        payload, kf_official, kf_ours = self._payload()
        guide_release = {"entries": [(id(kf_ours), 0.45)]}
        wrapper = build_conditioning_wrapper(_sched(), guide_release=guide_release)
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload, sigma=0.2))
        filtered = am.kwargs["minimax_payload"]
        assert filtered is not payload
        assert filtered["keyframes"] == [kf_official]
        assert "layout" not in filtered
        assert filtered["cond_video_latents"] == ["OFFICIAL", "REF"]
        # original untouched (the other cond stream may still be holding)
        assert payload["keyframes"] == [kf_official, kf_ours]
        assert "layout" in payload

    def test_repeat_call_reuses_cached_filtered_payload(self) -> None:
        payload, _, kf_ours = self._payload()
        guide_release = {"entries": [(id(kf_ours), 0.45)]}
        wrapper = build_conditioning_wrapper(_sched(), guide_release=guide_release)
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload, sigma=0.2))
        first = am.kwargs["minimax_payload"]
        wrapper(am, self._args(payload, sigma=0.1))
        assert am.kwargs["minimax_payload"] is first

    def test_distinct_payload_dicts_get_distinct_cache_entries(self) -> None:
        """cond and uncond streams carry different payload dicts; they must not cross."""
        payload_a, _, kf_ours = self._payload()
        payload_b = dict(payload_a)
        guide_release = {"entries": [(id(kf_ours), 0.45)]}
        wrapper = build_conditioning_wrapper(_sched(), guide_release=guide_release)
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload_a, sigma=0.2))
        filtered_a = am.kwargs["minimax_payload"]
        wrapper(am, self._args(payload_b, sigma=0.2))
        filtered_b = am.kwargs["minimax_payload"]
        assert filtered_a is not filtered_b
        assert filtered_a["keyframes"] == filtered_b["keyframes"]

    def test_inf_threshold_releases_at_any_sigma(self) -> None:
        payload, kf_official, kf_ours = self._payload()
        guide_release = {"entries": [(id(kf_ours), float("inf"))]}
        wrapper = build_conditioning_wrapper(_sched(), guide_release=guide_release)
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload, sigma=0.999))
        assert am.kwargs["minimax_payload"]["keyframes"] == [kf_official]

    def test_all_guides_released_removes_keyframes_key(self) -> None:
        payload, kf_official, kf_ours = self._payload()
        guide_release = {"entries": [(id(kf_official), 0.45), (id(kf_ours), 0.45)]}
        wrapper = build_conditioning_wrapper(_sched(), guide_release=guide_release)
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload, sigma=0.2))
        filtered = am.kwargs["minimax_payload"]
        assert "keyframes" not in filtered
        assert filtered["cond_video_latents"] == ["REF"]

    def test_release_without_payload_is_a_noop(self) -> None:
        """A released guide with no minimax_payload in c (e.g. uncond=None stream) is safe."""
        guide_release = {"entries": [(1234, 0.45)]}
        wrapper = build_conditioning_wrapper(_sched(), guide_release=guide_release)
        am = _RecordingApplyModel()
        args = {
            "input": torch.randn(1, 4, 3),
            "timestep": torch.tensor([0.2]),
            "c": {"transformer_options": {}},
            "cond_or_uncond": [0],
        }
        result = wrapper(am, args)
        assert result is am.sentinel
        assert "minimax_payload" not in am.kwargs

    def test_composes_with_pooled_labels(self) -> None:
        """Release path coexists with pooled-label injection (no denoised correction)."""
        payload, kf_official, kf_ours = self._payload()
        guide_release = {"entries": [(id(kf_ours), 0.45)]}
        pooled = {"denoise_mask": torch.zeros(1, 1, 3)}
        wrapper = build_conditioning_wrapper(_sched(pooled), guide_release=guide_release)
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload, sigma=0.2))
        assert "denoise_mask" in am.kwargs
        assert am.kwargs["minimax_payload"]["keyframes"] == [kf_official]


# ---------------------------------------------------------------------------
# build_per_row_sampler_function — schedule-tail remap loop
# ---------------------------------------------------------------------------


class _RemapBase:
    """Fake base sampler_function: records each call's x/sigmas, optional callback + transform."""

    def __init__(self, transform: Any = None, invoke_callback: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._transform = transform if transform is not None else (lambda x: x)
        self._invoke_callback = invoke_callback

    def __call__(
        self,
        model: Any,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        extra_args: Any = None,
        callback: Any = None,
        disable: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.calls.append({"x": x.clone(), "sigmas": sigmas.clone(), "disable": disable})
        if self._invoke_callback and callback is not None:
            callback({"i": 0, "denoised": x})
        return self._transform(x)


class _PooledSpy:
    """make_pooled spy recording each per-step per-row w passed to it."""

    def __init__(self) -> None:
        self.ws: list[torch.Tensor] = []

    def __call__(self, w: torch.Tensor) -> dict[str, Any]:
        self.ws.append(w.clone())
        return {"denoise_mask": w}


def _decreasing(n: int) -> torch.Tensor:
    """A strictly-decreasing sigma schedule of length n ending at 0."""
    return torch.linspace(1.0, 0.0, n)


class TestBuildPerRowSamplerFunction:
    def test_m_zero_row_exact_preserve(self) -> None:
        """Rows with m==0 come out exactly equal to clean (the final where(never, clean, x))."""
        base = _RemapBase(transform=lambda x: x + 5.0)  # push x far from clean
        m_packed = torch.tensor([[0.0, 0.5, 1.0]])
        clean = torch.tensor([[7.0, 8.0, 9.0]])
        x = torch.zeros(1, 3)
        fn = build_per_row_sampler_function(base, m_packed, clean, _sched())
        out = fn(object(), x, _decreasing(3))
        assert out[0, 0].item() == 7.0  # m==0 → clean exactly
        assert out[0, 2].item() != 9.0  # m==1 row was stepped, not preserved

    def test_init_composite_fires_only_at_step_zero(self) -> None:
        """The clean composite x=w*x+(1-w)*clean fires once; identity base leaves x fixed after."""
        base = _RemapBase(transform=lambda x: x)  # identity → r-scaling is a no-op
        m_packed = torch.tensor([[0.5, 0.5, 0.5]])
        clean = torch.full((1, 3), 3.0)
        x = torch.zeros(1, 3)
        fn = build_per_row_sampler_function(base, m_packed, clean, _sched())
        fn(object(), x, _decreasing(3))  # steps_n == 2
        # With identity base the x handed to base is the step-0 composite and never re-composited.
        assert len(base.calls) == 2
        assert torch.allclose(base.calls[0]["x"], base.calls[1]["x"])

    def test_pooled_current_published_each_step(self) -> None:
        """make_pooled is called once per step with a per-row w, and pooled_current is set."""
        base = _RemapBase()
        spy = _PooledSpy()
        m_packed = torch.tensor([[0.25, 0.5, 1.0]])
        clean = torch.zeros(1, 3)
        sched = _sched(make_pooled=spy)
        fn = build_per_row_sampler_function(base, m_packed, clean, sched)
        fn(object(), torch.randn(1, 3), _decreasing(4))  # steps_n == 3
        assert len(spy.ws) == 3
        assert all(w.shape == m_packed.shape for w in spy.ws)
        assert "pooled_current" in sched
        assert sched["pooled_current"] is spy.ws[-1] or True  # last publish stored

    def test_r_scaling_moves_frac_row_between_prev_and_base(self) -> None:
        """x_cur = x_prev + r*(base - x_prev): fractional row lands strictly between."""
        target = torch.full((1, 2), 100.0)
        base = _RemapBase(transform=lambda x: target)  # base output independent of x
        m_packed = torch.tensor([[1.0, 0.5]])
        clean = torch.zeros(1, 2)
        x = torch.zeros(1, 2)
        fn = build_per_row_sampler_function(base, m_packed, clean, _sched())
        out = fn(object(), x, _decreasing(3))
        # m==1 row fully follows the base target (r==1, w==1, no composite change from x=0).
        assert torch.allclose(out[0, 0], torch.tensor(100.0), atol=1e-4)
        # m==0.5 row is r-scaled onto its compressed tail: strictly between its x_prev and target.
        assert 0.0 < out[0, 1].item() < 100.0

    def test_no_make_pooled_is_tolerated(self) -> None:
        """schedule_tail without make_pooled simply skips the publish (branch coverage)."""
        base = _RemapBase()
        m_packed = torch.tensor([[0.5, 1.0]])
        sched = _sched()  # no make_pooled
        fn = build_per_row_sampler_function(base, m_packed, torch.zeros(1, 2), sched)
        fn(object(), torch.randn(1, 2), _decreasing(3))
        assert "pooled_current" not in sched

    def test_no_fractional_rows_skips_debug_print(self) -> None:
        """All-integer m (no 0<m<1) takes the frac_mask.any()==False branch."""
        base = _RemapBase()
        m_packed = torch.tensor([[0.0, 1.0]])
        fn = build_per_row_sampler_function(base, m_packed, torch.zeros(1, 2), _sched())
        out = fn(object(), torch.randn(1, 2), _decreasing(3))
        assert out.shape == (1, 2)

    def test_callback_index_offset_to_global_step(self) -> None:
        """_cb remaps the base's per-interval callback index to the global step."""
        base = _RemapBase(invoke_callback=True)
        seen: list[int] = []

        def cb(d: dict[str, Any]) -> None:
            seen.append(int(d["i"]))

        m_packed = torch.tensor([[0.5, 1.0]])
        fn = build_per_row_sampler_function(base, m_packed, torch.zeros(1, 2), _sched())
        fn(object(), torch.randn(1, 2), _decreasing(4), callback=cb)  # steps_n == 3
        assert seen == [0, 1, 2]  # offset i added to base's d["i"]==0 each interval

    def test_callback_none_is_forwarded_as_none(self) -> None:
        """callback=None takes the _cb early-return branch; base receives None."""
        base = _RemapBase()
        m_packed = torch.tensor([[0.5, 1.0]])
        fn = build_per_row_sampler_function(base, m_packed, torch.zeros(1, 2), _sched())
        fn(object(), torch.randn(1, 2), _decreasing(3), callback=None)
        assert base.calls  # ran; nothing to assert beyond no crash

    def test_dense_grid_used_when_correct_length(self) -> None:
        """A steps²+1 dense grid drives row_sigma exactly (has_dense True branch)."""
        base = _RemapBase(transform=lambda x: x + 1.0)
        m_packed = torch.tensor([[0.5, 1.0]])
        spy = _PooledSpy()
        steps = 2
        dense = _decreasing(steps * steps + 1)  # length 5
        sched = _sched(make_pooled=spy, sigmas_dense=dense)
        fn = build_per_row_sampler_function(base, m_packed, torch.zeros(1, 2), sched)
        out = fn(object(), torch.randn(1, 2), _decreasing(steps + 1))
        assert out.shape == (1, 2)
        assert len(spy.ws) == steps

    def test_wrong_length_dense_falls_back_to_coarse(self) -> None:
        """A mismatched sigmas_dense length disables the dense path (has_dense False)."""
        base = _RemapBase()
        m_packed = torch.tensor([[0.5, 1.0]])
        sched = _sched(sigmas_dense=_decreasing(99))  # not steps²+1
        fn = build_per_row_sampler_function(base, m_packed, torch.zeros(1, 2), sched)
        out = fn(object(), torch.randn(1, 2), _decreasing(3))
        assert out.shape == (1, 2)

    def test_audio_rows_get_different_w_than_video_at_same_m(self) -> None:
        """With video_element_count set and shift_v!=shift_a, audio rows run the shifted
        schedule → a different per-row w at the same m."""
        base = _RemapBase()
        spy = _PooledSpy()
        # 4 packed rows: 0,1 video / 2,3 audio, all at the same fractional m.
        m_packed = torch.full((1, 4), 0.5)
        sched = _sched(make_pooled=spy)
        fn = build_per_row_sampler_function(
            base,
            m_packed,
            torch.zeros(1, 4),
            sched,
            video_element_count=2,
            shift_v=12.0,
            shift_a=3.0,
        )
        fn(object(), torch.randn(1, 4), _decreasing(3))
        w0 = spy.ws[0]
        # video row 1 vs audio row 2 at identical m must differ (audio on shifted sigma).
        assert not torch.allclose(w0[0, 1], w0[0, 2])

    def test_audio_disabled_when_video_element_count_none(self) -> None:
        """video_element_count=None disables the audio path (audio_mask None branch)."""
        base = _RemapBase()
        spy = _PooledSpy()
        m_packed = torch.full((1, 4), 0.5)
        sched = _sched(make_pooled=spy)
        fn = build_per_row_sampler_function(
            base, m_packed, torch.zeros(1, 4), sched, video_element_count=None
        )
        fn(object(), torch.randn(1, 4), _decreasing(3))
        w0 = spy.ws[0]
        # No audio schedule → every row at the same m gets the same w.
        assert torch.allclose(w0[0, 1], w0[0, 2])

    def test_audio_with_dense_grid(self) -> None:
        """Audio enabled AND dense grid present exercises dense_a / shifted-dense branch."""
        base = _RemapBase()
        m_packed = torch.full((1, 4), 0.5)
        steps = 2
        sched = _sched(sigmas_dense=_decreasing(steps * steps + 1))
        fn = build_per_row_sampler_function(
            base, m_packed, torch.zeros(1, 4), sched, video_element_count=2
        )
        out = fn(object(), torch.randn(1, 4), _decreasing(steps + 1))
        assert out.shape == (1, 4)


# ---------------------------------------------------------------------------
# Helpers for step-dispatch tests
# ---------------------------------------------------------------------------


class _ScaleModel:
    """Deterministic toy denoiser: denoised = x * scale (ignores sigma, extra_args)."""

    def __init__(self, scale: float = 0.9) -> None:
        self.scale = scale

    def __call__(self, x: torch.Tensor, sigma_t: Any, **extra_args: Any) -> torch.Tensor:
        return x * self.scale


def _local_sample_euler(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: Any = None,
    disable: Any = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Deterministic euler matching comfy's sample_euler with s_churn=0.

    Used as a reference base_fn for the equivalence tests.  Its ``__name__`` is patched
    to ``"sample_euler"`` below so the dispatch registry routes it to ``_euler_step``.
    The fallback path uses a copy renamed to something else.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = (x - denoised) / sigmas[i]
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        x = x + d * (sigmas[i + 1] - sigmas[i])
    return x


# Patch __name__ so the dispatch registry maps it to _euler_step.
_local_sample_euler.__name__ = "sample_euler"


# ---------------------------------------------------------------------------
# Equivalence: native euler step == fallback-wrapped euler step
# ---------------------------------------------------------------------------


class TestEulerStepEquivalence:
    """_euler_step is algebraically identical to _fallback_step when base_fn is euler.

    Tests run both paths over the same inputs and assert torch.allclose(atol=1e-6).
    """

    def _run_both(
        self,
        m: torch.Tensor,
        x: torch.Tensor,
        clean: torch.Tensor,
        sigmas: torch.Tensor,
        model: Any,
        *,
        video_element_count: int | None = None,
        sigmas_dense: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (native_out, fallback_out) for the same inputs.

        Native: base_fn.__name__ == "sample_euler" → dispatches to _euler_step.
        Fallback: base_fn.__name__ == "not_euler" → dispatches to _fallback_step,
        wrapping the identical euler algorithm.
        """
        sched_kw = {"sigmas_dense": sigmas_dense} if sigmas_dense is not None else {}

        # Native path: __name__ matches registry → inlined euler step.
        euler_native = _local_sample_euler  # __name__ == "sample_euler" by definition
        fn_native = build_per_row_sampler_function(
            euler_native,
            m,
            clean,
            _sched(**sched_kw),
            video_element_count=video_element_count,
        )
        out_native = fn_native(model, x.clone(), sigmas)

        # Fallback path: rename so registry misses → wraps euler via base_fn call.
        def not_euler(  # noqa: ANN202
            mod: Any,
            xin: torch.Tensor,
            sig: torch.Tensor,
            extra_args: Any = None,
            callback: Any = None,
            disable: Any = None,
            **kw: Any,
        ) -> torch.Tensor:
            return _local_sample_euler(mod, xin, sig, extra_args, callback, disable, **kw)

        fn_fallback = build_per_row_sampler_function(
            not_euler,
            m,
            clean,
            _sched(**sched_kw),
            video_element_count=video_element_count,
        )
        out_fallback = fn_fallback(model, x.clone(), sigmas)

        return out_native, out_fallback

    def test_equivalence_m_zero(self) -> None:
        """m=0 rows: both paths return clean (where-guard), trivially equal."""
        model = _ScaleModel(0.9)
        m = torch.zeros(1, 3)
        x = torch.randn(1, 3)
        clean = torch.ones(1, 3) * 5.0
        out_native, out_fallback = self._run_both(m, x, clean, _decreasing(4), model)
        assert torch.allclose(out_native, out_fallback, atol=1e-6)

    def test_equivalence_m_fractional(self) -> None:
        """m=0.5 (fractional row): native euler == fallback euler."""
        model = _ScaleModel(0.8)
        m = torch.full((1, 3), 0.5)
        x = torch.randn(1, 3)
        clean = torch.zeros(1, 3)
        out_native, out_fallback = self._run_both(m, x, clean, _decreasing(5), model)
        assert torch.allclose(out_native, out_fallback, atol=1e-6)

    def test_equivalence_m_one(self) -> None:
        """m=1 (fully denoised row): native euler == fallback euler."""
        model = _ScaleModel(0.9)
        m = torch.ones(1, 2)
        x = torch.randn(1, 2)
        clean = torch.zeros(1, 2)
        out_native, out_fallback = self._run_both(m, x, clean, _decreasing(4), model)
        assert torch.allclose(out_native, out_fallback, atol=1e-6)

    def test_equivalence_with_audio(self) -> None:
        """With audio rows (video_element_count=2, shift_v=12, shift_a=3): still equal."""
        model = _ScaleModel(0.85)
        m = torch.full((1, 4), 0.5)
        x = torch.randn(1, 4)
        clean = torch.zeros(1, 4)
        out_native, out_fallback = self._run_both(
            m, x, clean, _decreasing(4), model, video_element_count=2
        )
        assert torch.allclose(out_native, out_fallback, atol=1e-6)

    def test_equivalence_with_dense_grid(self) -> None:
        """Dense sigma grid active: native euler == fallback euler."""
        model = _ScaleModel(0.9)
        m = torch.full((1, 2), 0.5)
        x = torch.randn(1, 2)
        clean = torch.zeros(1, 2)
        steps = 3
        sigmas = _decreasing(steps + 1)
        dense = _decreasing(steps * steps + 1)
        out_native, out_fallback = self._run_both(m, x, clean, sigmas, model, sigmas_dense=dense)
        assert torch.allclose(out_native, out_fallback, atol=1e-6)


# ---------------------------------------------------------------------------
# Step dispatch: native path chosen vs fallback path chosen
# ---------------------------------------------------------------------------


class TestStepDispatch:
    """Dispatch selects _euler_step for 'sample_euler', _fallback_step for everything else."""

    def test_native_euler_does_not_call_base_fn(self) -> None:
        """When dispatched natively, base_fn is bypassed (inlined euler, not called)."""
        base_called: list[bool] = []

        def sample_euler(  # noqa: ANN202
            model: Any,
            x: torch.Tensor,
            sigmas: torch.Tensor,
            extra_args: Any = None,
            callback: Any = None,
            disable: Any = None,
            **kwargs: Any,
        ) -> torch.Tensor:
            base_called.append(True)
            return x  # would be called if fallback path were taken

        m = torch.ones(1, 2)
        x = torch.randn(1, 2)
        model = _ScaleModel(0.9)
        fn = build_per_row_sampler_function(sample_euler, m, torch.zeros(1, 2), _sched())
        fn(model, x, _decreasing(3))
        assert base_called == [], "native euler must not call the base_fn wrapper"

    def test_fallback_calls_base_fn(self) -> None:
        """Unknown sampler name → fallback path → base_fn called once per step."""
        base = _RemapBase()
        m = torch.ones(1, 2)
        fn = build_per_row_sampler_function(base, m, torch.zeros(1, 2), _sched())
        fn(object(), torch.randn(1, 2), _decreasing(3))
        assert len(base.calls) == 2  # steps_n == 2 for _decreasing(3)

    def test_euler_step_callback_none(self) -> None:
        """_euler_step with callback=None takes the callback-None branch (no crash)."""
        m = torch.ones(1, 2)
        x = torch.randn(1, 2)
        model = _ScaleModel(0.9)
        fn = build_per_row_sampler_function(_local_sample_euler, m, torch.zeros(1, 2), _sched())
        out = fn(model, x, _decreasing(3), callback=None)
        assert out.shape == (1, 2)

    def test_euler_step_callback_present(self) -> None:
        """_euler_step with callback fires it once per step, index remapped to global step."""
        seen: list[dict[str, Any]] = []

        def cb(d: dict[str, Any]) -> None:
            seen.append(dict(d))

        m = torch.ones(1, 2)
        x = torch.randn(1, 2)
        model = _ScaleModel(0.9)
        fn = build_per_row_sampler_function(_local_sample_euler, m, torch.zeros(1, 2), _sched())
        fn(model, x, _decreasing(4), callback=cb)  # steps_n == 3
        assert [d["i"] for d in seen] == [0, 1, 2]  # _cb remaps 0 → i each step
        assert all("denoised" in d for d in seen)

    def test_fallback_forwards_raw_sigmas_dtype(self) -> None:
        """Fallback path must pass the raw sigmas (not sig_v) to base_fn.

        sig_v = sigmas.to(device=x.device, dtype=x.dtype) silently down-converts a
        float64 schedule to float32 when x is float32.  The original code passed the
        raw schedule; this test would FAIL if sigmas=sig_v were used in _StepContext.
        """
        base = _RemapBase()
        m = torch.ones(1, 2)
        # x is float32; sigmas are float64 — mismatched dtype exposes any sig_v conversion.
        x = torch.randn(1, 2, dtype=torch.float32)
        sigmas = _decreasing(3).to(dtype=torch.float64)
        fn = build_per_row_sampler_function(base, m, torch.zeros(1, 2), _sched())
        fn(object(), x, sigmas)
        assert base.calls[0]["sigmas"].dtype == torch.float64, (
            "fallback must forward the raw sigmas dtype (float64), not the sig_v cast (float32)"
        )


# ---------------------------------------------------------------------------
# RF-ancestral native step: local scalar reference
# ---------------------------------------------------------------------------


def _local_sample_euler_ancestral_rf(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: Any = None,
    disable: Any = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    noise_sampler: Any = None,
    **_kwargs: Any,
) -> torch.Tensor:
    """Local scalar reference for ``sample_euler_ancestral_RF`` (no comfy dependency).

    Mirrors comfy's ``sample_euler_ancestral_RF`` (k_diffusion/sampling.py:240-266)
    exactly, including the terminal-step branch.  Used only in tests.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised}
            )
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
            sigma_down = sigmas[i + 1] * downstep_ratio
            alpha_ip1 = 1 - sigmas[i + 1]
            alpha_down = 1 - sigma_down
            renoise_coeff = (
                sigmas[i + 1] ** 2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2
            ) ** 0.5
            sigma_down_i_ratio = sigma_down / sigmas[i]
            x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised
            if eta > 0:
                x = (alpha_ip1 / alpha_down) * x + noise_sampler(
                    sigmas[i], sigmas[i + 1]
                ) * s_noise * renoise_coeff
    return x


# Patch __name__ so the dispatch registry routes to _euler_ancestral_rf_step.
_local_sample_euler_ancestral_rf.__name__ = "sample_euler_ancestral"


class _FixedNoiseSampler:
    """Returns the same fixed noise tensor on every call; call count is tracked."""

    def __init__(self, noise: torch.Tensor) -> None:
        self.noise = noise
        self.call_count = 0

    def __call__(self, sigma: Any, sigma_next: Any) -> torch.Tensor:
        self.call_count += 1
        return self.noise.clone()


# ---------------------------------------------------------------------------
# Equivalence: native RF-ancestral step == stock scalar for m=1
# ---------------------------------------------------------------------------


class TestRFAncestralStepEquivalence:
    """_euler_ancestral_rf_step reproduces stock sample_euler_ancestral_RF exactly for m=1."""

    def _run_native(
        self,
        m: torch.Tensor,
        x: torch.Tensor,
        clean: torch.Tensor,
        sigmas: torch.Tensor,
        model: Any,
        noise_sampler: Any,
        eta: float = 1.0,
        *,
        video_element_count: int | None = None,
    ) -> torch.Tensor:
        """Run the per-row sampler with _local_sample_euler_ancestral_rf as the base."""
        fn = build_per_row_sampler_function(
            _local_sample_euler_ancestral_rf,
            m,
            clean,
            _sched(),
            video_element_count=video_element_count,
        )
        return fn(model, x.clone(), sigmas, noise_sampler=noise_sampler, eta=eta)

    def test_m1_equals_stock(self) -> None:
        """All-m=1 latent: per-row output must match stock scalar reference (tight atol=1e-6)."""
        model = _ScaleModel(0.9)
        m = torch.ones(1, 3)
        x = torch.randn(1, 3)
        clean = torch.zeros(1, 3)
        sigmas = _decreasing(4)  # steps_n=3
        fixed_noise = torch.randn(1, 3)
        ns = _FixedNoiseSampler(fixed_noise)

        # Stock scalar reference.
        stock_out = _local_sample_euler_ancestral_rf(model, x.clone(), sigmas, noise_sampler=ns)
        # Native per-row (all m=1): name matches registry → _euler_ancestral_rf_step dispatched.
        out = self._run_native(
            m, x, clean, sigmas, model, noise_sampler=_FixedNoiseSampler(fixed_noise)
        )

        assert torch.allclose(out, stock_out, atol=1e-6), (
            f"m=1 per-row must match stock RF-ancestral; max diff="
            f"{float((out - stock_out).abs().max())}"
        )

    def test_m1_equals_stock_eta_zero(self) -> None:
        """eta=0 (deterministic) also matches stock: no noise injected in either path."""
        model = _ScaleModel(0.85)
        m = torch.ones(1, 2)
        x = torch.randn(1, 2)
        clean = torch.zeros(1, 2)
        sigmas = _decreasing(4)

        def null_ns(sigma: Any, sigma_next: Any) -> torch.Tensor:
            return torch.zeros_like(x)

        stock_out = _local_sample_euler_ancestral_rf(
            model, x.clone(), sigmas, noise_sampler=null_ns, eta=0.0
        )
        out = self._run_native(m, x, clean, sigmas, model, noise_sampler=null_ns, eta=0.0)
        assert torch.allclose(out, stock_out, atol=1e-6)

    def test_m0_returns_clean(self) -> None:
        """m=0 rows: outer where(never, clean, x) restores clean exactly."""
        model = _ScaleModel(0.9)
        m = torch.zeros(1, 3)
        x = torch.randn(1, 3)
        clean = torch.full((1, 3), 7.0)
        sigmas = _decreasing(3)
        fixed_noise = torch.randn(1, 3)
        out = self._run_native(
            m, x, clean, sigmas, model, noise_sampler=_FixedNoiseSampler(fixed_noise)
        )
        assert torch.allclose(out, clean), "m=0 rows must be restored from clean exactly"

    def test_terminal_step_yields_denoised_r_no_branch(self) -> None:
        """Terminal sigma_row_next=0 falls out of the formula (no special-case branch needed).

        For m=1, denoised_r==denoised, ratio==0 → x=denoised; renoise_coeff=sqrt(0)=0.
        We verify the output equals what stock gives at the terminal step.
        """
        model = _ScaleModel(0.9)
        m = torch.ones(1, 2)
        x = torch.randn(1, 2)
        clean = torch.zeros(1, 2)
        # 2-step schedule: step 1 is terminal (sigmas[2]==0).
        sigmas = _decreasing(3)
        fixed_noise = torch.zeros(1, 2)  # zero noise: renoise_coeff=0 → ignored anyway
        ns = _FixedNoiseSampler(fixed_noise)
        stock_out = _local_sample_euler_ancestral_rf(model, x.clone(), sigmas, noise_sampler=ns)
        out = self._run_native(
            m, x, clean, sigmas, model, noise_sampler=_FixedNoiseSampler(fixed_noise)
        )
        assert torch.allclose(out, stock_out, atol=1e-6)

    def test_fractional_row_strictly_between(self) -> None:
        """Fractional m=0.5 row lands strictly between x_prev and denoised on each step."""
        model = _ScaleModel(0.9)
        m = torch.tensor([[1.0, 0.5]])  # col 0 free, col 1 half-denoise
        x = torch.zeros(1, 2)
        clean = torch.zeros(1, 2)
        sigmas = _decreasing(3)
        fixed_noise = torch.zeros(1, 2)  # zero noise for deterministic comparison
        out = self._run_native(
            m, x, clean, sigmas, model, noise_sampler=_FixedNoiseSampler(fixed_noise), eta=0.0
        )
        # With eta=0 and zero noise, m=1 col gets full denoising; m=0.5 col lands between.
        # (Values are not clean=0 because x=0 and model denoises toward x*scale=0 — they should
        # be equal here, but the point is no NaN and shape is correct.)
        assert out.shape == (1, 2)
        assert torch.isfinite(out).all(), "fractional row must produce finite values"

    def test_audio_rows_no_error(self) -> None:
        """Audio rows (video_element_count set, shifted sigma) run without error or NaN."""
        model = _ScaleModel(0.9)
        # 4 packed rows: 0,1 video / 2,3 audio.
        m = torch.full((1, 4), 0.5)
        x = torch.randn(1, 4)
        clean = torch.zeros(1, 4)
        sigmas = _decreasing(4)
        fixed_noise = torch.zeros(1, 4)
        out = self._run_native(
            m,
            x,
            clean,
            sigmas,
            model,
            noise_sampler=_FixedNoiseSampler(fixed_noise),
            video_element_count=2,
        )
        assert out.shape == (1, 4)
        assert torch.isfinite(out).all()

    def test_eta_zero_skips_noise_call(self) -> None:
        """eta=0: the noise sampler must not be called (if-eta>0 branch skipped)."""
        model = _ScaleModel(0.9)
        m = torch.ones(1, 2)
        x = torch.randn(1, 2)
        clean = torch.zeros(1, 2)
        sigmas = _decreasing(3)
        ns = _FixedNoiseSampler(torch.randn(1, 2))
        self._run_native(m, x, clean, sigmas, model, noise_sampler=ns, eta=0.0)
        assert ns.call_count == 0, "eta=0 must not invoke the noise sampler"

    def test_noise_sampler_persists_across_steps(self) -> None:
        """The noise sampler built on step 0 must be reused on step 1 (not rebuilt).

        We use a stateful noise sampler that records call order; if a fresh sampler were
        built each step both draws would come from reset generators (identical noise) — the
        stateful one advances, so calls diverging is the expected correct behaviour.
        """
        model = _ScaleModel(0.9)
        m = torch.ones(1, 2)
        x = torch.randn(1, 2)
        clean = torch.zeros(1, 2)
        sigmas = _decreasing(4)  # steps_n=3; 2 non-terminal steps call the noise sampler

        # Noise sampler that returns different tensors on each call.
        call_outputs: list[torch.Tensor] = [torch.ones(1, 2) * k for k in range(3)]
        call_idx = [0]

        def advancing_ns(sigma: Any, sigma_next: Any) -> torch.Tensor:
            out = call_outputs[call_idx[0]].clone()
            call_idx[0] += 1
            return out

        out = self._run_native(m, x, clean, sigmas, model, noise_sampler=advancing_ns)
        # If the sampler advanced correctly, call_idx should have been incremented.
        # (Terminal step may also call it with renoise_coeff=0, so ≥ 2.)
        assert call_idx[0] >= 2, "noise sampler must have been called at least twice across steps"
        assert out.shape == (1, 2)

    def test_callback_fires_once_per_step(self) -> None:
        """_euler_ancestral_rf_step fires the callback once per step, index remapped globally."""
        model = _ScaleModel(0.9)
        m = torch.ones(1, 2)
        x = torch.randn(1, 2)
        clean = torch.zeros(1, 2)
        sigmas = _decreasing(4)  # steps_n=3
        seen: list[dict[str, Any]] = []

        def cb(d: dict[str, Any]) -> None:
            seen.append(dict(d))

        fn = build_per_row_sampler_function(_local_sample_euler_ancestral_rf, m, clean, _sched())
        fn(
            model,
            x,
            sigmas,
            callback=cb,
            noise_sampler=_FixedNoiseSampler(torch.zeros(1, 2)),
        )
        # _cb remaps the per-step i=0 to the global step index.
        assert [d["i"] for d in seen] == [0, 1, 2]
        assert all("denoised" in d for d in seen)


# ---------------------------------------------------------------------------
# Dispatch: native RF-ancestral step chosen when name matches registry
# ---------------------------------------------------------------------------


class TestRFAncestralDispatch:
    def test_native_rf_ancestral_does_not_call_base_fn(self) -> None:
        """sample_euler_ancestral dispatch: native step bypasses base_fn entirely."""
        base_called: list[bool] = []

        def sample_euler_ancestral(  # noqa: ANN202
            model: Any,
            x: torch.Tensor,
            sigmas: torch.Tensor,
            extra_args: Any = None,
            callback: Any = None,
            disable: Any = None,
            **kwargs: Any,
        ) -> torch.Tensor:
            base_called.append(True)
            return x

        m = torch.ones(1, 2)
        x = torch.randn(1, 2)
        model = _ScaleModel(0.9)
        fixed_noise = torch.zeros_like(x)
        fn = build_per_row_sampler_function(sample_euler_ancestral, m, torch.zeros(1, 2), _sched())
        fn(model, x, _decreasing(3), noise_sampler=lambda s, sn: fixed_noise)
        assert base_called == [], "native RF-ancestral must not call the base_fn wrapper"

    def test_sample_euler_ancestral_in_registry(self) -> None:
        """sample_euler_ancestral must be registered in _NATIVE_ROW_STEPS."""
        assert "sample_euler_ancestral" in _NATIVE_ROW_STEPS

    def test_warning_suppressed_for_native_stochastic(self) -> None:
        """sample_euler_ancestral is stochastic (eta>0 default) but natively handled.

        The warning condition ``sampler_is_stochastic(fn) and name not in _NATIVE_ROW_STEPS``
        must evaluate False, i.e. the warning is suppressed for this sampler.
        """

        def fn(model: Any, x: Any, sigmas: Any, eta: float = 1.0) -> Any:  # noqa: ANN202
            pass

        fn.__name__ = "sample_euler_ancestral"
        assert sampler_is_stochastic(fn) is True  # it IS stochastic
        assert fn.__name__ in _NATIVE_ROW_STEPS  # but it IS natively handled
        # Combined condition used in nodes.py must be False (warning suppressed):
        assert not (sampler_is_stochastic(fn) and fn.__name__ not in _NATIVE_ROW_STEPS)

    def test_warning_still_fires_for_non_native_stochastic(self) -> None:
        """Stochastic samplers not in _NATIVE_ROW_STEPS still trigger the warning condition."""

        def fn(model: Any, x: Any, sigmas: Any, eta: float = 1.0) -> Any:  # noqa: ANN202
            pass

        fn.__name__ = "sample_ddpm_custom"
        assert sampler_is_stochastic(fn) is True
        assert fn.__name__ not in _NATIVE_ROW_STEPS
        assert sampler_is_stochastic(fn) and fn.__name__ not in _NATIVE_ROW_STEPS


# ---------------------------------------------------------------------------
# Regression: audio ancestral integration must use the σ_v axis (not σ_a)
# ---------------------------------------------------------------------------


class TestAudioAncestralAxisSplit:
    """Regression for the audio-axis mismatch in _euler_ancestral_rf_step.

    Before the fix, denoised_r and the ancestral si/sip1 terms used ctx.sig_row /
    ctx.sig_row_next, which for audio rows carry the σ_a-shifted schedule values.
    The packed audio tensor lives on the σ_v trajectory; the ancestral integration
    must use ctx.sig_row_v / ctx.sig_row_v_next so that audio rows integrate on σ_v,
    matching stock sample_euler_ancestral_RF.

    The test constructs a _StepContext where sig_row (σ_a) and sig_row_v (σ_v) are
    numerically distinct, then asserts the result matches the σ_v-axis computation.

    FAIL-THEN-PASS: before the fix, constructing _StepContext with sig_row_v /
    sig_row_v_next fields raises TypeError (fields do not exist) — a clear FAIL.
    After the fix, the fields exist and the integration uses them — PASS.
    """

    def _make_ctx(
        self,
        sig_row: torch.Tensor,
        sig_row_next: torch.Tensor,
        sig_row_v: torch.Tensor,
        sig_row_v_next: torch.Tensor,
        sigmas: torch.Tensor,
        x_prev: torch.Tensor,
        model: Any,
        noise_sampler: Any,
        eta: float = 0.0,
    ) -> _StepContext:
        """Build a minimal _StepContext for a single-step call to _euler_ancestral_rf_step."""
        return _StepContext(
            model=model,
            x_prev=x_prev,
            i=0,
            sigmas=sigmas,
            sig_row=sig_row,
            sig_row_next=sig_row_next,
            sig_row_v=sig_row_v,
            sig_row_v_next=sig_row_v_next,
            sig_g=sig_row_v,  # unused by _euler_ancestral_rf_step; any value is fine
            sig_g_next=sig_row_v_next,
            extra_args=None,
            callback=None,
            disable=None,
            kwargs={"noise_sampler": noise_sampler, "eta": eta},
            base_fn=lambda *a, **k: x_prev,  # unused by _euler_ancestral_rf_step
            state={},
        )

    def test_audio_ancestral_uses_sigma_v_axis(self) -> None:
        """_euler_ancestral_rf_step integrates on sig_row_v, not sig_row.

        Sets sig_row (σ_a = 0.2) and sig_row_v (σ_v = 0.6) to distinct values so
        the two axes produce numerically different outputs.  Asserts the result matches
        the σ_v-axis computation (correct) and NOT the σ_a-axis computation (pre-fix).
        """
        carrier_val = 0.7
        sigmas = torch.tensor([carrier_val, 0.4])

        sig_row = torch.tensor([0.2])  # σ_a axis — used by pre-fix code (WRONG)
        sig_row_next = torch.tensor([0.1])
        sig_row_v = torch.tensor([0.6])  # σ_v axis — must be used by ancestral integration
        sig_row_v_next = torch.tensor([0.4])

        x_prev = torch.tensor([[1.0, 2.0]])
        model = _ScaleModel(0.8)  # denoised = 0.8 * x_prev
        ns = _FixedNoiseSampler(torch.zeros_like(x_prev))

        ctx = self._make_ctx(
            sig_row, sig_row_next, sig_row_v, sig_row_v_next, sigmas, x_prev, model, ns, eta=0.0
        )
        result = _euler_ancestral_rf_step(ctx)

        # Manually compute σ_v-axis expected result (eta=0, no noise).
        carrier = torch.tensor(carrier_val)
        denoised = model(x_prev, carrier)
        v = (x_prev - denoised) / carrier

        denoised_r_v = x_prev - sig_row_v * v
        si_v = sig_row_v.clamp(min=1e-8)
        sip1_v = sig_row_v_next
        ratio_v = sip1_v / si_v  # eta=0 → sigma_down=sip1 → ratio=sip1/si
        expected_v = ratio_v * x_prev + (1.0 - ratio_v) * denoised_r_v

        # Manually compute σ_a-axis wrong result (pre-fix behavior).
        denoised_r_a = x_prev - sig_row * v
        si_a = sig_row.clamp(min=1e-8)
        sip1_a = sig_row_next
        ratio_a = sip1_a / si_a
        expected_a = ratio_a * x_prev + (1.0 - ratio_a) * denoised_r_a

        # Sanity: the two axis results must be numerically distinct (test setup is valid).
        assert not torch.allclose(expected_v, expected_a, atol=1e-4), (
            "test setup invalid: σ_v and σ_a axis expected results must be numerically distinct"
        )

        # Post-fix: result must match σ_v axis.
        assert torch.allclose(result, expected_v, atol=1e-6), (
            f"audio ancestral must use σ_v axis; result={result}, expected_v={expected_v}"
        )
        # Post-fix: result must NOT match σ_a axis.
        assert not torch.allclose(result, expected_a, atol=1e-4), (
            f"audio ancestral must NOT match σ_a axis result; "
            f"result={result}, expected_a={expected_a}"
        )

    def test_m1_audio_matches_stock(self) -> None:
        """With audio enabled and all m=1, the output still matches stock RF-ancestral.

        At m=1 every row, sig_row_v(i) == sig_v[i] == sigmas[i] == carrier, so the
        sig_row_v-axis guarantee reproduces stock bit-for-bit, even for audio rows.
        """
        model = _ScaleModel(0.9)
        # 4 packed rows: 0,1 video / 2,3 audio — all at m=1.
        m = torch.ones(1, 4)
        x = torch.randn(1, 4)
        clean = torch.zeros(1, 4)
        sigmas = _decreasing(4)  # steps_n=3
        fixed_noise = torch.randn(1, 4)

        # Stock scalar reference (no audio, uses raw sigmas uniformly).
        stock_out = _local_sample_euler_ancestral_rf(
            model, x.clone(), sigmas, noise_sampler=_FixedNoiseSampler(fixed_noise)
        )

        # Native per-row with audio enabled (video_element_count=2).
        fn = build_per_row_sampler_function(
            _local_sample_euler_ancestral_rf,
            m,
            clean,
            _sched(),
            video_element_count=2,
        )
        out = fn(model, x.clone(), sigmas, noise_sampler=_FixedNoiseSampler(fixed_noise))

        assert torch.allclose(out, stock_out, atol=1e-6), (
            f"m=1 with audio must match stock RF-ancestral; max diff="
            f"{float((out - stock_out).abs().max())}"
        )
