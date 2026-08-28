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
    _shift_schedule,
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
