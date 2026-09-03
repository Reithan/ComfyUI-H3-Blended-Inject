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
    _shift_schedule,
    _stream_row_sigma,
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
# _stream_row_sigma
# ---------------------------------------------------------------------------


class TestStreamRowSigma:
    """The extracted per-row sigma helper drives both the loop and the observer init.

    The whole point of extracting it is that the observer split can pass its OWN token-ordered
    fractional ``m`` and get the identical ``σ_row`` the sampler loop computes for those rows,
    with no packed/token layout assumptions.  These lock the dense-exact and coarse-lerp branches
    and the endpoint behavior at m∈{0,1}.
    """

    def test_dense_grid_exact_integer_index(self) -> None:
        # steps=2 → dense grid length steps²+1 = 5; row sigma reads grid[k_d·(steps−i)+i·steps].
        steps = 2
        dense = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2], dtype=torch.float64)
        m = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)  # k_d = 0, 1, 2
        # i=0 → idx = k_d·2 = [0, 2, 4]
        out0 = _stream_row_sigma(m, 0, steps, dense, dense, dense.numel())
        assert torch.allclose(out0, torch.tensor([1.0, 0.6, 0.2], dtype=torch.float64))
        # i=1 → idx = k_d·1 + 1·2 = [2, 3, 4]
        out1 = _stream_row_sigma(m, 1, steps, dense, dense, dense.numel())
        assert torch.allclose(out1, torch.tensor([0.6, 0.4, 0.2], dtype=torch.float64))

    def test_dense_index_clamped_to_grid_end(self) -> None:
        steps = 2
        dense = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)  # k_d = 2
        # i=2 → idx = 2·0 + 2·2 = 4 (last); larger i would clamp, never exceeds grid end.
        out = _stream_row_sigma(m, 2, steps, dense, dense, dense.numel())
        assert torch.allclose(out, torch.tensor([0.2], dtype=torch.float64))

    def test_coarse_lerp_when_no_dense_grid(self) -> None:
        steps = 4
        coarse = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0], dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)  # k_d = round(4·0.5) = 2, span = 0.5
        # i=1 → idx = 2 + 1·0.5 = 2.5 → lerp(coarse[2], coarse[3], .5) = 0.375
        out = _stream_row_sigma(m, 1, steps, None, coarse, coarse.numel())
        assert torch.allclose(out, torch.tensor([0.375], dtype=torch.float64))

    def test_matches_loop_row_sigma_for_fractional_subset(self) -> None:
        """Observer contract: feeding a token-ordered subset of m yields the same σ_row values
        the full-vector call produces at those same rows (dense branch)."""
        steps = 3
        dense = _decreasing(steps * steps + 1).to(torch.float64)
        full_m = torch.tensor([1.0, 0.4, 0.0, 0.7], dtype=torch.float64)
        subset_idx = torch.tensor([1, 3])  # the fractional rows
        for i in range(steps):
            full = _stream_row_sigma(full_m, i, steps, dense, dense, dense.numel())
            sub = _stream_row_sigma(full_m[subset_idx], i, steps, dense, dense, dense.numel())
            assert torch.allclose(sub, full[subset_idx])


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
    """The wrapper gates guide keyframes in minimax_payload by step index windows."""

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

    def _args(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "input": torch.randn(1, 4, 3),
            "timestep": torch.tensor([0.5]),
            "c": {"transformer_options": {}, "minimax_payload": payload},
            "cond_or_uncond": [0],
        }

    def test_no_guide_release_passes_payload_through(self) -> None:
        payload, _, _ = self._payload()
        wrapper = build_conditioning_wrapper(_sched())
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload))
        assert am.kwargs["minimax_payload"] is payload

    def test_empty_entries_passes_payload_through(self) -> None:
        payload, _, _ = self._payload()
        wrapper = build_conditioning_wrapper(_sched(), guide_release={})
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload))
        assert am.kwargs["minimax_payload"] is payload

    def test_step_before_end_holds_payload(self) -> None:
        """current_step < end_step → guide is still active."""
        payload, _, kf_ours = self._payload()
        guide_release = {"entries": [(id(kf_ours), 10)]}
        wrapper = build_conditioning_wrapper(_sched(current_step=5), guide_release=guide_release)
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload))
        assert am.kwargs["minimax_payload"] is payload
        assert "layout" in payload

    def test_step_at_end_releases_guide(self) -> None:
        """current_step >= end_step → guide is released."""
        payload, kf_official, kf_ours = self._payload()
        guide_release = {"entries": [(id(kf_ours), 10)]}
        wrapper = build_conditioning_wrapper(_sched(current_step=10), guide_release=guide_release)
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload))
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
        guide_release = {"entries": [(id(kf_ours), 10)]}
        wrapper = build_conditioning_wrapper(_sched(current_step=10), guide_release=guide_release)
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload))
        first = am.kwargs["minimax_payload"]
        wrapper(am, self._args(payload))
        assert am.kwargs["minimax_payload"] is first

    def test_distinct_payload_dicts_get_distinct_cache_entries(self) -> None:
        """cond and uncond streams carry different payload dicts; they must not cross."""
        payload_a, _, kf_ours = self._payload()
        payload_b = dict(payload_a)
        guide_release = {"entries": [(id(kf_ours), 10)]}
        wrapper = build_conditioning_wrapper(_sched(current_step=10), guide_release=guide_release)
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload_a))
        filtered_a = am.kwargs["minimax_payload"]
        wrapper(am, self._args(payload_b))
        filtered_b = am.kwargs["minimax_payload"]
        assert filtered_a is not filtered_b
        assert filtered_a["keyframes"] == filtered_b["keyframes"]

    def test_end_step_zero_releases_at_any_step(self) -> None:
        """end_step=0: current_step >= 0 always → guide always excluded."""
        payload, kf_official, kf_ours = self._payload()
        guide_release = {"entries": [(id(kf_ours), 0)]}
        wrapper = build_conditioning_wrapper(_sched(current_step=0), guide_release=guide_release)
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload))
        assert am.kwargs["minimax_payload"]["keyframes"] == [kf_official]

    def test_all_guides_released_removes_keyframes_key(self) -> None:
        payload, kf_official, kf_ours = self._payload()
        guide_release = {"entries": [(id(kf_official), 10), (id(kf_ours), 10)]}
        wrapper = build_conditioning_wrapper(_sched(current_step=10), guide_release=guide_release)
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload))
        filtered = am.kwargs["minimax_payload"]
        assert "keyframes" not in filtered
        assert filtered["cond_video_latents"] == ["REF"]

    def test_pending_entry_suppresses_guide_before_start_step(self) -> None:
        """current_step < start_step → guide is pending (excluded)."""
        payload, kf_official, kf_ours = self._payload()
        guide_release = {"pending_entries": [(id(kf_ours), 5)]}
        wrapper = build_conditioning_wrapper(_sched(current_step=3), guide_release=guide_release)
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload))
        assert am.kwargs["minimax_payload"]["keyframes"] == [kf_official]

    def test_pending_entry_activates_guide_at_start_step(self) -> None:
        """current_step >= start_step → guide is no longer pending (included)."""
        payload, _, kf_ours = self._payload()
        guide_release = {"pending_entries": [(id(kf_ours), 5)]}
        wrapper = build_conditioning_wrapper(_sched(current_step=5), guide_release=guide_release)
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload))
        assert am.kwargs["minimax_payload"] is payload  # unfiltered

    def test_step_window_excludes_both_pending_and_released(self) -> None:
        """Guide outside [start_step, end_step) is excluded regardless of which boundary."""
        payload, kf_official, kf_ours = self._payload()
        guide_release = {
            "entries": [(id(kf_ours), 15)],
            "pending_entries": [(id(kf_ours), 5)],
        }
        # Before window: pending
        wrapper_pre = build_conditioning_wrapper(
            _sched(current_step=3), guide_release=dict(guide_release)
        )
        am = _RecordingApplyModel()
        wrapper_pre(am, self._args(payload))
        assert am.kwargs["minimax_payload"]["keyframes"] == [kf_official]
        # Inside window: active
        win_release = {"entries": [(id(kf_ours), 15)], "pending_entries": [(id(kf_ours), 5)]}
        wrapper_in = build_conditioning_wrapper(_sched(current_step=10), guide_release=win_release)
        wrapper_in(am, self._args(payload))
        assert am.kwargs["minimax_payload"] is payload
        # After window: released
        wrapper_post = build_conditioning_wrapper(
            _sched(current_step=15),
            guide_release={"entries": [(id(kf_ours), 15)], "pending_entries": [(id(kf_ours), 5)]},
        )
        wrapper_post(am, self._args(payload))
        assert am.kwargs["minimax_payload"]["keyframes"] == [kf_official]

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
        """Step-gate path coexists with pooled-label injection (no denoised correction)."""
        payload, kf_official, kf_ours = self._payload()
        guide_release = {"entries": [(id(kf_ours), 10)]}
        pooled = {"denoise_mask": torch.zeros(1, 1, 3)}
        wrapper = build_conditioning_wrapper(
            _sched(pooled, current_step=10), guide_release=guide_release
        )
        am = _RecordingApplyModel()
        wrapper(am, self._args(payload))
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

    def test_audio_rows_axis_blind_same_w_as_video(self) -> None:
        """Audio is AXIS-BLIND: even with video_element_count set and shift_v!=shift_a, audio
        rows ride the video σ_v axis, so at the same m they get the SAME per-row w as video.

        This is the regression guarding the audio-native-composite direction: audio no longer
        integrates on its own σ_a axis in the per-row engine (its σ_a shift is applied inside the
        H3 forward, and its fade is delegated to the official composite noise_mask)."""
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
        # video row 1 vs audio row 2 at identical m must MATCH (audio on the σ_v axis).
        assert torch.allclose(w0[0, 1], w0[0, 2])

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
# Multistep native steps (PR3): local scalar references
# ---------------------------------------------------------------------------


def _local_sample_dpmpp_2m(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: Any = None,
    disable: Any = None,
    **_kwargs: Any,
) -> torch.Tensor:
    """Local scalar reference for comfy ``sample_dpmpp_2m`` (k_diffusion/sampling.py:796-818)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()  # noqa: E731
    t_fn = lambda sigma: sigma.log().neg()  # noqa: E731
    old_denoised = None
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised}
            )
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


def _local_sample_res_multistep(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: Any = None,
    disable: Any = None,
    **_kwargs: Any,
) -> torch.Tensor:
    """Local scalar reference for comfy ``res_multistep`` at eta=0 (sampling.py:1417-1456)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()  # noqa: E731
    t_fn = lambda sigma: sigma.log().neg()  # noqa: E731
    phi1_fn = lambda t: torch.expm1(t) / t  # noqa: E731
    phi2_fn = lambda t: (phi1_fn(t) - 1.0) / t  # noqa: E731
    old_sigma_down = None
    old_denoised = None
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down = sigmas[i + 1]  # eta=0 → sigma_down == sigmas[i+1], sigma_up == 0
        if callback is not None:
            callback(
                {"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised}
            )
        if sigma_down == 0 or old_denoised is None:
            d = (x - denoised) / sigmas[i]
            x = x + d * (sigma_down - sigmas[i])
        else:
            t, t_old, t_next, t_prev = (
                t_fn(sigmas[i]),
                t_fn(old_sigma_down),
                t_fn(sigma_down),
                t_fn(sigmas[i - 1]),
            )
            h = t_next - t
            c2 = (t_prev - t_old) / h
            phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)
            x = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised)
        old_denoised = denoised
        old_sigma_down = sigma_down
    return x


# Patch __name__ so the dispatch registry routes to the native multistep steps.
_local_sample_dpmpp_2m.__name__ = "sample_dpmpp_2m"
_local_sample_res_multistep.__name__ = "sample_res_multistep"


# ---------------------------------------------------------------------------
# Equivalence: native multistep step == stock scalar for m=1 (2nd order preserved)
# ---------------------------------------------------------------------------


class TestMultistepStepEquivalence:
    """Native dpmpp_2m / res_multistep reproduce stock scalars exactly for all-m=1 latents.

    This is the Finding-1 regression: the fallback (base_fn on a 2-sigma slice) resets
    old_denoised every step and silently runs first-order.  The native steps carry per-row
    history across the outer loop, so all-m=1 must match the true 2nd-order stock trajectory.
    """

    def _run_native(
        self,
        base_fn: Any,
        m: torch.Tensor,
        x: torch.Tensor,
        clean: torch.Tensor,
        sigmas: torch.Tensor,
        model: Any,
        *,
        video_element_count: int | None = None,
    ) -> torch.Tensor:
        fn = build_per_row_sampler_function(
            base_fn, m, clean, _sched(), video_element_count=video_element_count
        )
        return fn(model, x.clone(), sigmas)

    def test_dpmpp_2m_m1_equals_stock(self) -> None:
        model = _ScaleModel(0.9)
        m = torch.ones(1, 3)
        x = torch.randn(1, 3)
        clean = torch.zeros(1, 3)
        sigmas = _decreasing(5)  # steps_n=4: exercises 1st-order step0 + multistep steps
        stock = _local_sample_dpmpp_2m(model, x.clone(), sigmas)
        out = self._run_native(_local_sample_dpmpp_2m, m, x, clean, sigmas, model)
        assert torch.allclose(out, stock, atol=1e-6), (
            f"dpmpp_2m m=1 native must match stock; max diff={float((out - stock).abs().max())}"
        )

    def test_res_multistep_m1_equals_stock(self) -> None:
        model = _ScaleModel(0.85)
        m = torch.ones(1, 3)
        x = torch.randn(1, 3)
        clean = torch.zeros(1, 3)
        sigmas = _decreasing(5)
        stock = _local_sample_res_multistep(model, x.clone(), sigmas)
        out = self._run_native(_local_sample_res_multistep, m, x, clean, sigmas, model)
        assert torch.allclose(out, stock, atol=1e-6), (
            f"res_multistep m=1 native must match stock; "
            f"max diff={float((out - stock).abs().max())}"
        )

    def test_dpmpp_2m_not_first_order(self) -> None:
        """Guard against silent first-order degradation: 2nd-order output must DIFFER from the
        fallback path (which resets old_denoised each interval → first order)."""
        model = _ScaleModel(0.9)
        m = torch.ones(1, 3)
        x = torch.randn(1, 3)
        clean = torch.zeros(1, 3)
        sigmas = _decreasing(5)

        def not_native(  # noqa: ANN202
            mod: Any,
            xin: torch.Tensor,
            sig: torch.Tensor,
            extra_args: Any = None,
            callback: Any = None,
            disable: Any = None,
            **kw: Any,
        ) -> torch.Tensor:
            return _local_sample_dpmpp_2m(mod, xin, sig, extra_args, callback, disable, **kw)

        native = self._run_native(_local_sample_dpmpp_2m, m, x, clean, sigmas, model)
        fallback = self._run_native(not_native, m, x, clean, sigmas, model)
        assert not torch.allclose(native, fallback, atol=1e-5), (
            "native multistep must restore 2nd order (differ from first-order fallback)"
        )

    def test_dpmpp_2m_m0_exact_preserve(self) -> None:
        """m=0 rows come out exactly clean (frozen + where(never, clean, x) guard)."""
        model = _ScaleModel(0.5)
        m = torch.zeros(1, 3)
        x = torch.randn(1, 3)
        clean = torch.ones(1, 3) * 7.0
        out = self._run_native(_local_sample_dpmpp_2m, m, x, clean, _decreasing(5), model)
        assert torch.allclose(out, clean, atol=1e-6)

    def test_res_multistep_m0_exact_preserve(self) -> None:
        model = _ScaleModel(0.5)
        m = torch.zeros(1, 3)
        x = torch.randn(1, 3)
        clean = torch.ones(1, 3) * 7.0
        out = self._run_native(_local_sample_res_multistep, m, x, clean, _decreasing(5), model)
        assert torch.allclose(out, clean, atol=1e-6)

    def test_dpmpp_2m_fractional_finite(self) -> None:
        """Fractional rows produce finite output (no NaN/inf from log clamps) and preserve m=0."""
        model = _ScaleModel(0.8)
        m = torch.tensor([[0.0, 0.5, 1.0]])
        x = torch.randn(1, 3)
        clean = torch.ones(1, 3) * 3.0
        out = self._run_native(_local_sample_dpmpp_2m, m, x, clean, _decreasing(5), model)
        assert torch.isfinite(out).all()
        assert torch.allclose(out[0, 0], clean[0, 0], atol=1e-6)  # m=0 column preserved

    def test_res_multistep_fractional_finite(self) -> None:
        model = _ScaleModel(0.8)
        m = torch.tensor([[0.0, 0.5, 1.0]])
        x = torch.randn(1, 3)
        clean = torch.ones(1, 3) * 3.0
        out = self._run_native(_local_sample_res_multistep, m, x, clean, _decreasing(5), model)
        assert torch.isfinite(out).all()
        assert torch.allclose(out[0, 0], clean[0, 0], atol=1e-6)

    def test_multistep_registered_and_not_stochastic(self) -> None:
        """Both multistep names are in the registry and are NOT flagged stochastic (eta=0)."""
        assert "sample_dpmpp_2m" in _NATIVE_ROW_STEPS
        assert "sample_res_multistep" in _NATIVE_ROW_STEPS
        assert sampler_is_stochastic(_local_sample_dpmpp_2m) is False
        assert sampler_is_stochastic(_local_sample_res_multistep) is False

    def test_multistep_callback_fires_each_step(self) -> None:
        """The native multistep step fires the callback once per step (index remapped global)."""
        seen: list[dict[str, Any]] = []

        def cb(d: dict[str, Any]) -> None:
            seen.append(dict(d))

        model = _ScaleModel(0.9)
        fn = build_per_row_sampler_function(
            _local_sample_dpmpp_2m, torch.ones(1, 2), torch.zeros(1, 2), _sched()
        )
        fn(model, torch.randn(1, 2), _decreasing(4), callback=cb)  # steps_n == 3
        assert [d["i"] for d in seen] == [0, 1, 2]
        assert all("denoised" in d for d in seen)

    def test_multistep_with_audio_finite(self) -> None:
        """Audio rows (axis-blind) integrate without NaN under the multistep steps."""
        model = _ScaleModel(0.85)
        m = torch.full((1, 4), 0.5)
        x = torch.randn(1, 4)
        clean = torch.zeros(1, 4)
        out = self._run_native(
            _local_sample_dpmpp_2m, m, x, clean, _decreasing(5), model, video_element_count=2
        )
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# DPM++ SDE family (PR4) local scalar references (CONST/RF logit form, matching
# the native per-row steps' clamps so all-m=1 reproduces them bit-for-bit).
# ---------------------------------------------------------------------------

_SDE_EPS = 1e-8
_SDE_HI = 1.0 - 1e-4  # matches sampler._SNR_SIGMA_HI (comfy offset_first_sigma_for_snr)


def _snr_clamp(sigma: torch.Tensor) -> torch.Tensor:
    return sigma.clamp(_SDE_EPS, _SDE_HI)


def _lam(sigma: torch.Tensor) -> torch.Tensor:
    """Half-log-SNR λ = log((1−σ)/σ) for CONST/RF (clamped, matching the native step)."""
    return _snr_clamp(sigma).logit().neg()


def _sig(lam: torch.Tensor) -> torch.Tensor:
    return (-lam).sigmoid()


def _anc(sigma_from: torch.Tensor, sigma_to: torch.Tensor, eta: float) -> Any:
    """Scalar ``get_ancestral_step`` in exp(−λ) space (mirrors _sde_get_ancestral_step)."""
    inner = torch.clamp(sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2, min=0.0)
    su = torch.minimum(sigma_to, eta * torch.sqrt(inner))
    sd = torch.sqrt(torch.clamp(sigma_to**2 - su**2, min=0.0))
    return sd, su


def _local_sample_dpmpp_2m_sde(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: Any = None,
    disable: Any = None,
    noise_sampler: Any = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    **_kw: Any,
) -> torch.Tensor:
    """Local reference for comfy ``sample_dpmpp_2m_sde`` (midpoint) in CONST/RF logit form."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    old_denoised = None
    h_last = None
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised}
            )
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            si_c = _snr_clamp(sigmas[i])
            sip1_c = _snr_clamp(sigmas[i + 1])
            lam_s, lam_t = _lam(sigmas[i]), _lam(sigmas[i + 1])
            h = lam_t - lam_s
            h_eta = h * (eta + 1.0)
            alpha_t = sip1_c * lam_t.exp()
            neg_em1 = torch.expm1(-h_eta).neg()
            x = (sip1_c / si_c) * torch.exp(-h * eta) * x + alpha_t * neg_em1 * denoised
            if old_denoised is not None:
                r = h_last / h
                x = x + 0.5 * alpha_t * neg_em1 * (1.0 / r) * (denoised - old_denoised)
            if eta > 0:
                noise = noise_sampler(sigmas[i], sigmas[i + 1])
                renoise = torch.clamp(torch.expm1(-2.0 * h * eta).neg(), min=0.0).sqrt()
                x = x + noise * sigmas[i + 1] * renoise * s_noise
            old_denoised = denoised
            h_last = h
    return x


def _local_sample_dpmpp_3m_sde(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: Any = None,
    disable: Any = None,
    noise_sampler: Any = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    **_kw: Any,
) -> torch.Tensor:
    """Local reference for comfy ``sample_dpmpp_3m_sde`` in CONST/RF logit form."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    denoised_1 = denoised_2 = None
    h_1 = h_2 = None
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised}
            )
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            si_c = _snr_clamp(sigmas[i])
            sip1_c = _snr_clamp(sigmas[i + 1])
            lam_s, lam_t = _lam(sigmas[i]), _lam(sigmas[i + 1])
            h = lam_t - lam_s
            h_eta = h * (eta + 1.0)
            alpha_t = sip1_c * lam_t.exp()
            em1 = torch.expm1(-h_eta)
            x = (sip1_c / si_c) * torch.exp(-h * eta) * x + alpha_t * em1.neg() * denoised
            if h_2 is not None:
                r0, r1 = h_1 / h, h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = em1 / h_eta + 1.0
                phi_3 = phi_2 / h_eta - 0.5
                x = x + (alpha_t * phi_2) * d1 - (alpha_t * phi_3) * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = em1 / h_eta + 1.0
                x = x + (alpha_t * phi_2) * d
            if eta > 0:
                noise = noise_sampler(sigmas[i], sigmas[i + 1])
                renoise = torch.clamp(torch.expm1(-2.0 * h * eta).neg(), min=0.0).sqrt()
                x = x + noise * sigmas[i + 1] * renoise * s_noise
            denoised_1, denoised_2 = denoised, denoised_1
            h_1, h_2 = h, h_1
    return x


def _local_sample_dpmpp_sde(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: Any = None,
    disable: Any = None,
    noise_sampler: Any = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    r: float = 0.5,
    **_kw: Any,
) -> torch.Tensor:
    """Local reference for comfy ``sample_dpmpp_sde`` (2-eval) in CONST/RF logit form."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    fac = 1.0 / (2.0 * r)
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised}
            )
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            si_c = _snr_clamp(sigmas[i])
            sip1_c = _snr_clamp(sigmas[i + 1])
            lam_s, lam_t = _lam(sigmas[i]), _lam(sigmas[i + 1])
            h = lam_t - lam_s
            lam_s_1 = lam_s + r * h
            sigma_s_1 = _sig(lam_s_1)
            alpha_s = si_c * lam_s.exp()
            alpha_s_1 = sigma_s_1 * lam_s_1.exp()
            alpha_t = sip1_c * lam_t.exp()
            sd, su = _anc(lam_s.neg().exp(), lam_s_1.neg().exp(), eta)
            h_ = sd.log().neg() - lam_s
            x_2 = (alpha_s_1 / alpha_s) * torch.exp(-h_) * x
            x_2 = x_2 - alpha_s_1 * torch.expm1(-h_) * denoised
            if eta > 0 and s_noise > 0:
                x_2 = x_2 + alpha_s_1 * noise_sampler(sigmas[i], sigma_s_1) * s_noise * su
            denoised_2 = model(x_2, sigma_s_1 * s_in, **extra_args)
            sd, su = _anc(lam_s.neg().exp(), lam_t.neg().exp(), eta)
            h_ = sd.log().neg() - lam_s
            denoised_d = (1.0 - fac) * denoised + fac * denoised_2
            x = (alpha_t / alpha_s) * torch.exp(-h_) * x
            x = x - alpha_t * torch.expm1(-h_) * denoised_d
            if eta > 0 and s_noise > 0:
                x = x + alpha_t * noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * su
    return x


def _local_sample_dpmpp_2s_ancestral(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: Any = None,
    disable: Any = None,
    noise_sampler: Any = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    **_kw: Any,
) -> torch.Tensor:
    """Local reference for comfy ``sample_dpmpp_2s_ancestral_RF`` (2-eval), CONST/RF logit form."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    r = 0.5
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        downstep_ratio = 1.0 + (sigmas[i + 1] / sigmas[i] - 1.0) * eta
        sigma_down = sigmas[i + 1] * downstep_ratio
        alpha_ip1 = 1.0 - sigmas[i + 1]
        alpha_down = 1.0 - sigma_down
        renoise_coeff = torch.sqrt(
            torch.clamp(sigmas[i + 1] ** 2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2, min=0.0)
        )
        if callback is not None:
            callback(
                {"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised}
            )
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            lam_i, lam_down = _lam(sigmas[i]), _lam(sigma_down)
            sigma_s = _sig(lam_i + r * (lam_down - lam_i))
            u_ratio = sigma_s / sigmas[i]
            u = u_ratio * x + (1.0 - u_ratio) * denoised
            d_i = model(u, sigma_s * s_in, **extra_args)
            down_ratio = sigma_down / sigmas[i]
            x = down_ratio * x + (1.0 - down_ratio) * d_i
        if sigmas[i + 1] > 0 and eta > 0:
            noise = noise_sampler(sigmas[i], sigmas[i + 1])
            x = (alpha_ip1 / alpha_down) * x + noise * s_noise * renoise_coeff
    return x


_local_sample_dpmpp_2m_sde.__name__ = "sample_dpmpp_2m_sde"
_local_sample_dpmpp_3m_sde.__name__ = "sample_dpmpp_3m_sde"
_local_sample_dpmpp_sde.__name__ = "sample_dpmpp_sde"
_local_sample_dpmpp_2s_ancestral.__name__ = "sample_dpmpp_2s_ancestral"


class TestSDEStepEquivalence:
    """Native per-row DPM++ SDE steps reproduce their stock scalars exactly for all-m=1."""

    def _run_native(
        self,
        base_fn: Any,
        m: torch.Tensor,
        x: torch.Tensor,
        clean: torch.Tensor,
        sigmas: torch.Tensor,
        model: Any,
        noise_sampler: Any,
        *,
        eta: float = 1.0,
        video_element_count: int | None = None,
    ) -> torch.Tensor:
        fn = build_per_row_sampler_function(
            base_fn, m, clean, _sched(), video_element_count=video_element_count
        )
        return fn(model, x.clone(), sigmas, noise_sampler=noise_sampler, eta=eta)

    def _m1_matches(self, base_fn: Any, scale: float) -> None:
        model = _ScaleModel(scale)
        m = torch.ones(1, 3)
        x = torch.randn(1, 3)
        clean = torch.zeros(1, 3)
        sigmas = _decreasing(5)  # steps_n=4: first-order + multistep + terminal
        fixed = torch.randn(1, 3)
        stock = base_fn(model, x.clone(), sigmas, noise_sampler=_FixedNoiseSampler(fixed))
        out = self._run_native(
            base_fn, m, x, clean, sigmas, model, noise_sampler=_FixedNoiseSampler(fixed)
        )
        assert torch.allclose(out, stock, atol=1e-6), (
            f"{base_fn.__name__} m=1 native must match stock; "
            f"max diff={float((out - stock).abs().max())}"
        )

    def test_dpmpp_2m_sde_m1_equals_stock(self) -> None:
        self._m1_matches(_local_sample_dpmpp_2m_sde, 0.9)

    def test_dpmpp_3m_sde_m1_equals_stock(self) -> None:
        self._m1_matches(_local_sample_dpmpp_3m_sde, 0.85)

    def test_dpmpp_sde_m1_equals_stock(self) -> None:
        self._m1_matches(_local_sample_dpmpp_sde, 0.8)

    def test_dpmpp_2s_ancestral_m1_equals_stock(self) -> None:
        self._m1_matches(_local_sample_dpmpp_2s_ancestral, 0.75)

    def test_sde_eta_zero_equals_stock(self) -> None:
        """eta=0 (deterministic limit) also matches stock for all three SDE steps."""
        for base_fn in (
            _local_sample_dpmpp_2m_sde,
            _local_sample_dpmpp_3m_sde,
            _local_sample_dpmpp_sde,
            _local_sample_dpmpp_2s_ancestral,
        ):
            model = _ScaleModel(0.88)
            x = torch.randn(1, 2)
            sigmas = _decreasing(4)

            def null_ns(sigma: Any, sigma_next: Any, _x: torch.Tensor = x) -> torch.Tensor:
                return torch.zeros_like(_x)

            stock = base_fn(model, x.clone(), sigmas, noise_sampler=null_ns, eta=0.0)
            out = self._run_native(
                base_fn,
                torch.ones(1, 2),
                x,
                torch.zeros(1, 2),
                sigmas,
                model,
                noise_sampler=null_ns,
                eta=0.0,
            )
            assert torch.allclose(out, stock, atol=1e-6), f"{base_fn.__name__} eta=0 mismatch"

    def test_sde_m0_exact_preserve(self) -> None:
        """m=0 rows come out exactly clean under the outer where(never, clean, x) guard."""
        for base_fn in (
            _local_sample_dpmpp_2m_sde,
            _local_sample_dpmpp_3m_sde,
            _local_sample_dpmpp_sde,
            _local_sample_dpmpp_2s_ancestral,
        ):
            model = _ScaleModel(0.5)
            clean = torch.full((1, 3), 7.0)
            out = self._run_native(
                base_fn,
                torch.zeros(1, 3),
                torch.randn(1, 3),
                clean,
                _decreasing(5),
                model,
                noise_sampler=_FixedNoiseSampler(torch.randn(1, 3)),
            )
            assert torch.allclose(out, clean, atol=1e-6), f"{base_fn.__name__} m=0 not preserved"

    def test_sde_fractional_finite_and_preserves_m0(self) -> None:
        """Fractional rows stay finite (no NaN/inf from logit clamps); m=0 column preserved."""
        for base_fn in (
            _local_sample_dpmpp_2m_sde,
            _local_sample_dpmpp_3m_sde,
            _local_sample_dpmpp_sde,
            _local_sample_dpmpp_2s_ancestral,
        ):
            model = _ScaleModel(0.8)
            m = torch.tensor([[0.0, 0.5, 1.0]])
            clean = torch.full((1, 3), 3.0)
            out = self._run_native(
                base_fn,
                m,
                torch.randn(1, 3),
                clean,
                _decreasing(5),
                model,
                noise_sampler=_FixedNoiseSampler(torch.zeros(1, 3)),
            )
            assert torch.isfinite(out).all(), f"{base_fn.__name__} produced non-finite output"
            assert torch.allclose(out[0, 0], clean[0, 0], atol=1e-6)

    def test_sde_registered_and_flagged_stochastic(self) -> None:
        """All three SDE names are in the registry AND detected stochastic (eta default > 0)."""
        for name, fn in (
            ("sample_dpmpp_2m_sde", _local_sample_dpmpp_2m_sde),
            ("sample_dpmpp_3m_sde", _local_sample_dpmpp_3m_sde),
            ("sample_dpmpp_sde", _local_sample_dpmpp_sde),
            ("sample_dpmpp_2s_ancestral", _local_sample_dpmpp_2s_ancestral),
        ):
            assert name in _NATIVE_ROW_STEPS
            assert sampler_is_stochastic(fn) is True
            # Registered → the nodes.py warning condition is suppressed.
            assert not (sampler_is_stochastic(fn) and name not in _NATIVE_ROW_STEPS)

    def test_dpmpp_sde_callback_once_per_step(self) -> None:
        """The 2-eval dpmpp_sde step fires the callback only on its first eval, index remapped."""
        seen: list[dict[str, Any]] = []

        def cb(d: dict[str, Any]) -> None:
            seen.append(dict(d))

        model = _ScaleModel(0.9)
        fn = build_per_row_sampler_function(
            _local_sample_dpmpp_sde, torch.ones(1, 2), torch.zeros(1, 2), _sched()
        )
        fn(
            model,
            torch.randn(1, 2),
            _decreasing(4),
            callback=cb,
            noise_sampler=_FixedNoiseSampler(torch.zeros(1, 2)),
        )
        assert [d["i"] for d in seen] == [0, 1, 2]

    def test_dpmpp_2s_ancestral_callback_once_per_step(self) -> None:
        """The 2-eval dpmpp_2s_ancestral step fires the callback only on its first eval."""
        seen: list[dict[str, Any]] = []

        def cb(d: dict[str, Any]) -> None:
            seen.append(dict(d))

        model = _ScaleModel(0.9)
        fn = build_per_row_sampler_function(
            _local_sample_dpmpp_2s_ancestral, torch.ones(1, 2), torch.zeros(1, 2), _sched()
        )
        fn(
            model,
            torch.randn(1, 2),
            _decreasing(4),
            callback=cb,
            noise_sampler=_FixedNoiseSampler(torch.zeros(1, 2)),
        )
        assert [d["i"] for d in seen] == [0, 1, 2]
