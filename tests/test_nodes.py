"""Tests for ComfyUI H3 Blended Inject nodes (comfyui_h3_blended_inject/nodes.py).

Pass/fail matrix at initial implementation stage:

PASS NOW (static schema already implemented):
  - All INPUT_TYPES / wiring tests (class attrs, key ordering, combo values, mappings).

FAIL NOW, PASS once implemented:
  - All behavioral tests in TestH3AddInjectBehavior and TestH3InjectSamplerBehavior.
    (add_inject / sample currently raise NotImplementedError.)
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from comfyui_h3_blended_inject.nodes import (
    INJECT_LIST,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    H3AddInject,
    H3InjectSampler,
)
from comfyui_h3_blended_inject.schedule import Inject

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INTERPOLATION_TYPES = ["ease_in", "ease_out", "ease_in_out", "linear", "none"]
AUDIO_MODES = ["fade", "drop", "keep"]


class FakeImages:
    """Minimal stand-in for a ComfyUI IMAGE tensor batch (shape only)."""

    def __init__(self, frames: int, h: int, w: int, c: int = 3) -> None:
        self.shape = (frames, h, w, c)


def _make_add_inject_args(**overrides: Any) -> dict[str, Any]:
    """Return a baseline valid set of kwargs for H3AddInject.add_inject.

    inject_at=34 is 2*17 — an exact multiple, so no snap warning is issued.
    images has H=W=64 (multiples of 32) and a degenerate-envelope-safe frame count.
    Envelope indices are ordered: 0 <= 5 <= 10 <= 15.
    """
    defaults: dict[str, Any] = {
        "inject_at": 34,
        "start_fade_in": 0,
        "start_keyframes": 5,
        "end_keyframes": 10,
        "end_fade_out": 15,
        "min_denoise": 0.4,
        "interpolation_type": "linear",
        "audio_mode": "fade",
        "inject_list": None,
        "images": FakeImages(16, 64, 64),
        "audio": None,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# INPUT_TYPES / wiring tests — PASS NOW
# ---------------------------------------------------------------------------


class TestH3AddInjectInputTypes:
    """Static INPUT_TYPES schema for H3AddInject.  All of these pass now."""

    def test_required_keys_order(self):
        required = list(H3AddInject.INPUT_TYPES()["required"].keys())
        assert required == [
            "inject_at",
            "start_fade_in",
            "start_keyframes",
            "end_keyframes",
            "end_fade_out",
            "min_denoise",
            "interpolation_type",
            "audio_mode",
        ]

    def test_optional_keys(self):
        optional = set(H3AddInject.INPUT_TYPES()["optional"].keys())
        assert optional == {"inject_list", "images", "audio", "vae", "audio_vae"}

    def test_interpolation_type_combo(self):
        combo = H3AddInject.INPUT_TYPES()["required"]["interpolation_type"][0]
        assert combo == ["ease_in", "ease_out", "ease_in_out", "linear", "none"]

    def test_audio_mode_combo(self):
        combo = H3AddInject.INPUT_TYPES()["required"]["audio_mode"][0]
        assert combo == ["fade", "drop", "keep"]

    def test_inject_list_optional_type_is_inject_list_string(self):
        inject_list_spec = H3AddInject.INPUT_TYPES()["optional"]["inject_list"]
        assert inject_list_spec[0] == INJECT_LIST

    def test_images_tooltip_mentions_24_fps(self):
        images_spec = H3AddInject.INPUT_TYPES()["optional"]["images"]
        tooltip: str = images_spec[1].get("tooltip", "")
        assert "24" in tooltip and "fps" in tooltip.lower()


class TestH3AddInjectClassAttrs:
    """Class-level attributes on H3AddInject.  Pass now."""

    def test_return_types(self):
        assert H3AddInject.RETURN_TYPES == (INJECT_LIST,)

    def test_function_name(self):
        assert H3AddInject.FUNCTION == "add_inject"

    def test_category_set(self):
        assert H3AddInject.CATEGORY  # non-empty string


class TestH3InjectSamplerInputTypes:
    """Static INPUT_TYPES schema for H3InjectSampler.  Pass now.

    Sampler/scheduler lists fall back to [] when comfy is absent, so calling
    INPUT_TYPES() in tests is safe.
    """

    def test_required_keys_present(self):
        required = set(H3InjectSampler.INPUT_TYPES()["required"].keys())
        expected = {
            "model",
            "add_noise",
            "noise_seed",
            "steps",
            "cfg",
            "sampler_name",
            "scheduler",
            "positive",
            "latent_image",
            "start_at_step",
            "end_at_step",
            "return_with_leftover_noise",
            "inject_list",
        }
        assert expected <= required

    def test_negative_is_optional_input(self):
        """H3 is CFG-distilled: negative must be optional, not required."""
        types = H3InjectSampler.INPUT_TYPES()
        required = set(types["required"].keys())
        optional = types.get("optional", {})
        assert "negative" not in required, "'negative' must not be in required"
        assert "negative" in optional, "'negative' must be in optional"
        # Type spec first element must be "CONDITIONING"
        assert optional["negative"][0] == "CONDITIONING"

    def test_optional_keys(self):
        optional = set(H3InjectSampler.INPUT_TYPES().get("optional", {}).keys())
        assert "negative" in optional

    def test_return_types(self):
        assert H3InjectSampler.RETURN_TYPES == ("LATENT",)

    def test_function_name(self):
        assert H3InjectSampler.FUNCTION == "sample"


class TestNodeMappings:
    """NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS.  Pass now."""

    def test_class_mappings_h3_add_inject(self):
        assert NODE_CLASS_MAPPINGS["H3AddInject"] is H3AddInject

    def test_class_mappings_h3_inject_sampler(self):
        assert NODE_CLASS_MAPPINGS["H3InjectSampler"] is H3InjectSampler

    def test_display_name_h3_add_inject(self):
        assert NODE_DISPLAY_NAME_MAPPINGS["H3AddInject"] == "H3 Add Inject"

    def test_display_name_h3_inject_sampler(self):
        assert NODE_DISPLAY_NAME_MAPPINGS["H3InjectSampler"] == "H3 Inject Sampler"

    def test_entry_point_re_exports_both_nodes(self):
        """Top-level __init__.py (the ComfyUI entry point) re-exports both node keys."""
        entry_path = Path(__file__).resolve().parents[1] / "__init__.py"
        spec = importlib.util.spec_from_file_location("h3_blended_inject_entry", entry_path)
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)

        assert "H3AddInject" in entry.NODE_CLASS_MAPPINGS
        assert "H3InjectSampler" in entry.NODE_CLASS_MAPPINGS

    def test_entry_point_loads_when_pack_not_on_sys_path(self):
        """Regression: entry __init__.py self-heals when pack dir is absent from sys.path.

        ComfyUI execs the entry by file path from ``custom_nodes/`` without adding the
        pack directory to ``sys.path``, so the sibling ``comfyui_h3_blended_inject``
        package would be unimportable by default.  The fix (``sys.path.insert`` in
        ``__init__.py``) must be present; this test fails if it is removed.

        Global state (``sys.path``, ``sys.modules``) is saved before the test and
        fully restored in ``finally`` so other tests are unaffected.
        """
        root = str(Path(__file__).resolve().parents[1])

        # Save global state that this test temporarily mutates.
        saved_path = list(sys.path)
        saved_modules = {
            k: v
            for k, v in sys.modules.items()
            if k == "comfyui_h3_blended_inject" or k.startswith("comfyui_h3_blended_inject.")
        }

        try:
            # Remove pack modules so import resolution starts from scratch.
            for k in list(saved_modules):
                del sys.modules[k]

            # Strip project-root entries from sys.path to replicate ComfyUI's env.
            sys.path[:] = [p for p in sys.path if p not in ("", root, root + os.sep)]

            # Precondition guard: the pack must now be unimportable.
            # (If the pack is ever pip-installed, this guard may stop raising;
            # the exec_module block below still validates the entry-point behaviour.)
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module("comfyui_h3_blended_inject")

            # Load exactly as ComfyUI does: exec the entry file by path.
            entry_path = Path(root) / "__init__.py"
            spec = importlib.util.spec_from_file_location("h3_entry_syspath_probe", entry_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            assert "H3AddInject" in mod.NODE_CLASS_MAPPINGS
            assert "H3InjectSampler" in mod.NODE_CLASS_MAPPINGS

        finally:
            # Restore global state so subsequent tests are unaffected.
            sys.path[:] = saved_path
            # Overwrite any freshly-loaded copies with the original module objects.
            sys.modules.update(saved_modules)


# ---------------------------------------------------------------------------
# Behavioral tests — FAIL NOW (NotImplementedError), PASS once implemented
# ---------------------------------------------------------------------------


class TestH3AddInjectBehavior:
    """Behavioral contract for H3AddInject.add_inject.

    All tests here FAIL NOW because add_inject raises NotImplementedError.
    They will PASS once the method body is implemented.
    """

    def test_returns_one_tuple_containing_one_inject(self):
        node = H3AddInject()
        result = node.add_inject(**_make_add_inject_args())

        assert isinstance(result, tuple)
        assert len(result) == 1
        inject_list = result[0]
        assert isinstance(inject_list, list)
        assert len(inject_list) == 1
        assert isinstance(inject_list[0], Inject)

    def test_inject_fields_reflect_args(self):
        node = H3AddInject()
        args = _make_add_inject_args(
            inject_at=34,
            start_fade_in=0,
            start_keyframes=5,
            end_keyframes=10,
            end_fade_out=15,
            min_denoise=0.3,
            interpolation_type="ease_in",
            audio_mode="drop",
        )
        (inject_list,) = node.add_inject(**args)
        inj = inject_list[0]

        assert inj.inject_at == 34
        assert inj.start_fade_in == 0
        assert inj.start_keyframes == 5
        assert inj.end_keyframes == 10
        assert inj.end_fade_out == 15
        assert inj.min_denoise == pytest.approx(0.3)
        assert inj.interpolation_type == "ease_in"
        assert inj.audio_mode == "drop"

    def test_inject_at_is_snapped_down_to_multiple_of_17(self):
        """inject_at=35 is not a multiple of 17; must snap down to 34 (= 2 * 17)."""
        node = H3AddInject()
        (inject_list,) = node.add_inject(**_make_add_inject_args(inject_at=35))
        assert inject_list[0].inject_at == 34

    def test_chaining_appends_second_inject_preserving_order(self):
        node = H3AddInject()

        # Use valid ordering: 0 <= 0 <= 4 <= 4 with end_fade_out=4 < source_length=16.
        (list_1,) = node.add_inject(
            **_make_add_inject_args(
                inject_at=17, start_fade_in=0, start_keyframes=0, end_keyframes=4, end_fade_out=4
            )
        )
        (list_2,) = node.add_inject(
            **_make_add_inject_args(
                inject_at=34,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=4,
                end_fade_out=4,
                inject_list=list_1,
            )
        )

        assert len(list_2) == 2
        assert list_2[0].inject_at == 17
        assert list_2[1].inject_at == 34

    @given(
        interpolation_type=st.sampled_from(INTERPOLATION_TYPES),
        audio_mode=st.sampled_from(AUDIO_MODES),
    )
    @settings(max_examples=25)
    def test_interpolation_type_and_audio_mode_round_trip(
        self, interpolation_type: str, audio_mode: str
    ) -> None:
        """Every documented enum value for interpolation_type and audio_mode
        round-trips onto the produced Inject unchanged."""
        node = H3AddInject()
        (inject_list,) = node.add_inject(
            **_make_add_inject_args(
                interpolation_type=interpolation_type,
                audio_mode=audio_mode,
            )
        )
        inj = inject_list[0]
        assert inj.interpolation_type == interpolation_type
        assert inj.audio_mode == audio_mode


class TestH3AddInjectBehaviorExtra:
    """Additional behavioral tests for H3AddInject.add_inject covering edge-case branches."""

    def test_ordering_violation_raises_value_error(self):
        """start_fade_in > start_keyframes must raise ValueError before constructing Inject."""
        node = H3AddInject()
        with pytest.raises(ValueError, match="ordering violated"):
            node.add_inject(**_make_add_inject_args(start_fade_in=10, start_keyframes=5))

    def test_add_inject_without_images_sets_source_length_zero(self):
        """images=None → source_length=0 and resolution=(0, 0) on the produced Inject."""
        node = H3AddInject()
        (inject_list,) = node.add_inject(**_make_add_inject_args(images=None, audio=None))
        inj = inject_list[0]
        assert inj.source_length == 0
        assert inj.resolution == (0, 0)
        assert inj.video_latent is None
        assert inj.images is None

    def test_add_inject_with_audio_and_images_sanitizes_audio(self):
        """When both audio and images are supplied, sanitize_audio is called (needs torch)."""
        import torch

        # 16 kHz waveform; sanitize_audio will resample to 32 kHz and trim/pad.
        audio = {"waveform": torch.zeros(1, 16000), "sample_rate": 16000}
        node = H3AddInject()
        (inject_list,) = node.add_inject(**_make_add_inject_args(audio=audio))
        inj = inject_list[0]
        assert inj.audio is not None
        # After resampling the sample_rate should be updated to the target.
        assert inj.audio["sample_rate"] == 32000

    def test_add_inject_encodes_images_with_fake_vae(self):
        """FakeVAE.encode() result is stored as video_latent on the produced Inject."""
        import torch

        class FakeVAE:
            sentinel = torch.zeros(1)

            def encode(self, x: Any) -> Any:
                return self.sentinel

        node = H3AddInject()
        (inject_list,) = node.add_inject(**_make_add_inject_args(vae=FakeVAE()))
        assert inject_list[0].video_latent is FakeVAE.sentinel

    def test_audio_latent_is_none_when_no_audio_vae(self):
        """audio_vae=None → audio_latent=None even when images and audio are present."""
        import torch

        audio = {"waveform": torch.zeros(1, 8000), "sample_rate": 16000}
        node = H3AddInject()
        (inject_list,) = node.add_inject(**_make_add_inject_args(audio=audio, audio_vae=None))
        assert inject_list[0].audio_latent is None


class TestH3InjectSamplerBehavior:
    """Behavioral contract for H3InjectSampler.sample.

    Deeper sampler mechanics (hold/release math, mask derivation, schedule merge) are
    covered by the dedicated modules: test_hold_release.py, test_mask.py, test_schedule.py.
    This file tests only the node's public validation contract.
    """

    def test_resolution_mismatch_raises_value_error(self):
        """An inject whose resolution does not match the target latent must raise ValueError.

        The inject has resolution (64, 64); the latent tensor implies a different size.
        Once implemented, sample() should validate resolution before attempting to sample
        and raise ValueError with the offending values.
        """
        import torch

        # Construct the mismatched inject directly to avoid depending on add_inject
        # (which also raises NotImplementedError at this stage).
        mismatched_inject = Inject(
            inject_at=0,
            start_fade_in=0,
            start_keyframes=0,
            end_keyframes=3,
            end_fade_out=3,
            min_denoise=0.5,
            interpolation_type="linear",
            audio_mode="fade",
            images=FakeImages(4, 64, 64),  # 64x64
            audio=None,
            resolution=(64, 64),
            source_length=4,
        )

        # Latent samples shape implying a different spatial resolution than 64x64.
        latent_image: dict[str, Any] = {"samples": torch.zeros(1, 16, 16, 16)}

        node = H3InjectSampler()
        with pytest.raises(ValueError):
            node.sample(
                model=object(),
                add_noise="enable",
                noise_seed=0,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                positive=object(),
                negative=object(),
                latent_image=latent_image,
                start_at_step=0,
                end_at_step=20,
                return_with_leftover_noise="disable",
                inject_list=[mismatched_inject],
            )

    def test_envelope_violation_raises_when_no_images(self):
        """Inject with no images skips check_resolution but hits validate_envelope_indices."""
        import torch

        # start_fade_in=10 > start_keyframes=0 violates ordering.
        invalid_inject = Inject(
            inject_at=0,
            start_fade_in=10,
            start_keyframes=0,
            end_keyframes=5,
            end_fade_out=9,
            min_denoise=0.5,
            interpolation_type="linear",
            audio_mode="fade",
            images=None,
            audio=None,
            resolution=(0, 0),
            source_length=20,
        )
        latent_image: dict[str, Any] = {"samples": torch.zeros(1, 100, 8, 8)}

        node = H3InjectSampler()
        with pytest.raises(ValueError):
            node.sample(
                model=object(),
                add_noise="enable",
                noise_seed=0,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                positive=object(),
                negative=object(),
                latent_image=latent_image,
                start_at_step=0,
                end_at_step=20,
                return_with_leftover_noise="disable",
                inject_list=[invalid_inject],
            )

    def test_sample_calls_merge_and_mask_before_gpu(self):
        """Valid inject (no images) passes both checks; merge_schedule and apply_derived_mask run.

        After the CPU-testable steps succeed, the call hits the GPU-only _run_sampler helper
        which tries model.clone() on a plain object() and raises AttributeError.
        """
        import torch

        # Degenerate still-inject with no images; envelope is valid.
        valid_inject = Inject(
            inject_at=0,
            start_fade_in=0,
            start_keyframes=0,
            end_keyframes=0,
            end_fade_out=0,
            min_denoise=0.5,
            interpolation_type="linear",
            audio_mode="drop",
            images=None,
            audio=None,
            resolution=(0, 0),
            source_length=1,
        )
        # 4-dim fake tensor; target_rows = shape[1] = 5.
        latent_image: dict[str, Any] = {"samples": torch.zeros(1, 5, 8, 8)}

        node = H3InjectSampler()
        with pytest.raises(AttributeError):
            node.sample(
                model=object(),
                add_noise="enable",
                noise_seed=0,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                positive=object(),
                negative=object(),
                latent_image=latent_image,
                start_at_step=0,
                end_at_step=20,
                return_with_leftover_noise="disable",
                inject_list=[valid_inject],
            )

    def _make_valid_inject_and_latent(self):
        """Return a (valid_inject, latent_image) pair usable in sampler stub tests."""
        import torch

        inj = Inject(
            inject_at=0,
            start_fade_in=0,
            start_keyframes=0,
            end_keyframes=0,
            end_fade_out=0,
            min_denoise=0.5,
            interpolation_type="linear",
            audio_mode="drop",
            images=None,
            audio=None,
            resolution=(0, 0),
            source_length=1,
        )
        latent_image: dict[str, Any] = {"samples": torch.zeros(1, 5, 8, 8)}
        return inj, latent_image

    def test_sample_accepts_none_negative_and_forwards_it(self, monkeypatch):
        """Regression: negative must be optional — sample() must accept no negative kwarg
        and must forward negative=None to _run_sampler unchanged.
        """
        import comfyui_h3_blended_inject.nodes as nodes_mod

        calls: list[dict] = []

        def stub_run_sampler(**kwargs):
            calls.append(kwargs)
            return ({"samples": None},)

        monkeypatch.setattr(nodes_mod, "_run_sampler", stub_run_sampler)

        inj, latent_image = self._make_valid_inject_and_latent()
        node = H3InjectSampler()
        # Do NOT pass negative — it must default to None.
        node.sample(
            model=object(),
            add_noise="enable",
            noise_seed=0,
            steps=20,
            cfg=7.0,
            sampler_name="euler",
            scheduler="normal",
            positive=object(),
            latent_image=latent_image,
            start_at_step=0,
            end_at_step=20,
            return_with_leftover_noise="disable",
            inject_list=[inj],
        )
        assert len(calls) == 1
        assert calls[0]["negative"] is None

    def test_sample_forwards_explicit_negative_verbatim(self, monkeypatch):
        """When a negative is provided, sample() must forward it verbatim to _run_sampler."""
        import comfyui_h3_blended_inject.nodes as nodes_mod

        calls: list[dict] = []

        def stub_run_sampler(**kwargs):
            calls.append(kwargs)
            return ({"samples": None},)

        monkeypatch.setattr(nodes_mod, "_run_sampler", stub_run_sampler)

        sentinel_negative = object()
        inj, latent_image = self._make_valid_inject_and_latent()
        node = H3InjectSampler()
        node.sample(
            model=object(),
            add_noise="enable",
            noise_seed=0,
            steps=20,
            cfg=7.0,
            sampler_name="euler",
            scheduler="normal",
            positive=object(),
            latent_image=latent_image,
            start_at_step=0,
            end_at_step=20,
            return_with_leftover_noise="disable",
            inject_list=[inj],
            negative=sentinel_negative,
        )
        assert len(calls) == 1
        assert calls[0]["negative"] is sentinel_negative

    def test_nested_tensor_latent_does_not_crash_at_dim(self, monkeypatch):
        """Regression: NestedTensor latent (FLOW_AV) must not raise AttributeError.

        Before the fix, sample() called samples.dim() which NestedTensor does not
        implement, crashing on GPU with a real H3 latent.  After the fix, sample()
        must detect is_nested and extract dims from .tensors[0] / .tensors[1].

        This test MUST fail before the fix and pass after.
        """
        import torch

        import comfyui_h3_blended_inject.nodes as nodes_mod

        calls: list[dict] = []

        def stub_run_sampler(**kwargs):
            calls.append(kwargs)
            return ({"samples": None},)

        monkeypatch.setattr(nodes_mod, "_run_sampler", stub_run_sampler)
        # Bypass mask creation so comfy is not needed on this CPU path.
        monkeypatch.setattr(nodes_mod, "apply_derived_mask", lambda lat, *a, **kw: lat)

        # Fake NestedTensor that matches the H3 FLOW_AV structure.
        video = torch.zeros(1, 24, 5, 4, 4)  # [B=1, C=24, T=5, Hl=4, Wl=4]
        audio = torch.zeros(1, 32, 2, 7)  # [B=1, C=32, 2, audio_t=7]

        class FakeNestedTensor:
            is_nested = True

            def __init__(self) -> None:
                self.tensors = [video, audio]

        valid_inject = Inject(
            inject_at=0,
            start_fade_in=0,
            start_keyframes=0,
            end_keyframes=0,
            end_fade_out=0,
            min_denoise=0.5,
            interpolation_type="linear",
            audio_mode="drop",
            images=None,
            audio=None,
            resolution=(0, 0),
            source_length=1,
        )

        node = H3InjectSampler()
        # Must NOT raise AttributeError ("NestedTensor has no attribute dim").
        node.sample(
            model=object(),
            add_noise="enable",
            noise_seed=0,
            steps=20,
            cfg=7.0,
            sampler_name="euler",
            scheduler="normal",
            positive=object(),
            latent_image={"samples": FakeNestedTensor()},
            start_at_step=0,
            end_at_step=20,
            return_with_leftover_noise="disable",
            inject_list=[valid_inject],
        )
        assert len(calls) == 1, "sample() must call _run_sampler for a nested latent"
