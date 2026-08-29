"""Tests for ComfyUI H3 Blended Inject nodes (comfyui_h3_blended_inject/nodes.py)."""

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

from comfyui_h3_blended_inject import grid
from comfyui_h3_blended_inject.nodes import (
    INJECT_LIST,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    H3AddGuide,
    H3AddInject,
    H3InjectSampler,
    _unpack_av,
)
from comfyui_h3_blended_inject.schedule import Guide, Inject

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INTERPOLATION_TYPES = ["ease_in", "ease_out", "ease_in_out", "linear", "none"]
AUDIO_MODES = ["fade", "drop", "keep"]


class FakeImages:
    """Minimal stand-in for a ComfyUI IMAGE tensor batch (shape only).

    Supports slicing (``fake[:n]``) so that add_inject's content-length trim path does not
    crash when FakeImages is used in tests where the snap-down path is exercised.
    """

    def __init__(self, frames: int, h: int, w: int, c: int = 3) -> None:
        self.shape = (frames, h, w, c)

    def __getitem__(self, key: slice) -> FakeImages:
        if isinstance(key, slice):
            start, stop, step = key.indices(self.shape[0])
            new_frames = len(range(start, stop, step or 1))
            return FakeImages(new_frames, self.shape[1], self.shape[2], self.shape[3])
        raise TypeError(f"FakeImages indices must be slices, not {type(key).__name__}")


class FakeEncodeVAE:
    """Minimal VAE stand-in: encode() returns a fresh tensor-ish sentinel.

    Baseline add_inject kwargs must include a vae whenever images are present — the
    node now errors on images-without-vae instead of silently dropping the content.
    """

    def encode(self, x: Any) -> Any:
        import torch

        return torch.zeros(1)


class FakeAudioVAE:
    """Minimal audio-VAE stand-in for the audio-without-audio_vae guard.

    ``audio_sample_rate`` matches the real H3 default (32 kHz) so sanitize/resample
    behavior is unchanged; encode() returns a shape-plausible zero latent.
    """

    audio_sample_rate = 32000

    def encode(self, x: Any) -> Any:
        import torch

        return torch.zeros(1, 32, 2, 4)


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
        "images": FakeImages(22, 64, 64),  # 22 = 5+17*1, a valid H3 clip length (17n+5)
        "audio": None,
        "vae": FakeEncodeVAE(),  # images-without-vae now raises; baseline must wire one
    }
    defaults.update(overrides)
    # Images without a VAE is now a hard error; drop the default vae only when the
    # override explicitly removed the images too.
    if defaults.get("images") is None and "vae" not in overrides:
        defaults["vae"] = None
    return defaults


# ---------------------------------------------------------------------------
# _unpack_av
# ---------------------------------------------------------------------------


class TestUnpackAv:
    """Direct unit tests for the _unpack_av helper (nodes.py:77).

    _unpack_av is called only from _run_sampler (# pragma: no cover), so its body
    never executes during the GPU path.  These tests exercise both branches directly.
    """

    def test_nested_with_audio_returns_both_components(self) -> None:
        """NestedTensor path: is_nested=True, unbind() returns [video, audio]."""
        video = object()
        audio = object()

        class FakeNested:
            is_nested = True

            def unbind(self) -> list[object]:
                return [video, audio]

        result_video, result_audio = _unpack_av(FakeNested())
        assert result_video is video
        assert result_audio is audio

    def test_nested_without_audio_returns_none_audio(self) -> None:
        """NestedTensor path with only one component: audio is None."""
        video = object()

        class FakeNestedVideoOnly:
            is_nested = True

            def unbind(self) -> list[object]:
                return [video]

        result_video, result_audio = _unpack_av(FakeNestedVideoOnly())
        assert result_video is video
        assert result_audio is None

    def test_plain_tensor_returns_samples_and_none(self) -> None:
        """Non-nested path: no is_nested attribute → (samples, None)."""
        plain = object()
        result_video, result_audio = _unpack_av(plain)
        assert result_video is plain
        assert result_audio is None

    def test_is_nested_false_treated_as_plain(self) -> None:
        """Explicit is_nested=False is treated the same as a plain tensor."""

        class FakeNotNested:
            is_nested = False

        fake = FakeNotNested()
        result_video, result_audio = _unpack_av(fake)
        assert result_video is fake
        assert result_audio is None


# ---------------------------------------------------------------------------
# INPUT_TYPES / wiring tests
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
        # Assert on the specific phrase "24 fps" (not just the digit "24") and on
        # "native" — which captures the key assumption that no fps conversion is done.
        assert "24 fps" in tooltip, f"tooltip should contain '24 fps': {tooltip!r}"
        assert "native" in tooltip.lower(), (
            f"tooltip should state the 24 fps assumption is native (no conversion): {tooltip!r}"
        )


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
            "noise_seed",
            "steps",
            "cfg",
            "sampler_name",
            "scheduler",
            "positive",
            "latent_image",
        }
        assert expected <= required

    def test_inject_list_is_optional_not_required(self):
        """inject_list must be optional so the node runs with zero injects (passthrough)."""
        types = H3InjectSampler.INPUT_TYPES()
        assert "inject_list" not in types["required"], "inject_list must not be required"
        optional = types.get("optional", {})
        assert "inject_list" in optional, "inject_list must be in optional"
        assert optional["inject_list"][0] == INJECT_LIST

    def test_chaining_widgets_hidden(self):
        """Chaining widgets are deliberately hidden while unsupported (per-row
        compression makes add_noise=disable / partial ranges / leftover noise
        per-row-incorrect).  Assert they stay out of the node surface."""
        types = H3InjectSampler.INPUT_TYPES()
        surface = set(types["required"].keys()) | set(types.get("optional", {}).keys())
        for key in (
            "add_noise",
            "start_at_step",
            "end_at_step",
            "return_with_leftover_noise",
        ):
            assert key not in surface, f"chaining widget {key!r} must stay hidden"

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

    def test_crossfade_widget_removed(self):
        """The crossfade toggle was removed with the per-row img2img rework."""
        optional = H3InjectSampler.INPUT_TYPES().get("optional", {})
        assert "crossfade" not in optional, "'crossfade' widget must no longer be present"

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

    def test_entry_point_loads_when_pack_not_on_sys_path(self, monkeypatch):
        """Regression: entry __init__.py self-heals when pack dir is absent from sys.path.

        ComfyUI execs the entry by file path from ``custom_nodes/`` without adding the
        pack directory to ``sys.path``, so the sibling ``comfyui_h3_blended_inject``
        package would be unimportable by default.  The fix (``sys.path.insert`` in
        ``__init__.py``) must be present; this test fails if it is removed.

        monkeypatch auto-restores ``sys.path`` and the evicted ``sys.modules`` entries
        at teardown, so no manual save/restore is needed.
        """
        root = str(Path(__file__).resolve().parents[1])

        # Evict pack modules so import resolution starts from scratch; monkeypatch
        # restores each entry at teardown.
        pack_keys = [
            k
            for k in sys.modules
            if k == "comfyui_h3_blended_inject" or k.startswith("comfyui_h3_blended_inject.")
        ]
        for k in pack_keys:
            monkeypatch.delitem(sys.modules, k)

        # Strip project-root entries from sys.path to replicate ComfyUI's env;
        # monkeypatch restores the original sys.path attribute at teardown.
        clean_path = [p for p in sys.path if p not in ("", root, root + os.sep)]
        monkeypatch.setattr(sys, "path", clean_path)

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


# ---------------------------------------------------------------------------
# Behavioral tests
# ---------------------------------------------------------------------------


class TestH3AddInjectBehavior:
    """Behavioral contract for H3AddInject.add_inject."""

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
        (inject_list,) = node.add_inject(
            **_make_add_inject_args(audio=audio, audio_vae=FakeAudioVAE())
        )
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

    def test_audio_without_audio_vae_raises(self):
        """audio + audio_mode!='drop' + no audio_vae → hard error (was silently dropped)."""
        import torch

        audio = {"waveform": torch.zeros(1, 8000), "sample_rate": 16000}
        node = H3AddInject()
        with pytest.raises(ValueError, match="audio_vae"):
            node.add_inject(**_make_add_inject_args(audio=audio, audio_vae=None))

    def test_audio_drop_mode_without_audio_vae_ok(self):
        """audio_mode='drop' ignores audio content, so no audio_vae is required."""
        import torch

        audio = {"waveform": torch.zeros(1, 8000), "sample_rate": 16000}
        node = H3AddInject()
        (inject_list,) = node.add_inject(
            **_make_add_inject_args(audio=audio, audio_mode="drop", audio_vae=None)
        )
        assert inject_list[0].audio_latent is None

    def test_images_without_vae_raises(self):
        """images + no vae → hard error instead of silently ignored inject content."""
        node = H3AddInject()
        with pytest.raises(ValueError, match="vae"):
            node.add_inject(**_make_add_inject_args(vae=None))


class TestH3InjectSamplerBehavior:
    """Behavioral contract for H3InjectSampler.sample.

    Deeper sampler mechanics (per-row img2img lerp, fractional mask, clean-reference
    composite, schedule merge) are covered by the dedicated modules: test_sampler.py,
    test_mask_fractional.py, test_composite.py, test_schedule.py.
    This file tests only the node's public validation contract.
    """

    def test_resolution_mismatch_raises_value_error(self):
        """An inject whose resolution does not match the target latent must raise ValueError.

        The inject has resolution (64, 64); the latent tensor implies a different size.
        Once implemented, sample() should validate resolution before attempting to sample
        and raise ValueError with the offending values.
        """
        import torch

        # Construct the mismatched inject directly to avoid depending on add_inject.
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
                noise_seed=0,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                positive=object(),
                negative=object(),
                latent_image=latent_image,
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
                noise_seed=0,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                positive=object(),
                negative=object(),
                latent_image=latent_image,
                inject_list=[invalid_inject],
            )

    def test_sample_calls_merge_before_gpu(self):
        """Valid inject (no images) passes both checks; merge_schedule runs.

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
                noise_seed=0,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                positive=object(),
                negative=object(),
                latent_image=latent_image,
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
            noise_seed=0,
            steps=20,
            cfg=7.0,
            sampler_name="euler",
            scheduler="normal",
            positive=object(),
            latent_image=latent_image,
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
            noise_seed=0,
            steps=20,
            cfg=7.0,
            sampler_name="euler",
            scheduler="normal",
            positive=object(),
            latent_image=latent_image,
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
        # The per-row img2img pipeline builds the clean reference + fractional mask inside
        # _run_sampler (stubbed here), so sample() itself needs no comfy on this CPU path.

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
            noise_seed=0,
            steps=20,
            cfg=7.0,
            sampler_name="euler",
            scheduler="normal",
            positive=object(),
            latent_image={"samples": FakeNestedTensor()},
            inject_list=[valid_inject],
        )
        assert len(calls) == 1, "sample() must call _run_sampler for a nested latent"


# ---------------------------------------------------------------------------
# E8: content-length snapping and audio tail-alignment warning
# ---------------------------------------------------------------------------


class TestE8LengthSnapping:
    """E8: snap injected content length down to the 17n+5 grid; trim image/audio tensors.

    Fail-then-pass requirement: each test that covers new behaviour must FAIL before
    snap_length_down is wired into add_inject and PASS after.
    """

    @staticmethod
    def _images(frames: int, h: int = 64, w: int = 64) -> Any:
        """Return a real torch tensor [frames, H, W, 3] for use as inject images."""
        import torch

        return torch.zeros(frames, h, w, 3)

    @staticmethod
    def _audio(samples: int = 8000, sr: int = 16000) -> dict:
        import torch

        return {"waveform": torch.zeros(1, samples), "sample_rate": sr}

    # -- Snap-down: tensor trimmed, source_length updated ----------------------

    def test_snap_100_to_90_trims_image_tensor(self):
        """Source 100 → snaps to 90; image tensor trimmed; source_length=90."""
        node = H3AddInject()
        images = self._images(100)
        with pytest.warns(UserWarning):
            (inject_list,) = node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=89,
                end_fade_out=89,
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="drop",
                images=images,
                vae=FakeEncodeVAE(),
            )
        inj = inject_list[0]
        assert inj.source_length == 90
        assert inj.images.shape[0] == 90

    def test_snap_45_to_39_trims_image_tensor(self):
        """Source 45 → snaps to 39 (= 5+17*2); image tensor trimmed; source_length=39."""
        node = H3AddInject()
        images = self._images(45)
        with pytest.warns(UserWarning):
            (inject_list,) = node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=38,
                end_fade_out=38,
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="drop",
                images=images,
                vae=FakeEncodeVAE(),
            )
        inj = inject_list[0]
        assert inj.source_length == 39
        assert inj.images.shape[0] == 39

    def test_snap_20_to_5_trims_image_tensor(self):
        """Source 20 → snaps to 5 (= 5+17*0); image tensor trimmed; source_length=5."""
        node = H3AddInject()
        images = self._images(20)
        with pytest.warns(UserWarning):
            (inject_list,) = node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=4,
                end_fade_out=4,
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="drop",
                images=images,
                vae=FakeEncodeVAE(),
            )
        inj = inject_list[0]
        assert inj.source_length == 5
        assert inj.images.shape[0] == 5

    def test_snap_trims_audio_to_match_snapped_length(self):
        """When source 100 → 90 and audio is present, audio is sanitized for 90 frames."""
        import torch

        node = H3AddInject()
        images = self._images(100)
        # Audio longer than 90 frames' worth at 32 kHz should be trimmed.
        audio = {"waveform": torch.zeros(1, 200000), "sample_rate": 32000}
        with pytest.warns(UserWarning):
            (inject_list,) = node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=89,
                end_fade_out=89,
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="drop",
                images=images,
                vae=FakeEncodeVAE(),
                audio=audio,
                audio_vae=FakeAudioVAE(),
            )
        inj = inject_list[0]
        assert inj.source_length == 90
        # Audio should be trimmed to 90-frame duration at 32 kHz (target_sample_rate default 32000).
        expected_samples = round(90 / 24 * 32000)
        assert inj.audio["waveform"].shape[-1] == expected_samples

    # -- No-op when already valid ----------------------------------------------

    def test_no_op_when_source_length_is_39(self):
        """Source 39 (= 5+17*2): valid 17n+5 → returned unchanged; no UserWarning from snap."""
        import warnings

        node = H3AddInject()
        images = self._images(39)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            (inject_list,) = node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=38,
                end_fade_out=38,
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="drop",
                images=images,
                vae=FakeEncodeVAE(),
            )
        snap_warns = [
            x for x in w if "17n+5" in str(x.message) or "snapping" in str(x.message).lower()
        ]
        assert not snap_warns, "No snap warning expected for already-valid length 39"
        assert inject_list[0].source_length == 39
        assert inject_list[0].images.shape[0] == 39

    def test_no_op_when_source_length_is_90(self):
        """Source 90 (= 5+17*5): valid 17n+5 → returned unchanged."""
        node = H3AddInject()
        images = self._images(90)
        # No pytest.warns needed — snap warning must NOT be emitted.
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            (inject_list,) = node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=89,
                end_fade_out=89,
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="drop",
                images=images,
                vae=FakeEncodeVAE(),
            )
        snap_warns = [
            x for x in w if "17n+5" in str(x.message) or "snapping" in str(x.message).lower()
        ]
        assert not snap_warns
        assert inject_list[0].source_length == 90

    # -- Hard error: source_length < 5 -----------------------------------------

    def test_error_source_length_4(self):
        """source_length=4 < 5 → ValueError (minimum valid H3 clip length is 5)."""
        node = H3AddInject()
        images = self._images(4)
        with pytest.raises(ValueError, match="5"):
            node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=3,
                end_fade_out=3,
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="drop",
                images=images,
                vae=FakeEncodeVAE(),
            )

    def test_error_source_length_1_is_now_valid(self):
        """source_length=1 at a valid chunk boundary with audio_mode='drop' succeeds.

        F=1 is now a first-class valid inject length (single-keyframe path).
        This test was updated from 'expects ValueError' to 'expects success' when the
        valid-length set changed from {17n+5} to {1} ∪ {17n+5}.
        inject_at=0 is a multiple of 17 and audio_mode='drop' → all guards pass.
        """
        node = H3AddInject()
        images = self._images(1)
        (inject_list,) = node.add_inject(
            inject_at=0,
            start_fade_in=0,
            start_keyframes=0,
            end_keyframes=0,
            end_fade_out=0,
            min_denoise=0.0,
            interpolation_type="linear",
            audio_mode="drop",
            images=images,
            vae=FakeEncodeVAE(),
        )
        inj = inject_list[0]
        assert inj.source_length == 1

    # -- Hard error: fade index outside post-trim context ----------------------

    def test_error_efo_exceeds_snapped_length(self):
        """Source 100 → snaps to 90; end_fade_out=95 > 90 → ValueError at add_inject time.

        FAIL-THEN-PASS: Before snap_length_down is wired in, efo=95 <= 100 passes; no error.
        After: snap to 90, efo=95 > 90 → ValueError.
        """
        node = H3AddInject()
        images = self._images(100)
        with pytest.raises(ValueError, match="end_fade_out"):
            node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=90,
                end_fade_out=95,  # > snapped_length=90 → must ERROR
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="drop",
                images=images,
                vae=FakeEncodeVAE(),
            )

    def test_error_efo_just_over_snapped_length(self):
        """Source 100, end_fade_out=91 > snapped_length=90 → ValueError."""
        node = H3AddInject()
        images = self._images(100)
        with pytest.raises(ValueError):
            node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=90,
                end_fade_out=91,
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="drop",
                images=images,
                vae=FakeEncodeVAE(),
            )

    # -- Half-open boundary: efo == snapped_length is ACCEPTED -----------------

    def test_half_open_efo_equals_snapped_length_accepted(self):
        """efo == snapped_length (90) is VALID: exclusive upper bound, half-open model.

        FAIL-THEN-PASS: Without snap, source_length=100, so efo=90 passes trivially
        (90 <= 100); BUT inj.source_length would be 100, not 90. After snap: source=90,
        efo=90 == 90 is valid, and inj.source_length==90.
        """
        node = H3AddInject()
        images = self._images(100)
        with pytest.warns(UserWarning):
            (inject_list,) = node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=89,
                end_fade_out=90,  # == snapped_length → half-open, valid
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="drop",
                images=images,
                vae=FakeEncodeVAE(),
            )
        inj = inject_list[0]
        assert inj.source_length == 90  # snapped; fails without implementation (would be 100)
        assert inj.end_fade_out == 90


# ---------------------------------------------------------------------------
# E8: audio tail-alignment warning (from add_inject)
# ---------------------------------------------------------------------------


class TestE8AudioTailAlignment:
    """E8 addition: warn when non-audio-aligned clip end is exposed without a fade-out.

    Fail-then-pass: Before warn_audio_tail_alignment is wired in, no warning is emitted;
    tests expecting pytest.warns fail. After wiring, tests pass.
    """

    @staticmethod
    def _images(frames: int) -> Any:
        import torch

        return torch.zeros(frames, 64, 64, 3)

    @staticmethod
    def _audio() -> dict:
        import torch

        return {"waveform": torch.zeros(1, 8000), "sample_rate": 16000}

    def _add_inject(
        self, frames: int, audio_mode: str, end_keyframes: int, end_fade_out: int
    ) -> Any:
        """Helper: call add_inject with given params; images=frames tensor, audio always present."""
        node = H3AddInject()
        # Suppress snap-length warnings so only the tail-alignment warning is checked.
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=end_keyframes,
                end_fade_out=end_fade_out,
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode=audio_mode,
                images=self._images(frames),
                audio=self._audio(),
                audio_vae=FakeAudioVAE(),
            )
        return [str(x.message) for x in w if issubclass(x.category, UserWarning)]

    # -- WARN cases ------------------------------------------------------------

    def test_warn_keep_mode_non_aligned_length(self):
        """keep-mode + non-aligned length (56 = 5+17*3) → tail warning emitted.

        FAIL-THEN-PASS: Without the warning logic, no UserWarning is emitted.
        """
        node = H3AddInject()
        # 56 is a valid 17n+5 length; ceil(56/17)=4, 4%3=1 → not audio-aligned.
        with pytest.warns(UserWarning, match="audio-sync-aligned"):
            node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=55,
                end_fade_out=55,  # no fade-out ramp: end_fade_out == end_keyframes
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="keep",
                images=self._images(56),
                vae=FakeEncodeVAE(),
                audio=self._audio(),
                audio_vae=FakeAudioVAE(),
            )

    def test_warn_fade_mode_unfaded_end_non_aligned(self):
        """fade-mode + clip end not faded through + non-aligned length → tail warning.

        Un-faded means end_fade_out < snapped_length (ramp exists but doesn't reach tail).
        """
        node = H3AddInject()
        # 56 frames; fade-out ramp [50, 55) doesn't reach tail at 56.
        with pytest.warns(UserWarning, match="audio-sync-aligned"):
            node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=50,
                end_fade_out=55,  # ramp exists but < snapped_length=56 → not faded through
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="fade",
                images=self._images(56),
                vae=FakeEncodeVAE(),
                audio=self._audio(),
                audio_vae=FakeAudioVAE(),
            )

    def test_warn_fade_mode_no_ramp_at_all_non_aligned(self):
        """fade-mode + no fade-out ramp (end_keyframes == end_fade_out) + non-aligned → warns."""
        node = H3AddInject()
        with pytest.warns(UserWarning, match="audio-sync-aligned"):
            node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=55,
                end_fade_out=55,  # no ramp → not faded through
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="fade",
                images=self._images(56),
                vae=FakeEncodeVAE(),
                audio=self._audio(),
                audio_vae=FakeAudioVAE(),
            )

    # -- NO-WARN cases ---------------------------------------------------------

    def test_warn_keep_mode_video_fadeout_reaching_tail(self):
        """Regression: keep-mode + video fade-out reaching tail → MUST warn (end-to-end).

        Bug: is_faded_through was not gated on fade mode; keep+non-aligned+faded-through
        video envelope wrongly suppressed the warning.  After fix, keep is never
        treated as faded-through and always warns when non-aligned + has_audio.

        FAIL-THEN-PASS: DID NOT WARN against the unfixed code; warns after fix.
        """
        node = H3AddInject()
        # 56 = 5+17*3, not audio-aligned; video fade-out end_keyframes=50, end_fade_out=56
        # reaches the tail of the 56-frame clip.  keep mode must still warn.
        with pytest.warns(UserWarning, match="audio-sync-aligned"):
            node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=50,
                end_fade_out=56,  # == snapped_length=56, video ramp reaches tail
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="keep",
                images=self._images(56),
                vae=FakeEncodeVAE(),
                audio=self._audio(),
                audio_vae=FakeAudioVAE(),
            )

    def test_no_warn_fade_mode_faded_through(self):
        """fade-mode WITH fade-out ramp reaching clip tail → no tail-alignment warning.

        end_fade_out == snapped_length (56) AND end_fade_out > end_keyframes → faded through.
        """
        import warnings

        node = H3AddInject()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=50,
                end_fade_out=56,  # == snapped_length=56 → faded through → no warn
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="fade",
                images=self._images(56),
                vae=FakeEncodeVAE(),
                audio=self._audio(),
                audio_vae=FakeAudioVAE(),
            )
        tail_warns = [x for x in w if "audio-sync-aligned" in str(x.message)]
        assert not tail_warns, "Faded-through tail must NOT trigger tail-alignment warning"

    def test_no_warn_audio_aligned_length_keep(self):
        """Audio-aligned length (39 = 51*0+39) + keep-mode → no tail-alignment warning.

        ceil(39/17)=3, 3%3=0 → aligned; condition 1 fails → no warning.
        """
        import warnings

        node = H3AddInject()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=38,
                end_fade_out=38,
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="keep",
                images=self._images(39),
                vae=FakeEncodeVAE(),
                audio=self._audio(),
                audio_vae=FakeAudioVAE(),
            )
        tail_warns = [x for x in w if "audio-sync-aligned" in str(x.message)]
        assert not tail_warns, "Audio-aligned length must NOT trigger tail-alignment warning"

    def test_no_warn_drop_mode(self):
        """drop-mode (no injected audio) → no tail-alignment warning regardless of length."""
        import warnings

        node = H3AddInject()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=55,
                end_fade_out=55,
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="drop",
                images=self._images(56),
                vae=FakeEncodeVAE(),
                audio=self._audio(),  # audio present but mode=drop
                audio_vae=FakeAudioVAE(),
            )
        tail_warns = [x for x in w if "audio-sync-aligned" in str(x.message)]
        assert not tail_warns, "drop-mode must NOT trigger tail-alignment warning"

    def test_no_warn_no_audio(self):
        """No audio provided → no tail-alignment warning (nothing to misalign)."""
        import warnings

        node = H3AddInject()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=55,
                end_fade_out=55,
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="keep",
                images=self._images(56),
                vae=FakeEncodeVAE(),
                audio=None,  # no audio
            )
        tail_warns = [x for x in w if "audio-sync-aligned" in str(x.message)]
        assert not tail_warns


# ---------------------------------------------------------------------------
# F=1 single-frame (keyframe) inject constraints
# ---------------------------------------------------------------------------


class TestF1SingleFrameInject:
    """F=1 keyframe inject: placement, audio, and %51 warning gating.

    All fail-then-pass tests in this class require two nodes.py changes:
    1. The single-frame placement guard (inject_at % 17 == 0 for F=1 injects).
    2. The %51 audio-tick warning gate (only fire when audio is present and audio_mode != 'drop').
    """

    @staticmethod
    def _images(frames: int, h: int = 64, w: int = 64) -> Any:
        import torch

        return torch.zeros(frames, h, w, 3)

    @staticmethod
    def _audio(samples: int = 8000, sr: int = 16000) -> dict:
        import torch

        return {"waveform": torch.zeros(1, samples), "sample_rate": sr}

    # -- (i) Valid single-frame inject succeeds and produces source_length=1 -------

    def test_single_frame_at_chunk_boundary_drop_succeeds(self):
        """inject_at=187 (=11*17) + 1 frame + audio_mode='drop' → succeeds; no %51 warning.

        inject_at=187 is a chunk boundary (11*17) but not a multiple of 51 (187 % 51 = 34).
        With audio_mode='drop' the %51 warning must NOT fire (gated by audio presence).
        source_length must be 1 and inject_at on the Inject must be 187.

        FAIL-THEN-PASS: Before the %51 gate, a warning fires even for drop mode.
        After the gate, no warning fires and the inject succeeds.
        """
        import warnings

        node = H3AddInject()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            (inject_list,) = node.add_inject(
                inject_at=187,  # 11 * 17 — chunk boundary, not multiple of 51
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=0,
                end_fade_out=0,
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="drop",
                images=self._images(1),
                vae=FakeEncodeVAE(),
                audio=None,
            )
        inj = inject_list[0]
        assert inj.source_length == 1
        assert inj.inject_at == 187
        # %51 warning must NOT fire when audio_mode='drop'.
        audio_tick_warns = [x for x in w if "multiple of 51" in str(x.message)]
        assert not audio_tick_warns, (
            f"%51 warning must be suppressed for audio_mode='drop'; got: {audio_tick_warns}"
        )

    def test_single_frame_total_rows_is_1(self):
        """Verify total_rows(1) == 1 so F=1 inject maps to exactly one schedule row.

        Cross-module regression guard confirming the F=1 path of both
        total_rows (grid.py) and snap_length_down (sanitize.py) are consistent.
        """
        from comfyui_h3_blended_inject.grid import total_rows

        node = H3AddInject()
        (inject_list,) = node.add_inject(
            inject_at=0,
            start_fade_in=0,
            start_keyframes=0,
            end_keyframes=0,
            end_fade_out=0,
            min_denoise=0.0,
            interpolation_type="linear",
            audio_mode="drop",
            images=self._images(1),
            vae=FakeEncodeVAE(),
            audio=None,
        )
        inj = inject_list[0]
        assert inj.source_length == 1
        assert total_rows(inj.source_length) == 1

    # -- (ii) Non-chunk-boundary inject_at raises ValueError for F=1 ---------------

    def test_single_frame_nonmultiple_inject_at_raises(self):
        """inject_at=188 (not a multiple of 17) + 1 frame → ValueError.

        Single-frame injects must land exactly on a chunk boundary (inject_at % 17 == 0).
        The guard must use the ORIGINAL inject_at (188), not the post-snap value (187).

        FAIL-THEN-PASS: Before the guard, inject_at=188 is silently snapped to 187 and
        the inject succeeds. After the guard, ValueError is raised naming inject_at=188.
        """
        node = H3AddInject()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress snap warning from snap_inject_at
            with pytest.raises(ValueError, match="inject_at=188"):
                node.add_inject(
                    inject_at=188,  # not a multiple of 17
                    start_fade_in=0,
                    start_keyframes=0,
                    end_keyframes=0,
                    end_fade_out=0,
                    min_denoise=0.0,
                    interpolation_type="linear",
                    audio_mode="drop",
                    images=self._images(1),
                    vae=FakeEncodeVAE(),
                    audio=None,
                )

    def test_single_frame_nonmultiple_inject_at_names_nearest_edges(self):
        """ValueError for non-chunk-boundary F=1 inject names the two nearest edges.

        inject_at=35 → nearest chunk edges: 34 (=2*17) and 51 (=3*17).
        """
        node = H3AddInject()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError) as exc_info:
                node.add_inject(
                    inject_at=35,  # 35 % 17 = 1, not a chunk boundary
                    start_fade_in=0,
                    start_keyframes=0,
                    end_keyframes=0,
                    end_fade_out=0,
                    min_denoise=0.0,
                    interpolation_type="linear",
                    audio_mode="drop",
                    images=self._images(1),
                    vae=FakeEncodeVAE(),
                    audio=None,
                )
        msg = str(exc_info.value)
        assert "34" in msg and "51" in msg, (
            f"Error message must mention edges 34 and 51; got: {msg}"
        )

    # -- (iii) Single-frame inject with non-drop audio raises ValueError -------------

    def test_single_frame_keep_audio_raises(self):
        """inject_at=0 + 1 frame + audio_mode='keep' + audio → ValueError.

        Single-frame (keyframe) injects have no audio extent to preserve.
        audio_mode='keep' with audio present must raise ValueError.

        FAIL-THEN-PASS: Before the guard the inject succeeds; after it raises ValueError.
        """
        node = H3AddInject()
        with pytest.raises(ValueError, match="audio_mode"):
            node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=0,
                end_fade_out=0,
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="keep",
                images=self._images(1),
                vae=FakeEncodeVAE(),
                audio=self._audio(),
                audio_vae=FakeAudioVAE(),
            )

    def test_single_frame_fade_audio_raises(self):
        """inject_at=0 + 1 frame + audio_mode='fade' + audio → ValueError."""
        node = H3AddInject()
        with pytest.raises(ValueError, match="audio_mode"):
            node.add_inject(
                inject_at=0,
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=0,
                end_fade_out=0,
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="fade",
                images=self._images(1),
                vae=FakeEncodeVAE(),
                audio=self._audio(),
                audio_vae=FakeAudioVAE(),
            )

    def test_single_frame_no_audio_with_keep_mode_succeeds(self):
        """inject_at=0 + 1 frame + audio_mode='keep' + audio=None → succeeds.

        audio_mode='keep' with audio=None is fine: no audio to inject.
        """
        node = H3AddInject()
        (inject_list,) = node.add_inject(
            inject_at=0,
            start_fade_in=0,
            start_keyframes=0,
            end_keyframes=0,
            end_fade_out=0,
            min_denoise=0.0,
            interpolation_type="linear",
            audio_mode="keep",
            images=self._images(1),
            vae=FakeEncodeVAE(),
            audio=None,  # no audio → guard does not fire
        )
        assert inject_list[0].source_length == 1

    # -- (iv) %51 warning is gated on audio presence and audio_mode ----------------

    def test_no_51_warning_for_drop_mode_at_nonmultiple_of_51(self):
        """audio_mode='drop' + audio + inject_at=17 (not mult of 51) → no %51 UserWarning.

        The %51 position warning must be suppressed when audio_mode='drop'.
        Uses a 22-frame inject (F>1) to stay in the multi-frame path.

        FAIL-THEN-PASS: Before the gate, the warning fires regardless of audio_mode.
        After the gate, it is suppressed when audio_mode='drop'.
        """
        import warnings

        import torch

        node = H3AddInject()
        images = torch.zeros(22, 64, 64, 3)  # 22 = 5+17*1, valid 17n+5
        audio = self._audio()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            node.add_inject(
                inject_at=17,  # multiple of 17 but not 51
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=21,
                end_fade_out=21,
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="drop",
                images=images,
                vae=FakeEncodeVAE(),
                audio=audio,
                audio_vae=FakeAudioVAE(),
            )
        audio_tick_warns = [x for x in w if "multiple of 51" in str(x.message)]
        assert not audio_tick_warns, (
            f"%51 warning must NOT fire for audio_mode='drop'; fired: {audio_tick_warns}"
        )

    def test_51_warning_fires_for_keep_mode_with_audio(self):
        """audio_mode='keep' + audio + inject_at=17 (not mult of 51) → %51 UserWarning fires.

        The %51 warning must still fire when audio IS being injected (audio_mode != 'drop').
        Uses a 22-frame inject to stay in the multi-frame path.
        """
        import torch

        node = H3AddInject()
        images = torch.zeros(22, 64, 64, 3)
        audio = self._audio()
        with pytest.warns(UserWarning, match="multiple of 51"):
            node.add_inject(
                inject_at=17,  # multiple of 17 but not 51
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=21,
                end_fade_out=21,
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="keep",
                images=images,
                vae=FakeEncodeVAE(),
                audio=audio,
                audio_vae=FakeAudioVAE(),
            )

    def test_51_warning_suppressed_when_no_audio(self):
        """%51 warning must not fire when audio=None (regardless of audio_mode).

        FAIL-THEN-PASS: Before the gate, warning fires even without audio.
        After: only fires when audio is present and audio_mode != 'drop'.
        """
        import warnings

        import torch

        node = H3AddInject()
        images = torch.zeros(22, 64, 64, 3)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            node.add_inject(
                inject_at=17,  # multiple of 17 but not 51 → would normally warn
                start_fade_in=0,
                start_keyframes=0,
                end_keyframes=21,
                end_fade_out=21,
                min_denoise=0.0,
                interpolation_type="linear",
                audio_mode="keep",
                images=images,
                vae=FakeEncodeVAE(),
                audio=None,  # no audio → %51 warning suppressed
            )
        audio_tick_warns = [x for x in w if "multiple of 51" in str(x.message)]
        assert not audio_tick_warns, (
            f"%51 warning must NOT fire when audio=None; fired: {audio_tick_warns}"
        )


# ---------------------------------------------------------------------------
# H3AddGuide
# ---------------------------------------------------------------------------


class RecordingVAE:
    """VAE stand-in that records the batch passed to encode()."""

    def __init__(self) -> None:
        self.encoded: Any = None
        self.latent = object()

    def encode(self, x: Any) -> Any:
        self.encoded = x
        return self.latent


class TestH3AddGuideInputTypes:
    def test_required_inputs_order(self):
        required = H3AddGuide.INPUT_TYPES()["required"]
        assert list(required.keys()) == ["frame_idx", "hold_frac"]

    def test_frame_idx_allows_negative(self):
        opts = H3AddGuide.INPUT_TYPES()["required"]["frame_idx"][1]
        assert opts["min"] < 0

    def test_frame_idx_tooltip_disambiguates_from_inject_at(self):
        """The tooltip must call out pixel-frame semantics vs inject_at's latent-frame."""
        tooltip = H3AddGuide.INPUT_TYPES()["required"]["frame_idx"][1]["tooltip"]
        assert "PIXEL" in tooltip.upper()
        assert "inject_at" in tooltip

    def test_hold_frac_defaults_to_official_behavior(self):
        opts = H3AddGuide.INPUT_TYPES()["required"]["hold_frac"][1]
        assert opts["default"] == 1.0
        assert opts["min"] == 0.0
        assert opts["max"] == 1.0

    def test_optional_inputs(self):
        optional = H3AddGuide.INPUT_TYPES()["optional"]
        assert set(optional.keys()) == {"inject_list", "image", "audio", "vae", "audio_vae"}
        assert optional["inject_list"][0] == INJECT_LIST
        assert optional["image"][0] == "IMAGE"
        assert optional["audio"][0] == "AUDIO"
        assert optional["vae"][0] == "VAE"
        assert optional["audio_vae"][0] == "VAE"

    def test_class_attrs(self):
        assert H3AddGuide.RETURN_TYPES == (INJECT_LIST,)
        assert H3AddGuide.RETURN_NAMES == ("inject_list",)
        assert H3AddGuide.FUNCTION == "add_guide"
        assert H3AddGuide.CATEGORY == "H3 Blended Inject"

    def test_node_registration(self):
        assert NODE_CLASS_MAPPINGS["H3AddGuide"] is H3AddGuide
        assert NODE_DISPLAY_NAME_MAPPINGS["H3AddGuide"] == "H3 Add Guide"


class TestH3AddGuideBehavior:
    def test_no_content_raises(self):
        with pytest.raises(ValueError, match="image or an audio"):
            H3AddGuide().add_guide(frame_idx=0, hold_frac=1.0)

    def test_image_without_vae_raises(self):
        with pytest.raises(ValueError, match="vae"):
            H3AddGuide().add_guide(frame_idx=0, hold_frac=1.0, image=FakeImages(1, 64, 64))

    def test_audio_without_audio_vae_raises(self):
        with pytest.raises(ValueError, match="audio_vae"):
            H3AddGuide().add_guide(frame_idx=0, hold_frac=1.0, audio=object())

    def test_single_image_guide_fields(self):
        vae = RecordingVAE()
        (result,) = H3AddGuide().add_guide(
            frame_idx=-1, hold_frac=0.6, image=FakeImages(1, 64, 96), vae=vae
        )
        assert len(result) == 1
        g = result[0]
        assert isinstance(g, Guide)
        assert g.frame_idx == -1  # raw, unresolved — negative kept as entered
        assert g.hold_frac == 0.6
        assert g.guide_frames == 1
        assert g.resolution == (96, 64)  # (width, height) from [frames, H, W, C]
        assert g.video_latent is vae.latent
        assert g.audio_latent is None

    def test_long_batch_snaps_down_before_encode(self):
        vae = RecordingVAE()
        with pytest.warns(UserWarning, match="snapped down"):
            (result,) = H3AddGuide().add_guide(
                frame_idx=0, hold_frac=1.0, image=FakeImages(8, 64, 64), vae=vae
            )
        assert result[0].guide_frames == 5
        assert vae.encoded.shape[0] == 5  # trimmed batch is what gets encoded

    def test_short_batch_uses_first_image_only(self):
        vae = RecordingVAE()
        with pytest.warns(UserWarning, match="snapped down"):
            (result,) = H3AddGuide().add_guide(
                frame_idx=0, hold_frac=1.0, image=FakeImages(3, 64, 64), vae=vae
            )
        assert result[0].guide_frames == 1
        assert vae.encoded.shape[0] == 1

    def test_valid_clip_length_kept_without_warning(self):
        vae = RecordingVAE()
        (result,) = H3AddGuide().add_guide(
            frame_idx=0, hold_frac=1.0, image=FakeImages(22, 64, 64), vae=vae
        )
        assert result[0].guide_frames == 22

    def test_chaining_appends_without_mutating_input(self):
        (chain,) = H3AddInject().add_inject(**_make_add_inject_args())
        snapshot = list(chain)
        (result,) = H3AddGuide().add_guide(
            frame_idx=0,
            hold_frac=1.0,
            image=FakeImages(1, 64, 64),
            vae=RecordingVAE(),
            inject_list=chain,
        )
        assert chain == snapshot  # input list untouched
        assert result[:1] == snapshot
        assert isinstance(result[0], Inject)
        assert isinstance(result[1], Guide)


class TestH3InjectSamplerGuideBehavior:
    """Sample-time guide resolution: partition, resolution gate, bounds, keyframe build.

    _run_sampler is monkeypatched to capture its kwargs — everything before it is the
    CPU-testable contract under test.
    """

    LATENT_ROWS = 32  # → 107 pixel frames, 128x128 target

    def _sample(self, inject_list: list[Any], monkeypatch: pytest.MonkeyPatch | None = None):
        import torch

        import comfyui_h3_blended_inject.nodes as nodes_module

        captured: dict[str, Any] = {}
        if monkeypatch is not None:

            def fake_run_sampler(**kwargs: Any) -> tuple[dict[str, Any]]:
                captured.update(kwargs)
                return ({"samples": None},)

            monkeypatch.setattr(nodes_module, "_run_sampler", fake_run_sampler)

        result = H3InjectSampler().sample(
            model=object(),
            noise_seed=0,
            steps=20,
            cfg=7.0,
            sampler_name="euler",
            scheduler="normal",
            positive=[],
            negative=None,
            latent_image={"samples": torch.zeros(1, self.LATENT_ROWS, 8, 8)},
            inject_list=inject_list,
        )
        return captured, result

    def _video_guide(self, **overrides: Any) -> Guide:
        defaults: dict[str, Any] = {
            "frame_idx": 0,
            "hold_frac": 1.0,
            "video_latent": object(),
            "resolution": (128, 128),
            "guide_frames": 1,
        }
        defaults.update(overrides)
        return Guide(**defaults)

    def test_resolution_mismatch_raises_with_resize_hint(self):
        guide = self._video_guide(resolution=(64, 64))
        with pytest.raises(ValueError, match="Resize Image v2"):
            self._sample([guide])

    def test_audio_only_guide_skips_resolution_check(self, monkeypatch):
        import torch

        guide = Guide(
            frame_idx=0,
            hold_frac=1.0,
            audio_latent=torch.zeros(1, 32, 2, 4),
            resolution=(0, 0),
        )
        captured, _ = self._sample([guide], monkeypatch)
        assert len(captured["guide_entries"]) == 1

    def test_frame_idx_out_of_bounds_raises(self):
        # 32 rows → 107 pixel frames; index 107 is one past the end.
        guide = self._video_guide(frame_idx=107)
        with pytest.raises(ValueError, match="107 frames"):
            self._sample([guide])

    def test_clip_that_does_not_fit_raises(self):
        guide = self._video_guide(frame_idx=100, guide_frames=22)
        with pytest.raises(ValueError, match="does not fit"):
            self._sample([guide])

    def test_negative_frame_idx_resolves_from_end(self, monkeypatch):
        guide = self._video_guide(frame_idx=-1, hold_frac=0.6)
        captured, _ = self._sample([guide], monkeypatch)
        (kf, hold_frac) = captured["guide_entries"][0]
        assert kf["resolved_frame_index"] == 106  # 107 frames, -1 → last
        assert kf["latent"] is guide.video_latent
        assert hold_frac == 0.6

    def test_audio_latent_cropped_to_video_duration(self, monkeypatch):
        import torch

        audio_ticks = grid.audio_ticks_for_rows(self.LATENT_ROWS)
        guide = Guide(
            frame_idx=0,
            hold_frac=1.0,
            audio_latent=torch.zeros(1, 32, 2, audio_ticks + 100),
        )
        captured, _ = self._sample([guide], monkeypatch)
        (kf, _) = captured["guide_entries"][0]
        assert kf["audio_latent"].shape[-1] == audio_ticks

    def test_mixed_chain_partitions_injects_and_guides(self, monkeypatch):
        # Degenerate still-inject (no images) + one guide: the inject feeds the per-row
        # schedule, the guide feeds guide_entries; neither path sees the other's entry.
        inject = Inject(
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
        guide = self._video_guide(frame_idx=40)
        captured, _ = self._sample([inject, guide], monkeypatch)
        assert len(captured["guide_entries"]) == 1
        assert all(rs.inject is inject for rs in captured["schedule"])

    def test_guide_only_chain_yields_empty_schedule(self, monkeypatch):
        guide = self._video_guide()
        captured, _ = self._sample([guide], monkeypatch)
        assert captured["schedule"] == []
        assert len(captured["guide_entries"]) == 1

    def test_none_inject_list_runs_as_passthrough(self, monkeypatch):
        """inject_list=None must not raise: empty schedule + empty guides = passthrough.

        Regression for the required-input bug — with no injects wired (or all bypassed),
        ComfyUI passes nothing and sample() previously died partitioning None. The node
        must sample natively: empty schedule, no guides, and return a LATENT dict.
        """
        captured, result = self._sample(None, monkeypatch)
        assert captured["schedule"] == []
        assert captured["guide_entries"] == []
        assert isinstance(result, tuple)
        assert "samples" in result[0]

    def test_empty_inject_list_runs_as_passthrough(self, monkeypatch):
        """inject_list=[] must behave identically to None: clean passthrough sampling."""
        captured, result = self._sample([], monkeypatch)
        assert captured["schedule"] == []
        assert captured["guide_entries"] == []
        assert isinstance(result, tuple)
        assert "samples" in result[0]

    def test_foreign_entry_in_chain_raises_type_error(self):
        with pytest.raises(TypeError, match="Inject or Guide"):
            self._sample([object()])
