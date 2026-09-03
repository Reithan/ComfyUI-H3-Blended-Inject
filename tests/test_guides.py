"""Tests for guide resolution + timed-removal helpers (comfyui_h3_blended_inject/guides.py)."""

from __future__ import annotations

import math

import pytest
import torch

from comfyui_h3_blended_inject import grid
from comfyui_h3_blended_inject.guides import (
    FRAME_RESCALE,
    build_keyframe,
    crop_audio_latent,
    filter_released_keyframes,
    frame_count_for_rows,
    partition_inject_list,
    resolve_frame_index,
    snap_guide_length,
)
from comfyui_h3_blended_inject.schedule import Guide, Inject


def _make_inject() -> Inject:
    return Inject(
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


# ---------------------------------------------------------------------------
# partition_inject_list
# ---------------------------------------------------------------------------


class TestPartitionInjectList:
    def test_mixed_list_partitions_preserving_order(self) -> None:
        i1, i2 = _make_inject(), _make_inject()
        g1, g2 = (
            Guide(inject_at=0, start_percent=0.0, end_percent=1.0),
            Guide(inject_at=5, start_percent=0.0, end_percent=0.5),
        )
        injects, guides = partition_inject_list([i1, g1, i2, g2])
        assert injects == [i1, i2]
        assert guides == [g1, g2]

    def test_empty_list_returns_two_empty_lists(self) -> None:
        assert partition_inject_list([]) == ([], [])

    def test_foreign_entry_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="Inject or Guide"):
            partition_inject_list([_make_inject(), object()])


# ---------------------------------------------------------------------------
# snap_guide_length
# ---------------------------------------------------------------------------


class TestSnapGuideLength:
    def test_single_frame_passes_through(self) -> None:
        assert snap_guide_length(1) == 1

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_short_batch_snaps_to_first_image_with_warning(self, n: int) -> None:
        with pytest.warns(UserWarning, match="snapped down"):
            assert snap_guide_length(n) == 1

    @pytest.mark.parametrize("n", [5, 22, 39, 56])
    def test_valid_clip_lengths_pass_through(self, n: int) -> None:
        assert snap_guide_length(n) == n

    @pytest.mark.parametrize(("n", "expected"), [(8, 5), (21, 5), (23, 22), (38, 22), (40, 39)])
    def test_long_batch_snaps_down_to_17k_plus_5_with_warning(self, n: int, expected: int) -> None:
        with pytest.warns(UserWarning, match="snapped down"):
            assert snap_guide_length(n) == expected

    def test_zero_frames_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            snap_guide_length(0)


# ---------------------------------------------------------------------------
# frame_count_for_rows
# ---------------------------------------------------------------------------


class TestFrameCountForRows:
    @pytest.mark.parametrize(
        ("rows", "frames"),
        [
            (1, 1),  # single-frame latent
            (2, 5),  # minimum multi-frame clip
            (5, 17),  # one full chunk
            (7, 22),  # 22 = 17 + 5
            (32, 107),  # 107 = 17*6 + 5
            (37, 124),  # the default video length
        ],
    )
    def test_matches_official_frame_per_token_sum(self, rows: int, frames: int) -> None:
        """Closed form equals the official sum(FRAME_PER_TOKEN[k % 5] for k in range(rows))."""
        assert frame_count_for_rows(rows) == frames
        assert frame_count_for_rows(rows) == sum(grid.FRAME_PER_TOKEN[k % 5] for k in range(rows))


# ---------------------------------------------------------------------------
# resolve_frame_index
# ---------------------------------------------------------------------------


class TestResolveFrameIndex:
    def test_non_negative_index_passes_through(self) -> None:
        assert resolve_frame_index(40, 107, 1) == 40

    def test_negative_index_counts_from_end(self) -> None:
        assert resolve_frame_index(-1, 107, 1) == 106

    def test_last_frame_single_guide_fits(self) -> None:
        assert resolve_frame_index(106, 107, 1) == 106

    def test_index_past_end_raises(self) -> None:
        with pytest.raises(ValueError, match="outside the video's 107 frames"):
            resolve_frame_index(107, 107, 1)

    def test_negative_overshoot_raises(self) -> None:
        with pytest.raises(ValueError, match="outside the video's 107 frames"):
            resolve_frame_index(-108, 107, 1)

    def test_clip_that_does_not_fit_raises_clip_message(self) -> None:
        with pytest.raises(ValueError, match="22 frame guide clip .* does not fit"):
            resolve_frame_index(100, 107, 22)

    def test_clip_that_exactly_fits_passes(self) -> None:
        assert resolve_frame_index(85, 107, 22) == 85


# ---------------------------------------------------------------------------
# crop_audio_latent
# ---------------------------------------------------------------------------


class TestCropAudioLatent:
    def test_short_audio_returned_unchanged(self) -> None:
        z = torch.zeros(1, 32, 2, 50)
        assert crop_audio_latent(z, audio_ticks=178, resolved_frame_index=0) is z

    def test_long_audio_cropped_to_remaining_track(self) -> None:
        z = torch.zeros(1, 32, 2, 500)
        out = crop_audio_latent(z, audio_ticks=178, resolved_frame_index=0)
        assert out.shape[-1] == 178

    def test_crop_accounts_for_anchor_position(self) -> None:
        # max_rt = floor(178 - (5/3) * 60) = floor(78.0) = 78
        z = torch.zeros(1, 32, 2, 500)
        out = crop_audio_latent(z, audio_ticks=178, resolved_frame_index=60)
        assert out.shape[-1] == math.floor(178 - FRAME_RESCALE * 60)

    def test_cropped_result_is_a_clone_not_a_view(self) -> None:
        z = torch.ones(1, 32, 2, 500)
        out = crop_audio_latent(z, audio_ticks=178, resolved_frame_index=0)
        out[...] = 0.0
        assert z.sum().item() == pytest.approx(1 * 32 * 2 * 500)

    def test_anchor_past_audio_end_raises(self) -> None:
        # max_rt = floor(178 - (5/3) * 107) < 1 → no room for even one audio frame.
        z = torch.zeros(1, 32, 2, 10)
        with pytest.raises(ValueError, match="past the end"):
            crop_audio_latent(z, audio_ticks=178, resolved_frame_index=107)


# ---------------------------------------------------------------------------
# build_keyframe
# ---------------------------------------------------------------------------


class TestBuildKeyframe:
    def test_video_only_guide(self) -> None:
        latent = torch.zeros(1, 24, 2, 4, 4)
        g = Guide(inject_at=0, start_percent=0.0, end_percent=1.0, video_latent=latent)
        kf = build_keyframe(g, resolved_frame_index=40, audio_ticks=178)
        assert kf == {"resolved_frame_index": 40, "latent": latent}

    def test_audio_only_guide_crops_audio(self) -> None:
        z = torch.zeros(1, 32, 2, 500)
        g = Guide(inject_at=0, start_percent=0.0, end_percent=1.0, audio_latent=z)
        kf = build_keyframe(g, resolved_frame_index=0, audio_ticks=178)
        assert "latent" not in kf
        assert kf["audio_latent"].shape[-1] == 178

    def test_video_and_audio_guide_has_both_keys(self) -> None:
        latent = torch.zeros(1, 24, 2, 4, 4)
        z = torch.zeros(1, 32, 2, 50)
        g = Guide(
            inject_at=0, start_percent=0.0, end_percent=1.0, video_latent=latent, audio_latent=z
        )
        kf = build_keyframe(g, resolved_frame_index=0, audio_ticks=178)
        assert kf["latent"] is latent
        assert kf["audio_latent"] is z


# ---------------------------------------------------------------------------
# filter_released_keyframes
# ---------------------------------------------------------------------------


def _payload() -> tuple[dict, dict, dict, dict]:
    """Payload with an official-style keyframe, one of ours, and a ref.

    Returns (payload, kf_official, kf_ours, ref).
    """
    kf_official = {"resolved_frame_index": 40, "latent": "OFFICIAL_LAT"}
    kf_ours = {"resolved_frame_index": 40, "latent": "OURS_LAT", "audio_latent": "OURS_AUD"}
    ref = {"latent": "REF_LAT", "audio_latent": "REF_AUD"}
    payload = {
        "text_token_tags": "TAGS",
        "keyframes": [kf_official, kf_ours],
        "refs": [ref],
        "cond_video_latents": ["OFFICIAL_LAT", "OURS_LAT", "REF_LAT"],
        "cond_audio_latents": ["OURS_AUD", "REF_AUD"],
        "layout": object(),
        "seed": 0,
    }
    return payload, kf_official, kf_ours, ref


class TestFilterReleasedKeyframes:
    def test_removes_only_released_entries_by_identity(self) -> None:
        payload, kf_official, kf_ours, _ = _payload()
        out = filter_released_keyframes(payload, frozenset({id(kf_ours)}))
        assert out["keyframes"] == [kf_official]
        assert out["keyframes"][0] is kf_official

    def test_same_frame_index_different_object_is_not_caught(self) -> None:
        """An official keyframe at the SAME frame as ours must survive the release."""
        payload, kf_official, kf_ours, _ = _payload()
        assert kf_official["resolved_frame_index"] == kf_ours["resolved_frame_index"]
        out = filter_released_keyframes(payload, frozenset({id(kf_ours)}))
        assert kf_official in out["keyframes"]

    def test_layout_is_popped(self) -> None:
        payload, _, kf_ours, _ = _payload()
        out = filter_released_keyframes(payload, frozenset({id(kf_ours)}))
        assert "layout" not in out

    def test_cond_lists_rebuilt_held_keyframes_then_refs(self) -> None:
        payload, _, kf_ours, _ = _payload()
        out = filter_released_keyframes(payload, frozenset({id(kf_ours)}))
        assert out["cond_video_latents"] == ["OFFICIAL_LAT", "REF_LAT"]
        # The official keyframe has no audio_latent, so only the ref's audio remains.
        assert out["cond_audio_latents"] == ["REF_AUD"]

    def test_all_released_removes_keyframes_key(self) -> None:
        payload, kf_official, kf_ours, _ = _payload()
        out = filter_released_keyframes(payload, frozenset({id(kf_official), id(kf_ours)}))
        assert "keyframes" not in out
        assert out["cond_video_latents"] == ["REF_LAT"]
        assert out["cond_audio_latents"] == ["REF_AUD"]

    def test_input_payload_is_not_mutated(self) -> None:
        payload, kf_official, kf_ours, _ = _payload()
        snapshot = dict(payload)
        keyframes_snapshot = list(payload["keyframes"])
        filter_released_keyframes(payload, frozenset({id(kf_ours)}))
        assert payload == snapshot
        assert payload["keyframes"] == keyframes_snapshot
        assert "layout" in payload

    def test_ref_video_predicate_is_key_presence(self) -> None:
        """Refs video uses ``"latent" in r`` (comfy's predicate), so an explicit None is kept."""
        payload, _, kf_ours, ref = _payload()
        ref_none = {"latent": None}
        payload["refs"] = [ref, ref_none]
        out = filter_released_keyframes(payload, frozenset({id(kf_ours)}))
        assert out["cond_video_latents"] == ["OFFICIAL_LAT", "REF_LAT", None]

    def test_keyframe_video_predicate_is_none_check(self) -> None:
        """Keyframes video uses ``kf.get("latent") is not None``; audio-only entries skipped."""
        payload, _, kf_ours, _ = _payload()
        kf_audio_only = {"resolved_frame_index": 10, "audio_latent": "AUD_ONLY"}
        payload["keyframes"] = list(payload["keyframes"]) + [kf_audio_only]
        out = filter_released_keyframes(payload, frozenset({id(kf_ours)}))
        assert out["cond_video_latents"] == ["OFFICIAL_LAT", "REF_LAT"]
        assert out["cond_audio_latents"] == ["AUD_ONLY", "REF_AUD"]

    def test_no_refs_key_handled(self) -> None:
        payload, kf_official, kf_ours, _ = _payload()
        del payload["refs"]
        out = filter_released_keyframes(payload, frozenset({id(kf_ours)}))
        assert out["cond_video_latents"] == ["OFFICIAL_LAT"]
        assert out["cond_audio_latents"] == []

    def test_other_payload_keys_carried_over(self) -> None:
        payload, _, kf_ours, _ = _payload()
        out = filter_released_keyframes(payload, frozenset({id(kf_ours)}))
        assert out["text_token_tags"] == "TAGS"
        assert out["seed"] == 0
