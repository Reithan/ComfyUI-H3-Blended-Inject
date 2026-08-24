"""Tests for comfyui_h3_blended_inject.composite.

The per-row img2img sampler needs two composite operations on the *unpacked* latent
components (video ``[1,C,T,Hl,Wl]``, audio ``[1,C,2,audio_t]``):

  - build_clean_reference: the ``clean`` reference the init lerp blends toward — the target
    latent with ALL inject video/audio content composited in (every covered row/tick, not
    just d==0), so fractional rows img2img *from* the inject content.
  - post_composite_preserve: after sampling, a binary exact-preserve overwrite of m==0
    video rows and audio-preserve ticks from the clean reference (no ghost).

Both are pure tensor ops, tested here with small fake latents.  Inject content rows/ticks
are filled with distinctive values so we can assert what got written where.
"""

from __future__ import annotations

import torch

from comfyui_h3_blended_inject.composite import build_clean_reference, post_composite_preserve
from comfyui_h3_blended_inject.schedule import Inject, RowSchedule

# Small latent dims for testing.
_CV, _HL, _WL = 2, 1, 1  # video channels, spatial
_CA = 2  # audio channels


def make_inject(
    inject_at: int = 0,
    n_clip_rows: int = 3,
    n_clip_ticks: int = 0,
    audio_mode: str = "fade",
    video_fill_base: float = 100.0,
    audio_fill_base: float = 200.0,
    video_dtype: torch.dtype = torch.float32,
) -> Inject:
    """Build an Inject with recognizable video/audio latent content.

    Clip video row k is filled with value ``video_fill_base + k``; clip audio tick k with
    ``audio_fill_base + k``.  ``n_clip_ticks == 0`` → no audio latent.
    """
    video_latent = torch.zeros(1, _CV, n_clip_rows, _HL, _WL, dtype=video_dtype)
    for k in range(n_clip_rows):
        video_latent[:, :, k, :, :] = video_fill_base + k
    audio_latent = None
    if n_clip_ticks > 0:
        audio_latent = torch.zeros(1, _CA, 2, n_clip_ticks, dtype=video_dtype)
        for k in range(n_clip_ticks):
            audio_latent[:, :, :, k] = audio_fill_base + k
    return Inject(
        inject_at=inject_at,
        start_fade_in=0,
        start_keyframes=0,
        end_keyframes=17,
        end_fade_out=39,
        min_denoise=0.0,
        interpolation_type="linear",
        audio_mode=audio_mode,
        images=None,
        audio=None,
        resolution=(0, 0),
        source_length=39,
        video_latent=video_latent,
        audio_latent=audio_latent,
    )


def rs(row_idx: int, denoise: float, inject: Inject, audio_frozen: bool = False) -> RowSchedule:
    return RowSchedule(row_idx=row_idx, denoise=denoise, inject=inject, audio_frozen=audio_frozen)


def target_video(rows: int = 6) -> torch.Tensor:
    """Target video latent filled with 0.0 everywhere (distinct from inject fills)."""
    return torch.zeros(1, _CV, rows, _HL, _WL)


def target_audio(ticks: int = 8) -> torch.Tensor:
    return torch.zeros(1, _CA, 2, ticks)


# ---------------------------------------------------------------------------
# build_clean_reference
# ---------------------------------------------------------------------------


class TestBuildCleanReferenceVideo:
    def test_writes_inject_rows_regardless_of_denoise(self) -> None:
        """Inject content is composited on every covered row, even fractional/full ones."""
        inj = make_inject(inject_at=0, n_clip_rows=3)
        schedule = [rs(0, 0.0, inj), rs(1, 0.5, inj), rs(2, 1.0, inj)]
        video = target_video(6)
        clean_v, _ = build_clean_reference(video, None, schedule, target_rows=6, audio_ticks=8)
        assert clean_v[0, 0, 0, 0, 0].item() == 100.0  # clip row 0
        assert clean_v[0, 0, 1, 0, 0].item() == 101.0  # clip row 1 (fractional)
        assert clean_v[0, 0, 2, 0, 0].item() == 102.0  # clip row 2 (full denoise)

    def test_uncovered_rows_keep_target(self) -> None:
        inj = make_inject(inject_at=0, n_clip_rows=2)
        schedule = [rs(0, 0.0, inj), rs(1, 0.5, inj)]
        video = target_video(6)
        clean_v, _ = build_clean_reference(video, None, schedule, target_rows=6, audio_ticks=8)
        for r in range(2, 6):
            assert clean_v[0, 0, r, 0, 0].item() == 0.0

    def test_does_not_mutate_input_video(self) -> None:
        inj = make_inject(inject_at=0, n_clip_rows=2)
        schedule = [rs(0, 0.0, inj)]
        video = target_video(6)
        build_clean_reference(video, None, schedule, target_rows=6, audio_ticks=8)
        assert torch.all(video == 0.0)

    def test_aligns_inject_dtype(self) -> None:
        inj = make_inject(inject_at=0, n_clip_rows=2, video_dtype=torch.float64)
        schedule = [rs(0, 0.0, inj)]
        video = target_video(6)  # float32
        clean_v, _ = build_clean_reference(video, None, schedule, target_rows=6, audio_ticks=8)
        assert clean_v.dtype == torch.float32

    def test_none_video_returns_none(self) -> None:
        inj = make_inject(inject_at=0, n_clip_rows=2, n_clip_ticks=4)
        schedule = [rs(0, 0.0, inj)]
        clean_v, clean_a = build_clean_reference(
            None, target_audio(8), schedule, target_rows=6, audio_ticks=8
        )
        assert clean_v is None
        assert clean_a is not None


class TestBuildCleanReferenceAudio:
    def test_fade_mode_writes_audio(self) -> None:
        inj = make_inject(inject_at=0, n_clip_rows=2, n_clip_ticks=4, audio_mode="fade")
        schedule = [rs(0, 0.0, inj), rs(1, 0.5, inj)]
        clean_v, clean_a = build_clean_reference(
            target_video(6), target_audio(8), schedule, target_rows=6, audio_ticks=8
        )
        # at least one audio tick got inject content (>= 200.0)
        assert (clean_a >= 200.0).any()

    def test_drop_mode_leaves_audio_untouched(self) -> None:
        inj = make_inject(inject_at=0, n_clip_rows=2, n_clip_ticks=4, audio_mode="drop")
        schedule = [rs(0, 0.0, inj), rs(1, 0.5, inj)]
        _, clean_a = build_clean_reference(
            target_video(6), target_audio(8), schedule, target_rows=6, audio_ticks=8
        )
        assert torch.all(clean_a == 0.0)

    def test_keep_mode_writes_audio(self) -> None:
        inj = make_inject(inject_at=0, n_clip_rows=2, n_clip_ticks=4, audio_mode="keep")
        schedule = [rs(0, 0.5, inj, audio_frozen=True)]
        _, clean_a = build_clean_reference(
            target_video(6), target_audio(8), schedule, target_rows=6, audio_ticks=8
        )
        assert (clean_a >= 200.0).any()


# ---------------------------------------------------------------------------
# post_composite_preserve
# ---------------------------------------------------------------------------


class TestPostCompositePreserveVideo:
    def test_overwrites_m0_rows_only(self) -> None:
        """d==0 rows are replaced from clean; d>0 rows keep the sampled value."""
        inj = make_inject(inject_at=0, n_clip_rows=3)
        schedule = [rs(0, 0.0, inj), rs(1, 0.5, inj), rs(2, 1.0, inj)]
        clean_v, _ = build_clean_reference(
            target_video(6), None, schedule, target_rows=6, audio_ticks=8
        )
        sampled = torch.full((1, _CV, 6, _HL, _WL), 7.0)  # sampler output
        out_v, _ = post_composite_preserve(
            sampled, None, clean_v, None, schedule, target_rows=6, audio_ticks=8
        )
        assert out_v[0, 0, 0, 0, 0].item() == 100.0  # d==0 → clean
        assert out_v[0, 0, 1, 0, 0].item() == 7.0  # d==0.5 → sampled kept
        assert out_v[0, 0, 2, 0, 0].item() == 7.0  # d==1.0 → sampled kept

    def test_does_not_mutate_sampled(self) -> None:
        inj = make_inject(inject_at=0, n_clip_rows=2)
        schedule = [rs(0, 0.0, inj)]
        clean_v, _ = build_clean_reference(
            target_video(6), None, schedule, target_rows=6, audio_ticks=8
        )
        sampled = torch.full((1, _CV, 6, _HL, _WL), 7.0)
        post_composite_preserve(
            sampled, None, clean_v, None, schedule, target_rows=6, audio_ticks=8
        )
        assert torch.all(sampled == 7.0)


class TestPostCompositePreserveAudio:
    def test_overwrites_preserve_ticks_only(self) -> None:
        """audio_preserve ticks are replaced from clean; others keep sampled."""
        inj = make_inject(inject_at=0, n_clip_rows=2, n_clip_ticks=8, audio_mode="fade")
        # row 0 is a d==0 fade row → audio_preserve True; row 1 fractional → not preserved
        schedule = [rs(0, 0.0, inj), rs(1, 0.5, inj)]
        _, clean_a = build_clean_reference(
            target_video(6), target_audio(8), schedule, target_rows=6, audio_ticks=8
        )
        sampled_a = torch.full((1, _CA, 2, 8), 9.0)
        _, out_a = post_composite_preserve(
            None, sampled_a, None, clean_a, schedule, target_rows=6, audio_ticks=8
        )
        # some ticks overwritten from clean (>=200), some remain sampled (9.0)
        assert (out_a >= 200.0).any()
        assert (out_a == 9.0).any()

    def test_no_preserve_leaves_audio_sampled(self) -> None:
        """Fractional fade rows (no d==0) do not preserve audio."""
        inj = make_inject(inject_at=0, n_clip_rows=2, n_clip_ticks=8, audio_mode="fade")
        schedule = [rs(0, 0.5, inj), rs(1, 0.7, inj)]
        _, clean_a = build_clean_reference(
            target_video(6), target_audio(8), schedule, target_rows=6, audio_ticks=8
        )
        sampled_a = torch.full((1, _CA, 2, 8), 9.0)
        _, out_a = post_composite_preserve(
            None, sampled_a, None, clean_a, schedule, target_rows=6, audio_ticks=8
        )
        assert torch.all(out_a == 9.0)
