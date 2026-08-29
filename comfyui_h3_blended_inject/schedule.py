"""Inject data model and per-row schedule merge.

An :class:`Inject` describes one inject configuration added to the chain by ``H3AddInject``.
:func:`merge_schedule` collapses a list of injects into a per-row schedule using last-in-wins
at row granularity: a later inject in the list overwrites earlier injects on every row it claims,
both schedule value and content.  No blending between overlapping injects occurs; the boundary
is a hard edge.

This module is pure Python / stdlib and must import without ``comfy`` or ``torch`` present.
Image and audio handles are typed as ``Any`` to avoid runtime torch/comfy dependencies here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from comfyui_h3_blended_inject.envelope import classify_row_region, evaluate_envelope

# Type alias for the list that flows between H3AddInject / H3AddGuide nodes and into
# H3InjectSampler.  It may hold both :class:`Inject` and :class:`Guide` entries; the sampler
# partitions by type.  The ComfyUI type *string* is defined separately in nodes.py as
# INJECT_LIST = "INJECT_LIST".
InjectList = list["Inject | Guide"]


# eq=False: identity equality — the same Inject object IS the same inject (no value-based dedup).
@dataclass(eq=False)
class Inject:
    """All parameters needed to schedule and apply one inject into the target latent.

    Fields are stored as provided by the node; sanitization and envelope evaluation happen
    downstream in :mod:`~comfyui_h3_blended_inject.sanitize` and
    :mod:`~comfyui_h3_blended_inject.envelope`.

    Attributes
    ----------
    inject_at:
        Latent FRAME index in the target latent where this inject begins.  Must be a
        multiple of 17 frames after snapping
        (see :func:`~comfyui_h3_blended_inject.sanitize.snap_inject_at`).
        The fade indices (start_fade_in, start_keyframes, end_keyframes, end_fade_out) are
        **clip frame indices** — positions within this inject's own source content.
        Clip frame ``k`` maps to latent frame ``inject_at + k``.
    start_fade_in:
        First source clip frame below 1.0.  The 1.0 anchor is at ``start_fade_in - 1``.
    start_keyframes:
        Source clip frame index where the hold at ``min_denoise`` begins.
    end_keyframes:
        EXCLUSIVE: first fade-out frame.  Last held frame is ``end_keyframes - 1``.
    end_fade_out:
        EXCLUSIVE upper bound: denoise returns to 1.0 here.  The last content frame is
        ``end_fade_out - 1``.
    min_denoise:
        Denoise floor during the hold region, in [0.0, 1.0].  For still-inject (degenerate
        envelope), this is the single frame's denoise value.
    interpolation_type:
        Curve applied to both fade-in and fade-out regions.  One of: "ease_in", "ease_out",
        "ease_in_out", "linear", "none".
    audio_mode:
        How the audio stream is handled for this inject.  One of:
        - "fade": audio envelope follows the video denoise schedule.
        - "drop": no audio inject; audio rows from this inject are left as generation.
        - "keep": audio inject at d=0 via the derived noise mask (exact preservation).
    images:
        IMAGE tensor batch ([batch, H, W, C] float32) or None if no video/image inject.
    audio:
        AUDIO dict ({"waveform": Tensor, "sample_rate": int}) or None if no audio inject.
    resolution:
        (width, height) in pixels.  Must be a multiple of 32 and match the target latent.
    source_length:
        Number of source frames in the inject content (used to validate envelope indices).
    video_latent:
        Pre-encoded video latent tensor (from the video VAE), or ``None`` if no VAE was
        supplied to ``H3AddInject``.  Used by ``H3InjectSampler`` to build the clean
        reference latent for the per-row img2img sampler without re-encoding at sample time.
    audio_latent:
        Pre-encoded audio latent tensor (from the audio VAE), or ``None`` if no audio VAE
        was supplied.  Used analogously to ``video_latent`` for the audio stream.
    """

    inject_at: int
    start_fade_in: int
    start_keyframes: int
    end_keyframes: int
    end_fade_out: int
    min_denoise: float
    interpolation_type: Literal["ease_in", "ease_out", "ease_in_out", "linear", "none"]
    audio_mode: Literal["fade", "drop", "keep"]
    images: Any | None
    audio: Any | None
    resolution: tuple[int, int]
    source_length: int
    video_latent: Any | None = None
    audio_latent: Any | None = None


# eq=False: identity equality — the same Guide object IS the same guide.  The sampler's
# timed cond removal tracks the keyframe dicts built from guides by object identity, so
# value-based equality would be actively wrong here.
@dataclass(eq=False)
class Guide:
    """One native keyframe/guide cond entry added to the chain by ``H3AddGuide``.

    A guide is the monadic counterpart of comfy's ``MiniMaxH3AddGuide``: it anchors an
    image / short clip / audio as a native H3 keyframe cond row (re-injected every step,
    never denoised), but rides in the ``INJECT_LIST`` chain instead of being applied to
    conditioning at node time.  ``H3InjectSampler`` partitions the list, resolves each
    guide against the target latent, and appends the keyframe dicts to the positive
    conditioning at sample time.

    Attributes
    ----------
    inject_at:
        PIXEL-frame index to anchor at (raw, as entered).  Negative values count from the
        end of the video.  NOTE: these are pixel-frame indices unlike
        :attr:`Inject.inject_at`, which is a latent-frame index snapped to the 17-frame
        grid.
    start_percent:
        Fraction of the sampling schedule at which this guide's cond row is first added,
        in [0.0, 1.0].  ``0.0`` = present from step 0 (official/native behavior).  Higher
        values delay the guide's activation; it is absent from conditioning while sigma is
        still above the start threshold.  Half-open with ``end_percent``:
        ``[start_percent, end_percent)``.
    end_percent:
        Fraction of the sampling schedule at which this guide's cond row is removed, in
        [0.0, 1.0].  ``1.0`` = held for the whole run (official/native behavior, never
        removed).  Lower values release the cond row partway so a co-located fractional
        latent inject can finish denoising without the cond-token attractor re-pulling it
        toward source.  ``0.0`` = removed from step 0 (no-reference ablation).
    video_latent:
        Pre-encoded video latent for the anchored frame(s) (from the video VAE at node
        time), or ``None`` for an audio-only guide.
    audio_latent:
        Pre-encoded audio latent (from the audio VAE at node time), or ``None``.  Cropped
        to the video's remaining duration at sample time, once the target is known.
    resolution:
        ``(width, height)`` in pixels of the encoded frames; ``(0, 0)`` when audio-only.
        Validated exact-match against the target latent at sample time (no in-node resize).
    guide_frames:
        Number of pixel frames anchored: 1 (single image / audio-only) or a valid clip
        length ``17k + 5``.  Used for the sample-time bounds check.
    """

    inject_at: int
    start_percent: float
    end_percent: float
    video_latent: Any | None = None
    audio_latent: Any | None = None
    resolution: tuple[int, int] = (0, 0)
    guide_frames: int = 1


@dataclass
class RowSchedule:
    """The resolved schedule entry for one target latent video row.

    Produced by :func:`merge_schedule`; consumed by :mod:`~comfyui_h3_blended_inject.mask`,
    :mod:`~comfyui_h3_blended_inject.composite`, and :mod:`~comfyui_h3_blended_inject.sampler`.

    Attributes
    ----------
    row_idx:
        Zero-based index of this row in the target latent.
    denoise:
        Resolved denoise value for this row in [0.0, 1.0].  0.0 = exact preserve; 1.0 = fully
        generated; fractional = per-row img2img start (the row's ``m_r``).
    inject:
        The winning :class:`Inject` for this row, or ``None`` if no inject covers it.
    audio_frozen:
        True iff the winning inject has ``audio_mode == "keep"``.  When True, the derived
        mask sets the audio ticks corresponding to this row to 0 (exact preserve).
    region:
        Envelope region for this row, set by
        :func:`~comfyui_h3_blended_inject.envelope.classify_row_region`.  One of:
        ``'preserve'`` (d==0 hold, routed to mask+composite), ``'hold'`` (fractional hold,
        binary is-held gate), ``'fade'`` (ramp row, permanent prediction blend),
        ``'free'`` (d==1.0, no intervention).  Defaults to ``'free'`` for rows constructed
        outside :func:`merge_schedule` (e.g. in tests).
    audio_preserve:
        Computed property.  True iff this row's audio should be composited and mask-frozen
        at d=0.  In ``keep`` mode (``audio_frozen == True``) every row is preserved.  In
        ``fade`` mode, only the rows where ``denoise == 0.0`` are preserved — mirroring the
        video-preserve set exactly so audio follows the video envelope through the hold
        region.
    """

    row_idx: int
    denoise: float
    inject: Inject | None
    audio_frozen: bool = field(default=False)
    region: str = field(default="free")

    @property
    def audio_preserve(self) -> bool:
        """True iff this row's audio should be composited + mask-frozen at d=0.

        Keep-mode (audio_frozen) preserves the inject's audio everywhere; fade-mode
        preserves audio exactly where the video schedule reaches exact preserve
        (denoise == 0.0), so the audio-preserve set mirrors the video-preserve set
        and audio follows the video envelope through the hold region.
        """
        if self.audio_frozen:
            return True
        return self.inject is not None and self.inject.audio_mode == "fade" and self.denoise == 0.0

    @property
    def audio_denoise(self) -> float:
        """The per-row fractional audio denoise ``m_r`` for the per-row img2img sampler.

        Fractional generalization of :attr:`audio_preserve`:

        - **keep** mode (``audio_frozen``): ``0.0`` everywhere — audio is frozen/preserved.
        - **fade** mode: follows the video envelope, so it equals this row's ``denoise``.
        - **drop** mode / no inject: ``1.0`` — audio is generated from scratch.

        Consistency: ``audio_preserve`` is ``True`` exactly when ``audio_denoise == 0.0``.
        """
        if self.audio_frozen:
            return 0.0
        if self.inject is not None and self.inject.audio_mode == "fade":
            return self.denoise
        return 1.0


def merge_schedule(
    inject_list: InjectList,
    target_rows: int,
) -> list[RowSchedule]:
    """Merge a list of injects into a flat per-row schedule using last-in-wins semantics.

    Each inject in ``inject_list`` is evaluated to produce per-row denoise values (via
    :func:`~comfyui_h3_blended_inject.envelope.evaluate_envelope`).  When two injects claim
    the same target row, the later inject in the list (higher index) wins: its denoise value
    and content overwrite the earlier inject entirely.  There is no blending at overlap
    boundaries; the earlier inject's envelope ends with a hard edge at the first row the later
    inject claims.

    Rows not claimed by any inject are represented with ``denoise = 1.0`` and ``inject = None``
    (normal generation; they are not included in the returned list to keep output sparse).

    Parameters
    ----------
    inject_list:
        Ordered list of :class:`Inject` instances.  Append order matches
        ``H3AddInject`` chain order; later entries win on overlap.
    target_rows:
        Total number of rows in the target latent.

    Returns
    -------
    list[RowSchedule]
        One :class:`RowSchedule` per target row *that has at least one inject claiming it*,
        sorted by ``row_idx`` ascending.  Rows with no inject are omitted.

    Notes
    -----
    The caller (``H3InjectSampler``) is responsible for iterating over uncovered rows (no
    entry in the result) and treating them as ``d = 1.0`` / pure generation.
    """
    row_map: dict[int, tuple[Inject, float]] = {}
    for inj in inject_list:
        for row_idx, d in evaluate_envelope(
            inj.start_fade_in,
            inj.start_keyframes,
            inj.end_keyframes,
            inj.end_fade_out,
            inj.min_denoise,
            inj.interpolation_type,
            inj.source_length,
            target_rows,
            inj.inject_at,
        ):
            if 0 <= row_idx < target_rows:
                row_map[row_idx] = (inj, d)  # last writer wins
    return [
        RowSchedule(
            row_idx=row_idx,
            denoise=d,
            inject=inj,
            audio_frozen=(inj.audio_mode == "keep"),  # "keep" mode freezes audio at d=0
            region=classify_row_region(
                row_idx=row_idx,
                inject_at=inj.inject_at,
                start_fade_in=inj.start_fade_in,
                start_keyframes=inj.start_keyframes,
                end_keyframes=inj.end_keyframes,
                end_fade_out=inj.end_fade_out,
                min_denoise=inj.min_denoise,
            ),
        )
        for row_idx, (inj, d) in sorted(row_map.items())
    ]
