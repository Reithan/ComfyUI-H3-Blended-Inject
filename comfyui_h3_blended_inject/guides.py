"""Guide (native keyframe cond row) resolution and timed-removal helpers.

Pure, CPU-testable logic backing ``H3AddGuide`` and the sampler's per-guide timed cond
removal (see the wiki's ``timed-cond-removal-prototype`` doc):

- :func:`snap_guide_length` / :func:`resolve_frame_index` / :func:`frame_count_for_rows` /
  :func:`crop_audio_latent` mirror comfy's ``MiniMaxH3AddGuide`` node logic
  (``comfy_extras/nodes_minimax_h3.py``), relocated to node/sample time on our side.
- :func:`filter_released_keyframes` builds the released payload copy: keyframe entries
  removed by object identity, ``layout`` popped (its signature does NOT encode keyframes,
  so a stale prebuilt layout would silently desync; forward rebuilds when absent), and the
  cond latent lists rebuilt exactly as ``model_base.MiniMaxH3.extra_conds`` builds them
  (held keyframes first, then refs, with the same presence predicates).

Guide start/end step indices are computed directly in ``nodes._run_sampler`` from
``start_percent`` / ``end_percent`` and the sampler's ``steps`` count; the conditioning
wrapper compares the sampler loop's ``schedule_tail["current_step"]`` against those indices.

This module imports only stdlib + :mod:`~comfyui_h3_blended_inject.grid` (also pure);
latents are duck-typed (``.shape`` / slicing / ``.clone()``) so no torch import is needed.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

from comfyui_h3_blended_inject import grid
from comfyui_h3_blended_inject.schedule import Guide, Inject

# Audio latent frames per pixel frame (the two streams share one time axis).
# Equals comfy's ldm.minimax.model.FRAME_RESCALE (5/3 = 40 Hz audio / 24 fps video).
FRAME_RESCALE: float = grid.AUDIO_HZ / grid.FPS

# Minimum image-batch size that anchors as a clip; smaller batches use the first image only.
MIN_CLIP_FRAMES: int = 5


def partition_inject_list(inject_list: list[Any]) -> tuple[list[Inject], list[Guide]]:
    """Split a mixed ``INJECT_LIST`` into ``(injects, guides)``, each in chain order.

    ``H3AddInject`` and ``H3AddGuide`` append to the same chain; the sampler routes
    :class:`~comfyui_h3_blended_inject.schedule.Inject` entries through the per-row
    img2img schedule and :class:`~comfyui_h3_blended_inject.schedule.Guide` entries
    through the native keyframe cond path.

    Raises
    ------
    TypeError
        If the list contains an entry that is neither an ``Inject`` nor a ``Guide``.
    """
    injects: list[Inject] = []
    guides: list[Guide] = []
    for entry in inject_list:
        if isinstance(entry, Inject):
            injects.append(entry)
        elif isinstance(entry, Guide):
            guides.append(entry)
        else:
            raise TypeError(
                f"INJECT_LIST entries must be Inject or Guide, got {type(entry).__name__}"
            )
    return injects, guides


def snap_guide_length(n_frames: int) -> int:
    """Snap an image-batch length to a valid guide clip length, mirroring the official node.

    Batches shorter than :data:`MIN_CLIP_FRAMES` anchor the first image only (returns 1);
    longer batches snap DOWN to the model's ``17k + 5`` clip grid (5, 22, 39, ...).  A
    warning is issued whenever frames are dropped (the official node trims silently).

    Raises
    ------
    ValueError
        If ``n_frames`` is less than 1.
    """
    if n_frames < 1:
        raise ValueError(f"guide image batch must have at least 1 frame, got {n_frames}")
    if n_frames < MIN_CLIP_FRAMES:
        snapped = 1
    else:
        snapped = n_frames
        while snapped % 17 != MIN_CLIP_FRAMES:
            snapped -= 1
    if snapped < n_frames:
        warnings.warn(
            f"H3AddGuide: image batch of {n_frames} frames snapped down to {snapped} "
            f"(valid guide clip lengths are 1 and 17k+5: 5, 22, 39, ...); "
            f"{n_frames - snapped} trailing frame(s) dropped.",
            UserWarning,
            stacklevel=2,
        )
    return snapped


def frame_count_for_rows(target_rows: int) -> int:
    """Return the pixel-frame count covered by ``target_rows`` latent video rows.

    Equals the official node's ``sum(FRAME_PER_TOKEN[k % 5] for k in range(rows))``;
    :func:`grid.row_start_frame` computes the same value in closed form (the frame at
    which a hypothetical next row would begin).
    """
    return grid.row_start_frame(target_rows)


def resolve_frame_index(frame_idx: int, frame_count: int, guide_frames: int) -> int:
    """Resolve a raw (possibly negative) pixel ``frame_idx`` and bounds-check the guide.

    Mirrors the official node: negative indices count from the end of the video; the
    anchored clip must fit inside ``frame_count`` pixel frames.

    Returns
    -------
    int
        The resolved non-negative frame index.

    Raises
    ------
    ValueError
        If the frame (or clip starting there) falls outside the video.
    """
    resolved = frame_idx if frame_idx >= 0 else frame_count + frame_idx
    if resolved < 0 or resolved + guide_frames > frame_count:
        if guide_frames == 1:
            raise ValueError(f"frame_idx {frame_idx} is outside the video's {frame_count} frames")
        raise ValueError(
            f"a {guide_frames} frame guide clip at frame_idx {frame_idx} does not fit "
            f"in the video's {frame_count} frames"
        )
    return resolved


def crop_audio_latent(audio_latent: Any, audio_ticks: int, resolved_frame_index: int) -> Any:
    """Crop a guide's audio latent to the video's remaining duration.

    The streams share one time axis: :data:`FRAME_RESCALE` audio latent frames per pixel
    frame.  An anchor at ``resolved_frame_index`` leaves
    ``floor(audio_ticks - FRAME_RESCALE * resolved_frame_index)`` audio latent frames of
    track; longer audio is cropped (a fresh ``.clone()``, matching the official node so
    the stored latent is never a view of the caller's tensor).

    Raises
    ------
    ValueError
        If the anchor sits past the end of the audio track (no room for even 1 frame).
    """
    max_rt = math.floor(audio_ticks - FRAME_RESCALE * resolved_frame_index)
    if max_rt < 1:
        raise ValueError(
            f"frame_idx resolving to {resolved_frame_index} is past the end of the "
            f"video's audio track ({audio_ticks} audio latent frames)"
        )
    if audio_latent.shape[-1] > max_rt:
        audio_latent = audio_latent[..., :max_rt].clone()
    return audio_latent


def build_keyframe(guide: Guide, resolved_frame_index: int, audio_ticks: int) -> dict[str, Any]:
    """Build the native keyframe cond dict for a resolved guide.

    The dict matches what comfy's guide nodes append to ``minimax_keyframes``:
    ``resolved_frame_index`` always present, ``latent`` / ``audio_latent`` only when the
    guide carries them (audio cropped to the remaining track here, now that the target's
    ``audio_ticks`` is known).  The returned dict's OBJECT IDENTITY is the sampler's
    release-tracking key; build once per run and pass the same object through.
    """
    keyframe: dict[str, Any] = {"resolved_frame_index": resolved_frame_index}
    if guide.video_latent is not None:
        keyframe["latent"] = guide.video_latent
    if guide.audio_latent is not None:
        keyframe["audio_latent"] = crop_audio_latent(
            guide.audio_latent, audio_ticks, resolved_frame_index
        )
    return keyframe


def filter_released_keyframes(
    payload: dict[str, Any], released_ids: frozenset[int] | set[int]
) -> dict[str, Any]:
    """Return a payload copy with the released keyframe cond entries stripped.

    ``released_ids`` holds ``id()`` values of keyframe dicts to remove; object identity,
    NOT frame index, so official-node / fl2va keyframes anchored at the same frame are
    never caught.  The copy:

    - filters ``payload["keyframes"]`` to still-held entries (key removed entirely when
      none remain, matching the GPU-confirmed all-released prototype);
    - pops ``layout`` — ``PackedLayout.signature`` does not encode keyframes, so keeping
      the prebuilt layout would silently desync the packed sequence; the DiT forward
      rebuilds the layout whenever ``payload.get("layout")`` is ``None``;
    - rebuilds ``cond_video_latents`` / ``cond_audio_latents`` exactly as
      ``model_base.MiniMaxH3.extra_conds`` does: held keyframes first, then refs,
      preserving the original presence predicates (refs video uses ``"latent" in r``).

    The input ``payload`` is never mutated.
    """
    held = [kf for kf in payload.get("keyframes", []) if id(kf) not in released_ids]
    released = dict(payload)
    released.pop("layout", None)
    if held:
        released["keyframes"] = held
    else:
        released.pop("keyframes", None)
    refs = payload.get("refs") or []
    released["cond_video_latents"] = [
        kf["latent"] for kf in held if kf.get("latent") is not None
    ] + [r["latent"] for r in refs if "latent" in r]
    released["cond_audio_latents"] = [
        kf["audio_latent"] for kf in held if kf.get("audio_latent") is not None
    ] + [r["audio_latent"] for r in refs if r.get("audio_latent") is not None]
    return released
