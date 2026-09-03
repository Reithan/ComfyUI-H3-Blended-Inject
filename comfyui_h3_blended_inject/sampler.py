"""Per-row img2img sampler helpers (schedule-tail remap).

This module turns ComfyUI's global sampler into a per-row img2img sampler for H3 blended
injects using the GPU-validated **schedule-tail remap** mechanism.  Each fractional row
(``0 < m < 1``) runs the LAST ``d``-fraction (``d = m``) of the schedule's sigma values,
stretched over ALL steps:

- Release step ``k_d = round(steps·(1−m))`` (clamped to ``[0, steps]``); ``never = k_d ≥ steps``
  marks ``m == 0`` (exact-preserve) rows.  Stretch factor ``span = (steps − k_d) / steps``.
- ``row_sigma(i)`` is read EXACTLY from a dense ``steps²+1`` sigma grid at grid index
  ``k_d·(steps − i) + i·steps`` — an integer index for every row/step, so there is no
  interpolation error against the shift-12 schedule curvature.  A coarse-grid lerp is used
  only if that dense grid is absent.
- The DiT label is ``w = (sig_row / sig_g).clamp(max=1.0)`` so the model computes the truthful
  ``t_row = 1 − w·σ_g = 1 − σ_row`` in both the held and free phases.
- Per-step per-row scaling ``r = (sig_row − sig_row_next) / (sig_g − sig_g_next)`` is applied
  as ``x_cur = x_prev + r·(x_cur − x_prev)`` so each row integrates its own compressed tail
  instead of the global interval — this replaces the old denoised correction entirely.
- An init-only clean composite fires ONCE at ``i == 0`` (``x = w·x + (1−w)·clean``) to place
  each row on its own noise-line at ``σ_row(0)``; it is never re-injected (per-region SDEdit
  on the stretched tail).  A final ``where(never, clean, x_cur)`` guarantees ``m == 0`` preserve.

**Audio is axis-blind.**  Audio does NOT ride this per-row engine.  Its conditioning is
full-generation (``m == 1``), so ``row_sigma``/``sig_g``/``w``/``r`` run on the VIDEO σ_v axis
for every row and audio integrates exactly as stock at σ_v (the σ_a shift is applied inside the
H3 forward, not here).  Audio's fade is applied by the official ``KSamplerX0Inpaint`` composite
via a ``noise_mask`` built in ``nodes.py``
(:func:`~comfyui_h3_blended_inject.mask.derive_audio_composite_noise_mask`).  This retires the
earlier σ_a-axis per-row audio path (the Bug-B static source); see ``audio-native-composite.md``.
The audio-modality boundary (packed video-prefix length) is still tracked for
:func:`scale_packed_audio` and the observer plumbing.

**Clean-K/V splice (single-forward, Option II).**  When the observer-split block patches are
installed and fractional rows exist, the euler step runs ONE forward per interval
(:func:`_single_forward_denoised`) that carries an exact band-only side stream, letting a
fractional row broadcast (and perceive itself) as its clean anchor at ``σ_obs = m·σ_g`` while its
own velocity stays truthful — no ghost.  It reproduces the (removed) two-forward capture/splice
mechanism bit-for-bit.  See ``observer_split.py`` and the wiki ``clean-kv-split.md``.

The sampler loop dispatches each step through a ``_NATIVE_ROW_STEPS`` registry (keyed by
``base_fn.__name__``); ``_fallback_step`` is the default for unknown samplers.  ``sample_euler``
and ``sample_euler_ancestral`` (H3's RF-ancestral path) are registered here (PR1 + PR2), as are
the deterministic multistep samplers ``sample_dpmpp_2m`` and ``sample_res_multistep`` (PR3) whose
native steps carry per-row ``old_denoised`` history in ``_StepContext.state`` to restore true 2nd
order under the remap (a black-box ``base_fn`` on a 2-σ slice degrades to first order — Finding 1).

Everything here is pure and CPU-testable; ``torch`` is the only heavy dependency.  ``comfy``
imports stay lazy so this module loads without a ComfyUI environment.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from comfyui_h3_blended_inject.guides import filter_released_keyframes
from comfyui_h3_blended_inject.observer_split import (
    _band_mod_index,
    _embed_ratio,
    _observer_time_embed,
    _observer_timestep,
)


def scale_packed_audio(
    packed: torch.Tensor,
    video_element_count: int,
    audio_scale: float,
) -> torch.Tensor:
    """Scale the audio tail of a packed AV latent in place to match ``process_latent_in``.

    ComfyUI's :meth:`MiniMaxH3.process_latent_in` leaves the video slice untouched
    (``scale_factor == 1.0``) but multiplies the AUDIO slice by ``audio_scale``
    (``shift / audio_shift`` = 4.0), carrying the audio stream onto the video schedule.  The
    sampler therefore holds ``x_global`` whose audio slice is already ``audio_scale``-scaled.
    The clean reference the remap composites against must live in that same scaled space, so
    its audio slice carries the identical scale here.

    ``packed`` is a flat ``[B, video_elems + audio_elems]`` latent; ``video_element_count`` is
    ``prod(video_shape[1:])`` — the packed video-prefix length (and the same boundary the
    audio-modality mask uses in :func:`build_per_row_sampler_function`).  A no-op when
    ``audio_scale == 1.0`` or there is no audio tail.

    **Dual contract:** mutates ``packed`` in place *and* returns the same tensor.  Both are
    relied on: call sites use the return value for assignment, and tests assert ``out is packed``
    (identity) to confirm no copy was made.

    Parameters
    ----------
    packed:
        The packed AV latent to scale in place.
    video_element_count:
        Number of packed elements belonging to the video stream (the prefix length).
    audio_scale:
        The audio carry scale (``MiniMaxH3.audio_scale()``); ``1.0`` is a no-op.

    Returns
    -------
    torch.Tensor
        ``packed`` — the same object, audio tail scaled in place.
    """
    if audio_scale != 1.0 and video_element_count < packed.shape[-1]:
        packed[..., video_element_count:] *= audio_scale
    return packed


def quantize_denoise(m: torch.Tensor) -> torch.Tensor:
    """Snap per-row denoise fractions to H3's native 1/256 mask grid (ceil).

    H3's ``_token_grid_masks`` quantizes the pooled denoise mask with
    ``ceil(mask * 256) / 256`` before it reaches the DiT, so the network's per-row
    timestep labels live on a 1/256 grid.  Quantizing ``m`` up front keeps the remap's
    release-step and label math on the *identical* per-row values the DiT will see (the
    native quantization becomes a no-op on already-quantized values).  ``0`` and ``1`` are
    fixed points.
    """
    return torch.ceil(m * 256.0) / 256.0


def sampler_is_stochastic(fn: Callable[..., Any]) -> bool:
    """Return ``True`` iff ``fn`` is a genuinely stochastic sampler step function.

    Detection is signature-based (no hardcoded sampler list): a sampler is stochastic
    when it declares an ``eta`` parameter whose default is > 0 (ancestral/SDE families
    inject fresh noise scaled by ``eta`` each step).  Deterministic samplers either omit
    ``eta`` (euler, dpmpp_2m, res_multistep — whose public wrapper hardcodes ``eta=0.``
    internally) or default it to 0.  Known blind spot: samplers that inject noise
    unconditionally without an ``eta`` knob (ddpm, lcm, er_sde) are not detected —
    acceptable for a warning heuristic; do not rely on this for a hard gate.  The
    schedule-tail remap steps the base sampler one interval at a time and is not
    scale-invariant under stochastic renoise on H3's rectified-flow path, so stochastic
    samplers corrupt fractional/preserved rows and only warrant a warning.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    eta = params.get("eta")
    return (
        eta is not None
        and isinstance(eta.default, (int, float))
        and not isinstance(eta.default, bool)
        and eta.default > 0
    )


def build_conditioning_wrapper(
    schedule_tail: dict[str, Any],
    *,
    guide_release: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any], dict[str, Any]], Any]:
    """Return a ``model_function_wrapper`` that injects the remap's per-row conditioning.

    In the schedule-tail remap the sampler loop stashes the per-step per-row label mask into
    ``schedule_tail["pooled_current"]`` (pooled DiT conditioning built from ``w`` so the model
    computes ``t_row = 1 − w·σ = 1 − σ_row``, truthful in both the held and free phases).  This
    wrapper places that conditioning into a COPY of ``c`` and returns the raw denoised
    prediction — there is NO denoised correction: the sampler loop's per-row ``r``-scaling
    confines each row's integral to its compressed tail.  Before the loop's first step
    ``pooled_current`` is absent, so ``pooled_ones`` (native full-denoise labels) is used.

    Two optional jobs layer on top:

    - **Observer-label K/V split.**  When ``schedule_tail["observer"]`` is populated (by
      :func:`~comfyui_h3_blended_inject.observer_split.install_observer_split`), the per-call
      observer labels (``t_obs = 1 − m·σ``) consumed by the DiT block patches are refreshed via
      ``observer_call_update`` — imported lazily so this module needs no ComfyUI at load time.
    - **Per-guide step-gated cond.**  When ``guide_release["entries"]`` and/or
      ``guide_release["pending_entries"]`` are armed, each guide's keyframe cond row is
      stripped from a COPY of ``minimax_payload`` whenever the step falls outside that
      guide's ``[start_step, end_step)`` window.  Current step is read from
      ``schedule_tail["current_step"]`` (set by the sampler loop each iteration).  Matching
      is by object identity, so foreign keyframes are never caught.

    Matches ComfyUI's ``model_function_wrapper`` contract:
    ``(apply_model, args_dict) -> prediction``.

    Parameters
    ----------
    schedule_tail:
        The schedule-tail config dict.  Read keys: ``"pooled_current"`` (per-step, set by the
        sampler loop), ``"pooled_ones"`` (native-label fallback), ``"observer"`` (optional
        observer state).
    guide_release:
        Mutable state dict for per-guide step-gated cond, shared with ``_run_sampler``:

        - ``"entries"``: list of ``(keyframe_id, end_step)`` pairs — guide is excluded when
          ``current_step >= end_step``.
        - ``"pending_entries"``: list of ``(keyframe_id, start_step)`` pairs — guide is
          excluded when ``current_step < start_step``.
        - ``"cache"``: filled here; filtered payload copies keyed by
          ``(id(payload), frozenset(excluded_ids))`` so cond/uncond streams never cross.

        ``None`` or empty/missing entry lists → inert.

    Returns
    -------
    Callable
        A ``model_function_wrapper`` for ``model_options["model_function_wrapper"]``.
    """

    def wrapper(apply_model: Callable[..., Any], args_dict: dict[str, Any]) -> Any:
        inp = args_dict["input"]
        c = dict(args_dict["c"])
        active_pooled = schedule_tail.get("pooled_current", schedule_tail["pooled_ones"])
        # Clean-K/V observer splice: bump the per-forward token so each forward
        # rebuilds its block splice plan. See observer_split.py.
        obs = schedule_tail.get("observer")
        if obs is not None:  # pragma: no cover - armed only on GPU runs
            from comfyui_h3_blended_inject.observer_split import observer_call_update

            observer_call_update(obs)
        for key, value in active_pooled.items():
            c[key] = value.to(device=inp.device, dtype=inp.dtype)
        # --- Per-guide step-gated cond (start/end step windows) ---
        # "entries" = (kf_id, end_step): guide excluded when current_step >= end_step.
        # "pending_entries" = (kf_id, start_step): guide excluded when current_step < start_step.
        if guide_release is not None:
            has_release = bool(guide_release.get("entries"))
            has_pending = bool(guide_release.get("pending_entries"))
            if has_release or has_pending:
                cur = schedule_tail.get("current_step", 0)
                released_ids = frozenset(
                    kf_id for kf_id, end_step in guide_release.get("entries", []) if cur >= end_step
                )
                pending_ids = frozenset(
                    kf_id
                    for kf_id, start_step in guide_release.get("pending_entries", [])
                    if cur < start_step
                )
                excluded_ids = released_ids | pending_ids
                payload = c.get("minimax_payload")
                if excluded_ids and isinstance(payload, dict) and payload.get("keyframes"):
                    cache = guide_release.setdefault("cache", {})
                    cache_key = (id(payload), excluded_ids)
                    filtered = cache.get(cache_key)
                    if filtered is None:
                        filtered = filter_released_keyframes(payload, excluded_ids)
                        cache[cache_key] = filtered
                    c["minimax_payload"] = filtered
        return apply_model(inp, args_dict["timestep"], **c)

    return wrapper


def _shift_schedule(sig: torch.Tensor, from_shift: float, to_shift: float) -> torch.Tensor:
    """Vectorized :func:`time_shift_sigma` over a whole sigma schedule/grid tensor.

    Applies the same two-step warp element-wise so an entire coarse schedule or dense grid can
    be re-shifted from the video shift to the audio shift in one shot.  ``f(0) = 0``, ``f(1) = 1``.
    """
    base = sig / (from_shift + sig * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def _stream_row_sigma(
    m: torch.Tensor,
    i: int,
    steps_n: int,
    dense_grid: torch.Tensor | None,
    coarse: torch.Tensor,
    n_sig: int,
) -> torch.Tensor:
    """Per-row target sigma for ONE modality's schedule at global step ``i``.

    Pure function of the per-row denoise fractions ``m`` and that modality's schedule grids.
    Release step ``k_d = round(steps·(1−m))`` compresses each row onto the last ``span`` of the
    schedule; with the dense ``steps²+1`` grid present the row sigma is read at an EXACT integer
    grid index ``k_d·(steps − i) + i·steps`` (no interpolation error), otherwise the coarse grid
    is lerped at ``k_d + i·span``.

    Extracted so the sampler loop's ``row_sigma`` AND the observer split's single-forward
    block-0 side-hidden init compute the identical ``σ_row`` from the same ``m`` — the observer
    passes its own token-ordered fractional ``m`` so there is no packed/token layout mismatch.
    """
    k_d = torch.round(steps_n * (1.0 - m)).clamp(0, steps_n)
    if dense_grid is not None:
        idx = (k_d * (steps_n - i) + i * steps_n).round().long()
        return dense_grid[idx.clamp(0, steps_n * steps_n)]
    span = (steps_n - k_d) / steps_n
    idx = (k_d + i * span).clamp(0.0, float(steps_n))
    lo = idx.floor().long().clamp(0, n_sig - 1)
    hi = (lo + 1).clamp(0, n_sig - 1)
    fr = (idx - lo.to(idx.dtype)).clamp(0.0, 1.0)
    return coarse[lo] * (1.0 - fr) + coarse[hi] * fr


@dataclass
class _StepContext:
    """Per-step context passed to each row step function.

    Carries all information a step function needs to compute the r-scaled row update for one
    global step interval.  ``state`` is a mutable per-row history store reserved for multistep
    samplers (PR3); it is unused in PR1.
    """

    model: Any
    x_prev: torch.Tensor
    i: int
    sigmas: torch.Tensor  # raw schedule arg — same tensor the original passed to base_fn
    sig_row: torch.Tensor  # per-row sigma at step i (audio rows on shifted schedule)
    sig_row_next: torch.Tensor  # per-row sigma at step i+1
    sig_g: torch.Tensor  # global_sigma(i).clamp(min=1e-8), per-row
    sig_g_next: torch.Tensor  # global_sigma(i+1), per-row
    extra_args: dict[str, Any] | None
    callback: Any  # _cb(i)-wrapped callback, or None
    disable: Any
    kwargs: dict[str, Any]
    base_fn: Callable[..., Any]
    state: dict[str, Any] = field(default_factory=dict)


#: Type alias for a per-row step function.
StepFn = Callable[[_StepContext], torch.Tensor]


def _fallback_step(ctx: _StepContext) -> torch.Tensor:
    """Reproduce the original loop body: delegate to ``base_fn`` then apply r-scale.

    This is the default path for any sampler not registered in ``_NATIVE_ROW_STEPS``.  It
    preserves the original behavior exactly — no numerical change for any sampler.
    """
    x_base = ctx.base_fn(
        ctx.model,
        ctx.x_prev,
        ctx.sigmas[ctx.i : ctx.i + 2],
        extra_args=ctx.extra_args,
        callback=ctx.callback,
        disable=True,
        **ctx.kwargs,
    )
    r = ((ctx.sig_row - ctx.sig_row_next) / (ctx.sig_g - ctx.sig_g_next).clamp(min=1e-8)).clamp(
        min=0.0
    )
    return ctx.x_prev + r * (x_base - ctx.x_prev)


def _single_forward_denoised(  # pragma: no cover - requires live H3 model (GPU)
    ctx: _StepContext,
    sigma: torch.Tensor,
    s_in: torch.Tensor,
    extra_args: dict[str, Any],
    x: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact single-forward clean-K/V denoise for fractional fade rows (Option II).

    Reproduces the (removed) two-forward clean-K/V mechanism bit-for-bit in ONE forward.  The
    two-forward mechanism — a capture forward on the static ``clean`` inject re-noised to the
    observed level ``σ_obs = m·σ_g`` whose fractional-band K/V are spliced into a truthful ``σ_row``
    self forward — is preserved in wiki ``observed-level-plant/clean-kv-split.md``.

    Primes the band-only side stream for this step (per-row embed ``ratio``, observer modulation
    ``t_emb_m`` + mod indices — computed from the observer's OWN token-ordered ``m`` so there is no
    packed/token layout mismatch), publishes the truthful ``σ_row`` labels for the main stream, and
    runs ONE forward in ``single`` mode.  The block patches carry ``h_m`` block-to-block and read
    one combined K/V with two queries, reproducing the two-forward ``denoised`` bit-for-bit at ~half
    the model cost.  See ``option-ii-single-forward.md``.
    """
    xin = ctx.x_prev if x is None else x
    st = ctx.state
    obs = st["observer"]
    schedule_tail = st["schedule_tail"]
    make_pooled = st["make_pooled"]

    st["prime_side_stream"](ctx.i)  # per-step token-ordered ratio / t_emb_m / mod indices
    if make_pooled is not None:
        w = (ctx.sig_row / ctx.sig_g).clamp(max=1.0)
        schedule_tail["pooled_current"] = make_pooled(w.to(device=st["pdev"], dtype=st["pdtype"]))
    obs["h_m"] = None
    obs["mode"] = "single"
    denoised = ctx.model(xin, sigma * s_in, **extra_args)
    obs["mode"] = None
    return denoised


def _euler_step(ctx: _StepContext) -> torch.Tensor:
    """Native per-row euler step: inline comfy's deterministic euler then apply r-scale.

    Numerically equivalent to ``_fallback_step`` when ``base_fn`` is ``sample_euler`` with
    default ``s_churn=0``.  Using the inlined version opens the seam for PR2 (RF-ancestral)
    and PR3 (multistep) to extend this protocol without going through ``base_fn``.

    The denom and dt use the raw ``ctx.sigmas[i]`` / ``ctx.sigmas[i+1]`` (the unmodified
    schedule tensor, same dtype/device as the original ``base_fn`` call), exactly as
    ``sample_euler`` would see; the per-row/audio behavior enters only through ``r``.
    """
    extra_args = {} if ctx.extra_args is None else ctx.extra_args
    s_in = ctx.x_prev.new_ones([ctx.x_prev.shape[0]])
    sigma_i = ctx.sigmas[ctx.i]
    st = ctx.state
    obs = st.get("observer")
    frac = st.get("frac_mask")
    if obs is not None and frac is not None and bool(frac.any()):  # pragma: no cover - GPU
        denoised = _single_forward_denoised(ctx, sigma_i, s_in, extra_args)
    else:
        denoised = ctx.model(ctx.x_prev, sigma_i * s_in, **extra_args)
    d = (ctx.x_prev - denoised) / sigma_i
    if ctx.callback is not None:
        ctx.callback(
            {
                "i": 0,
                "denoised": denoised,
                "x": ctx.x_prev,
                "sigma": sigma_i,
                "sigma_hat": sigma_i,
            }
        )
    x_base = ctx.x_prev + d * (ctx.sigmas[ctx.i + 1] - sigma_i)
    r = ((ctx.sig_row - ctx.sig_row_next) / (ctx.sig_g - ctx.sig_g_next).clamp(min=1e-8)).clamp(
        min=0.0
    )
    return ctx.x_prev + r * (x_base - ctx.x_prev)


def _default_row_noise_sampler(ctx: _StepContext, extra_args: dict[str, Any]) -> Any:
    """Build (once) or fetch the seeded ``default_noise_sampler`` for RF-ancestral steps.

    Shared by :func:`_euler_ancestral_rf_step` and :func:`_dpmpp_2s_ancestral_step` (both use
    comfy's plain per-interval Gaussian noise, not the SDE BrownianTree).  Tests inject a
    deterministic sampler via ``ctx.kwargs["noise_sampler"]``; the GPU path imports comfy's
    ``default_noise_sampler``.  Persisted across steps via ``ctx.state`` so the generator advances
    once per interval, matching stock's single pre-loop build.
    """
    if "noise_sampler" not in ctx.state:
        ns = ctx.kwargs.get("noise_sampler")
        if ns is None:
            from comfy.k_diffusion.sampling import default_noise_sampler  # pragma: no cover

            ns = default_noise_sampler(  # pragma: no cover
                ctx.x_prev, seed=extra_args.get("seed")
            )
        ctx.state["noise_sampler"] = ns
    return ctx.state["noise_sampler"]


def _euler_ancestral_rf_step(ctx: _StepContext) -> torch.Tensor:
    """Native per-row RF-ancestral step for euler_ancestral on H3.

    Implements one step of ``sample_euler_ancestral_RF`` elementwise over the per-row sigma
    tensors from the schedule-tail remap.  No r-scaling — the per-row sigma integration is
    handled directly by the ancestral math.

    **Audio is axis-blind here.**  ``sig_row``/``sig_g`` run on the video σ_v axis for every row
    (:func:`row_sigma`), so audio rows integrate exactly as stock ancestral at σ_v (the σ_a shift
    is applied inside the H3 forward) — no per-row σ_a compression.  Audio's fade is applied by the
    official ``KSamplerX0Inpaint`` composite via ``noise_mask``, which acts inside ``ctx.model``
    here.  This retires the earlier σ_a-axis audio-ancestral path (the Bug-B static source); see
    ``audio-native-composite.md``.

    One model eval per interval at the global carrier sigma.  Per-row ``x0`` is recovered as
    ``denoised_r = x_prev − σ_row·(x_prev − denoised)/σ_carrier`` (the conditioning wrapper
    labels the model with ``t_row = 1 − σ_row`` so it returns the per-row denoised; the
    velocity ``v = (x − denoised) / σ_carrier`` then maps back to per-row ``x0``).  All
    ancestral terms (``sigma_down``, ``alpha_ip1/alpha_down``, ``renoise_coeff``, ``ratio``)
    are computed elementwise over the ``sig_row`` / ``sig_row_next`` tensors.

    Guarantees:

    - **m=1 rows reproduce stock** ``sample_euler_ancestral_RF`` **exactly**: for m=1,
      ``sig_row == sigmas[i]``, so ``denoised_r == denoised`` and all per-row terms match the
      scalar stock values — given an identical noise draw the outputs are bit-for-bit equal.
    - **Terminal step** (``sig_row_next == 0``) falls out without a branch: ``sigma_down=0``,
      ``ratio=0`` → ``x = denoised_r``; ``renoise_coeff=0`` → no noise added.
    - **m=0 rows** freeze: ``sig_row=0`` clamped to ``eps`` → ``ratio→0``, ``coeff→0`` →
      ``x = denoised_r = x_prev``; the outer ``where(never, clean, x)`` guard restores clean.

    The seeded noise sampler is built once on the first call (or taken from
    ``ctx.kwargs["noise_sampler"]`` for CPU tests) and persists across steps via
    ``ctx.state`` — the outer loop shares one ``step_state`` dict across all iterations so the
    generator advances correctly, matching stock's single pre-loop build.
    GPU-only paths (``noise_scale`` attribute, ``default_noise_sampler`` import) are
    ``# pragma: no cover``.
    """
    extra_args = {} if ctx.extra_args is None else ctx.extra_args
    eta = ctx.kwargs.get("eta", 1.0)
    s_noise = ctx.kwargs.get("s_noise", 1.0)
    if hasattr(ctx.model, "inner_model"):  # pragma: no cover
        s_noise = s_noise * getattr(
            ctx.model.inner_model.model_patcher.get_model_object("model_sampling"),
            "noise_scale",
            1.0,
        )

    # Build or retrieve the seeded noise sampler (persists across steps via ctx.state).
    noise_sampler = _default_row_noise_sampler(ctx, extra_args)

    carrier = ctx.sigmas[ctx.i]
    s_in = ctx.x_prev.new_ones([ctx.x_prev.shape[0]])
    # Fractional VIDEO rows must denoise through the clean-K/V observer splice, exactly like
    # _euler_step — otherwise a fractional row's noisy K/V leak into co-located rows via joint
    # attention and re-imprint the fade artifact (Bug F).  Audio rows carry no observer stream
    # (audio conditioning is full-gen m=1; the fade is handled by the official composite), so this
    # forward stays axis-blind for audio.
    st = ctx.state
    obs = st.get("observer")
    frac = st.get("frac_mask")
    if obs is not None and frac is not None and bool(frac.any()):  # pragma: no cover - GPU
        denoised = _single_forward_denoised(ctx, carrier, s_in, extra_args)
    else:
        denoised = ctx.model(ctx.x_prev, carrier * s_in, **extra_args)
    if ctx.callback is not None:
        ctx.callback(
            {
                "i": 0,
                "denoised": denoised,
                "x": ctx.x_prev,
                "sigma": carrier,
                "sigma_hat": carrier,
            }
        )

    # Recover per-row x0 from global-carrier velocity.
    v = (ctx.x_prev - denoised) / carrier
    denoised_r = ctx.x_prev - ctx.sig_row * v

    # Elementwise ancestral update over per-row sigma tensors.
    eps = 1e-8
    si = ctx.sig_row.clamp(min=eps)
    sip1 = ctx.sig_row_next
    downstep_ratio = 1.0 + (sip1 / si - 1.0) * eta
    sigma_down = sip1 * downstep_ratio
    alpha_ip1 = 1.0 - sip1
    alpha_down = 1.0 - sigma_down
    renoise_coeff = torch.sqrt(
        torch.clamp(sip1**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2, min=0.0)
    )
    ratio = sigma_down / si
    x = ratio * ctx.x_prev + (1.0 - ratio) * denoised_r
    if eta > 0:
        noise = noise_sampler(carrier, ctx.sigmas[ctx.i + 1])
        x = (alpha_ip1 / alpha_down) * x + noise * s_noise * renoise_coeff
    return x


def _recover_at(
    ctx: _StepContext,
    carrier: torch.Tensor,
    sig_row_target: torch.Tensor,
    x_input: torch.Tensor,
    *,
    fire_callback: bool,
) -> torch.Tensor:
    """One model eval at ``carrier`` on ``x_input``; recover per-row ``x0`` at ``sig_row_target``.

    Generalises :func:`_recover_row_denoised` for callers that must denoise a tensor other than
    ``ctx.x_prev`` and/or map to a per-row sigma other than ``ctx.sig_row`` — specifically the
    second (midpoint) eval of :func:`_dpmpp_sde_step`.  The conditioning wrapper labels the model
    with ``t = 1 − w·σ`` so a single forward returns a velocity ``v = (x_input − denoised)/carrier``
    that maps back to each row's own ``x0`` at the requested level::

        denoised_r = x_input − sig_row_target · v

    Fractional VIDEO rows denoise through the clean-K/V observer splice
    (:func:`_single_forward_denoised`) so their noisy K/V do not leak into co-located rows via
    joint attention (Bug F); audio rows carry no observer stream (axis-blind, fade rides the
    official composite).  ``fire_callback`` gates the euler-shaped callback so a multi-eval step
    reports only its first eval.
    """
    extra_args = {} if ctx.extra_args is None else ctx.extra_args
    s_in = x_input.new_ones([x_input.shape[0]])
    st = ctx.state
    obs = st.get("observer")
    frac = st.get("frac_mask")
    if obs is not None and frac is not None and bool(frac.any()):  # pragma: no cover - GPU
        denoised = _single_forward_denoised(ctx, carrier, s_in, extra_args, x=x_input)
    else:
        denoised = ctx.model(x_input, carrier * s_in, **extra_args)
    if fire_callback and ctx.callback is not None:
        ctx.callback(
            {
                "i": 0,
                "denoised": denoised,
                "x": x_input,
                "sigma": carrier,
                "sigma_hat": carrier,
            }
        )
    v = (x_input - denoised) / carrier
    return x_input - sig_row_target * v


def _recover_row_denoised(ctx: _StepContext, carrier: torch.Tensor) -> torch.Tensor:
    """One model eval at the global carrier σ; recover per-row ``x0`` via the velocity identity.

    Shared spine for the deterministic multistep steps (``_dpmpp_2m_step`` /
    ``_res_multistep_step``) and the SDE steps' first eval, mirroring the recovery inside
    :func:`_euler_ancestral_rf_step`.  The conditioning wrapper labels the model with
    ``t_row = 1 − σ_row`` so a single forward at ``carrier`` returns a velocity
    ``v = (x_prev − denoised) / carrier`` that maps back to each row's own ``x0``::

        denoised_r = x_prev − sig_row · v

    Fractional VIDEO rows denoise through the clean-K/V observer splice
    (:func:`_single_forward_denoised`) exactly as the euler steps do, so their noisy K/V do not
    leak into co-located rows via joint attention (Bug F).  Audio rows carry no observer stream
    (their fade rides the official composite), so the forward stays axis-blind for audio.  Fires
    the euler-shaped callback once.
    """
    return _recover_at(ctx, carrier, ctx.sig_row, ctx.x_prev, fire_callback=True)


def _dpmpp_2m_step(ctx: _StepContext) -> torch.Tensor:
    """Native per-row DPM-Solver++(2M) step (deterministic).

    Ports comfy's ``sample_dpmpp_2m`` (k_diffusion/sampling.py:796-818) to the schedule-tail
    remap by integrating each row over its OWN sigma tail.  The ``_fallback_step`` path calls
    ``base_fn`` on a 2-σ slice per interval, which re-initializes ``old_denoised = None`` on every
    call and silently degrades to first order everywhere (Finding 1); carrying per-row
    ``old_denoised`` across the outer loop restores true 2nd order, including on free ``m == 1``
    rows.

    One model eval per interval at the global carrier σ; per-row ``x0`` recovered via
    :func:`_recover_row_denoised`.  All time coefficients (``t = −log σ``, ``h``, ``r``) are
    elementwise over the ``sig_row`` / ``sig_row_next`` tensors.

    Guarantees:

    - **m=1 rows reproduce stock** ``sample_dpmpp_2m`` **exactly**: ``sig_row == sigmas[i]`` and
      ``denoised_r == denoised``, so every per-row term equals the scalar stock value.
    - **First step** (no history) and **terminal rows** (``sig_row_next == 0``) take the
      first-order exponential branch, matching stock's ``old_denoised is None or sigmas[i+1] == 0``.
    - **m=0 rows freeze**: ``sig_row ≤ ε`` → held at ``x_prev``; the outer
      ``where(never, clean, ·)`` guard then restores clean.
    """
    eps = 1e-8
    carrier = ctx.sigmas[ctx.i]
    denoised_r = _recover_row_denoised(ctx, carrier)

    # Only ``si`` is clamped: leaving ``sig_row_next`` raw makes the terminal step EXACT
    # (``t_next = +inf`` → ``expm1(-inf) == -1``, ``ratio == 0``), reproducing stock's ``sigmas[i+1]
    # == 0`` branch bit-for-bit.  Clamping ``si`` alone still avoids the frozen-row ``inf − inf``
    # NaN (``m=0`` rows are discarded by the ``sig_row <= eps`` guard below regardless).
    si = ctx.sig_row.clamp(min=eps)
    sip1 = ctx.sig_row_next
    t = -torch.log(si)
    t_next = -torch.log(sip1)
    h = t_next - t
    ratio = sip1 / si  # sigma_fn(t_next) / sigma_fn(t)
    em1 = torch.expm1(-h)  # (-h).expm1()

    first_order = ratio * ctx.x_prev - em1 * denoised_r
    old = ctx.state.get("dpmpp_2m_old")
    if old is None:
        x_new = first_order
    else:
        old_denoised_r, old_sig_row = old
        h_last = t - (-torch.log(old_sig_row.clamp(min=eps)))
        r = h_last / h
        denoised_d = (1.0 + 1.0 / (2.0 * r)) * denoised_r - (1.0 / (2.0 * r)) * old_denoised_r
        second_order = ratio * ctx.x_prev - em1 * denoised_d
        x_new = torch.where(ctx.sig_row_next <= eps, first_order, second_order)

    ctx.state["dpmpp_2m_old"] = (denoised_r, ctx.sig_row)
    return torch.where(ctx.sig_row <= eps, ctx.x_prev, x_new)


def _res_multistep_step(ctx: _StepContext) -> torch.Tensor:
    """Native per-row res_multistep step (deterministic, ``eta=0``).

    Ports comfy's ``res_multistep`` (k_diffusion/sampling.py:1417-1456, ``eta=0``/``cfg_pp=False``)
    to the schedule-tail remap.  Same Finding-1 fix as :func:`_dpmpp_2m_step`: per-row
    ``old_denoised`` and ``old_sigma_down`` history are carried across the outer loop so the RES
    exponential integrator runs at 2nd order per row instead of degrading to first order.

    ``eta=0`` makes ``sigma_down == sig_row_next`` and adds no renoise (deterministic).  One model
    eval per interval; per-row ``x0`` via :func:`_recover_row_denoised`.

    Guarantees mirror :func:`_dpmpp_2m_step`: m=1 rows reproduce stock ``res_multistep`` exactly;
    first-step / terminal rows take the Euler branch (stock's ``sigma_down == 0 or old_denoised is
    None``); m=0 rows freeze under the outer ``where(never, clean, ·)`` guard.
    """
    eps = 1e-8
    carrier = ctx.sigmas[ctx.i]
    denoised_r = _recover_row_denoised(ctx, carrier)

    si = ctx.sig_row
    sigma_down = ctx.sig_row_next  # eta=0 → sigma_down == sig_row_next, sigma_up == 0
    # Euler (first-order) branch — used on the first step and terminal rows.
    d = (ctx.x_prev - denoised_r) / si.clamp(min=eps)
    euler = ctx.x_prev + d * (sigma_down - si)

    old = ctx.state.get("res_multistep_old")
    if old is None:
        x_new = euler
    else:
        old_denoised_r, old_sigma_down, old_sig_row = old
        t = -torch.log(si.clamp(min=eps))
        t_old = -torch.log(old_sigma_down.clamp(min=eps))
        t_next = -torch.log(sigma_down.clamp(min=eps))
        t_prev = -torch.log(old_sig_row.clamp(min=eps))
        h = t_next - t
        c2 = (t_prev - t_old) / h
        phi1 = torch.expm1(-h) / (-h)
        phi2 = (phi1 - 1.0) / (-h)
        b1 = torch.nan_to_num(phi1 - phi2 / c2, nan=0.0)
        b2 = torch.nan_to_num(phi2 / c2, nan=0.0)
        second = torch.exp(-h) * ctx.x_prev + h * (b1 * denoised_r + b2 * old_denoised_r)
        x_new = torch.where(sigma_down <= eps, euler, second)

    ctx.state["res_multistep_old"] = (denoised_r, sigma_down, ctx.sig_row)
    return torch.where(ctx.sig_row <= eps, ctx.x_prev, x_new)


# ---------------------------------------------------------------------------
# Per-row DPM++ SDE family (PR4): dpmpp_sde, dpmpp_2m_sde, dpmpp_3m_sde.
#
# Ports comfy's SDE samplers (k_diffusion/sampling.py @b78cec87: sample_dpmpp_sde 737-792,
# sample_dpmpp_2m_sde 821-873, sample_dpmpp_3m_sde 881-941) to the schedule-tail remap.  H3 is
# ``CONST`` (rectified flow) so ``sigma_to_half_log_snr(σ) = logit(σ).neg() = log((1−σ)/σ)`` and
# ``alpha_t = σ·exp(λ) = 1 − σ`` — the RF ``α = 1 − σ`` identity lives INSIDE the SNR helpers.  The
# per-row steps run the logit form elementwise over ``sig_row`` clamped to ``[ε, 1−ε]`` (comfy's
# ``offset_first_sigma_for_snr`` only fixes the global scalar σ≈1).  One model eval per interval at
# the global carrier σ recovers per-row ``x0`` via the shared velocity spine; SDE renoise is queried
# from a scalar-interval BrownianTree at the GLOBAL carrier interval (unit-variance output, so all
# per-row scaling stays in the elementwise coefficients — noise correlation follows the global
# schedule, an accepted approximation).  Axis-blind: σ_v for every row, audio fade rides the
# official composite.  See ``sampler-class-support/delivery-plan.md`` (PR4), ``native-step-design``.
# ---------------------------------------------------------------------------


#: Upper σ clamp for the half-log-SNR transform.  Mirrors comfy's ``offset_first_sigma_for_snr``
#: (percent offset ``1e-4`` → σ≈0.9999 for CONST/RF): σ=1 gives λ=−∞ and ``exp(−λ)`` overflow that
#: makes ``get_ancestral_step`` (called in exp(−λ) space) lose all precision.  ``1e-4`` keeps λ
#: finite/moderate at the first step; the exact boundary value is an accepted approximation (the
#: task-#73 GPU spike validates the real schedule).  Lower clamp stays ``1e-8`` (terminal rows are
#: handled by the ``sig_row_next <= eps`` branch, so this only guards ``log(0)``).
_SNR_SIGMA_HI = 1.0 - 1e-4


def _snr_clamp(sigma: torch.Tensor) -> torch.Tensor:
    """Clamp σ into the stable half-log-SNR domain ``[1e-8, 1−1e-4]``."""
    return sigma.clamp(1e-8, _SNR_SIGMA_HI)


def _snr_lambda(sigma_c: torch.Tensor) -> torch.Tensor:
    """Half-log-SNR ``λ = log((1−σ)/σ)`` for CONST/RF (``sigma_c`` pre-clamped to ``(0, 1)``)."""
    return sigma_c.logit().neg()


def _snr_sigma(lam: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_snr_lambda` for CONST/RF: ``σ = sigmoid(−λ)``."""
    return (-lam).sigmoid()


def _sde_get_ancestral_step(
    sigma_from: torch.Tensor, sigma_to: torch.Tensor, eta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Elementwise ``get_ancestral_step`` (sampling.py:68-75) over tensors.

    Comfy's scalar version uses python ``min()``; the per-row reimpl uses ``torch.minimum``.  Called
    in ``exp(−λ) = σ/α`` space by the SDE steps.  ``eta == 0`` yields ``(sigma_to, 0)`` naturally
    (``su = min(sigma_to, 0) = 0``, ``sd = sqrt(sigma_to²) = sigma_to``).
    """
    inner = torch.clamp(sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2, min=0.0)
    sigma_up = torch.minimum(sigma_to, eta * torch.sqrt(inner))
    sigma_down = torch.sqrt(torch.clamp(sigma_to**2 - sigma_up**2, min=0.0))
    return sigma_down, sigma_up


def _sde_noise_sampler(ctx: _StepContext, extra_args: dict[str, Any]) -> Any:
    """Build (once) or fetch the SDE noise sampler, persisted across steps via ``ctx.state``.

    Tests inject a deterministic sampler via ``ctx.kwargs["noise_sampler"]``; the GPU path builds a
    scalar-interval ``BrownianTreeNoiseSampler`` (queried at the global carrier interval).
    """
    if "noise_sampler" not in ctx.state:
        ns = ctx.kwargs.get("noise_sampler")
        if ns is None:  # pragma: no cover - GPU (BrownianTree needs the real schedule)
            from comfy.k_diffusion.sampling import BrownianTreeNoiseSampler

            sigmas = ctx.sigmas
            pos = sigmas[sigmas > 0]
            ns = BrownianTreeNoiseSampler(
                ctx.x_prev, pos.min(), sigmas.max(), seed=extra_args.get("seed"), cpu=True
            )
        ctx.state["noise_sampler"] = ns
    return ctx.state["noise_sampler"]


def _sde_s_noise(ctx: _StepContext) -> float:
    """``s_noise`` scaled by the model_sampling ``noise_scale`` attr (GPU-only), matching stock."""
    s_noise = ctx.kwargs.get("s_noise", 1.0)
    if hasattr(ctx.model, "inner_model"):  # pragma: no cover
        s_noise = s_noise * getattr(
            ctx.model.inner_model.model_patcher.get_model_object("model_sampling"),
            "noise_scale",
            1.0,
        )
    return s_noise


def _dpmpp_2m_sde_step(ctx: _StepContext) -> torch.Tensor:
    """Native per-row DPM-Solver++(2M) SDE step (``solver_type='midpoint'``).

    Ports comfy's ``sample_dpmpp_2m_sde`` (sampling.py:821-873): PR3's 2M ``old_denoised`` history
    carry ∩ PR2's stochastic renoise.  One model eval per interval; per-row ``x0`` via the shared
    spine.  Per-row ``(denoised_r, h)`` history in ``ctx.state``.  m=1 rows reproduce stock exactly
    (``sig_row == sigmas[i]``); terminal rows (``sig_row_next == 0``) take the denoising branch;
    m=0 rows freeze under the outer ``where(never, clean, ·)`` guard.
    """
    extra_args = {} if ctx.extra_args is None else ctx.extra_args
    eps = 1e-8
    eta = ctx.kwargs.get("eta", 1.0)
    carrier = ctx.sigmas[ctx.i]
    denoised_r = _recover_row_denoised(ctx, carrier)

    si_c = _snr_clamp(ctx.sig_row)
    sip1_c = _snr_clamp(ctx.sig_row_next)
    lam_s = _snr_lambda(si_c)
    lam_t = _snr_lambda(sip1_c)
    h = lam_t - lam_s
    h_eta = h * (eta + 1.0)
    alpha_t = sip1_c * lam_t.exp()
    neg_em1 = torch.expm1(-h_eta).neg()  # (-h_eta).expm1().neg() = 1 − exp(−h_eta)

    x_new = (sip1_c / si_c) * torch.exp(-h * eta) * ctx.x_prev + alpha_t * neg_em1 * denoised_r

    old = ctx.state.get("dpmpp_2m_sde_old")
    if old is not None:
        old_denoised_r, h_last = old
        r = h_last / h
        # midpoint solver term (comfy default)
        x_new = x_new + 0.5 * alpha_t * neg_em1 * (1.0 / r) * (denoised_r - old_denoised_r)

    if eta > 0:
        s_noise = _sde_s_noise(ctx)
        ns = _sde_noise_sampler(ctx, extra_args)
        noise = ns(carrier, ctx.sigmas[ctx.i + 1])
        renoise = torch.clamp(torch.expm1(-2.0 * h * eta).neg(), min=0.0).sqrt()
        x_new = x_new + noise * ctx.sig_row_next * renoise * s_noise

    x_new = torch.where(ctx.sig_row_next <= eps, denoised_r, x_new)
    ctx.state["dpmpp_2m_sde_old"] = (denoised_r, h)
    return torch.where(ctx.sig_row <= eps, ctx.x_prev, x_new)


def _dpmpp_3m_sde_step(ctx: _StepContext) -> torch.Tensor:
    """Native per-row DPM-Solver++(3M) SDE step.

    Ports comfy's ``sample_dpmpp_3m_sde`` (sampling.py:881-941): same leading term + stochastic
    renoise as :func:`_dpmpp_2m_sde_step`, but a 2-deep history ``(denoised_1, denoised_2, h_1,
    h_2)`` drives the 3M combiner (with the 2M term as the one-history fallback and no correction on
    the first step).  One model eval per interval; per-row ``x0`` via the shared spine.  Guarantees
    mirror :func:`_dpmpp_2m_sde_step`.
    """
    extra_args = {} if ctx.extra_args is None else ctx.extra_args
    eps = 1e-8
    eta = ctx.kwargs.get("eta", 1.0)
    carrier = ctx.sigmas[ctx.i]
    denoised_r = _recover_row_denoised(ctx, carrier)

    si_c = _snr_clamp(ctx.sig_row)
    sip1_c = _snr_clamp(ctx.sig_row_next)
    lam_s = _snr_lambda(si_c)
    lam_t = _snr_lambda(sip1_c)
    h = lam_t - lam_s
    h_eta = h * (eta + 1.0)
    alpha_t = sip1_c * lam_t.exp()
    em1 = torch.expm1(-h_eta)  # h_eta.neg().expm1()

    x_new = (sip1_c / si_c) * torch.exp(-h * eta) * ctx.x_prev + alpha_t * em1.neg() * denoised_r

    d1, d2, h_1, h_2 = ctx.state.get("dpmpp_3m_sde_old", (None, None, None, None))
    if h_2 is not None:
        r0 = h_1 / h
        r1 = h_2 / h
        d1_0 = (denoised_r - d1) / r0
        d1_1 = (d1 - d2) / r1
        d1c = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
        d2c = (d1_0 - d1_1) / (r0 + r1)
        phi_2 = em1 / h_eta + 1.0
        phi_3 = phi_2 / h_eta - 0.5
        x_new = x_new + (alpha_t * phi_2) * d1c - (alpha_t * phi_3) * d2c
    elif h_1 is not None:
        r = h_1 / h
        d = (denoised_r - d1) / r
        phi_2 = em1 / h_eta + 1.0
        x_new = x_new + (alpha_t * phi_2) * d

    if eta > 0:
        s_noise = _sde_s_noise(ctx)
        ns = _sde_noise_sampler(ctx, extra_args)
        noise = ns(carrier, ctx.sigmas[ctx.i + 1])
        renoise = torch.clamp(torch.expm1(-2.0 * h * eta).neg(), min=0.0).sqrt()
        x_new = x_new + noise * ctx.sig_row_next * renoise * s_noise

    x_new = torch.where(ctx.sig_row_next <= eps, denoised_r, x_new)
    ctx.state["dpmpp_3m_sde_old"] = (denoised_r, d1, h, h_1)
    return torch.where(ctx.sig_row <= eps, ctx.x_prev, x_new)


def _dpmpp_sde_step(ctx: _StepContext) -> torch.Tensor:
    """Native per-row DPM-Solver++ SDE step (stochastic, TWO model evals, no history).

    Ports comfy's ``sample_dpmpp_sde`` (sampling.py:737-792).  Eval 1 recovers per-row ``x0`` at the
    global carrier σ (fires the callback).  A per-row midpoint ``σ_row_mid = sigmoid(−(λ_s + r·h))``
    is formed; eval 2 runs at the GLOBAL midpoint carrier after :func:`publish_labels` republishes
    ``w_mid = σ_row_mid/σ_mid_g`` so the model returns each row's midpoint velocity, recovered via
    :func:`_recover_at`.  Ancestral steps run in ``exp(−λ)`` space
    (:func:`_sde_get_ancestral_step`).

    m=1 rows reproduce stock (``σ_row_mid == σ_mid_g`` → ``w_mid = 1``); terminal rows take the
    denoising branch; m=0 rows freeze under the outer guard.

    Fractional-row caveat: eval 2's clean-K/V side stream is primed at step ``i``'s ``sig_row`` (not
    the midpoint) — the task-#73 GPU risk.  CPU/non-observer and audio paths publish ``w_mid``
    correctly; the fractional VIDEO midpoint refinement is deferred to GPU validation.
    """
    extra_args = {} if ctx.extra_args is None else ctx.extra_args
    eps = 1e-8
    eta = ctx.kwargs.get("eta", 1.0)
    r = ctx.kwargs.get("r", 0.5)
    fac = 1.0 / (2.0 * r)
    carrier = ctx.sigmas[ctx.i]

    denoised_r = _recover_row_denoised(ctx, carrier)  # eval 1 (fires callback)

    si_c = _snr_clamp(ctx.sig_row)
    sip1_c = _snr_clamp(ctx.sig_row_next)
    lam_s = _snr_lambda(si_c)
    lam_t = _snr_lambda(sip1_c)
    h = lam_t - lam_s
    lam_s_1 = lam_s + r * h
    sigma_s_1 = _snr_sigma(lam_s_1)  # per-row midpoint sigma
    alpha_s = si_c * lam_s.exp()
    alpha_s_1 = sigma_s_1 * lam_s_1.exp()
    alpha_t = sip1_c * lam_t.exp()

    # Global midpoint carrier (scalar) for the second forward + BrownianTree query interval.
    sg_c = _snr_clamp(ctx.sig_g)
    sgn_c = _snr_clamp(ctx.sig_g_next)
    lam_sg = _snr_lambda(sg_c)
    lam_tg = _snr_lambda(sgn_c)
    sigma_s_1_g = _snr_sigma(lam_sg + r * (lam_tg - lam_sg))

    s_noise = _sde_s_noise(ctx)
    do_noise = eta > 0 and s_noise > 0
    ns = _sde_noise_sampler(ctx, extra_args) if do_noise else None

    # Step 1: ancestral sub-step to the midpoint (exp(−λ) space).
    sd1, su1 = _sde_get_ancestral_step(lam_s.neg().exp(), lam_s_1.neg().exp(), eta)
    h1_ = sd1.log().neg() - lam_s
    x_2 = (alpha_s_1 / alpha_s) * torch.exp(-h1_) * ctx.x_prev
    x_2 = x_2 - alpha_s_1 * torch.expm1(-h1_) * denoised_r
    if do_noise:
        x_2 = x_2 + alpha_s_1 * ns(carrier, sigma_s_1_g) * s_noise * su1

    # Eval 2 at the global midpoint: republish per-row midpoint labels, recover per-row x0.
    w_mid = (sigma_s_1 / sigma_s_1_g).clamp(max=1.0)
    ctx.state["publish_labels"](w_mid)
    denoised_2_r = _recover_at(ctx, sigma_s_1_g, sigma_s_1, x_2, fire_callback=False)

    # Step 2: ancestral sub-step to the interval end.
    sd2, su2 = _sde_get_ancestral_step(lam_s.neg().exp(), lam_t.neg().exp(), eta)
    h2_ = sd2.log().neg() - lam_s
    denoised_d = (1.0 - fac) * denoised_r + fac * denoised_2_r
    x_new = (alpha_t / alpha_s) * torch.exp(-h2_) * ctx.x_prev
    x_new = x_new - alpha_t * torch.expm1(-h2_) * denoised_d
    if do_noise:
        x_new = x_new + alpha_t * ns(carrier, ctx.sigmas[ctx.i + 1]) * s_noise * su2

    x_new = torch.where(ctx.sig_row_next <= eps, denoised_r, x_new)
    return torch.where(ctx.sig_row <= eps, ctx.x_prev, x_new)


def _dpmpp_2s_ancestral_step(ctx: _StepContext) -> torch.Tensor:
    """Native per-row DPM-Solver++(2S) ancestral step (stochastic, TWO model evals, no history).

    Ports comfy's ``sample_dpmpp_2s_ancestral_RF`` (sampling.py:686-734): PR2's exact RF-ancestral
    renoise algebra (``downstep_ratio`` → ``sigma_down``, ``alpha_ip1/alpha_down``,
    ``renoise_coeff``, ``default_noise_sampler``) wrapped around a second-order midpoint refine
    (the DPM++(2S) inner solve), shaped like :func:`_dpmpp_sde_step`'s 2-eval structure with a
    :func:`publish_labels` refresh — but the midpoint is deterministic (no ancestral noise on the
    inner ``u``) and the noise draw is comfy's plain per-interval Gaussian, not the SDE Brownian
    tree.

    Eval 1 recovers per-row ``x0`` at the global carrier σ (fires the callback).  The per-row
    midpoint ``σ_s = sigmoid(−(λ_i + r·(λ_down − λ_i)))`` sits in half-log-SNR space between the
    row's σ and its ancestral-reduced ``sigma_down`` (r=½).  Eval 2 runs at the GLOBAL midpoint
    carrier after :func:`publish_labels` republishes ``w_mid = σ_s/σ_s_g`` so the model returns each
    row's midpoint velocity, recovered via :func:`_recover_at`.  The final ancestral renoise is
    identical to :func:`_euler_ancestral_rf_step`.

    m=1 rows reproduce stock (``σ_s == σ_s_g`` → ``w_mid = 1``, identical noise draw); terminal rows
    (``sig_row_next == 0``) take the denoising branch (matching stock's Euler terminal → denoised);
    m=0 rows freeze under the outer guard.  Fractional-row caveat matches :func:`_dpmpp_sde_step`:
    eval 2's clean-K/V side stream is primed at step ``i``'s ``sig_row`` (the task-#73 GPU risk).
    """
    extra_args = {} if ctx.extra_args is None else ctx.extra_args
    eps = 1e-8
    eta = ctx.kwargs.get("eta", 1.0)
    r = 0.5
    carrier = ctx.sigmas[ctx.i]

    denoised_r = _recover_row_denoised(ctx, carrier)  # eval 1 (fires callback)

    # Per-row ancestral geometry (elementwise, same algebra as _euler_ancestral_rf_step).
    si_div = ctx.sig_row.clamp(min=eps)
    sip1 = ctx.sig_row_next
    sigma_down = sip1 * (1.0 + (sip1 / si_div - 1.0) * eta)
    alpha_ip1 = 1.0 - sip1
    alpha_down = 1.0 - sigma_down
    renoise_coeff = torch.sqrt(
        torch.clamp(sip1**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2, min=0.0)
    )

    # Per-row midpoint σ_s in half-log-SNR space, from the row's σ to its sigma_down.
    lam_i = _snr_lambda(_snr_clamp(ctx.sig_row))
    lam_down = _snr_lambda(_snr_clamp(sigma_down))
    sigma_s = _snr_sigma(lam_i + r * (lam_down - lam_i))

    # Global midpoint carrier (scalar) for eval 2, formed the same way from the global schedule.
    sg_down = ctx.sig_g_next * (1.0 + (ctx.sig_g_next / _snr_clamp(ctx.sig_g) - 1.0) * eta)
    lam_ig = _snr_lambda(_snr_clamp(ctx.sig_g))
    lam_downg = _snr_lambda(_snr_clamp(sg_down))
    sigma_s_g = _snr_sigma(lam_ig + r * (lam_downg - lam_ig))

    # Inner DPM++(2S) solve: deterministic midpoint point u, then eval 2 at the global midpoint.
    u_ratio = sigma_s / si_div
    u = u_ratio * ctx.x_prev + (1.0 - u_ratio) * denoised_r
    w_mid = (sigma_s / sigma_s_g).clamp(max=1.0)
    ctx.state["publish_labels"](w_mid)
    d_i_r = _recover_at(ctx, sigma_s_g, sigma_s, u, fire_callback=False)

    down_ratio = sigma_down / si_div
    x_new = down_ratio * ctx.x_prev + (1.0 - down_ratio) * d_i_r

    # Final ancestral renoise (comfy's plain per-interval Gaussian, PR2 algebra).
    if eta > 0:
        s_noise = _sde_s_noise(ctx)
        noise_sampler = _default_row_noise_sampler(ctx, extra_args)
        noise = noise_sampler(carrier, ctx.sigmas[ctx.i + 1])
        x_new = (alpha_ip1 / alpha_down) * x_new + noise * s_noise * renoise_coeff

    x_new = torch.where(ctx.sig_row_next <= eps, denoised_r, x_new)
    return torch.where(ctx.sig_row <= eps, ctx.x_prev, x_new)


#: Registry mapping ``base_fn.__name__`` to a native per-row step function.  Samplers not in
#: this dict fall back to ``_fallback_step`` (original behavior, no change).
_NATIVE_ROW_STEPS: dict[str, StepFn] = {
    "sample_euler": _euler_step,
    "sample_euler_ancestral": _euler_ancestral_rf_step,
    "sample_dpmpp_2m": _dpmpp_2m_step,
    "sample_res_multistep": _res_multistep_step,
    "sample_dpmpp_sde": _dpmpp_sde_step,
    "sample_dpmpp_2m_sde": _dpmpp_2m_sde_step,
    "sample_dpmpp_3m_sde": _dpmpp_3m_sde_step,
    "sample_dpmpp_2s_ancestral": _dpmpp_2s_ancestral_step,
}


def build_per_row_sampler_function(
    base_fn: Callable[..., Any],
    m_packed: torch.Tensor,
    clean_packed: torch.Tensor,
    schedule_tail: dict[str, Any],
    *,
    video_element_count: int | None = None,
    shift_v: float = 12.0,
    shift_a: float = 3.0,
) -> Callable[..., Any]:
    """Wrap a k-diffusion ``sampler_function`` to run the per-row schedule-tail remap.

    The returned callable matches ComfyUI's ``sampler_function`` contract
    ``(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs)``.  It runs
    the ``rescheduled`` remap described in the module docstring: an init-only clean composite
    at step 0, per-step per-row label ``w`` (stashed for the conditioning wrapper), a single
    global-schedule step of ``base_fn`` per interval, then per-row ``r``-scaling onto each
    row's own compressed tail.  ``m == 0`` rows are restored from ``clean`` at the end.

    Audio rows (packed elements at/after ``video_element_count``) run the remap on the
    sigma-shifted audio schedule (:func:`time_shift_sigma` with ``shift_v``/``shift_a``); only
    the sigma VALUES fed to ``row_sigma``/``sig_g``/``w``/``r`` differ.  ``base_fn`` still steps
    the whole packed latent on the raw video schedule.

    Parameters
    ----------
    base_fn:
        The underlying k-diffusion sampler step function (e.g. ``sample_res_multistep``).
    m_packed:
        Per-row denoise fractions in the sampler's packed latent layout, broadcastable to x.
    clean_packed:
        The clean reference latent in the packed layout, broadcastable to x.
    schedule_tail:
        Config dict.  Read keys: ``"make_pooled"`` (``c_vec -> pooled conds``, called per step
        to publish ``pooled_current``), ``"sigmas_dense"`` (the ``steps²+1`` dense grid).  Sets
        ``"pooled_current"`` each step.
    video_element_count:
        Packed video-prefix length; elements at/after it are audio and run the shifted
        schedule.  ``None`` (no audio) disables the audio path entirely.
    shift_v, shift_a:
        Video/audio sigma shifts for the audio schedule warp.

    Returns
    -------
    Callable
        A drop-in ``sampler_function``.
    """

    def sampler_function(
        model: Any,
        x: torch.Tensor,
        sigmas: Any,
        extra_args: dict[str, Any] | None = None,
        callback: Any = None,
        disable: Any = None,
        **kwargs: Any,
    ) -> Any:
        n_sig = int(sigmas.shape[-1])
        steps_n = max(1, n_sig - 1)
        clean = clean_packed.to(device=x.device, dtype=x.dtype)
        m_dev = m_packed.to(device=x.device, dtype=x.dtype)
        make_pooled = schedule_tail.get("make_pooled")

        # k_d / span depend only on m → identical for video and audio rows.
        k_d = torch.round(steps_n * (1.0 - m_dev)).clamp(0, steps_n)
        never = k_d >= steps_n  # d == 0 → exact preserve

        # Audio-modality mask over packed elements: same boundary scale_packed_audio uses.
        audio_mask = None
        if video_element_count is not None and 0 <= video_element_count < m_dev.shape[-1]:
            flat = torch.zeros(m_dev.shape[-1], dtype=torch.bool, device=x.device)
            flat[video_element_count:] = True
            audio_mask = flat.view(*([1] * (m_dev.dim() - 1)), -1)

        # Video schedule + (if audio) its sigma-shifted counterpart, coarse and dense.
        sig_v = sigmas.to(device=x.device, dtype=x.dtype)
        sig_a = _shift_schedule(sig_v, shift_v, shift_a) if audio_mask is not None else None
        dense = schedule_tail.get("sigmas_dense")
        has_dense = dense is not None and int(dense.shape[-1]) == steps_n * steps_n + 1
        if has_dense:
            dense_v = dense.to(device=x.device, dtype=x.dtype)
            dense_a = _shift_schedule(dense_v, shift_v, shift_a) if audio_mask is not None else None

        def row_sigma(i: int) -> torch.Tensor:
            """Per-row target sigma on the video σ_v axis for ALL rows (axis-blind).

            Audio no longer integrates on its own σ_a axis in our per-row engine: audio
            conditioning is full-generation (``m == 1``) so this returns ``sig_v[i]`` for every
            audio element, matching the official axis-blind ``sample_euler_ancestral_RF`` (the σ_a
            shift is applied inside the H3 forward).  The audio fade is delegated to the official
            KSamplerX0Inpaint composite via ``noise_mask``; see ``audio-native-composite.md``.
            """
            return _stream_row_sigma(
                m_dev, i, steps_n, dense_v if has_dense else None, sig_v, n_sig
            )

        def global_sigma(i: int) -> torch.Tensor:
            """Per-row global sigma at step ``i`` — σ_v for all rows (audio axis-blind)."""
            return sig_v[i]

        def _cb(offset: int) -> Any:  # remap per-step callback index to the global step
            if callback is None:
                return None

            def inner(d: dict[str, Any]) -> Any:
                d = dict(d)
                d["i"] = offset + int(d.get("i", 0))
                return callback(d)

            return inner

        frac_mask = (m_dev > 0.0) & (m_dev < 1.0)
        if bool(frac_mask.any()):
            ks = k_d[frac_mask]
            print(
                f"[H3_INJECT] schedule-tail remap: {steps_n} steps; fractional rows release at "
                f"step [{int(ks.min())}..{int(ks.max())}] "
                f"(d in [{float(m_dev[frac_mask].min()):.3f},"
                f"{float(m_dev[frac_mask].max()):.3f}])",
                flush=True,
            )

        # Resolve the per-row step function once before the loop (keyed by base sampler name).
        step = _NATIVE_ROW_STEPS.get(getattr(base_fn, "__name__", ""), _fallback_step)

        # Shared mutable state for native steps that need persistence across iterations
        # (e.g. the seeded noise sampler in _euler_ancestral_rf_step, old_denoised for PR3).
        # Must be created once here so each step call shares the same dict object.
        step_state: dict[str, Any] = {}

        # Clean-K/V root fix (Option II, single-forward): when the observer-split patches are
        # installed and fractional rows exist, each euler step runs ONE forward that carries an
        # exact band-only side stream (:func:`_single_forward_denoised`), reproducing the
        # two-forward clean-K/V denoise bit-for-bit.  The two-forward code was removed on branch
        # single-forward-clean-kv-splice; the removed mechanism is preserved in wiki
        # c2-rho-fix-paths/observed-level-plant/clean-kv-split.md.  observer is None (no fractional
        # rows / install skipped) → single-forward euler, unchanged.
        step_state["observer"] = schedule_tail.get("observer")
        step_state["clean"] = clean
        step_state["m_dev"] = m_dev
        step_state["frac_mask"] = frac_mask
        step_state["make_pooled"] = make_pooled
        step_state["schedule_tail"] = schedule_tail
        step_state["pdev"] = m_packed.device
        step_state["pdtype"] = m_packed.dtype

        def publish_labels(w: torch.Tensor) -> None:
            """Republish pooled per-row labels mid-step (dpmpp_sde's 2nd, midpoint eval).

            Wraps the loop's ``make_pooled`` → ``schedule_tail["pooled_current"]`` publish so a
            multi-eval step can relabel the model at ``w_mid`` between its two forwards.  A no-op
            when ``make_pooled`` is absent (CPU tests, models that ignore conditioning).
            """
            if make_pooled is not None:  # pragma: no cover - GPU (real conditioning)
                schedule_tail["pooled_current"] = make_pooled(
                    w.to(device=m_packed.device, dtype=m_packed.dtype)
                )

        step_state["publish_labels"] = publish_labels

        def prime_side_stream(i: int) -> None:  # pragma: no cover - GPU
            """Per-step token-ordered side-stream setup: embed ``ratio``, ``t_emb_m``, mod indices.

            Computed from the observer's OWN token-ordered ``m`` (``stream["m"]``) via the shared
            :func:`_stream_row_sigma` so ``σ_row`` matches the sampler loop exactly with no
            packed/token layout assumption.  Video runs on the raw schedule, audio on the shifted
            one; both index a single shared ``t_emb_m`` table (tag disambiguates the modality).
            """
            obs = step_state["observer"]
            if obs is None:
                return
            dm = obs.get("dm")
            modalities = [
                (
                    "video",
                    0,
                    sig_v[i],
                    dense_v if has_dense else None,
                    sig_v,
                    max(1.0 - float(sig_v[i]), 0.999),
                ),
            ]
            if audio_mask is not None and sig_a is not None:
                modalities.append(
                    ("audio", 2, sig_a[i], dense_a if has_dense else None, sig_a, 1.0)
                )
            active: list[dict[str, Any]] = []
            obs_parts: list[torch.Tensor] = []
            for key, tag, glob_i, dgrid, cgrid, pin in modalities:
                stream = obs.get(key)
                if stream is None:
                    continue
                mrow = stream["m"].to(device=x.device, dtype=x.dtype)
                sig_row_band = _stream_row_sigma(mrow, i, steps_n, dgrid, cgrid, n_sig)
                stream["ratio"] = _embed_ratio(mrow * glob_i, sig_row_band)
                obslev = _observer_timestep(mrow, glob_i, pin)
                stream["_obs"] = obslev
                stream["_tag"] = tag
                active.append(stream)
                obs_parts.append(obslev)
            if not active:
                return
            levels = torch.cat(obs_parts).unique()
            obs["t_emb_m"] = _observer_time_embed(dm, levels, x.dtype, x.device)
            for stream in active:
                stream["mod_index"] = _band_mod_index(levels, stream["_obs"], stream["_tag"])

        step_state["prime_side_stream"] = prime_side_stream

        # One-time embed capture: snapshot the clean inject's block-0 band hidden (h_clean).  One
        # extra forward on the STATIC clean latent, amortized over all steps.
        if step_state["observer"] is not None:  # pragma: no cover - GPU
            obs0 = step_state["observer"]
            s_in0 = x.new_ones([x.shape[0]])
            if make_pooled is not None:
                schedule_tail["pooled_current"] = make_pooled(
                    m_dev.clamp(max=1.0).to(device=m_packed.device, dtype=m_packed.dtype)
                )
            obs0["mode"] = "embed_capture"
            model(clean, sigmas[0] * s_in0, **(extra_args or {}))
            obs0["mode"] = None

        x_cur = x
        sig_row = row_sigma(0)
        schedule_tail["total_steps"] = steps_n
        for i in range(steps_n):
            schedule_tail["current_step"] = i
            sig_g = global_sigma(i).clamp(min=1e-8)
            sig_g_next = global_sigma(i + 1)
            # Truthful label: model computes t_row = 1 − w·σ_g = 1 − σ_row (per modality).
            w = (sig_row / sig_g).clamp(max=1.0)
            if make_pooled is not None:
                schedule_tail["pooled_current"] = make_pooled(
                    w.to(device=m_packed.device, dtype=m_packed.dtype)
                )
            # Init-only clean composite: plant each row's ACTUAL latent noise at the truthful
            # self-attention level σ_row (w = sig_row/σ_g).  The single-forward side stream sources
            # the observed-level content for neighbours from block-0 embed capture, so x_prev
            # carries the σ_row content the main forward integrates.
            if i == 0:
                x_cur = w * x_cur + (1.0 - w) * clean
            x_prev = x_cur
            sig_row_next = row_sigma(i + 1)
            ctx = _StepContext(
                model=model,
                x_prev=x_prev,
                i=i,
                sigmas=sigmas,  # raw schedule arg — matches what the original passed to base_fn
                sig_row=sig_row,
                sig_row_next=sig_row_next,
                sig_g=sig_g,
                sig_g_next=sig_g_next,
                extra_args=extra_args,
                callback=_cb(i),
                disable=disable,
                kwargs=kwargs,
                base_fn=base_fn,
                state=step_state,  # shared across iterations; native steps persist state here
            )
            # Per-row step scaling: each row integrates its OWN Δσ_row, not the global Δσ.
            x_cur = step(ctx)
            sig_row = sig_row_next
        return torch.where(never, clean, x_cur)  # belt-and-braces d == 0 exact preserve

    return sampler_function


# ---------------------------------------------------------------------------
# Audio sigma shift — lives here (not grid.py) because it is a scheduling
# formula used at sample time, not a grid geometry helper.
# ---------------------------------------------------------------------------


def time_shift_sigma(sigma: float, from_shift: float = 12.0, to_shift: float = 3.0) -> float:
    """Return the shifted audio sigma for a given video sigma.

    Mirrors ``time_shift_sigma`` from ``comfy/ldm/minimax/model.py``.  Audio rows run their
    remap against this shifted sigma, not the raw video sigma, to keep audio and video fades
    temporally aligned.

    Parameters
    ----------
    sigma:
        Current video sigma value (scalar, in [0, 1] space).
    from_shift:
        Video sigma shift (``sigma_shift_video``).  Defaults to 12.0, the H3 DiT
        constructor default.  In production, pass the runtime value from the diffusion
        model's ``sigma_shift_video`` so audio timing stays aligned when the user changes
        the video shift via the ``MiniMax H3 Sigma Shift`` node.
    to_shift:
        Audio sigma shift (``sigma_shift_audio``).  Defaults to 3.0, the H3 DiT
        constructor default.  Pass the runtime ``sigma_shift_audio`` when available.

    Returns
    -------
    float
        Shifted sigma value appropriate for the audio stream.
    """
    # Two-step warp from comfy/ldm/minimax/model.py ~36-38: invert the video shift to
    # recover the base grid, then re-apply the audio shift.
    # Returns the raw warp value; ComfyUI applies `1.0 - warp` at the model
    # boundary as the audio conditioning timestep.  Contract: f(0)=0, f(1)=1.
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return float(to_shift * base / (1.0 + (to_shift - 1.0) * base))
