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

**Audio.**  The base sampler steps the whole packed (video+audio) latent on the VIDEO sigma
schedule uniformly.  Audio rows, however, live on the sigma-shifted audio schedule
(:func:`time_shift_sigma`), so ``row_sigma``, ``sig_g``, ``w`` and ``r`` use audio-shifted
sigmas for audio rows and raw video sigmas for video rows.  ``k_d`` / ``span`` depend only on
``m`` and are identical for both modalities.  Audio rows are located by the packed video
prefix length (the same boundary :func:`scale_packed_audio` uses).

The sampler loop dispatches each step through a ``_NATIVE_ROW_STEPS`` registry (keyed by
``base_fn.__name__``); ``_fallback_step`` is the default for unknown samplers.  ``sample_euler``
and ``sample_euler_ancestral`` (H3's RF-ancestral path, kills Bug B) are registered here (PR1 +
PR2); multistep (PR3) steps plug in via the same ``_StepContext`` protocol.

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
        # Observer-label K/V split: refresh the per-call observer labels (t_obs = 1 − m·σ)
        # consumed by the DiT block patches. See observer_split.py.
        obs = schedule_tail.get("observer")
        if obs is not None:  # pragma: no cover - armed only on GPU runs
            from comfyui_h3_blended_inject.observer_split import observer_call_update

            observer_call_update(obs, float(args_dict["timestep"].flatten()[0]))
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
    sig_row_v: torch.Tensor  # per-row sigma on the σ_v axis for ALL rows (audio rows too);
    # used by the ancestral integration so audio integrates on the σ_v trajectory,
    # matching stock sample_euler_ancestral_RF.
    sig_row_v_next: torch.Tensor  # per-row sigma on the σ_v axis at step i+1
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


def _euler_ancestral_rf_step(ctx: _StepContext) -> torch.Tensor:
    """Native per-row RF-ancestral step (kills Bug B for euler_ancestral on H3).

    Implements one step of ``sample_euler_ancestral_RF`` elementwise over the per-row sigma
    tensors from the schedule-tail remap.  No r-scaling — the per-row sigma integration is
    handled directly by the ancestral math.

    One model eval per interval at the global carrier sigma.  Per-row ``x0`` is recovered by
    projecting the velocity onto the σ_v-axis per-row sigma:
    ``denoised_r = x_prev − σ_row_v·(x_prev − denoised)/σ_carrier``
    (the conditioning wrapper labels the model with ``t_row = 1 − σ_row`` so it returns the
    per-row denoised; the velocity ``v = (x − denoised) / σ_carrier`` maps back to per-row
    ``x0`` via ``sig_row_v``).  All ancestral integration terms (``sigma_down``,
    ``alpha_ip1/alpha_down``, ``renoise_coeff``, ``ratio``) are computed elementwise over the
    ``sig_row_v`` / ``sig_row_v_next`` tensors so that AUDIO rows integrate on the σ_v
    trajectory (matching stock ``sample_euler_ancestral_RF``), while the per-row model LABEL
    ``w = sig_row/sig_g`` remains on σ_a as required.

    Guarantees:

    - **m=1 rows reproduce stock** ``sample_euler_ancestral_RF`` **exactly**: for m=1,
      ``sig_row_v == sigmas[i] == carrier``, so ``denoised_r == denoised`` and all per-row
      terms match the scalar stock values — given an identical noise draw the outputs are
      bit-for-bit equal.
    - **Terminal step** (``sig_row_v_next == 0``) falls out without a branch: ``sigma_down=0``,
      ``ratio=0`` → ``x = denoised_r``; ``renoise_coeff=0`` → no noise added.
    - **m=0 rows** freeze: ``sig_row_v=0`` clamped to ``eps`` → ``ratio→0``, ``coeff→0`` →
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
    if "noise_sampler" not in ctx.state:
        ns = ctx.kwargs.get("noise_sampler")
        if ns is None:
            from comfy.k_diffusion.sampling import default_noise_sampler  # pragma: no cover

            ns = default_noise_sampler(  # pragma: no cover
                ctx.x_prev, seed=extra_args.get("seed")
            )
        ctx.state["noise_sampler"] = ns
    noise_sampler = ctx.state["noise_sampler"]

    carrier = ctx.sigmas[ctx.i]
    s_in = ctx.x_prev.new_ones([ctx.x_prev.shape[0]])
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

    # Recover per-row x0 from global-carrier velocity (project onto σ_v axis for all rows).
    v = (ctx.x_prev - denoised) / carrier
    denoised_r = ctx.x_prev - ctx.sig_row_v * v

    # Elementwise ancestral update over σ_v-axis per-row sigma tensors.
    # Audio rows integrate on σ_v (matching stock sample_euler_ancestral_RF); the per-row
    # model LABEL (w = sig_row/sig_g) remains on σ_a and is computed separately in the loop.
    eps = 1e-8
    si = ctx.sig_row_v.clamp(min=eps)
    sip1 = ctx.sig_row_v_next
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


# ---------------------------------------------------------------------------
# DPM++ / multistep shared spine (PR3 helpers, PR4 SDE composition entry points)
#
# Shared spine for deterministic DPM++ steps (PR3) and the SDE family (PR4):
# recover → time_coeffs → second_order.  PR4 will add SDE renoise and
# mid-evaluation steps that compose with these three helpers directly.
# ---------------------------------------------------------------------------


def _recover_row_denoised(ctx: _StepContext) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the model and recover the per-row x0 estimate.

    Runs a single model evaluation with the global carrier sigma, fires ``ctx.callback``
    with the stock k-diffusion dict, then recovers ``denoised_r`` by projecting the
    velocity onto each row's σ_v-axis sigma ``sig_row_v``.

    The projection axis is ``sig_row_v`` (NOT the σ_a-shifted ``sig_row``): the packed
    latent — video AND audio rows — lives on the σ_v carrier trajectory, so x0 recovery
    must use σ_v.  This mirrors ``_euler_ancestral_rf_step`` after the audio-axis fix
    (Fix A, ``audio-axis-verdict.md``).  For video rows ``sig_row_v == sig_row``; only
    audio rows differ.  The σ_a LABEL (``w = sig_row/sig_g``) is applied in the outer
    loop and is unaffected.

    Returns
    -------
    tuple[denoised, denoised_r]
        ``denoised`` — raw model output (global carrier scale).
        ``denoised_r`` — per-row x0 estimate at ``σ_row_v``.
    """
    extra_args = {} if ctx.extra_args is None else ctx.extra_args
    s_in = ctx.x_prev.new_ones([ctx.x_prev.shape[0]])
    carrier = ctx.sigmas[ctx.i]
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
    v = (ctx.x_prev - denoised) / carrier
    denoised_r = ctx.x_prev - ctx.sig_row_v * v
    return denoised, denoised_r


def _dpmpp_time_coeffs(
    sig_a: torch.Tensor,
    sig_b: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DPM++ log-time coefficients for a (σ_a → σ_b) step.

    Returns
    -------
    tuple[t_a, t_b, h, sigma_ratio]
        ``t_a = −log σ_a``, ``t_b = −log σ_b``, ``h = t_b − t_a``,
        ``sigma_ratio = exp(−h)`` (the damping factor; equals ``σ_b / σ_a`` for exact logs).
    """
    t_a = -sig_a.clamp(min=eps).log()
    t_b = -sig_b.clamp(min=eps).log()
    h = t_b - t_a
    sigma_ratio = (-h).exp()
    return t_a, t_b, h, sigma_ratio


def _dpmpp_2m_second_order(
    denoised_r: torch.Tensor,
    old_denoised_r: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    """Blend current and previous x0 estimates for the DPM++ 2M second-order update.

    Returns the multistep predictor ``D = (1 + 1/(2r))·d_r − (1/(2r))·d_r_prev``.
    PR4 (dpmpp_sde, 2M SDE, 3M SDE) reuses this as the deterministic mid-step predictor.
    """
    return (1.0 + 1.0 / (2.0 * r)) * denoised_r - (1.0 / (2.0 * r)) * old_denoised_r


def _dpmpp_2m_step(ctx: _StepContext) -> torch.Tensor:
    """Native per-row DPM-Solver++(2M) step — restores true 2nd-order under the remap.

    Implements one step of ``sample_dpmpp_2m`` elementwise over the per-row sigma tensors.
    No noise term (deterministic, ``eta=0`` equivalent).

    Per-row x0 is recovered as
    ``denoised_r = x_prev − σ_row_v·(x_prev − denoised)/σ_carrier`` (same σ_v-axis
    identity as the RF-ancestral step after Fix A).  All DPM++ 2M update math then runs
    on the σ_v-axis per-row sigmas ``σ_row_v`` / ``σ_row_v_next`` directly, not on the
    global carrier schedule and not on the σ_a-shifted ``σ_row``.  Audio rows follow the
    σ_v trajectory the packed latent really lives on; the σ_a LABEL is separate.

    Cross-step history (``old_denoised_r``, ``h_prev_row``) lives in ``ctx.state`` so the
    outer loop shares one persistent dict across all iterations.  On the first step
    ``old_denoised_r`` is ``None`` and the first-order DDIM branch is taken, exactly as
    ``sample_dpmpp_2m`` does.

    Guarantees:

    - **m=1 rows reproduce stock** ``sample_dpmpp_2m`` **within atol=1e-5**: for a m=1
      VIDEO row ``σ_row_v = σ_carrier`` and ``denoised_r = denoised``, so every t / h / r
      matches the scalar stock values.  (Audio m=1 rows run on their own σ_v trajectory,
      which is the correct target — not the global carrier.)  The terminal step deviates
      by ~1e-8 because this implementation eps-clamps ``t_next`` where stock uses exact
      ``σ_fn(+∞) = 0``; the equivalence test uses ``atol=1e-5``.
    - **Terminal rows** (``σ_row_v_next == 0``) fall back to first-order elementwise via
      ``torch.where``, mirroring stock's ``if sigmas[i+1] == 0`` branch.
    - **m=0 rows** produce safe output (h≈0, ``x ≈ x_prev``); the outer
      ``where(never, clean, x)`` restores clean.

    GPU-only paths (``model.inner_model`` attr) are ``# pragma: no cover``.
    """
    _, denoised_r = _recover_row_denoised(ctx)

    # t = −log σ;  σ_fn(t) = exp(−t) = σ.  σ_v-axis per-row sigmas (audio on σ_v).
    t_i, _, h, sigma_ratio = _dpmpp_time_coeffs(ctx.sig_row_v, ctx.sig_row_v_next)

    old_denoised_r = ctx.state.get("old_denoised_r")
    h_prev: torch.Tensor | None = ctx.state.get("h_prev_row")

    if old_denoised_r is None:
        # Step 0: first-order DDIM (stock's `old_denoised is None` branch).
        x = sigma_ratio * ctx.x_prev - (-h).expm1() * denoised_r
    else:
        # Steps 1+: second-order; fall back to first-order for terminal rows.
        eps = 1e-8
        r = h_prev / h.clamp(min=eps)
        denoised_d = _dpmpp_2m_second_order(denoised_r, old_denoised_r, r)
        x_2nd = sigma_ratio * ctx.x_prev - (-h).expm1() * denoised_d
        x_1st = sigma_ratio * ctx.x_prev - (-h).expm1() * denoised_r
        # Elementwise: terminal rows (σ_row_v_next == 0) use first-order, mirroring stock.
        x = torch.where(ctx.sig_row_v_next == 0.0, x_1st, x_2nd)

    ctx.state["old_denoised_r"] = denoised_r
    ctx.state["h_prev_row"] = h
    return x


def _res_multistep_step(ctx: _StepContext) -> torch.Tensor:
    """Native per-row Restart-Multistep step — deterministic (``eta=0``, no noise).

    Implements one step of ``sample_res_multistep`` (which calls ``res_multistep`` with
    ``eta=0, cfg_pp=False``) elementwise over the σ_v-axis per-row sigma tensors.  With
    ``eta=0``: ``sigma_down = σ_row_v_next``, ``sigma_up = 0`` — no noise is ever added.

    Per-row x0 recovered identically to ``_dpmpp_2m_step`` (σ_v axis).  The RES 2nd-order
    update formula (``phi1``, ``phi2``, ``b1``, ``b2``) runs on ``σ_row_v`` /
    ``σ_row_v_next`` with ``torch.nan_to_num`` guards for near-zero denominators, matching
    stock.  Audio rows integrate on the σ_v trajectory (per Fix A); the σ_a label is
    separate.

    Cross-step history: ``old_denoised_r`` and ``prev_sig_row`` (= σ_row_v from the
    previous step, needed to form ``t_prev``).  On the first step ``old_denoised_r`` is
    ``None``; first-order Euler is used, exactly as stock.

    With ``eta=0``, ``old_sigma_down`` in stock always equals the current ``σ_row_v``
    (previous step's ``sigma_down = σ_prev_next = σ_cur``), so ``t_old = t`` and
    ``c2 = (t_prev − t) / h`` — all terms resolve from ``prev_sig_row``.

    Guarantees:

    - **m=1 VIDEO rows reproduce stock** ``sample_res_multistep`` **bit-for-bit**: all
      per-row quantities collapse to the stock scalar values.  (Audio m=1 rows run on
      their own σ_v trajectory, the correct target.)
    - **Terminal rows** (``σ_row_v_next == 0``) use first-order elementwise.
    - **m=0 rows** produce safe output; ``where(never, clean, x)`` restores clean.

    GPU-only paths (``model.inner_model`` attr) are ``# pragma: no cover``.
    """
    _, denoised_r = _recover_row_denoised(ctx)

    # With eta=0: sigma_down = σ_row_v_next, sigma_up = 0 (no noise).  σ_v-axis integration.
    sigma_down = ctx.sig_row_v_next
    eps = 1e-8

    old_denoised_r = ctx.state.get("old_denoised_r")
    prev_sig_row: torch.Tensor | None = ctx.state.get("prev_sig_row")

    if old_denoised_r is None:
        # Step 0: first-order Euler (stock's `old_denoised is None` branch).
        d = (ctx.x_prev - denoised_r) / ctx.sig_row_v.clamp(min=eps)
        x = ctx.x_prev + d * (sigma_down - ctx.sig_row_v)
    else:
        # Steps 1+: 2nd-order RES; fall back to 1st-order for terminal rows.
        # t = −log σ;  σ_fn(t) = exp(−t);  φ1(t) = expm1(t)/t;  φ2(t) = (φ1(t)−1)/t.
        t_i, _, h, sigma_ratio = _dpmpp_time_coeffs(ctx.sig_row_v, sigma_down, eps)
        # With eta=0: old_sigma_down = prev sigma_down = prev sig_row_v_next = cur sig_row_v.
        # So t_old = t_i; c2 = (t_prev − t_old) / h = (t_prev − t_i) / h.
        t_prev = -prev_sig_row.clamp(min=eps).log()
        c2 = torch.nan_to_num((t_prev - t_i) / h, nan=0.0)

        neg_h = -h
        phi1_val = torch.nan_to_num(neg_h.expm1() / neg_h, nan=1.0)
        phi2_val = torch.nan_to_num((phi1_val - 1.0) / neg_h, nan=0.5)
        b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
        b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)

        x_2nd = sigma_ratio * ctx.x_prev + h * (b1 * denoised_r + b2 * old_denoised_r)
        # First-order fallback for terminal rows.
        d = (ctx.x_prev - denoised_r) / ctx.sig_row_v.clamp(min=eps)
        x_1st = ctx.x_prev + d * (sigma_down - ctx.sig_row_v)
        x = torch.where(sigma_down == 0.0, x_1st, x_2nd)

    ctx.state["old_denoised_r"] = denoised_r
    ctx.state["prev_sig_row"] = ctx.sig_row_v
    return x


#: Registry mapping ``base_fn.__name__`` to a native per-row step function.  Samplers not in
#: this dict fall back to ``_fallback_step`` (original behavior, no change).
_NATIVE_ROW_STEPS: dict[str, StepFn] = {
    "sample_euler": _euler_step,
    "sample_euler_ancestral": _euler_ancestral_rf_step,
    "sample_dpmpp_2m": _dpmpp_2m_step,
    "sample_res_multistep": _res_multistep_step,
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
        span = (steps_n - k_d) / steps_n

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

        def _stream_row_sigma(i: int, dense_grid: torch.Tensor | None, coarse: torch.Tensor):
            """Per-row sigma for ONE modality's schedule at global step ``i``."""
            if has_dense:
                # Exact stretched-tail sigma: schedule position (k_d·steps + i·(steps−k_d))/steps²
                # is grid point k_d·(steps−i) + i·steps of the SAME scheduler run at steps² steps.
                idx = (k_d * (steps_n - i) + i * steps_n).round().long()
                return dense_grid[idx.clamp(0, steps_n * steps_n)]
            # Fallback (dense grid absent): lerp the coarse grid at k_d + i·span.
            idx = (k_d + i * span).clamp(0.0, float(steps_n))
            lo = idx.floor().long().clamp(0, n_sig - 1)
            hi = (lo + 1).clamp(0, n_sig - 1)
            fr = (idx - lo.to(idx.dtype)).clamp(0.0, 1.0)
            return coarse[lo] * (1.0 - fr) + coarse[hi] * fr

        def row_sigma(i: int) -> torch.Tensor:
            """Per-row target sigma, audio rows on the shifted schedule."""
            sv = _stream_row_sigma(i, dense_v if has_dense else None, sig_v)
            if audio_mask is None:
                return sv
            sa = _stream_row_sigma(i, dense_a if has_dense else None, sig_a)
            return torch.where(audio_mask, sa, sv)

        def row_sigma_v(i: int) -> torch.Tensor:
            """Per-row sigma on the σ_v axis for ALL rows (audio rows included).

            Returns the same value as row_sigma for video rows; for audio rows it returns
            the σ_v-axis sigma rather than the σ_a-shifted value.  Used by the ancestral
            integration so that audio rows integrate on the σ_v trajectory, matching
            stock sample_euler_ancestral_RF.  When audio_mask is None this is identical
            to row_sigma (no audio present).
            """
            return _stream_row_sigma(i, dense_v if has_dense else None, sig_v)

        def global_sigma(i: int) -> torch.Tensor:
            """Per-row global sigma at step ``i``, audio rows on the shifted schedule."""
            if audio_mask is None:
                return sig_v[i]
            return torch.where(audio_mask, sig_a[i], sig_v[i])

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

        x_cur = x
        sig_row = row_sigma(0)
        sig_row_v = row_sigma_v(0)
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
            # Init-only clean composite: place each row on its noise-line at σ_row(0), then
            # never re-inject (per-region SDEdit on the stretched tail).
            if i == 0:
                x_cur = w * x_cur + (1.0 - w) * clean
            x_prev = x_cur
            sig_row_next = row_sigma(i + 1)
            sig_row_v_next = row_sigma_v(i + 1)
            ctx = _StepContext(
                model=model,
                x_prev=x_prev,
                i=i,
                sigmas=sigmas,  # raw schedule arg — matches what the original passed to base_fn
                sig_row=sig_row,
                sig_row_next=sig_row_next,
                sig_row_v=sig_row_v,
                sig_row_v_next=sig_row_v_next,
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
            sig_row_v = sig_row_v_next
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
