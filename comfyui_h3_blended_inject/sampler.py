"""Per-row img2img sampler helpers.

This module owns the three levers that turn ComfyUI's global sampler into a per-row
img2img sampler for H3 blended injects (see the ``per-row-img2img-architecture`` design):

1. **Per-row initial noise** (:func:`per_row_init_lerp`): after ComfyUI's global
   ``noise_scaling`` produces ``x_global = sigma_max*eps + (1-sigma_max)*clean``, the
   img2img start for row ``r`` with denoise ``m_r`` is exactly
   ``x_r = m_r*x_global + (1-m_r)*clean`` — a per-row lerp between the fully-noised latent
   and the clean reference.  ``m=1`` → full generation, ``m=0`` → preserve, ``0<m<1`` →
   img2img start.  Applied at the top of a thin wrapped ``sampler_function``
   (:func:`build_per_row_sampler_function`) because the sampler steps its own outer ``x``
   — a model_function_wrapper cannot set the initial ``x``.

2. **Per-row DiT conditioning** — handled by the conditioning wrapper (built elsewhere)
   that feeds the fractional ``denoise_mask`` so the DiT compresses each row's schedule.

3. **No native noise_mask** — the caller passes ``noise_mask=None`` so ``KSamplerX0Inpaint``
   never composites, removing the compounding re-pin ghost.

For deterministic samplers (euler, res_multistep, dpmpp_2m, ...) the per-step update is
invariant under scaling all sigmas by ``m_r``, so running the global sampler on the global
schedule with per-row-conditioned ``x0`` and per-row initial noise reproduces per-row
img2img exactly — no sampler modification needed.  Stochastic samplers (ancestral/SDE,
euler s_churn>0) add fresh noise per step; :func:`_make_per_row_noise_sampler` scales that
noise per-row, but this is INSUFFICIENT for H3's RF-ancestral path (Bug B): the
``sample_euler_ancestral_RF`` renoise sub-step is affine (not linear) in sigma via its
``alpha = 1 - sigma`` terms, so no noise-magnitude shim alone can make it scale-invariant.
Stochastic samplers are unsupported/deferred; see ``bugs.md`` Bug B and
``stochastic-recovery-theory.md``.

Everything here is pure and CPU-testable; ``torch`` is the only heavy dependency.

This module is imported lazily inside ``nodes._run_sampler`` (a ``# pragma: no cover`` GPU
path), so it has no module-level importers at load time — that is intentional, not an orphan.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import torch


def per_row_init_lerp(
    x: torch.Tensor,
    m: torch.Tensor,
    clean: torch.Tensor,
) -> torch.Tensor:
    """Blend the fully-noised latent ``x`` toward the clean reference per row.

    Computes ``m * x + (1 - m) * clean`` with ``m`` and ``clean`` cast to ``x``'s device
    and dtype so mixed-precision inputs do not raise.  ``m`` is the per-row denoise fraction
    broadcast over the non-row dimensions (e.g. shape ``[1, rows, 1]`` against ``x`` of
    shape ``[1, rows, cols]``); ``m == 0`` yields ``clean``, ``m == 1`` yields ``x``.

    Parameters
    ----------
    x:
        The globally noise-scaled latent (``sigma_max*eps + (1-sigma_max)*clean``).
    m:
        Per-row denoise fractions, broadcastable to ``x``.
    clean:
        The clean reference latent (target with all inject content composited in),
        broadcastable to ``x``.

    Returns
    -------
    torch.Tensor
        The per-row img2img starting latent, same shape/dtype/device as ``x``.
    """
    m = m.to(device=x.device, dtype=x.dtype)
    clean = clean.to(device=x.device, dtype=x.dtype)
    return m * x + (1.0 - m) * clean


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
    The per-row init lerp (:func:`per_row_init_lerp`) blends toward the clean reference in that
    same space, so the lerp's clean term must carry the identical audio scale — otherwise
    fractional-denoise (``0 < m < 1``) audio rows img2img *from* a mismatched reference and
    decode as static.  ``m == 1`` rows drop the clean term and ``m == 0`` rows are restored
    post-sampling, so only fractional audio is affected (matching the observed fade-region
    garble under deterministic samplers).

    ``packed`` is a flat ``[B, video_elems + audio_elems]`` latent; ``video_element_count`` is
    ``prod(video_shape[1:])`` — the packed video-prefix length.  A no-op when
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
    timestep labels live on a 1/256 grid.  Levers 1 and 3 (init lerp, denoised
    correction) must use the *identical* per-row ``m`` or the lever-3 identity
    ``corrected = x - m*sigma*v`` is off by up to 1/256 per row.  Quantizing ``m``
    up front makes all three levers consistent: the native quantization becomes a
    no-op on already-quantized values.  ``0`` and ``1`` are fixed points.
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
    acceptable for a warning heuristic; do not rely on this for a hard gate.
    Stochastic samplers are unsupported by the per-row compression (Bug B): the RF
    renoise sub-step is affine in sigma and not scale-invariant under ``sigma -> m*sigma``.
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


# ---------------------------------------------------------------------------
# Deferred stochastic shim (Bug B)
# ---------------------------------------------------------------------------


def sampler_accepts_noise_sampler(fn: Callable[..., Any]) -> bool:
    """Return ``True`` iff ``fn`` declares an explicit ``noise_sampler`` parameter.

    A bare ``**kwargs`` does *not* count: only samplers that name ``noise_sampler`` actually
    consume an injected noise source (stochastic ancestral/SDE samplers).  Deterministic
    samplers omit it, so the shim is skipped for them.  When this returns ``True``,
    ``build_per_row_sampler_function`` installs ``_make_per_row_noise_sampler`` — a shim
    that is currently deferred (Bug B: insufficient for H3's RF-ancestral path).
    """
    try:
        return "noise_sampler" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _make_per_row_noise_sampler(
    base_noise_sampler: Callable[[Any, Any], torch.Tensor],
    m: torch.Tensor,
) -> Callable[[Any, Any], torch.Tensor]:
    """Wrap a noise_sampler so its output is scaled per-row by ``m``.

    Stochastic samplers add ``noise_sampler(sigma, sigma_next) * s_noise * sigma_up`` each
    step, where ``sigma_up`` is a bare sigma.  Row ``r`` running at compressed schedule
    ``m_r*sigma`` should receive ``m_r*sigma_up`` of noise, so this wrapper pre-scales the
    sampled noise by ``m_r``.

    CAVEAT (Bug B): this correctness argument only holds for Karras-style ancestral
    (``get_ancestral_step`` is homogeneous degree 1).  H3's CONST model routes to
    ``sample_euler_ancestral_RF``, which does NOT use ``get_ancestral_step`` — its renoise
    coefficients are affine in sigma (``alpha = 1 - sigma``), so this magnitude shim is
    insufficient there and stochastic samplers remain unsupported.  Kept as a dead shim;
    see ``stochastic-recovery-theory.md`` for the proposed real fix.

    Parameters
    ----------
    base_noise_sampler:
        The underlying noise source, ``(sigma, sigma_next) -> tensor``.
    m:
        Per-row denoise fractions, broadcastable to the noise tensor.

    Returns
    -------
    Callable
        A ``(sigma, sigma_next) -> tensor`` noise sampler with per-row-scaled output.
    """

    def noise_sampler(sigma: Any, sigma_next: Any) -> torch.Tensor:
        noise = base_noise_sampler(sigma, sigma_next)
        return noise * m.to(device=noise.device, dtype=noise.dtype)

    return noise_sampler


def _default_noise_sampler_factory(
    x: torch.Tensor, seed: int | None = None
) -> Callable[[Any, Any], torch.Tensor]:
    """Lazily build ComfyUI's ``default_noise_sampler(x, seed=seed)``.

    Imported lazily so this module stays importable without ``comfy`` present (tests inject
    their own factory).  Falls back to the no-seed call when the API does not declare a
    ``seed`` parameter (older ComfyUI builds).
    """
    from comfy.k_diffusion import sampling as k_sampling  # noqa: PLC0415

    try:
        sig = inspect.signature(k_sampling.default_noise_sampler)
    except (TypeError, ValueError):
        # Uninspectable (C extension or similar); skip the seed-aware path.
        return k_sampling.default_noise_sampler(x)
    if "seed" in sig.parameters:
        return k_sampling.default_noise_sampler(x, seed=seed)
    return k_sampling.default_noise_sampler(x)


def build_conditioning_wrapper(
    pooled_conds: dict[str, torch.Tensor],
    m_packed: torch.Tensor | None = None,
) -> Callable[[Callable[..., Any], dict[str, Any]], Any]:
    """Return a ``model_function_wrapper`` that injects per-row conditioning and corrects
    the denoised prediction so each row integrates at its compressed rate.

    Two jobs, both required for per-row img2img:

    **1. Inject pooled per-row conditioning.**  Each key/tensor in ``pooled_conds`` is placed
    directly into a copy of ``c`` (device/dtype-aligned to the current ``input``) so it flows
    through ``apply_model(**c)`` → DiT ``extra_conds`` params.  ``c`` is copied, never mutated.

    **2. Denoised correction.**  H3's ``process_timestep`` compresses the *embedding*
    (``v_timestep = m*t``) but ``calculate_denoised`` still uses the outer sigma, so the
    sampler would integrate every row over the full global interval.  Blending
    ``corrected = m*denoised + (1-m)*x`` makes the effective velocity ``m*v`` and confines
    each row's integral to its compressed ``m*sigma`` interval.  ``m==1`` → raw denoised;
    ``m==0`` → frozen at the (clean) init.  The correction commutes with CFG.

    Matches ComfyUI's ``model_function_wrapper`` contract:
    ``(apply_model, args_dict) -> prediction``.

    Parameters
    ----------
    pooled_conds:
        DiT conditioning key → pooled mask tensor from ``MiniMaxH3._denoise_mask_values``.
        May be empty (pure denoised-correction pass-through).
    m_packed:
        Per-row denoise fractions broadcastable to the model input/output.  ``None`` skips
        the denoised correction (used by conditioning-injection unit tests).

    Returns
    -------
    Callable
        A ``model_function_wrapper`` for ``model_options["model_function_wrapper"]``.
    """

    def wrapper(apply_model: Callable[..., Any], args_dict: dict[str, Any]) -> Any:
        inp = args_dict["input"]
        c = dict(args_dict["c"])
        for key, value in pooled_conds.items():
            c[key] = value.to(device=inp.device, dtype=inp.dtype)
        denoised = apply_model(inp, args_dict["timestep"], **c)
        if m_packed is None:
            return denoised
        m = m_packed.to(device=denoised.device, dtype=denoised.dtype)
        return m * denoised + (1.0 - m) * inp

    return wrapper


def build_per_row_sampler_function(
    base_fn: Callable[..., Any],
    m_packed: torch.Tensor,
    clean_packed: torch.Tensor,
    *,
    scale_stochastic_noise: bool = False,
    noise_sampler_factory: Callable[..., Callable[[Any, Any], torch.Tensor]] | None = None,
) -> Callable[..., Any]:
    """Wrap a k-diffusion ``sampler_function`` to run per-row img2img.

    The returned callable matches ComfyUI's ``sampler_function`` contract
    ``(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs)``.  On
    entry it lerps ``x`` toward the clean reference per row (:func:`per_row_init_lerp`) so
    each row starts from its img2img noise level, then delegates to ``base_fn`` unchanged.

    When ``scale_stochastic_noise`` is set (used for stochastic samplers, detected via
    :func:`sampler_accepts_noise_sampler`) and the caller did not already supply a
    ``noise_sampler``, a per-row-scaling noise_sampler is built from the lerp'd ``x`` (so it
    has the right shape/device) and injected into ``kwargs``.  Deterministic samplers leave
    ``scale_stochastic_noise`` off and need no noise source at all.

    Parameters
    ----------
    base_fn:
        The underlying k-diffusion sampler step function (e.g. ``sample_res_multistep``).
    m_packed:
        Per-row denoise fractions in the sampler's packed latent layout, broadcastable to x.
    clean_packed:
        The clean reference latent in the packed layout, broadcastable to x.
    scale_stochastic_noise:
        Whether to inject a per-row-scaling noise_sampler for stochastic samplers.
    noise_sampler_factory:
        ``(x, seed=...) -> noise_sampler`` used to build the base noise source; defaults to
        ComfyUI's ``default_noise_sampler``.  Injected by tests.

    Returns
    -------
    Callable
        A drop-in ``sampler_function``.
    """
    factory = noise_sampler_factory or _default_noise_sampler_factory

    def sampler_function(
        model: Any,
        x: torch.Tensor,
        sigmas: Any,
        extra_args: dict[str, Any] | None = None,
        callback: Any = None,
        disable: Any = None,
        **kwargs: Any,
    ) -> Any:
        x0 = per_row_init_lerp(x, m_packed, clean_packed)
        if scale_stochastic_noise and kwargs.get("noise_sampler") is None:
            seed = (extra_args or {}).get("seed")
            base_ns = factory(x0, seed=seed)
            kwargs["noise_sampler"] = _make_per_row_noise_sampler(base_ns, m_packed)
        return base_fn(
            model,
            x0,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            **kwargs,
        )

    return sampler_function


# ---------------------------------------------------------------------------
# Audio sigma shift — lives here (not grid.py) because it is a scheduling
# formula used at sample time, not a grid geometry helper.
# ---------------------------------------------------------------------------


def time_shift_sigma(sigma: float, from_shift: float = 12.0, to_shift: float = 3.0) -> float:
    """Return the shifted audio sigma for a given video sigma.

    Mirrors ``time_shift_sigma`` from ``comfy/ldm/minimax/model.py``.  Audio rows release
    against this shifted sigma, not the raw video sigma, to keep audio and video fades
    temporally aligned.

    Parameters
    ----------
    sigma:
        Current video sigma value (scalar, in [0, 1] space).
    from_shift:
        Video sigma shift (``sigma_shift_video``).  Defaults to 12.0, the H3 DiT
        constructor default.  In production, pass the runtime value from
        ``transformer_options["minimax_h3_sigma_shift_video"]`` so audio timing stays
        aligned when the user changes the video shift via the ``MiniMax H3 Sigma Shift``
        node.
    to_shift:
        Audio sigma shift (``sigma_shift_audio``).  Defaults to 3.0, the H3 DiT
        constructor default.  Pass the runtime value from
        ``transformer_options["minimax_h3_sigma_shift_audio"]`` when available.

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
