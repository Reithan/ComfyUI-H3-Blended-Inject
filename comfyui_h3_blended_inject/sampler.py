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
img2img exactly — no sampler modification needed.  Only stochastic samplers
(ancestral/SDE, euler s_churn>0) add a bare ``sigma_up`` of fresh noise per step; those
need a per-row-scaling noise_sampler (:func:`make_per_row_noise_sampler`) so row ``r`` gets
``m_r*sigma_up`` of noise.  See the ``per-row-sampler-scale-invariance`` note for the proof.

Everything here is pure and CPU-testable; ``torch`` is the only heavy dependency.
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


def sampler_accepts_noise_sampler(fn: Callable[..., Any]) -> bool:
    """Return ``True`` iff ``fn`` declares an explicit ``noise_sampler`` parameter.

    A bare ``**kwargs`` does *not* count: only samplers that name ``noise_sampler`` actually
    consume an injected noise source (stochastic ancestral/SDE samplers).  Deterministic
    samplers omit it, so we skip the shim for them.
    """
    try:
        return "noise_sampler" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def make_per_row_noise_sampler(
    base_noise_sampler: Callable[[Any, Any], torch.Tensor],
    m: torch.Tensor,
) -> Callable[[Any, Any], torch.Tensor]:
    """Wrap a noise_sampler so its output is scaled per-row by ``m``.

    Stochastic samplers add ``noise_sampler(sigma, sigma_next) * s_noise * sigma_up`` each
    step, where ``sigma_up`` is a bare sigma.  Row ``r`` running at compressed schedule
    ``m_r*sigma`` should receive ``m_r*sigma_up`` of noise, so pre-scaling the sampled noise
    by ``m_r`` reproduces the correct per-row stochastic term.  ``get_ancestral_step`` is
    homogeneous degree 1, so ``sigma_up`` already scales correctly with the deterministic
    part; only the fresh-noise magnitude needs this correction.

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
    their own factory).  Falls back to a plain ``randn_like`` source if the seed-aware API
    is unavailable.
    """
    from comfy.k_diffusion import sampling as k_sampling  # noqa: PLC0415

    try:
        return k_sampling.default_noise_sampler(x, seed=seed)
    except TypeError:
        return k_sampling.default_noise_sampler(x)


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
            kwargs["noise_sampler"] = make_per_row_noise_sampler(base_ns, m_packed)
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
