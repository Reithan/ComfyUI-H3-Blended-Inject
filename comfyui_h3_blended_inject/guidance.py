"""Guidance resolution helper for H3 optional-negative CFG/NRS logic.

H3 is CFG-distilled (positive-only by default).  This module contains the
single pure function :func:`resolve_guidance` that decides how to forward the
optional negative conditioning, cfg scale, and model_options to ComfyUI's
sampler — without importing torch or comfy.

Background (from ``comfy/samplers.py``):

- ``sampling_function``: if ``math.isclose(cond_scale, 1.0)`` and
  ``model_options.get("disable_cfg1_optimization", False) == False``, then
  ``uncond_ = None`` — i.e. the uncond pass is dropped at cfg==1.0 UNLESS
  ``disable_cfg1_optimization`` is True.
- ``cfg_function``: if ``"sampler_cfg_function" in model_options``,
  that custom hook is called regardless of cfg or whether an uncond pass ran
  (with ``uncond = x - uncond_pred``; if no uncond pass ran uncond_pred=0 so it
  gets ``x``, i.e. garbage).  Standard blend otherwise is
  ``uncond_pred + (cond_pred - uncond_pred) * cond_scale``, which at
  no-negative (uncond_pred=0) gives ``cond_pred * cond_scale`` — a silent gain
  if cfg != 1.0.
- NRS and other guidance nodes attach via
  ``model.set_model_sampler_cfg_function(...)`` stored as
  ``model_options["sampler_cfg_function"]``.  This key is treated agnostically
  as "a custom guidance hook is present" — not assumed to be NRS specifically.
"""

from __future__ import annotations

import math
import warnings
from typing import Any


def resolve_guidance(
    negative: Any,
    cfg: float,
    model_options: dict[str, Any],
) -> tuple[Any, float, dict[str, Any]]:
    """Return (effective_negative, effective_cfg, model_options) per the H3 guidance rule.

    Three branches, in priority order:

    1. **negative is not None** (wired, even a zeroed ConditioningZeroOut):
       Returns the negative unchanged, cfg unchanged, and a *new* model_options
       dict with ``disable_cfg1_optimization=True`` so the uncond pass runs
       even at cfg==1.0 (required for cfg-independent hooks like NRS).
       The input ``model_options`` dict is never mutated.

    2. **negative is None AND ``"sampler_cfg_function" in model_options``**
       (custom guidance patched but no negative supplied — user error):
       Emits a UserWarning and returns ``([], cfg, model_options)`` unchanged.
       The hook is NOT stripped; cfg is NOT overridden — the hook runs as-is
       but without a real uncond pass (user's accepted risk).

    3. **negative is None AND no hook** (official CFG-distilled cond-only path):
       Forces ``effective_cfg=1.0`` to avoid the silent ``cond*cfg`` gain.
       Emits a UserWarning if the input cfg was not already 1.0.
       Returns ``([], 1.0, model_options)`` with model_options unchanged.

    Parameters
    ----------
    negative:
        Negative conditioning list, or ``None`` if not connected.
    cfg:
        Classifier-free guidance scale as entered by the user.
    model_options:
        ``dict[str, Any]`` from the cloned model patcher (``m.model_options``).
        This dict is never mutated; when a change is needed a new dict is returned.

    Returns
    -------
    tuple[Any, float, dict]
        ``(effective_negative, effective_cfg, effective_model_options)``
    """
    if negative is not None:
        # Branch 1: wired negative — enable uncond pass even at cfg==1.0 so
        # cfg-independent hooks (NRS, etc.) receive a real uncond prediction.
        new_options = {**model_options, "disable_cfg1_optimization": True}
        return negative, cfg, new_options

    if "sampler_cfg_function" in model_options:
        # Branch 2: custom guidance hook present but no negative — user error.
        warnings.warn(
            "A custom sampler_cfg_function is patched on the model but no negative "
            "conditioning is connected.  The guidance hook will run without an "
            "unconditional pass and may behave incorrectly (uncond_pred=0 means the "
            "hook sees x instead of a real uncond prediction).  Connect a negative "
            "conditioning (e.g. Conditioning Zero Out) to suppress this warning and "
            "provide a valid uncond pass.",
            UserWarning,
            stacklevel=2,
        )
        return [], cfg, model_options

    # Branch 3: no negative, no hook — official H3 CFG-distilled cond-only path.
    if not math.isclose(cfg, 1.0):
        warnings.warn(
            f"No negative conditioning is connected (cfg={cfg!r}).  The node runs "
            "positive-only (the official H3 CFG-distilled mode) and the cfg value is "
            "ignored — effective_cfg is forced to 1.0 to avoid a silent cond*cfg gain.  "
            "Connect a negative conditioning to enable CFG or NRS-style guidance.",
            UserWarning,
            stacklevel=2,
        )
    return [], 1.0, model_options
