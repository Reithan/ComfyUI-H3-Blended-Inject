"""Tests for comfyui_h3_blended_inject.guidance.resolve_guidance.

All four branches of the optional-negative CFG/NRS rule are covered:

  1. negative wired → (negative, cfg, model_options + disable_cfg1_optimization=True);
     original model_options dict is NOT mutated.
  2. negative None + sampler_cfg_function present → ([], cfg, model_options unchanged);
     UserWarning emitted; hook key still present.
  3. negative None + no hook + cfg != 1.0 → ([], 1.0, model_options unchanged);
     UserWarning emitted about ignored cfg.
  4. negative None + no hook + cfg == 1.0 → ([], 1.0, model_options unchanged);
     NO warning emitted.

No torch or comfy imports are used; this module is pure Python.
"""

from __future__ import annotations

import math
import warnings

import pytest

from comfyui_h3_blended_inject.guidance import resolve_guidance

# ---------------------------------------------------------------------------
# Branch 1: negative wired
# ---------------------------------------------------------------------------


class TestResolveGuidanceNegativeWired:
    """Branch 1: negative is not None."""

    def test_returns_same_negative_object(self):
        sentinel = object()
        eff_neg, _, _ = resolve_guidance(sentinel, 7.0, {})
        assert eff_neg is sentinel

    def test_returns_same_cfg(self):
        _, eff_cfg, _ = resolve_guidance([], 5.5, {})
        assert math.isclose(eff_cfg, 5.5)

    def test_disable_cfg1_optimization_set_true(self):
        _, _, opts = resolve_guidance([], 7.0, {})
        assert opts.get("disable_cfg1_optimization") is True

    def test_other_model_options_preserved(self):
        original = {"foo": "bar", "baz": 42}
        _, _, opts = resolve_guidance([], 7.0, original)
        assert opts["foo"] == "bar"
        assert opts["baz"] == 42

    def test_input_model_options_not_mutated(self):
        original = {"some_key": "some_value"}
        resolve_guidance([], 7.0, original)
        assert "disable_cfg1_optimization" not in original

    def test_returned_dict_is_new_object(self):
        original = {}
        _, _, opts = resolve_guidance([], 7.0, original)
        assert opts is not original

    def test_existing_disable_cfg1_optimization_is_overridden(self):
        """If model_options already had disable_cfg1_optimization=False, it is set True."""
        original = {"disable_cfg1_optimization": False}
        _, _, opts = resolve_guidance([], 7.0, original)
        assert opts["disable_cfg1_optimization"] is True

    def test_no_warning_emitted_when_negative_wired(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resolve_guidance([], 7.0, {})
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert not user_warnings

    def test_wired_empty_list_is_treated_as_wired(self):
        """An empty list is a valid wired negative (e.g. ConditioningZeroOut result)."""
        eff_neg, _, opts = resolve_guidance([], 1.0, {})
        assert eff_neg == []
        assert opts.get("disable_cfg1_optimization") is True

    def test_cfg_1_with_wired_negative_still_sets_disable_optimization(self):
        """At cfg==1.0, disable_cfg1_optimization must still be True when negative is wired."""
        _, eff_cfg, opts = resolve_guidance(["cond"], 1.0, {})
        assert math.isclose(eff_cfg, 1.0)
        assert opts.get("disable_cfg1_optimization") is True


# ---------------------------------------------------------------------------
# Branch 2: negative None + sampler_cfg_function present
# ---------------------------------------------------------------------------


class TestResolveGuidanceNoneNegativeWithHook:
    """Branch 2: negative is None and sampler_cfg_function is in model_options."""

    def _opts_with_hook(self):
        return {"sampler_cfg_function": lambda *a, **kw: None, "other": "preserved"}

    def test_emits_user_warning(self):
        with pytest.warns(UserWarning, match="sampler_cfg_function"):
            resolve_guidance(None, 7.0, self._opts_with_hook())

    def test_warning_mentions_unconditional_pass(self):
        with pytest.warns(UserWarning, match="unconditional"):
            resolve_guidance(None, 7.0, self._opts_with_hook())

    def test_warning_suggests_connecting_negative(self):
        with pytest.warns(UserWarning, match="negative"):
            resolve_guidance(None, 7.0, self._opts_with_hook())

    def test_returns_empty_list_as_effective_negative(self):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            eff_neg, _, _ = resolve_guidance(None, 7.0, self._opts_with_hook())
        assert eff_neg == []

    def test_cfg_is_unchanged(self):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            _, eff_cfg, _ = resolve_guidance(None, 5.5, self._opts_with_hook())
        assert math.isclose(eff_cfg, 5.5)

    def test_model_options_returned_unchanged(self):
        opts = self._opts_with_hook()
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            _, _, returned_opts = resolve_guidance(None, 7.0, opts)
        assert returned_opts is opts

    def test_hook_key_still_present_in_returned_options(self):
        opts = self._opts_with_hook()
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            _, _, returned_opts = resolve_guidance(None, 7.0, opts)
        assert "sampler_cfg_function" in returned_opts

    def test_other_keys_still_present(self):
        opts = self._opts_with_hook()
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            _, _, returned_opts = resolve_guidance(None, 7.0, opts)
        assert returned_opts.get("other") == "preserved"

    def test_input_model_options_not_mutated(self):
        opts = self._opts_with_hook()
        original_len = len(opts)
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            resolve_guidance(None, 7.0, opts)
        assert len(opts) == original_len


# ---------------------------------------------------------------------------
# Branch 3: negative None + no hook + cfg != 1.0
# ---------------------------------------------------------------------------


class TestResolveGuidanceNoneNegativeNoHookCfgNotOne:
    """Branch 3: negative is None, no hook, cfg != 1.0."""

    def test_emits_user_warning(self):
        with pytest.warns(UserWarning):
            resolve_guidance(None, 7.0, {})

    def test_warning_mentions_no_negative(self):
        with pytest.warns(UserWarning, match="negative"):
            resolve_guidance(None, 7.0, {})

    def test_warning_mentions_cfg_value(self):
        with pytest.warns(UserWarning, match="7.0"):
            resolve_guidance(None, 7.0, {})

    def test_effective_cfg_forced_to_one(self):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            _, eff_cfg, _ = resolve_guidance(None, 7.0, {})
        assert math.isclose(eff_cfg, 1.0)

    def test_returns_empty_list_as_effective_negative(self):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            eff_neg, _, _ = resolve_guidance(None, 7.0, {})
        assert eff_neg == []

    def test_model_options_returned_unchanged(self):
        opts = {"some_key": 123}
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            _, _, returned_opts = resolve_guidance(None, 7.0, opts)
        assert returned_opts is opts

    def test_small_cfg_deviation_also_warns(self):
        """Even a small non-1.0 value triggers the warning."""
        with pytest.warns(UserWarning):
            resolve_guidance(None, 1.01, {})

    def test_cfg_forced_to_one_for_small_deviation(self):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            _, eff_cfg, _ = resolve_guidance(None, 1.01, {})
        assert math.isclose(eff_cfg, 1.0)


# ---------------------------------------------------------------------------
# Branch 4: negative None + no hook + cfg == 1.0
# ---------------------------------------------------------------------------


class TestResolveGuidanceNoneNegativeNoHookCfgOne:
    """Branch 4: negative is None, no hook, cfg == 1.0 — the silent happy path."""

    def test_no_warning_emitted(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resolve_guidance(None, 1.0, {})
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert not user_warnings, f"Unexpected warnings: {[str(x.message) for x in user_warnings]}"

    def test_effective_cfg_is_one(self):
        _, eff_cfg, _ = resolve_guidance(None, 1.0, {})
        assert math.isclose(eff_cfg, 1.0)

    def test_returns_empty_list_as_effective_negative(self):
        eff_neg, _, _ = resolve_guidance(None, 1.0, {})
        assert eff_neg == []

    def test_model_options_returned_unchanged(self):
        opts = {"key": "val"}
        _, _, returned_opts = resolve_guidance(None, 1.0, opts)
        assert returned_opts is opts

    def test_math_isclose_used_for_one_comparison(self):
        """Values within float tolerance of 1.0 must not warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resolve_guidance(None, 1.0 + 1e-12, {})
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert not user_warnings
