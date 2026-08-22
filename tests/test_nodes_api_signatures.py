"""Regression tests for ComfyUI API call signatures in nodes.py.

These tests parse the source statically (no torch/comfy import required) and
assert that call sites match the real ComfyUI API.  They catch kwargs that
ComfyUI never accepted and would raise TypeError at runtime.

Real signatures (from ComfyUI source / documented):
  - comfy.sample.prepare_noise(latent_image, seed, noise_inds=None)
      * NO noise_mask parameter — noise_mask is a separate parameter of
        comfy.sample.sample(), not prepare_noise().
"""

from __future__ import annotations

import ast
from pathlib import Path

NODES_PY = Path(__file__).resolve().parents[1] / "comfyui_h3_blended_inject" / "nodes.py"


def _collect_prepare_noise_calls(tree: ast.AST) -> list[ast.Call]:
    """Return every Call node whose function resolves to prepare_noise."""
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # comfy.sample.prepare_noise(...)  → Attribute
        if isinstance(func, ast.Attribute) and func.attr == "prepare_noise":
            calls.append(node)
        # bare prepare_noise(...)  → Name
        elif isinstance(func, ast.Name) and func.id == "prepare_noise":
            calls.append(node)
    return calls


class TestPrepareNoiseSignature:
    """Regression guard: prepare_noise must not receive a noise_mask keyword."""

    def test_no_noise_mask_kwarg_on_prepare_noise(self):
        source = NODES_PY.read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = _collect_prepare_noise_calls(tree)

        assert calls, "No prepare_noise call sites found — test may be stale"

        bad_calls: list[int] = []
        for call in calls:
            for kw in call.keywords:
                if kw.arg == "noise_mask":
                    bad_calls.append(call.lineno)

        assert not bad_calls, (
            f"prepare_noise() called with noise_mask= kwarg at line(s) {bad_calls} "
            f"in {NODES_PY}. "
            "The real ComfyUI signature is prepare_noise(latent_image, seed, noise_inds=None) "
            "— noise_mask is not a valid parameter."
        )
