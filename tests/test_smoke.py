"""Smoke tests for the package scaffold.

These verify the repo imports and exposes the ComfyUI registration surface.
Replace/extend as real nodes and logic modules land (see plan Test plan).
"""

import comfyui_h3_blended_inject


def test_package_has_version():
    assert isinstance(comfyui_h3_blended_inject.__version__, str)


def test_entry_point_exposes_mappings():
    import importlib.util
    from pathlib import Path

    entry_path = Path(__file__).resolve().parents[1] / "__init__.py"
    spec = importlib.util.spec_from_file_location("h3_blended_inject_entry", entry_path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)

    assert isinstance(entry.NODE_CLASS_MAPPINGS, dict)
    assert isinstance(entry.NODE_DISPLAY_NAME_MAPPINGS, dict)
