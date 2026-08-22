"""Core package for ComfyUI-H3-Blended-Inject.

Holds the node implementations and the pure-logic modules they build on
(envelope evaluation, schedule merge, sanitization, derived mask, and the
hold-and-release sampler wrapper). Modules that do not touch ``comfy`` or
``torch`` runtime state stay importable on their own so they can be tested
CPU-side without a running ComfyUI.
"""

__version__ = "0.0.1"
