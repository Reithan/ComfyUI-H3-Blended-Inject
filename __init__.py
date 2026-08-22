"""ComfyUI-H3-Blended-Inject.

ComfyUI entry point. ComfyUI imports this module from ``custom_nodes/`` and reads
``NODE_CLASS_MAPPINGS`` / ``NODE_DISPLAY_NAME_MAPPINGS`` to register nodes.

Node implementations live in the ``comfyui_h3_blended_inject`` package. They are
kept separate from this entry point so the pure-logic modules (envelope,
schedule, sanitization, derived mask) can be unit-tested without a running
ComfyUI. The mappings below are empty until the nodes land.
"""

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
