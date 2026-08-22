"""ComfyUI-H3-Blended-Inject.

ComfyUI entry point. ComfyUI imports this module from ``custom_nodes/`` and reads
``NODE_CLASS_MAPPINGS`` / ``NODE_DISPLAY_NAME_MAPPINGS`` to register nodes.

Node implementations live in the ``comfyui_h3_blended_inject`` package. They are
kept separate from this entry point so the pure-logic modules (envelope,
schedule, sanitization, derived mask) can be unit-tested without a running
ComfyUI. Both mappings are re-exported directly from
``comfyui_h3_blended_inject.nodes``.

ComfyUI loads this file by path from ``custom_nodes/`` without adding the pack
directory to ``sys.path``, so the sibling ``comfyui_h3_blended_inject`` package
is not importable by default. Prepend this directory to ``sys.path`` so the
package (and its internal absolute imports) resolves as a top-level module,
matching how the test suite imports it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from comfyui_h3_blended_inject.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
