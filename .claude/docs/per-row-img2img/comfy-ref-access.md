<!-- provenance: reference (operational meta — how to reach comfy source + read verification stamps) -->
<!-- verified: 2026-08-28 · sparse-checkout policy user-confirmed 2026-08-24 -->
# comfy-ref access & verification stamps (meta)

Operational notes split out of [PER_ROW_IMG2IMG_NOTES.md](../PER_ROW_IMG2IMG_NOTES.md) to keep the
index under budget. Read when you need to reach comfy source or interpret a doc's verified stamp.

## comfy-ref access

`/home/reithan/projects/comfy-ref` is a SPARSE checkout (files we reference, ON DISK). When a
needed file is missing, ADD it (user-confirmed policy 2026-08-24):
`cd /home/reithan/projects/comfy-ref && git sparse-checkout add /path/to/file` (leading slash
for a single file). Run `git sparse-checkout list` there to see what is currently checked out
(typically `comfy/{sample,samplers,utils,nested_tensor,model_base,model_sampling,latent_formats}.py`,
`comfy/k_diffusion/sampling.py`, `comfy/ldm/minimax/`,
`comfy_extras/{nodes_differential_diffusion,nodes_minimax_h3}.py`).

## Verification stamps

Line 2 of every wiki doc — `<!-- verified: <date> · <source> @<sha> -->`. Refresh date + SHA
whenever you re-verify. Stale SHA ⇒ treat line numbers as hints; navigate by symbol name instead.
