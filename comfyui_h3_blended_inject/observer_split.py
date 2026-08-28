"""Observer-label K/V split for fractional inject rows.

The H3 DiT applies each row's mask label exactly once per block — an adaLN
scale/shift of ``norm1(x)`` (``_mod_scale_shift``) — and that single modulated
tensor feeds the fused qkv projection.  A row's K/V (what other rows READ of
it) and its Q / residual path (what the row COMPUTES for itself) therefore
inherit the label at a single seam and are separable inside the block.

This module patches every DiT block (``patches_replace["dit"][("double_block",
i)]``) so that fractional inject rows present their K and V under the official
label ``t_obs = 1 − m·σ`` (a constant observed ratio m:1 across the whole run)
while Q, gates, and the MLP path keep the truthful remapped label
``t_row = 1 − σ_row``.  The row's own velocity prediction never sees the
observer label — only what neighbouring rows read of it changes — which is
what makes this split safe.

The splice is performed at the qkv seam because K/V and Q are separable there:
K and V are what other rows attend to, so relabeling them changes the row's
presented denoise state without perturbing its own Q addressing or residual.

GPU-only: everything below the per-call helpers requires a live ComfyUI + H3
model and is exercised only on GPU runs.  ``comfy`` imports stay lazy so the
per-call helpers remain importable and testable on CPU.
"""

from __future__ import annotations

from typing import Any

import torch

# Cond-timestep pins, mirrored from comfy/ldm/minimax/model.py:32-33 so the
# CPU-testable per-call helper needs no comfy import.
VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0


def observer_call_update(obs: dict[str, Any], sigma_v: float) -> None:
    """Refresh per-model-call observer labels; called by the conditioning wrapper.

    Computes ``t_obs = clamp(1 − m·σ, max=t_pin)`` per fractional row for the
    video stream (raw ``σ_v``) and audio stream (shifted ``σ_a``), mirroring the
    model's own label formula (model.py:596/605).  Stored into ``obs["call"]``
    for the block patches, with a fresh ``token`` so per-forward plan caches
    invalidate.
    """
    from comfyui_h3_blended_inject.sampler import time_shift_sigma

    call: dict[str, Any] = {"token": obs.get("_token", 0) + 1}
    obs["_token"] = call["token"]
    video = obs.get("video")
    if video is not None:
        t_pin_v = max(1.0 - sigma_v, VISUAL_COND_TIMESTEP)
        call["t_obs_v"] = (1.0 - video["m"] * sigma_v).clamp(max=t_pin_v)
    audio = obs.get("audio")
    if audio is not None:
        sigma_a = time_shift_sigma(sigma_v, obs.get("shift_v", 12.0), obs.get("shift_a", 3.0))
        t_pin_a = max(1.0 - sigma_a, AUDIO_COND_TIMESTEP)
        call["t_obs_a"] = (1.0 - audio["m"] * sigma_a).clamp(max=t_pin_a)
    obs["call"] = call


def _fractional_rows(rows: torch.Tensor) -> dict[str, Any] | None:
    """Index/value bundle for strictly fractional (0<m<1) rows; None if none.

    Uses the model's own 1e-3 full-denoise tolerance (model.py:83) on the high
    side; m==0 rows are excluded — they already get native pin treatment.
    """
    frac = (rows > 1e-6) & (rows < 1.0 - 1e-3)
    if not bool(frac.any()):
        return None
    return {
        "n": int(rows.numel()),
        "pos": frac.nonzero(as_tuple=False).reshape(-1),
        "m": rows[frac],
    }


def install_observer_split(  # pragma: no cover - requires live ComfyUI model (GPU)
    m: Any,
    schedule_tail_cfg: dict[str, Any],
    m_packed: torch.Tensor,
    latent_shapes: list[Any],
) -> bool:
    """Install the observer-label K/V split block patches on model clone ``m``.

    Computes the per-token-row official mask values exactly as the model will
    (same ``_denoise_mask_values`` cond tensor → same ``mask_row_values`` call
    with post-pad latent dims), stores the static observer state into
    ``schedule_tail_cfg["observer"]``, and registers a replace-patch for every
    DiT block.  Returns False (nothing installed) when no fractional rows exist.
    """
    import comfy.ldm.common_dit
    from comfy.ldm.minimax.model import mask_row_values

    dm = m.get_model_object("diffusion_model")
    pooled_obs = m.model._denoise_mask_values(m_packed, latent_shapes)
    state: dict[str, Any] = {
        "shift_v": float(getattr(dm, "sigma_shift_video", 12.0)),
        "shift_a": float(getattr(dm, "sigma_shift_audio", 3.0)),
    }

    vm = pooled_obs.get("denoise_mask")
    if vm is not None:
        # The model pads video_x to patch size BEFORE taking the row-grid dims
        # (model.py:556-563); replicate the padding on the mask tensor itself so
        # our row count/order match its ``mask_row_values`` output exactly.
        padded = comfy.ldm.common_dit.pad_to_patch_size(vm, dm.patch_size)
        lat_t, lat_h, lat_w = int(padded.shape[2]), int(padded.shape[3]), int(padded.shape[4])
        rows = mask_row_values(vm[0, 0].to(torch.float32), lat_t, lat_h, lat_w)
        if rows is not None:
            state["video"] = _fractional_rows(rows)
    am = pooled_obs.get("audio_denoise_mask")
    if am is not None:
        state["audio"] = _fractional_rows(am[0, 0].to(torch.float32).reshape(-1))

    if state.get("video") is None and state.get("audio") is None:
        return False

    schedule_tail_cfg["observer"] = state
    for i, block in enumerate(dm.blocks):
        m.set_model_patch_replace(_make_block_patch(dm, block, state), "dit", "double_block", i)
    return True


def _call_plan(  # pragma: no cover - GPU
    dm: Any, state: dict[str, Any], call: dict[str, Any], segs: list[Any], t_emb: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Per-forward splice plan: (global token positions, adaLN mod rows, t_emb_obs).

    Locates the per-row video/audio segments in ``mod_segments`` (the only
    entries with TENSOR mod rows; modality = mod_index % 3 — video tag 0, audio
    tag 2, model.py:615), maps our fractional-row indices to global token
    positions, and embeds the unique observer t values.  Cached in ``call`` —
    all blocks of one forward share segments and labels.
    """
    plan = call.get("plan", "unset")
    if plan != "unset":
        return plan

    import comfy.model_management

    pos_parts: list[torch.Tensor] = []
    t_parts: list[torch.Tensor] = []
    tag_parts: list[torch.Tensor] = []
    for a, b, row in segs:
        if not torch.is_tensor(row):
            continue
        tag = int(row[0].item() % 3)
        for key, t_key, want_tag in (("video", "t_obs_v", 0), ("audio", "t_obs_a", 2)):
            stream = state.get(key)
            if stream is None or tag != want_tag or t_key not in call:
                continue
            if (b - a) != stream["n"]:
                continue  # layout mismatch — skip rather than mis-splice
            dev = row.device
            pos_parts.append(stream["pos"].to(dev) + a)
            t_parts.append(call[t_key].to(dev, torch.float32))
            tag_parts.append(
                torch.full((stream["pos"].numel(),), want_tag, dtype=torch.long, device=dev)
            )
    if not pos_parts:
        call["plan"] = None
        return None

    pos = torch.cat(pos_parts)
    t_obs = torch.cat(t_parts)
    tags = torch.cat(tag_parts)
    levels = t_obs.unique()
    # AdalnProj lays outputs out as [n_t * modalities, hidden] with mod row
    # t_idx*3 + tag (model.py:219-228) — same indexing as the truthful table.
    obs_rows = torch.searchsorted(levels, t_obs) * 3 + tags

    device = t_emb.device
    t_vals = levels.to(device=device, dtype=torch.float32)
    if getattr(dm, "use_adaln_curves", False):
        # Interpolated time-embedding curve, replicated from model.py:684-689.
        table = comfy.model_management.cast_to(dm.adaln_t_table, device=device)
        grid = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = grid.floor().long().clamp(max=table.shape[0] - 2)
        t_emb_obs = torch.lerp(table[i0], table[i0 + 1], (grid - i0).unsqueeze(1))
    else:
        t_emb_obs = dm.time_embedder(t_vals).to(t_emb.dtype)

    plan = (pos.to(device), obs_rows.to(device), t_emb_obs)
    call["plan"] = plan
    return plan


def _attention_with_observer_kv(  # pragma: no cover - GPU
    attn: Any,
    x: torch.Tensor,
    x_obs: torch.Tensor,
    pos: torch.Tensor,
    rope_freqs: Any,
    transformer_options: dict[str, Any],
) -> torch.Tensor:
    """``Attention.forward`` (model.py:169-196) with observer K+V spliced in.

    The observer projections are written into the qkv buffer BEFORE the fused
    RMSNorm+rope pass, so spliced keys and values receive identical norm/rope
    treatment at their true token positions.  Q is never touched — the row's
    own addressing stays truthful.
    """
    import comfy.model_management
    import comfy.quant_ops
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

    s = x.shape[0]
    inner = attn.heads * attn.head_dim
    q, k, v = attn.qkv_proj(x).split(inner, dim=-1)
    obs_kv = attn.qkv_proj(x_obs).split(inner, dim=-1)
    k[pos] = obs_kv[1].to(k.dtype)
    v[pos] = obs_kv[2].to(v.dtype)
    v = v.view(s, attn.heads, attn.head_dim)
    if rope_freqs is not None:
        q = q.view(1, s, attn.heads, attn.head_dim)
        k = k.view(1, s, attn.heads, attn.head_dim)
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
            )
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
            )
        q = q[0]
        k = k[0]
    else:
        q = attn.q_norm(q.view(s, attn.heads, attn.head_dim))
        k = attn.k_norm(k.view(s, attn.heads, attn.head_dim))
    v = v.clone()
    q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
    k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
    v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
    out = optimized_attention(
        q, k, v, attn.heads, mask=None, skip_reshape=True, transformer_options=transformer_options
    )
    return attn.out_proj(out.squeeze(0))


def _make_block_patch(dm: Any, block: Any, state: dict[str, Any]) -> Any:  # pragma: no cover - GPU
    """Replace-patch for one DiT block: truthful everything except inject-row K/V.

    Replicates ``DiTBlock.forward`` (model.py:286-291) with the attn input for
    inject-token K/V modulated under the observer label.  Falls through to the
    original block whenever no per-call observer state is armed.
    """

    def patch(args: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        call = state.get("call")
        if not call:
            return extra["original_block"](args)

        from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift

        h = args["img"]
        t_emb = args["t_emb"]
        segs = args["mod_segments"]
        plan = _call_plan(dm, state, call, segs, t_emb)
        if plan is None:
            return extra["original_block"](args)
        pos, obs_rows, t_emb_obs = plan

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
        o_shift, o_scale = block.adaln_proj(t_emb_obs)[:2]
        hn = block.norm1(h)
        # Observer view sliced BEFORE the truthful in-place modulation.
        h_obs = hn[pos].clone()
        h_obs.mul_(1.0 + o_scale[obs_rows].to(h_obs.dtype)).add_(o_shift[obs_rows].to(h_obs.dtype))
        h_mod = _mod_scale_shift(hn, shift_msa, scale_msa, segs)
        attn_out = _attention_with_observer_kv(
            block.attn,
            h_mod,
            h_obs,
            pos,
            args["rope_freqs"],
            args["transformer_options"],
        )
        x = _mod_gate(h, gate_msa, attn_out, segs)
        h2 = _mod_scale_shift(block.norm2(x), shift_mlp, scale_mlp, segs)
        return {"img": _mod_gate(x, gate_mlp, block.mlp(h2), segs)}

    return patch
