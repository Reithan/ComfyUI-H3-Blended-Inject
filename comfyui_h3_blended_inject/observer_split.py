"""Clean-K/V observer content splice for fractional inject rows.

The H3 DiT applies each row's mask label exactly once per block — an adaLN
scale/shift of ``norm1(x)`` (``_mod_scale_shift``) — and that single modulated
tensor feeds the fused qkv projection.  A row's K/V (what other rows READ of it,
AND what the row reads of ITSELF in self-attention) and its Q / residual path
(what the row COMPUTES for its own velocity) inherit the label at a single seam
and are separable there.

This module patches every DiT block so a fractional inject row's K and V are
sourced from GENUINELY CLEANER CONTENT — the row's x0 estimate re-noised to the
observed level ``m·σ_g`` — while Q, gates, and the MLP path keep the truthful
remapped label ``t_row = 1 − σ_row``.  It runs as two forwards per euler step:

- ``mode == "capture"``: forward the clean observer content (labels published as
  ``m`` through ``pooled_current`` like any other row); snapshot each block's
  fractional-band raw ``qkv_proj`` K/V (pre-RMSNorm/rope) into ``kv_cache[idx]``.
- ``mode == "splice"``: the truthful ``σ_row`` self forward; overwrite the band's
  K/V with the captured clean K/V before the fused RMSNorm+rope pass.  Q is never
  touched — the row's own velocity stays truthful (no ghost), and because
  self-attention reads the row's own K/V, the row also perceives itself as the
  clean anchor at ``m``.

This SUPERSEDES the old label-only relabel (which re-modulated the still-noisy
hidden and leaked residual fade-noise) and the second-stream sampler path (which
disabled the split and lost self-reception).  See
``c2-rho-fix-paths/observed-level-plant/second-stream.md``.

The pure helpers — ``observer_call_update``, ``_fractional_rows``, ``_call_plan``
— are importable and unit-tested on CPU.  The comfy-coupled functions
(``install_observer_split``, ``_attention_with_cached_kv``, ``_make_block_patch``)
require a live ComfyUI + H3 model and are exercised only on GPU runs; their
``comfy`` imports stay lazy so this module imports cleanly on CPU.
"""

from __future__ import annotations

from typing import Any

import torch


def observer_call_update(obs: dict[str, Any]) -> None:
    """Bump the per-forward token so each forward rebuilds its splice plan.

    Called once per model forward by the conditioning wrapper.  The clean-K/V
    mechanism carries no per-call observer LABELS (the capture forward publishes
    the ``m`` labels through ``pooled_current`` like any other row), so this only
    needs to invalidate the cached position plan between the capture and splice
    forwards of a step.
    """
    token = obs.get("_token", 0) + 1
    obs["_token"] = token
    obs["call"] = {"token": token}


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


def _observer_timestep(m: torch.Tensor, sigma: torch.Tensor, pin: float) -> torch.Tensor:
    """Per-row observer timestep ``t_obs = clamp(1 − m·σ, max=pin)`` (model.py:593-609).

    ``m`` is the per-row fractional denoise and ``sigma`` the modality's global sigma at this
    step (``sigma_v`` for video, the shifted ``sigma_a`` for audio).  The single forward carries a
    SECOND time-embedding modulated at these observer levels so the side stream can be gated at
    ``m`` while the main stream stays at ``σ_row`` — the two-forward capture's adaLN table is not
    available in a splice-labeled forward, so we recompute the observer timesteps here and feed
    them back through the model's own time-embedder/curve on GPU.

    ``pin`` is the modality cond-timestep ceiling (``t_pin_v = max(t_v, 0.999)`` for video,
    ``t_pin_a = 1.0`` for audio); a fully-clean row (m→0) pins at the cond level exactly like the
    native inject path.
    """
    return (1.0 - m * sigma).clamp(max=pin)


def _embed_ratio(sig_obs: torch.Tensor, sig_row: torch.Tensor) -> torch.Tensor:
    """Block-0 embed-blend weight ``ratio = clamp(σ_obs / σ_row, 0, 1)``.

    The observer content is ``x_obs = clean + ratio·(x_prev − clean) = (1−ratio)·clean +
    ratio·x_prev`` with ``ratio = σ_obs/σ_row`` (clean-kv-split.md).  Because the H3 patch-embed is
    affine and row-wise, ``embed(x_obs) = ratio·embed(x_prev) + (1−ratio)·embed(clean)`` EXACTLY
    when the two weights sum to 1 — so the side stream's block-0 hidden is this same ratio-blend of
    the main and clean block-0 hiddens, reproducing the two-forward capture's embed bit-for-bit
    without a second patch-embed pass.  Clamped because ``σ_obs = m·σ_g ≤ σ_row`` should hold but
    schedule rounding can nudge the quotient just past 1.
    """
    return (sig_obs / sig_row.clamp(min=1e-6)).clamp(0.0, 1.0)


def _blend_hidden(h_main: torch.Tensor, h_clean: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
    """Row-wise ``ratio·h_main + (1−ratio)·h_clean`` over the fractional band.

    ``h_main``/``h_clean`` are ``(band, dim)`` block-0 hiddens (main stream at ``x_prev``, captured
    clean stream at the static ``clean`` inject); ``ratio`` is the per-row ``(band,)`` embed weight
    from :func:`_embed_ratio`.  The weight broadcasts over the feature dim so each fractional row
    blends at its own observer level.
    """
    r = ratio.reshape(-1, *([1] * (h_main.dim() - 1))).to(h_main.dtype)
    return r * h_main + (1.0 - r) * h_clean


def _band_mod_index(levels: torch.Tensor, row_obs: torch.Tensor, tag: int) -> torch.Tensor:
    """Per-row adaLN mod-row index into the side stream's own ``t_emb_m`` table.

    Mirrors the model's ``rows_to_mod_index`` (model.py:617-622): each row's observer timestep is
    located in ``levels`` (the SORTED-UNIQUE set of observer timesteps carried by ``t_emb_m``) and
    mapped to ``searchsorted(levels, obs)·3 + tag`` — the AdalnProj expands ``[M, t_dim]`` into
    ``M·3`` mod-rows (3 modalities), so level index ``i`` and modality ``tag`` (video 0, audio 2)
    select row ``i·3 + tag``.  ``levels`` spans BOTH modalities' observer levels so video and audio
    band rows index a single shared table; the tag disambiguates the modality slice.
    """
    idx = torch.searchsorted(levels, row_obs)
    return idx.to(torch.long) * 3 + tag


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

    state["kv_cache"] = {}
    state["mode"] = None
    schedule_tail_cfg["observer"] = state
    for i, block in enumerate(dm.blocks):
        m.set_model_patch_replace(_make_block_patch(dm, block, state, i), "dit", "double_block", i)
    return True


def _call_plan(state: dict[str, Any], call: dict[str, Any], segs: list[Any]) -> torch.Tensor | None:
    """Per-forward splice plan: global token positions of the fractional rows.

    Locates the per-row video/audio segments in ``mod_segments`` (the only
    entries with TENSOR mod rows; modality = mod_index % 3 — video tag 0, audio
    tag 2, model.py:615) and maps our fractional-row indices to global token
    positions.  Cached in ``call`` — all blocks of one forward share segments;
    the per-forward token bump invalidates the cache between the two forwards.
    """
    plan = call.get("plan", "unset")
    if plan != "unset":
        return plan

    pos_parts: list[torch.Tensor] = []
    for a, b, row in segs:
        if not torch.is_tensor(row):
            continue
        tag = int(row[0].item() % 3)
        for key, want_tag in (("video", 0), ("audio", 2)):
            stream = state.get(key)
            if stream is None or tag != want_tag:
                continue
            if (b - a) != stream["n"]:
                continue  # layout mismatch — skip rather than mis-splice
            pos_parts.append(stream["pos"].to(row.device) + a)
    if not pos_parts:
        call["plan"] = None
        return None

    pos = torch.cat(pos_parts)
    call["plan"] = pos
    return pos


def _attention_with_cached_kv(  # pragma: no cover - GPU
    attn: Any,
    x: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: torch.Tensor,
    rope_freqs: Any,
    transformer_options: dict[str, Any],
) -> torch.Tensor:
    """``Attention.forward`` (model.py:169-196) with cached K+V spliced in.

    The cached (clean-content) K/V rows are written into the qkv buffer BEFORE
    the fused RMSNorm+rope pass, so spliced keys and values receive identical
    norm/rope treatment at their true token positions.  Q is never touched —
    the row's own addressing stays truthful.
    """
    import comfy.model_management
    import comfy.quant_ops
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

    s = x.shape[0]
    inner = attn.heads * attn.head_dim
    q, k, v = attn.qkv_proj(x).split(inner, dim=-1)
    k[pos] = k_cache.to(k.dtype)
    v[pos] = v_cache.to(v.dtype)
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


def _make_block_patch(  # pragma: no cover - GPU
    dm: Any, block: Any, state: dict[str, Any], idx: int
) -> Any:
    """Replace-patch for one DiT block, two-forward clean-K/V splice.

    ``capture`` mode (clean-content forward): snapshot this block's fractional-row
    K/V (raw PRE-rope ``qkv_proj`` output) into ``state["kv_cache"][idx]``, then
    run the original block untouched.  ``splice`` mode (truthful σ_row forward):
    replay ``DiTBlock.forward`` (model.py:286-291) with the fractional rows' K/V
    overwritten by the cached clean K/V — so neighbours (and the row itself) read
    the frame at its clean denoise level while its Q/gate/MLP stay truthful.
    Falls through to the original block when no observer state is armed.
    """

    def patch(args: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        call = state.get("call")
        mode = state.get("mode")
        if not call or mode is None:
            return extra["original_block"](args)

        from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift

        h = args["img"]
        t_emb = args["t_emb"]
        segs = args["mod_segments"]
        pos = _call_plan(state, call, segs)
        if pos is None:
            return extra["original_block"](args)

        if mode == "capture":
            shift_msa, scale_msa = block.adaln_proj(t_emb)[:2]
            h_mod = _mod_scale_shift(block.norm1(h), shift_msa, scale_msa, segs)
            inner = block.attn.heads * block.attn.head_dim
            _, k, v = block.attn.qkv_proj(h_mod[pos]).split(inner, dim=-1)
            state["kv_cache"][idx] = (k.detach().clone(), v.detach().clone())
            return extra["original_block"](args)

        cached = state["kv_cache"].get(idx)
        if cached is None:
            return extra["original_block"](args)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
        h_mod = _mod_scale_shift(block.norm1(h), shift_msa, scale_msa, segs)
        attn_out = _attention_with_cached_kv(
            block.attn,
            h_mod,
            cached[0],
            cached[1],
            pos,
            args["rope_freqs"],
            args["transformer_options"],
        )
        x = _mod_gate(h, gate_msa, attn_out, segs)
        h2 = _mod_scale_shift(block.norm2(x), shift_mlp, scale_mlp, segs)
        return {"img": _mod_gate(x, gate_mlp, block.mlp(h2), segs)}

    return patch
