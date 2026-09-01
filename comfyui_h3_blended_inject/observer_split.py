"""Clean-K/V observer content splice for fractional inject rows.

The H3 DiT applies each row's mask label exactly once per block — an adaLN
scale/shift of ``norm1(x)`` (``_mod_scale_shift``) — and that single modulated
tensor feeds the fused qkv projection.  A row's K/V (what other rows READ of it,
AND what the row reads of ITSELF in self-attention) and its Q / residual path
(what the row COMPUTES for its own velocity) inherit the label at a single seam
and are separable there.

This module patches every DiT block so a fractional inject row's K and V are
sourced from GENUINELY CLEANER CONTENT — the static ``clean`` inject re-noised to
the observed level ``m·σ_g`` — while Q, gates, and the MLP path keep the truthful
remapped label ``t_row = 1 − σ_row``.  This lets a fractional row broadcast (and,
because self-attention reads its own K/V, perceive itself) as its clean anchor at
``m`` while its own velocity stays truthful — no ghost.

It runs as ONE forward per euler step (Option II, exact single-forward):

- ``mode == "single"``: carry a band-only side stream ``h_m`` (init
  ``ratio·h_main0 + (1−ratio)·h_clean`` — exact by patch-embed linearity) and,
  per block, build ONE combined K/V set = the main stream's K/V with the band rows
  overwritten by the side stream's, read by two queries: the main query at ``σ_row``
  (→ ``denoised``) and the band query at observer ``m`` (→ evolves ``h_m``).
- ``mode == "embed_capture"``: a one-time forward on the static ``clean`` inject
  that snapshots the block-0 band hidden ``h_clean`` for the side-stream init.

This reproduces bit-for-bit the earlier two-forward capture/splice mechanism (a
capture forward on the clean content that cached each block's band K/V, then a
splice forward that overwrote the band's K/V before the fused RMSNorm+rope pass).
That two-forward code was removed on branch single-forward-clean-kv-splice; it is
preserved in ``c2-rho-fix-paths/observed-level-plant/clean-kv-split.md``.  Both
SUPERSEDE the old label-only relabel (re-modulated the still-noisy hidden, leaked
fade-noise) and the second-stream path (lost self-reception); see
``c2-rho-fix-paths/observed-level-plant/second-stream.md`` and
``option-ii-single-forward.md``.

The pure helpers — ``observer_call_update``, ``_fractional_rows``,
``_observer_timestep``, ``_embed_ratio``, ``_blend_hidden``, ``_band_mod_index``,
``_call_plan`` — are importable and unit-tested on CPU.  The comfy-coupled
functions (``install_observer_split``, ``_observer_time_embed``,
``_norm_rope_query``, ``_dual_attention``, ``_single_plan``, ``_make_block_patch``)
require a live ComfyUI + H3 model and are exercised only on GPU runs; their
``comfy`` imports stay lazy so this module imports cleanly on CPU.
"""

from __future__ import annotations

import os
from typing import Any

import torch

#: Env toggle for the AUDIO-band K/V splice (A/B discriminator, claude-opus-4-8).  The splice
#: was built for the VIDEO keyframe ghost; the audio band was spliced for symmetry.  Set
#: ``H3BI_SPLICE_AUDIO=0`` to leave audio K/V on the main stream (band sits at its truthful
#: σ_c level) — if the residual mid-fade audio crackle under euler_ancestral disappears, the
#: audio splice is the remaining re-excited residual; if unchanged, it is the estimation floor.
SPLICE_AUDIO_ENV = "H3BI_SPLICE_AUDIO"


def _splice_audio_enabled() -> bool:
    """True unless ``H3BI_SPLICE_AUDIO=0`` is set (read at install time)."""
    return os.environ.get(SPLICE_AUDIO_ENV, "1") != "0"


def observer_call_update(obs: dict[str, Any]) -> None:
    """Bump the per-forward token so each forward rebuilds its splice plan.

    Called once per model forward by the conditioning wrapper.  The clean-K/V
    mechanism carries no per-call observer LABELS (the single forward publishes the
    truthful ``σ_row`` labels through ``pooled_current`` like any other row, and the
    one-time embed-capture publishes ``m``), so this only needs to invalidate the
    cached position/side-stream plan between forwards (embed-capture vs each step).
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
        # dm is captured so the single-forward path can build the side stream's observer
        # time-embedding (t_emb_m) from the model's own adaLN curve / time-embedder.
        "dm": dm,
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
    if am is not None and _splice_audio_enabled():
        state["audio"] = _fractional_rows(am[0, 0].to(torch.float32).reshape(-1))

    if state.get("video") is None and state.get("audio") is None:
        return False

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
    the per-forward token bump invalidates the cache each forward.  Used by the
    one-time ``embed_capture`` forward to locate the band's block-0 hidden.
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


def _single_plan(  # pragma: no cover - GPU
    state: dict[str, Any], call: dict[str, Any], segs: list[Any]
) -> dict[str, Any] | None:
    """Single-forward splice plan: band token positions + aligned per-row side-stream data.

    Like :func:`_call_plan` but also gathers, in the SAME seg-iteration order as ``pos``, the
    per-step ``ratio`` (block-0 embed-blend weight) and ``mod`` (adaLN mod-row index into the
    side stream's ``t_emb_m`` table) that the sampler primed onto each stream.  Concatenating all
    three in lockstep guarantees the side hidden ``h_m``, its modulation, and its clean-embed init
    stay row-aligned to the global token positions regardless of video/audio segment order.
    """
    plan = call.get("splan", "unset")
    if plan != "unset":
        return plan

    pos_parts: list[torch.Tensor] = []
    ratio_parts: list[torch.Tensor] = []
    mod_parts: list[torch.Tensor] = []
    for a, b, row in segs:
        if not torch.is_tensor(row):
            continue
        tag = int(row[0].item() % 3)
        for key, want_tag in (("video", 0), ("audio", 2)):
            stream = state.get(key)
            if stream is None or tag != want_tag:
                continue
            if (b - a) != stream["n"] or "ratio" not in stream or "mod_index" not in stream:
                continue  # layout mismatch or side stream not primed this step — skip
            pos_parts.append(stream["pos"].to(row.device) + a)
            ratio_parts.append(stream["ratio"].to(row.device))
            mod_parts.append(stream["mod_index"].to(row.device))
    if not pos_parts:
        call["splan"] = None
        return None

    plan = {
        "pos": torch.cat(pos_parts),
        "ratio": torch.cat(ratio_parts),
        "mod": torch.cat(mod_parts),
    }
    call["splan"] = plan
    return plan


def _observer_time_embed(  # pragma: no cover - GPU
    dm: Any, levels: torch.Tensor, dtype: Any, device: Any
) -> torch.Tensor:
    """Side stream's ``t_emb_m`` table for the observer timestep ``levels`` (model.py:683-691).

    Reuses the model's own adaLN curve (when ``use_adaln_curves``) or ``time_embedder`` so the side
    stream is modulated at the observer levels ``m`` exactly as the two-forward capture would be —
    the capture forward's adaLN table is not present in a σ_row-labelled single forward, so we
    rebuild just the band's ``m`` levels here.  Returns an ``[len(levels), t_dim]`` table indexed by
    :func:`_band_mod_index`.
    """
    import comfy.model_management

    t_vals = levels.to(torch.float32).to(device)
    if getattr(dm, "use_adaln_curves", False):
        table = comfy.model_management.cast_to(dm.adaln_t_table, device=device)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
    return dm.time_embedder(t_vals).to(dtype)


def _norm_rope_query(  # pragma: no cover - GPU
    attn: Any, q_raw: torch.Tensor, rope_slice: torch.Tensor, qw: Any, kw: Any, rot: int
) -> torch.Tensor:
    """RMSNorm + split-half rope for a QUERY-only band tensor at its own token positions.

    Mirrors the q half of ``Attention.forward`` (model.py:173-187) but for just the ``[B, inner]``
    band queries; ``rope_slice`` is ``rope_freqs[:, pos]`` so each side query is rotated at its
    TRUE global token position (identical to the two-forward capture's band q).  The fused kernel
    needs a k argument, so a throwaway clone is passed and discarded.
    """
    import comfy.model_management
    import comfy.quant_ops

    b = q_raw.shape[0]
    q = q_raw.view(1, b, attn.heads, attn.head_dim)
    k_dummy = q.clone()
    if comfy.model_management.in_training:
        q, _ = comfy.quant_ops.ck.rms_rope_split_half(
            q, k_dummy, rope_slice, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
        )
    else:
        comfy.quant_ops.ck.rms_rope_split_half_(
            q, k_dummy, rope_slice, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
        )
    return q[0]


def _dual_attention(  # pragma: no cover - GPU
    attn: Any,
    x_main: torch.Tensor,
    x_m: torch.Tensor,
    pos: torch.Tensor,
    rope_freqs: Any,
    transformer_options: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """One combined K/V set read by two queries — the single-forward core.

    Builds the ONE combined K/V set = the main stream's K/V (from ``x_main`` at σ_row) with the
    band rows overwritten by the side stream's raw K/V (from ``x_m`` at observer ``m``), then reads
    it with an extended query ``[q_main ; q_m]``.  ``q_main`` reproduces the splice forward's
    attention (feeds ``denoised``); ``q_m`` reproduces the capture forward's band attention (evolves
    ``h_m``).  Because the band K/V are spliced RAW before the fused norm+rope, they are rotated at
    their true positions by the full-length pass — exactly as the two-forward splice does.
    """
    import comfy.model_management
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

    s = x_main.shape[0]
    inner = attn.heads * attn.head_dim
    q, k, v = attn.qkv_proj(x_main).split(inner, dim=-1)
    q_m, k_m, v_m = attn.qkv_proj(x_m).split(inner, dim=-1)
    k[pos] = k_m.to(k.dtype)
    v[pos] = v_m.to(v.dtype)
    v = v.view(s, attn.heads, attn.head_dim)
    if rope_freqs is not None:
        q = q.view(1, s, attn.heads, attn.head_dim)
        k = k.view(1, s, attn.heads, attn.head_dim)
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x_main.device)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x_main.device)
        rot = rope_freqs.shape[-3] * 2
        import comfy.quant_ops

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
        q_m = _norm_rope_query(attn, q_m, rope_freqs[:, pos], qw, kw, rot)
    else:
        q = attn.q_norm(q.view(s, attn.heads, attn.head_dim))
        k = attn.k_norm(k.view(s, attn.heads, attn.head_dim))
        q_m = attn.q_norm(q_m.view(q_m.shape[0], attn.heads, attn.head_dim))
    v = v.clone()
    q_ext = torch.cat([q, q_m], dim=0)
    qc = AttentionTensorContainer(q_ext.transpose(0, 1).unsqueeze(0))
    kc = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
    vc = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
    out = optimized_attention(
        qc,
        kc,
        vc,
        attn.heads,
        mask=None,
        skip_reshape=True,
        transformer_options=transformer_options,
    ).squeeze(0)
    return attn.out_proj(out[:s]), attn.out_proj(out[s:])


def _make_block_patch(  # pragma: no cover - GPU
    dm: Any, block: Any, state: dict[str, Any], idx: int
) -> Any:
    """Replace-patch for one DiT block, clean-K/V splice (single-forward, Option II).

    Modes:
    - ``embed_capture`` (one-time, clean inject): snapshot the block-0 band hidden ``h_clean``.
    - ``single``: carry a band-only side stream ``h_m`` (init ``ratio·h_main0 + (1−ratio)·h_clean``)
      alongside the main stream; per block build ONE combined K/V (main K/V, band overwritten by the
      side stream's) read by two queries — main at σ_row (→ ``denoised``) and band at observer ``m``
      (→ evolves ``h_m``).  Reproduces bit-for-bit the removed two-forward capture/splice mechanism
      (preserved in ``c2-rho-fix-paths/observed-level-plant/clean-kv-split.md``) in ~1 forward.  See
      ``c2-rho-fix-paths/observed-level-plant/option-ii-single-forward.md``.

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

        if mode == "embed_capture":
            # One-time: snapshot the clean inject's block-0 band hidden (embed of `clean`).
            if idx == 0:
                cpos = _call_plan(state, call, segs)
                if cpos is not None:
                    state["h_clean"] = h[cpos].detach().clone()
            return extra["original_block"](args)

        if mode == "single":
            plan = _single_plan(state, call, segs)
            if plan is None or state.get("h_clean") is None:
                return extra["original_block"](args)
            pos = plan["pos"]
            band_mod = plan["mod"]
            t_emb_m = state["t_emb_m"]
            segs_m = [(0, int(pos.shape[0]), band_mod)]
            if idx == 0:
                h_m = _blend_hidden(h[pos], state["h_clean"].to(h.dtype), plan["ratio"])
            else:
                h_m = state.get("h_m")
                if h_m is None:
                    return extra["original_block"](args)
            smsa, scmsa, gmsa, smlp, scmlp, gmlp = block.adaln_proj(t_emb)
            sm_m, scm_m, gm_m, sm2_m, scm2_m, gm2_m = block.adaln_proj(t_emb_m)
            h_mod_main = _mod_scale_shift(block.norm1(h), smsa, scmsa, segs)
            h_mod_m = _mod_scale_shift(block.norm1(h_m), sm_m, scm_m, segs_m)
            main_out, band_out = _dual_attention(
                block.attn,
                h_mod_main,
                h_mod_m,
                pos,
                args["rope_freqs"],
                args["transformer_options"],
            )
            x = _mod_gate(h, gmsa, main_out, segs)
            h2 = _mod_scale_shift(block.norm2(x), smlp, scmlp, segs)
            x = _mod_gate(x, gmlp, block.mlp(h2), segs)
            x_m = _mod_gate(h_m, gm_m, band_out, segs_m)
            h2_m = _mod_scale_shift(block.norm2(x_m), sm2_m, scm2_m, segs_m)
            state["h_m"] = _mod_gate(x_m, gm2_m, block.mlp(h2_m), segs_m)
            return {"img": x}

        # Only embed_capture and single modes exist; any other value runs the block untouched.
        return extra["original_block"](args)

    return patch
