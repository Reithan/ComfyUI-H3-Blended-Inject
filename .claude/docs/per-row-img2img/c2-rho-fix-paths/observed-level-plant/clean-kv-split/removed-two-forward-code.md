<!-- provenance: reference (preserved removed code — the two-forward clean-K/V implementation, deleted from code) -->
<!-- verified: GPU-CONFIRMED 2026-08-31 (as prose in ../clean-kv-split.md); code removed 2026-09-01 branch single-forward-clean-kv-splice; recoverable from git history -->
# Removed implementation — two-forward clean-K/V code reference (preserved)

Parent: [../clean-kv-split.md](../clean-kv-split.md).

AS OF branch `single-forward-clean-kv-splice` the two-forward clean-K/V code below was REMOVED
from `sampler.py` / `observer_split.py`; the single-forward path
([../option-ii-single-forward.md](../option-ii-single-forward.md)) is the SOLE runtime mechanism.
The two-forward remains the conceptual/mathematical reference — the single forward reproduces it
bit-for-bit — and is recoverable from git history. These snippets are its authoritative record now
that the code is gone.

## 1. Observer content + the two publishes + the two forwards (`_clean_kv_denoised`, sampler.py)

```python
# Observer content: static clean inject re-noised to σ_obs (no cross-step feedback).
anchor = clean
ratio = (sig_obs / sig_row.clamp(min=eps)).clamp(max=1.0)
x_obs = torch.where(frac, anchor + ratio * (x_prev - anchor), x_prev)

# Capture forward: clean content, m labels; block patches snapshot fractional-band K/V.
obs["kv_cache"] = {}
obs["mode"] = "capture"
_publish(m_dev.clamp(max=1.0))
model(x_obs, sigma * s_in, **extra_args)  # discard output; fills kv_cache

# Splice forward: truthful σ_row self forward; block patches overwrite band K/V with the cache.
obs["mode"] = "splice"
_publish((sig_row / sig_g).clamp(max=1.0))
denoised = model(x_prev, sigma * s_in, **extra_args)
```

`sig_obs = m·σ_g`; `_publish` sets `pooled_current = make_pooled(w_vec)`. The capture forward
publishes the observer level `m`; the splice forward publishes the truthful per-row fraction
`σ_row/σ_g`. For integer neighbour rows these coincide (`σ_row = m·σ_g` at `m ∈ {0,1}`), which is
exactly what makes the single-forward's neighbour-K/V reuse exact.

## 2. Capture-mode and splice-mode block patch (`_make_block_patch`, observer_split.py)

```python
if mode == "capture":
    shift_msa, scale_msa = block.adaln_proj(t_emb)[:2]
    h_mod = _mod_scale_shift(block.norm1(h), shift_msa, scale_msa, segs)
    inner = block.attn.heads * block.attn.head_dim
    _, k, v = block.attn.qkv_proj(h_mod[pos]).split(inner, dim=-1)
    state["kv_cache"][idx] = (k.detach().clone(), v.detach().clone())  # raw PRE-rope K/V
    return extra["original_block"](args)

# splice mode: overwrite the fractional band's K/V with the cached clean K/V, Q untouched
cached = state["kv_cache"].get(idx)
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
h_mod = _mod_scale_shift(block.norm1(h), shift_msa, scale_msa, segs)
attn_out = _attention_with_cached_kv(
    block.attn, h_mod, cached[0], cached[1], pos, args["rope_freqs"], args["transformer_options"]
)
x = _mod_gate(h, gate_msa, attn_out, segs)
h2 = _mod_scale_shift(block.norm2(x), shift_mlp, scale_mlp, segs)
return {"img": _mod_gate(x, gate_mlp, block.mlp(h2), segs)}
```

## 3. The cached-K/V attention splice (`_attention_with_cached_kv`, observer_split.py)

The key detail is that the cached K/V are written into the qkv buffer BEFORE the fused RMSNorm+rope
pass, so spliced keys/values get identical norm/rope at their true positions; Q is never touched.

```python
q, k, v = attn.qkv_proj(x).split(inner, dim=-1)
k[pos] = k_cache.to(k.dtype)  # splice cached clean K/V at the band's true token positions
v[pos] = v_cache.to(v.dtype)
# ... then the normal fused RMSNorm + split-half rope pass runs over the full sequence,
#     so spliced rows are normed/roped exactly as native rows at their positions.
```
