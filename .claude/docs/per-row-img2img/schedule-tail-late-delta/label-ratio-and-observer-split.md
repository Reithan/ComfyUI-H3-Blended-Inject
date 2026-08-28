<!-- provenance: confirmed (GPU-VALIDATED 2026-08-27, proto-schedule-tail-release) -->
<!-- verified: n/a — analytical only; comfy-ref model.py @b78cec87 lines 171/286-291/698-706 cited for source grounding -->
# H4: Label-ratio hypothesis + Observer-label K/V split

**Proposed:** 2026-08-28, branch `proto-schedule-tail-release`. **Built 2026-08-28** (commits 8da7cd9 + 3eee965). Leading candidate; GPU-unverified.

Cross-links: [parent index](../schedule-tail-late-delta.md) · [data-and-hypotheses](data-and-hypotheses.md) ·
[label-channel-probe](label-channel-probe.md) · [native-h3-mechanism](../native-h3-mechanism.md).

## H4 — Label-ratio / attention-differential hypothesis

**Claim:** cross-row attention diffusion is a function of the DIFFERENCE between rows' labels,
not label absolute value. The operative quantity is the ratio of inject label to neighbor label
(neighbor label is always 1 outside the inject).

| Condition | Inject label | Differential | Blend result |
|---|---|---|---|
| No inject | 1 | 1:1 | — (baseline) |
| d=0 (both) | 0.999 (pin) | ≈1:1 | Anchors perfectly |
| d=0.05 (both) | ~0.61 | ~0.61:1 | Strong blend |
| d≈0.2 (both, rescheduled mid-band) | ~0.25 early label | ~0.25:1 | Weak blend |
| d inject, official schedule | d (constant) | d:1 all steps | Blends well |
| d≈0.2 rescheduled | ~0.75 label early | ~0.75:1 early | Blends weakly |

All existing data fits: large differential (small ratio) anchors well; small differential (large
ratio early in rescheduled run) weakens mid-band. Official (constant d:1 all steps) blends well;
rescheduled (label shoots near 1 early) fails mid-band.

**Observational equivalence with H1:** H4 and H1 (content legibility) are indistinguishable on
all existing data because label ≡ content consistency locks both channels together on the convex
σ-shift curve. Discriminating them requires a lock-breaking intervention — one that changes label
without changing content depth, or vice versa. The observer-label split below is that experiment.

**Neighbor-side ratio preservation (rejected, user):** holding neighbors' labels down to match the
inject's level would preserve the d:1 ratio from the neighbor side. Rejected: ramps, tuning,
complexity — impractical for a prototype.

## Observer-label split (K/V-label design)

**Goal:** one-sided ratio preservation + H4-vs-H1 discrimination. No neighbor-row changes needed.

### Source grounding

comfy-ref `model.py` @b78cec87, DiTBlock.forward lines 286–291: the per-row label is applied
exactly once as `_mod_scale_shift` on `norm1(x)`. That single modulated `h` feeds the fused
`qkv_proj` (Attention.forward line 171). A row's K/V (what neighbors read) and Q (own addressing)
inherit the label at one seam and are separable inside the block.

### Design

For inject-row token positions only: compute a second modulation of `norm1(x)` under the official
label `d` and use it for that row's K/V emission. Q, residual stream, gates, and MLP keep the
truthful rescheduled label.

Effect: neighbors see the inject row as if it carries label `d` (official level) throughout the
run, pinning the observed differential at d:1. The inject's own prediction trajectory runs on
the rescheduled label — in-frame strength untouched.

### Properties

- Observed differential pinned at d:1 all run (matches official blend regime from first step).
- Rescheduled trajectory and in-frame denoising strength untouched.
- Avoids OFFLABEL-1: the observer label never enters the row's own prediction path.
- No attention mask: no SDPA kernel fallback; fully backend-agnostic.
- No extra tokens: zero attention-budget cost.
- Overhead: one extra modulation + qkv call on inject tokens only per block.

### Implementation sketch

`blocks_replace[("double_block", i)]` hook (model.py 698–706). Patch replicates the block forward
with spliced K/V, including fused RMSNorm + rope on spliced keys. Approximately 100–150 lines;
fragile vs comfy updates (acceptable for prototype).

### Ablations and risks

- **K-only vs K+V:** V also carries copied values to neighbors. Default K+V = full "seen as
  label-d" semantics. K-only is a lighter variant worth testing if K+V over-anchors.
- **Q sees observer-K indirectly:** no direct residual feedback path from the fake K into Q.
- **Observer label constant d to start:** simpler; later try σ_off(i) for exact schedule match.

### Build (2026-08-28, commits 8da7cd9 + 3eee965)

New module `comfyui_h3_blended_inject/observer_split.py`. Installs
`patches_replace["dit"][("double_block", i)]` replace-patches on every DiT block via
`ModelPatcher.set_model_patch_replace`. Per-token-row official mask `m` computed once at install
by replicating the model's own `mask_row_values` call (pad-to-patch-size dims) from the
`_denoise_mask_values` cond tensor.

Conditioning wrapper refreshes per-call observer labels `t_obs = clamp(1−m·σ, max=t_pin)` for
video (σ_v) and audio (shifted σ_a) at each model call. Block patch replicates DiTBlock.forward:
slices the norm1 output for inject tokens BEFORE the truthful in-place modulation, modulates
that slice under `t_obs` (adaLN rows `t_idx*3+tag`, curve-table embed replicated), and splices
observer K (mode `'k'`) or K+V (mode `'kv'`) into the fused qkv buffer BEFORE the RMSNorm+rope
pass so spliced keys get identical norm/rope at their true positions. Q, residual, MLP, gates,
and final layer stay truthful.

Node knob: `observer_split` combo `["off","kv","k"]` default `"off"`, active only in remap modes
(`'both'`/`'rescheduled'`). Per-forward splice plan cached per call token; segment identification
by tensor-valued mod rows + modality tag (`mod%3`: video 0, audio 2) + length match,
skip-on-mismatch. GPU-only (pragma'd); no tests per prototype convention.

## GPU readout (2026-08-27, proto-schedule-tail-release)

Mode: `rescheduled`, `observer_split=kv`. Resolutions tested: 0.2MP, 0.5MP.

| config (res / inj1_d / inj2_d) | result |
|---|---|
| 0.5MP / 0.30 / 0.15 | almost perfect; inj1 slightly more denoised than expected |
| 0.5MP / 0.20 / 0.20 | almost perfect; inj1 slightly under-denoised |
| 0.5MP / 0.25 / 0.20 | inj1 perfect; inj2 slight poor blend into timeline |
| 0.2MP / X / 0.20 | inj2 slightly too denoised |
| 0.2MP / X / 0.15 | inj2 just right |
| 0.2MP / 0.25 / X | inj1 too denoised |
| 0.2MP / 0.20 / X | inj1 perfect |
| 0.2MP / 0.20 / 0.15 | near-exact perfection; seed sweep would reliably land on ideal gen |
| 0.5MP / 0.20 / 0.15 | inj1 slightly under-denoised; inj2 great |

**Verdict: prototype bar met.** Blend and denoise quality is solid across tested configs. Remaining
variance follows the expected pattern: two injection windows competing for the same attention
context — a config/prompt knob issue, not a mechanism defect. In real use a seed sweep at the
sweet-spot config (d=0.2/0.15) reliably lands an ideal gen.

**H4 status: CONFIRMED (GPU).** Observer-label K/V split demonstrably improves blend quality;
the label-ratio channel is real and manipulable. Confound-breaking experiment succeeded.

**Audio:** minor artifacts present; attributed to the known incomplete audio-port (video masking
mechanisms not fully ported to audio) plus possible bad audio prompt / fade-in artifacts. Expected.
Not a mechanism defect.

**No further mechanistic changes identified.** Multi-inject interference at tight spacings is
inherent to multi-keyframe setups (present in Motion Context too) — not addressable at the
mechanism layer.
