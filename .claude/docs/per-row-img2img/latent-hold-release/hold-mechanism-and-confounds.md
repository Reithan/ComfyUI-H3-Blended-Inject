<!-- provenance: status (SUPERSEDED — route-1; live GPU debug thread + code trace; central bug: provenance-blind anchor_mask) -->
<!-- verified: 2026-08-25 · GPU runs (md=0.0/0.5 hold A/B) + sampler.py/nodes.py source @proto-latent-hold-release -->
# Hold mechanism, invariance, and the provenance confound (Findings 7–11, SUPERSEDED)

Index: [index](index.md). The provenance fix is GPU-confirmed; the next problem (held keyframe
under-denoises) continues in [anchor-denoise-after-clean-fix](anchor-denoise-after-clean-fix.md).

## Finding 7: the clean puzzle — only the RELEASE separated blenders from the cutter

For the degenerate 1-frame config with `m_hold == m_packed` element-wise (`{40:0, else:1}`) and
`pooled_hold == pooled_conds`: hold's steps 0–12 are CODE-IDENTICAL to denoise=0.0's. Same seed, same math.
Three outcomes: co-evolving md=0.5 (no hold) BLENDS; frozen md=0 (denoise=0.0) BLENDS; frozen-then-released
(hold) CUTS. The only thing the cutter had that the blenders lacked = the release/re-noise at step 12.
**SUPERSEDED by Finding 10** — the md=0.5 hold runs also froze the opening video fade-out, so "release" was
never cleanly isolated. Finding: pure static-OOD is challenged (denoise=0.0 frozen still blends).

## Finding 8: mechanism trace — what the hold does to a targeted row (summary)

- **denoise=0.0 (no hold):** anchor `init_m=0` → `x0[anchor]=clean`; m=0 correction re-freezes it every step.
- **hold md=0.5:** during hold `init_m=0` → `x0[anchor]=clean` (identical). First noise at release step k_sw:
  `level = m_release·sigma_sw = 0.5·0.932 ≈ 0.466`, correlated eps.
- **Neighbors:** `init_m=1` in both → `x0 = x_global` (full noise), identical.

Hold mechanism: present a targeted row as m=0 — clean content AND `t_row=0.999` (pooled_hold; adaLN + attention K/V).
Both identical to what denoise=0.0 presents for that row.

## Finding 9: min_denoise=0.0 did NOT no-op the hold — it froze 1,084,864 fractional elems

GPU test (2026-08-25): with `latent_hold_frac` changed 0.0 → 0.5, `anchor_mask` caught 1,084,864 elements
and armed the hold. `min_denoise` magnitude is not the axis; the hold freezes whatever fractional rows
`anchor_mask` catches. **RESOLVED by Finding 10** — those elems were the opening video fade-out ramp, not keyframes.

## Finding 10: the hold is PROVENANCE-BLIND — it froze the opening video inject's fade-out

**Standard test config** (unchanged across every prototype test): (1) opening VIDEO inject fading OUT at start,
(2) 1-frame inject at r40, (3) 1-frame inject at r60.

`anchor_mask=(m>0)&(m<1)` (`nodes.py:251`) is provenance-blind — it catches every fractional row by value.
The 1,084,864 fractional elems were the opening video fade-out ramp, which should co-evolve. The r40/r60
keyframes at md=0.0 are m=0 (uncatchable), so the md=0.0 A/B tested "freeze the opening fade-out," NOT the
keyframe hold. At md=0.5 the hold froze the fade-out AND r40/r60 together — every prior "hold cuts" observation
was confounded. A clean keyframe-only hold was NEVER tested before the provenance fix.

**Propagated, not local (user, 2026-08-25):** H3's temporal attention propagates changes globally. Freezing the
opening video fade-out changed how r40 was handled: the frozen fade-out shifted r40 from blend to cut via
attention propagation, not a local effect at the fade-out region. Holding ANY unintended fractional row can
corrupt a distant keyframe's blend.

**FIX:** give the hold provenance — tag intended keyframe-inject rows at inject construction. Hold only rows
whose winning inject has `source_length==1` with `0<min_denoise<1`; never hold clip fade-outs.

## Finding 11: the provenance fix (implemented 2026-08-25)

Provenance is an inject property, decided at creation. A keyframe = single-frame inject (`source_length==1`)
at fractional hold (`0<min_denoise<1`). Both fields already live on `Inject`.

Implementation:
- `mask.derive_hold_provenance_mask(...)` emits **1.0 only for rows whose winning inject has `source_length==1`
  with `0<min_denoise<1`**, 0.0 for every other video row and **0.0 for ALL audio ticks**.
- `nodes.py` packs it to `prov_packed` and narrows arming to
  `anchor_mask = (m_packed>0) & (m_packed<1) & (prov_packed>0.5)`. Opening clip excluded; r40/r60 included.
- Fade-ramp rows now co-evolve as ordinary fractional img2img rows; only single-frame keyframes are held.

DEFERRED: a MULTI-frame clip at sustained `0<min_denoise<1` is never held under this rule; if such a plateau
ever ghosts, widen the predicate to the row-level `'hold'` region classifier.

**GPU result (2026-08-25):** armed-elem count dropped ~1.08M to 97,920 (r40+r60 keyframe rows only); r40's
blend SURVIVES the hold (attracts/blends correctly). The surviving problem — r40 itself under-denoises — is
Finding 12 in [anchor-denoise-after-clean-fix](anchor-denoise-after-clean-fix.md).
