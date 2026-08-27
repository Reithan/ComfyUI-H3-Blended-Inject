<!-- provenance: status (live GPU debug thread + code trace; route-1 latent hold-and-release) -->
<!-- verified: 2026-08-25 · GPU runs (md=0.0/0.5 hold A/B) + sampler.py/nodes.py source @proto-latent-hold-release -->
# Hold mechanism, invariance, and the provenance confound (Findings 7–11)

Index: [index](index.md). The attraction/envelope thread: [attraction-and-envelope](attraction-and-envelope.md).
The provenance fix is now GPU-confirmed; the next problem (held keyframe under-denoises) continues in
[anchor-denoise-after-clean-fix](anchor-denoise-after-clean-fix.md) (Finding 12).

## Finding 7 — the clean puzzle (as posed): only the RELEASE separated blenders from the cutter

For the degenerate 1-frame config, with `m_hold == m_packed` element-wise (`{40:0, else:1}`) and
`pooled_hold == pooled_conds`:
- **hold's steps 0–12 are CODE-IDENTICAL to denoise=0.0's** — same init-lerp, pooled t_row, denoised
  correction, deterministic segment-1 math. Same seed+scheduler+steps → previews bit-identical to step 12.
- Three outcomes: co-evolving md=0.5 (no hold) BLENDS; frozen md=0 (denoise=0.0) BLENDS;
  frozen-then-**released** (hold) CUTS. The only thing the cutter had that the blenders lacked = the
  release/re-noise at step 12 ⇒ release was the leading suspect. **SUPERSEDED by Finding 10** — the
  md=0.5 hold runs also froze the opening video fade-out, so "release" was never cleanly isolated.
- Tension with the user's OOD/static-island hypothesis: it predicts a frozen island should NOT blend, but
  denoise=0.0 (frozen) DOES — so pure static-OOD is challenged.

## Finding 8 — mechanism trace (code): what the hold does to a targeted row

Seed/scheduler confound is DEAD (all runs one fixed seed + `linear_quadratic`).

Init-noise trace (`sampler.py:416`, `nodes.py:253-258`):
- **denoise=0.0 (no hold):** anchor `init_m=m_packed=0` → `x0[anchor]=clean`; the `m=0` correction
  re-freezes it every step. **Zero noise into the anchor, ever.**
- **hold md=0.5:** during hold `init_m=m_hold=0` → `x0[anchor]=clean` (identical). First noise at RELEASE
  (step k_sw): `level = m_release·σ_sw = 0.5·0.932 ≈ 0.466`, correlated eps.
- **Neighbors:** `init_m=1` in both → `x0 = x_global` (full noise), identical.

Hold mechanism = present a targeted row as **m=0**: clean content (init + correction) AND `t_row=0.999`
(`pooled_hold`; feeds adaLN + attention K/V). Both IDENTICAL to what denoise=0.0 presents for that row.

## Finding 9 — min_denoise=0.0 did NOT no-op the hold: it froze 1,084,864 fractional elems

GPU test (2026-08-25): euler / linear_quadratic / seed 326859996473190 / 20 steps; ONLY
`latent_hold_frac` changed (0.0 → 0.5):
- **run 1 (0.0, no hold):** no hold log; attraction/blending occurs. ✅
- **run 2 (0.0, hold):** `armed: 1,084,864 anchor row-elems, hold_frac=0.5`; hold 10/20 steps, release
  σ=0.975; `|clean|=|x0|=0.8392`. No attraction. ❌

**CORRECTION (I over-called "bug" in Finding 8):** md=0.0 is a hold no-op ONLY if there are NO fractional
(0<m<1) rows. `anchor_mask` caught 1,084,864 → the hold armed and FROZE real fractional rows. So
run1-vs-run2 is the SAME co-evolve-vs-freeze A/B as Finding 4 at md=0.0 — the hold doing its job, **not a
machinery bug.** Bug-vs-no-bug open question → RESOLVED by Finding 10 (faded config, no envelope bug).

⇒ **min_denoise magnitude is not the axis; freeze-vs-co-evolve of the fractional rows is.** Freezing a
fractional row to clean (m=0, t_row=0.999) KILLS the attraction it produces when it co-evolves — which
INVERTS route-1's "freeze early to create attraction" premise for those rows.

## Finding 10 — the hold is PROVENANCE-BLIND: it froze the opening video inject's fade-out

User identified the standard test config (2026-08-25, unchanged across every prototype test): (1) an
opening VIDEO inject that fades OUT at the start, (2) a 1-frame inject at r40, (3) a 1-frame inject at
r60.

- The 1,084,864 fractional elems = the **opening video fade-out ramp**. `anchor_mask=(m>0)&(m<1)`
  (`nodes.py:251`) is provenance-blind — it catches every fractional row by value, so it grabbed the
  opening fade-out (which should co-evolve) rather than only the keyframe injects the hold is FOR.
  ⇒ **no envelope bug; the hold froze the wrong rows.**
- **Confounds ALL prior hold tests:**
  - **md=0.0:** keyframes are m=0 (uncatchable) → hold froze ONLY the opening fade-out; r40/r60 frozen
    clean in BOTH runs. So the md=0.0 A/B tested "freeze the opening fade-out," NOT the keyframe hold.
  - **md=0.5:** hold froze the opening fade-out AND r40/r60 together → every "hold cuts" obs is
    confounded; we have NEVER cleanly tested holding the keyframe injects alone.
- Route-1's hold selects on 0<m<1 → inherently a FRACTIONAL-keyframe mechanism; at md=0.0 it cannot
  target the keyframes at all.

**PROPAGATED, not local (user, 2026-08-25).** Changes to one inject globally affect the rest of the
timeline (H3's forward diffusion propagation — every row attends every other row each step). Freezing the
opening video inject's fade-out changed how the **r40 keyframe** was handled downstream: the frozen
fade-out is what shifted r40 from blend to cut. So the md=0.0 cut is an **attention-propagation** effect
from the wrong-row freeze, not a local cut at the fade-out region. This is the concrete mechanism behind
"the hold froze the wrong rows" — and it means the damage is non-local: holding ANY unintended fractional
row can corrupt a distant keyframe's blend.

**FIX:** give the hold PROVENANCE — tag intended keyframe-inject rows at inject construction (the builder
knows 1-frame keyframe vs clip content); hold only those, never clip fade-outs. Then run the clean
keyframe-only hold A/B (never yet run). Because the effect propagates, the provenance filter is not a
nicety — an over-broad freeze poisons the whole timeline via attention.

## Finding 11 — the provenance fix (implemented 2026-08-25)

Provenance is an INJECT property, decided at creation, not something to re-derive from mask values: a
keyframe is a **single-frame inject** (`source_length == 1`) at a **fractional hold** (`0 < min_denoise <
1`). Both fields already live on `Inject` — no new flag needed. (An earlier draft used
`classify_row_region(..., crossfade=True) == 'hold'`; for 1-frame injects that reduces to exactly this
rule — the degenerate envelope is one row classified `'hold'` iff `min_denoise>0` — so the simpler
inject-level check is equivalent and clearer. The two differ ONLY for multi-frame clips, which we do not
hold; see the deferred case below.)

Implementation:
- `mask.derive_hold_provenance_mask(...)` mirrors `derive_fractional_mask`'s broadcast/pack path but emits
  **1.0 only for rows whose winning inject is `source_length==1` with `0<min_denoise<1`**, 0.0 for every
  other video row and **0.0 for ALL audio ticks** (the hold never targets audio — resolves the "audio
  silently held" note).
- `nodes.py` packs it to `prov_packed` (same layout as `m_packed`) and narrows the arming to
  **`anchor_mask = (m_packed>0) & (m_packed<1) & (prov_packed>0.5)`**. The opening clip (multi-frame) is
  excluded outright; r40/r60 single-frame keyframes are included.
- Fade-ramp rows now co-evolve as ordinary fractional img2img rows (their `m_packed` still drives per-row
  denoise); only single-frame keyframes are frozen/released.

DEFERRED (out of scope for the keyframe fix): a MULTI-frame clip injected at a sustained `0<min_denoise<1`
is never held under this rule. If such a sustained-clip plateau ever ghosts (untested), widen the
predicate then — the row-level `'hold'` region classifier is the ready generalization.

DONE (GPU, 2026-08-25): the clean keyframe-only md=0.5 hold A/B ran. Armed-elem count dropped ~1.08M →
97,920 (r40+r60 keyframe rows only); r40's blend SURVIVES the hold (attracts/blends correctly). The
surviving problem — r40 itself under-denoises — is Finding 12 in
[anchor-denoise-after-clean-fix](anchor-denoise-after-clean-fix.md).

## Remaining discriminators

- **Attribute the co-evolving attraction** (guide already removed): (a) **wrong-content** — m=0 with a
  DIFFERENT image; neighbors chase it ⇒ attraction is content-driven, latent-resident. (b) **label-gate**
  — `min_denoise=1/256`: content ≈ clean but label fractional; if attraction collapses, the *label* is
  the gate ([lanpaint-langevin-corrector](../lanpaint-langevin-corrector.md) "kill risk").

## Diagnostic gaps to close

- Seed/scheduler confound is DEAD. Still move the σ-schedule print OUT of the armed-hold branch
  (`sampler.py:445`) so every run logs it.
- **Log the per-row m vector once** — would show which rows are fractional (fade-ramp vs keyframe vs
  audio) instantly, confirming Finding 10 directly.
- Note: audio ticks are silently anchor-held ("fade" mode fractionalizes audio even for image-only
  injects) — worth an explicit decision later.

## Diagnostics already in code

Commits 92a5f22 / 4b0edf7 / (level-print fix): per-phase anchor mean-abs (`|clean|`/`|x0|`/`|x_mid|`) +
σ-schedule print, emitted inside the armed `hold_release` branch (move it out per the gap above).
