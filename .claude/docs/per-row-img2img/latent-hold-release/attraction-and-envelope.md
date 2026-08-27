<!-- provenance: status (live GPU debug thread + Fable analysis; route-1 latent hold-and-release) -->
<!-- verified: 2026-08-25 · GPU runs (hold A/B, denoise=0.0) + Fable code audit @proto-latent-hold-release -->
# Attraction & the envelope fork (Findings 4–6)

Index: [index](index.md). Build + earlier findings: [mechanism-and-early-findings](mechanism-and-early-findings.md).
Mechanism trace, invariance, and the provenance confound (Findings 7–10):
[hold-mechanism-and-confounds](hold-mechanism-and-confounds.md).

## Finding 4 — A/B: attraction needs no hold; hold-run cut (why = OPEN)

GPU A/B (only `latent_hold_frac` changed, 0.6 → 0.0, all else fixed): **hold OFF** (keyframe at
fractional min_denoise) → neighbors blend toward the keyframe; **hold ON** → hard cut. So attraction is
a BASELINE property of the co-evolving per-row inject and does not need a hold; route-1's stated premise
("hold clean early to CREATE attraction") is not what's doing the work. What it does NOT establish is
*why* the hold run cut — do not jump to "the hold suppresses attraction." (Later resolved: the hold froze
the WRONG rows — see [hold-mechanism-and-confounds](hold-mechanism-and-confounds.md) Finding 10.)

## Finding 5 — freeze BLENDS (denoise=0.0, VALID); my two reasoning errors, retracted

Config (user, 2026-08-25): 1-frame inject on the singleton "1" row of a 1/4/4/4/4 chunk, markers
0/0/1/1, interpolation **none**. Two GPU observations:
- **OBS-A:** hold-ON run — no neighbor blend formed DURING the hold (before release).
- **OBS-B:** `denoise=0.0` — neighbors DO attract, early. **VALID as a freeze test** (Finding 6 decodes
  the envelope): under `none`, row 40 = md exactly = 0.0 → a TRUE frozen clean m=0 anchor, and there is
  NO shoulder row. So a frozen clean interior anchor DID attract.

Two corrections to my earlier reasoning (both retracted):
- "OBS-A ⇒ every release theory is dead" — a non-sequitur.
- "blend commits only post-σ_sw (~93% of σ-space)" — wrong (user): H3 resolves in **wavefronts, not
  uniformly**; blend visibly forms around the keyframe in steps 2–3 at σ≈0.99. Differences show up
  during the EARLY steps, not only after release.

## Finding 6 — the envelope decode (code-verified); Fable's +0.5 trap is a LINEAR-only artifact

`evaluate_envelope(0,0,1,1, md, interp, source_length=1, target_rows=60, inject_at=136)` computed
directly (`/tmp/env_check.py`, `PYTHONPATH=.`):

| interp | md | rows (row: d) |
|---|---|---|
| **none** | 0.0 | `{39: 1.0, 40: 0.0}` |
| **none** | 0.5 | `{39: 1.0, 40: 0.5}` |
| linear | 0.0 | `{39: 0.875, 40: 0.5}` |
| linear | 0.5 | `{39: 0.9375, 40: 0.75}` |

- **Under `none` (the user's setting):** `evaluate_curve('none')` is a STEP (w=0 for fade_t<1, w=1 at
  fade_t==1), so row 40 = md EXACTLY (0.0 → true frozen clean m=0) and row 39 = **1.0 = full gen** — it
  is NOT an anchor and is NOT frozen. So for this config there is **no gray shoulder wall and no +0.5
  trap.** RUN B genuinely tested a frozen clean m=0 anchor. ✅ OBS-B is valid.
- **Fable's two "breaks" were `linear`/eased artifacts.** Under a non-step curve the single-frame row's
  center (`frame+0.5`) lands mid-fade-out → row 40 → 0.5 at md=0 (never `min_denoise`), and the shoulder
  is fractional. **Real, but only for linear/eased interpolation — they do NOT apply to the user's run.**
- **Keep the +0.5 trap as a general warning for linear/eased 1-frame injects:** consider
  auto-degenerating the envelope (or forcing `none`) when `source_length==1`.

## Where this thread continues

Findings 7–10 (mechanism trace, the code-identical-prefix puzzle, the min_denoise=0.0 no-op correction,
and the provenance-blind `anchor_mask` confound that resolves it) live in
[hold-mechanism-and-confounds](hold-mechanism-and-confounds.md).
