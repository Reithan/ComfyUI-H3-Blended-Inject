<!-- provenance: theory (HOLD-26 design — IMPLEMENTED commit 074e443, GPU-FALSIFIED HOLD-27 2026-08-27) -->
<!-- verified: 2026-08-27 · HOLD-27 GPU run (see section below); code-confirmed vs sampler.py ~502-570 / nodes.py 305-316 @074e443; branch proto-latent-hold-release -->
# Min-free-steps floor for per-frame scheduled release

**Design status:** HOLD-26 — implemented (`074e443`), GPU-FALSIFIED (HOLD-27, 2026-08-27).
Motivated by the HOLD-25 QA smudge (low-d rows lose spatial resolution) plus the pre-refinement
A/B sweep (below). Supersedes the earlier "Option B rescales the LEVEL" design on THIS page.

## Problem it solves

Free (denoising) steps = `steps − k_row`, `k_row = floor(steps·(1−d))`. A low-d row gets few free
steps and smudges. To add free steps you must release earlier — but an earlier step is a HIGHER
schedule sigma, so pulling release earlier ALSO deepens the entry noise. The pre-refinement build
(`e5996c0`) rescaled `d` to move release earlier, which conflated the two: `d=0.05` released so
early it LOOKED like `d≈0.3` (over-denoised, lost inject fidelity). See sweep below.

## The fix: decouple LEVEL from TIMING (sampler.py ~502–570)

Two per-row quantities are now separated:

- **Level (ALWAYS intended-d):** `level_step = floor(steps·(1−d))`, `sig_L = sigmas[level_step]`.
  `sig_L` drives BOTH the content pin (`pin_release = sig_L·eps + (1−sig_L)·clean`) AND the label
  numerator, in EVERY mode. A row therefore always releases with the noise its own `d` asked for —
  this is what fixes the Option-2 over-denoise defect.
- **Timing (mode-dependent) `k_rel`:** the ONLY thing `min_ratio`/`rescale` move. It sets WHEN a
  row stops being held, pulled earlier so ≥`min_ratio·steps` free steps remain.
  - **Option 1** (`rescale_step_release=False`, FLOOR): `k_rel = min(level_step, cap)`,
    `cap = steps − round(min_ratio·steps)` — a hard floor; only low-d rows move earlier.
  - **Option 2** (`rescale_step_release=True`, RESCALE): `k_rel` from `d` rescaled into
    `[min_ratio, 1]` (`m_eff = d·(1−min_ratio)+min_ratio`) — every row pulled proportionally earlier.

## Descending sub-schedule after release

Per step the label is `c_i = clamp(sig_L / max(σ_i, σ_{k_rel}), max=1)`:

- held (`i < k_rel`): `σ_i > σ_{k_rel}` → `t_row = c_i·σ_i = L` (neighbors see release at L, HOLD-25);
- released (`i ≥ k_rel`): `t_row = L·σ_i/σ_{k_rel}` descends L→0 across the STRETCHED `[k_rel, steps_n]`
  interval, instead of snapping the released row back onto the global schedule.

## Strict generalization (why 580 tests still pass)

For a non-clamped row `σ_{k_rel} = sig_L`, so `c` reverts to the old `min(1, L/σ_i)` EXACTLY —
byte-identical at `min_ratio=0`; the 580 existing tests pass unchanged. `d=0` preserves via
`L=0 → c=0`, backed by the belt-and-braces `never`-restore (`torch.where(never, clean, x_cur)`).

## Caveat

The descent is label-only (the per_frame path sets NO denoised correction), so it carries the
per-row-σ-vs-global-Euler mismatch — absorbed for deterministic samplers, NOT free. NOT GPU-verified.

## Alternative not taken (verdict revised by HOLD-27)

A plan artifact proposed a **schedule-matched** descent rather than the constant-compression
implemented here. HOLD-27 shows the failure is level-semantics (no denoised correction) plus
provenance-blindness, NOT descent shape — making schedule-matched descent NOT the next step.

## Empirical A/B sweep (user GPU, pre-refinement build e5996c0)

These motivated the refinement above:

- `min_ratio=0.00` → pop still present (per_frame mode).
- Option 2 (rescale) ON + `min_ratio` as low as 0.1 → removes the `d=0.2` pop. Option 1 (no rescale)
  needs `min_ratio ~0.3` to remove the same pop.
- `d=0.15 / min_ratio=0.1` → tiny barely-noticeable pop (≈ the limit).
- `d=0.05 / min_ratio=0.15` → minor pop; `min_ratio=0.2` → only barely present.
- Scaled to 0.5MP: `d=0.05 / min_ratio=0.25` → smeary inject frame; `min_ratio=0.3` → pop removed.
- **KEY fidelity observation:** with rescale ON, `min_ratio` 0.2–0.3, `d=0.05` → the visual denoise
  looks like 0.2–0.3, NOT 0.05. With NO rescale (Option 1), `d=0.05 / min_ratio=0.3` → an
  "accurate"-looking 0.05 denoise, no pop. ⇒ rescale conflated level with timing; the refinement
  (always intended-d level) resolves it.

## hold_prenoise_step0 is INERT in per_frame mode (code-confirmed)

Two gates make `hold_prenoise_step0` provably inert whenever per_frame release is on:

- `nodes.py:316` — the `hold_release` dict (the only holder of `prenoise_step0`) is built only
  `if latent_hold_frac > 0.0 and per_frame_cfg is None`.
- `sampler.py:582` — the per_frame branch `return`s before the hold path (line 627+) that reads
  `prenoise`. The mode print (`nodes.py:312`) says "hold/prenoise disabled."

Consequence: `min_ratio`/`rescale_step_release` act ONLY in per_frame mode; `hold_prenoise_step0`
acts ONLY in hold mode — never both live. Any "hold_prenoise killed the pop" observation came from
per_frame-OFF (hold-path) runs. In per_frame mode the intended-d refinement alone must carry the
pop-kill.

See [per-frame-scheduled-release](per-frame-scheduled-release.md) for the release mechanism this floors.

## HOLD-27 GPU result (2026-08-27) — FALSIFIED

Build `074e443`, `per_frame_release=True`, euler, same content/seed/steps as the e5996c0 motivating sweep.

| # | Res | d | min_ratio | rescale | Result |
|---|---|---|---|---|---|
| 1 | 0.2MP | 0.2 | 0.1 | ON | Denoised too much; smooth and well blended |
| 2 | 0.2MP | 0.15 | 0.1 | ON | Keyframe pops with smudged frame; neighbors contaminated |
| 3 | 0.2MP | 0.05 | 0.2 | ON | Smooth, well blended; far too denoised |
| 4 | 0.2MP | 0.05 | 0.3 | OFF | New pop at fade-in transition; injects far too denoised (REGRESSION — this was accurate-0.05/no-pop on e5996c0) |
| 5 | 0.5MP | 0.05 | 0.3 | ON | Big color flash at end of video fade-in; both injects far too denoised |

**Verdict:** Level/timing decoupling FAILED on GPU. Every run over-denoises including run 4's
previously-accurate Option-1 config — the intended-d level pin did NOT restore level accuracy.

**Leading suspect (mechanism, HYPOTHESIS):** the per_frame path applies NO denoised correction.
After release each free step applies the full global Euler update with only the LABEL scaled
(the "label-only descent" caveat on this doc). Nothing clamps total displacement to `d`, so
realized redraw is governed by free-step COUNT (which `min_ratio` increases), not by the level
pin. The descending sub-schedule also keeps `t_row` below global sigma for the whole free
interval, worsening the per-row-σ-vs-global-Euler mismatch.

**Second, separate bug (code-CONFIRMED):** the floor is provenance-blind. `k_rel` is computed
from the full per-row envelope `m_dev` with no anchor-provenance filter, so the opening video
inject's fractional fade-out ramp rows get clamped (Option 1) / rescaled (Option 2) like
keyframes — producing run 4's new fade pop and run 5's color flash. Same bug class as the old
provenance-blind `anchor_mask` confound (see hold-mechanism-and-confounds.md Findings 7–11).

**Consequence:** the failure is level-semantics (no displacement clamp) plus provenance-blindness.
Open question: whether adding a per-row denoised correction (`m·denoised+(1−m)·x` style) to the
per_frame path can fix the level issue; the floor must also be provenance-filtered before any
re-test.
