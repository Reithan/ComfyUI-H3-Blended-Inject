<!-- provenance: bug (Bug F: ATTRIBUTED retrodiction post-#32; plant-axis: SHIPPED+FALSIFIED GPU 2026-09-01;
     content-axis: GPU PARTIAL 2026-09-01; C2 durable port: GPU CONFIRMED; round-8 anchor fix fe0343a+91078cc, GPU pending) -->
<!-- verified: 2026-09-01 (branch fix-euler-ancestral-per-row-renoise) · C2 durable port GPU CONFIRMED; residual localized low-m band (round 8); anchor fix fe0343a+91078cc -->
# euler_ancestral per-row fix — index

Design and shipped record for the combined fix that clears both `euler_ancestral` per-row
artifacts. Neither half alone is sufficient — that is why every prior single-sided attempt failed.

Read alongside [bugs.md](bugs.md) Bug C (axis) + Bug F (clean-K/V gap) and
[audio-axis-verdict.md](audio-axis-verdict.md).

## Two artifacts under euler_ancestral (both on main; both absent under euler)

Stock H3 handles `euler_ancestral` cleanly, so OUR per-row machinery is the cause. A same-branch,
same-prompt sampler swap isolates it: `euler` shows NEITHER artifact; `euler_ancestral` shows BOTH.

1. **VIDEO ghost (Bug F).** `_euler_ancestral_rf_step` (sampler.py:412) calls `ctx.model(...)`
   directly at sampler.py:467 with NO observer/frac gate. `_euler_step` routes through the
   clean-K/V splice `_single_forward_denoised` (sampler.py:391). The ancestral step never fires
   it → fractional rows (video AND audio) receive the ghost-contaminated denoised.
2. **AUDIO noise (Bug C).** NOT a release-schedule problem. Real cause = observer side-stream K/V
   CONTENT wired to the wrong axis for fractional audio rows. Two sequential explanations; see
   children below.

## THE FIX = both halves together

1. **Wire the clean-K/V gate into `_euler_ancestral_rf_step`.** Add the same observer/frac gate
   that `_euler_step` uses so the forward routes through `_single_forward_denoised`.
2. **Fix observer band K/V content axis.** The audio observer band K/V content was primed on the
   σ_a axis; post-plant-fix the audio content sits on the σ_v axis. `_audio_observer_ratio`
   computes content via `shift⁻¹(m·σ_a)` on σ_v (the Möbius shift inverse).

**σ_a stays load-bearing for the LABEL only** — per-row `w`, pooled conds, observer-split labels.
The σ_a-label proof is untouched ([audio-axis-verdict/sigma-a-label-proof.md](audio-axis-verdict/sigma-a-label-proof.md)).
Do NOT apply the `/sig_g` velocity change from the #76 thread (σ_a-coherent, abandoned direction).

## Current status

| Item | Status |
|---|---|
| VIDEO ghost fix (Bug F wiring) | ATTRIBUTED retrodiction post-#32 (both symptoms cleared) |
| Plant-axis fix (PR #32, commits 3e82dba+e3ec742) | SHIPPED; FALSIFIED for fade-audio GPU 2026-09-01 |
| Content-axis fix (PR #32 revised, commits 4644fcf+e4a9940) | SHIPPED; GPU PARTIAL — ring narrowed mid-m; both ends clean; hiss shorter |
| Euler regression check | CONFIRMED CLEAN GPU 2026-09-01 (prime_side_stream shared path; no regression) |
| Fade-audio hiss | OPEN — audio band anchor fix pending GPU; see audio-anchor-scale.md |
| C2 carry-compression durable port (c2-durable-port.md) | GPU CONFIRMED; residual localized low-m band (round 8); anchor fix fe0343a+91078cc implemented |

## Child docs

- [plant-axis.md](euler-ancestral-per-row-fix/plant-axis.md) — plant-axis fix record + GPU FALSIFICATION + bypass sub-theory refutation + Bug B refinement
- [content-axis.md](euler-ancestral-per-row-fix/content-axis.md) — content-axis observer fix (GPU PARTIAL 2026-09-01; residual stochastic-only)
- [c2-durable-port.md](euler-ancestral-per-row-fix/c2-durable-port.md) — C2 carry-compression durable port: Fable round-5 verdict, mechanism, decision, spec (GPU CONFIRMED; residual localized low-m band)
- [audio-anchor-scale.md](euler-ancestral-per-row-fix/audio-anchor-scale.md) — round-8: hot audio band anchor theory, impl (fe0343a+91078cc), GPU plan

## Co-location verdict (resolves earlier mis-attribution)

- Bug C (audio axis) is TIMELINE-WIDE — all audio rows are m=1 in drop mode (`audio_denoise=1.0`).
  "Single-frame drop-mode audio noise = Bug C" was WRONG.
- The inject-local co-located audio noise was Bug F (H3's shared A/V attention imprinted the
  ghost-contaminated VIDEO denoised on co-located audio — consistent with Bug E audio-tracks-visual
  GPU precedent). NOT A/B-isolated — retrodiction only.
- Fade hiss = observer band content mis-axis (m-dependent, peaks mid-m, vanishes at m→0 and m=1).
  See [content-axis.md](euler-ancestral-per-row-fix/content-axis.md).
- Bug C stays REAL, timeline-wide.

## Stale-thread reconciliation — PR #20 was a PHANTOM code fix

PR #20 (commit b87535c, "Fix #76 audio hiss: divide velocity recovery by per-modality sig_g")
claims a code fix, but its diff is **WIKI-ONLY** — 4 doc files, ZERO code changed (verified via
`git show --stat b87535c`). The `/sig_g` (σ_a-coherent) formulation is the abandoned direction.
Flagged so the #76 thread is not mistaken for a shipped code fix.
