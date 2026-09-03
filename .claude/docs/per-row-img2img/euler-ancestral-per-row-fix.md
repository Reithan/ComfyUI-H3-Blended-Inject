<!-- provenance: bug (Bug F: ATTRIBUTED; plant-axis: SHIPPED+FALSIFIED; content-axis: GPU PARTIAL;
     C2 durable port: GPU CONFIRMED; anchor fix: FALSIFIED round 9; PLANT_AXIS revert: GPU CONFIRMED round 10 — muffling fixed, static persists;
     δ-reinjection: δ-residual CONFIRMED Branch 1; noise-carry fix GPU CONFIRMED PARTIAL round 11 — low-m δ leak gone;
     mid-m ANCESTRAL-SPECIFIC (Round-11b falsifies residual-accounting PRIMARY); mechanism = denoiser noise-leak committed by ancestral replacement;
     packed-axis clamp = minor bookkeeping wart (k_d 6–9, ≤6%); CPU-probe-gated fix candidates pending — see mid-m-denoiser-leak.md) -->
<!-- verified: 2026-09-02 (branch fix-euler-ancestral-per-row-renoise) · round-11 GPU CONFIRMED PARTIAL + Round-11b euler-deterministic control:
     low-m δ leak gone; mid-m = ANCESTRAL-SPECIFIC (residual-accounting PRIMARY FALSIFIED);
     mechanism = denoiser noise-leak (λσ_c·εc in ĉ; ancestral commits it; euler re-denoises it);
     packed-axis split = minor bookkeeping only (k_d 6–9, ≤6%); fix = CPU probe → excess-leak cancellation (design in mid-m-denoiser-leak.md) -->
# euler_ancestral per-row fix — index

> SUPERSEDED (2026-09-02): the C2 packed-axis audio correction is abandoned. Audio fades now route
> through the official mask composite — see [audio-native-composite](audio-native-composite.md).

Design and shipped record for the combined fix that clears both `euler_ancestral` per-row
artifacts. Neither half alone is sufficient — that is why every prior single-sided attempt failed.

Read alongside [bugs.md](bugs.md) Bug C (axis) + Bug F (clean-K/V gap) and
[audio-axis-verdict.md](audio-axis-verdict.md).

## Two artifacts under euler_ancestral (both on main; both absent under euler)

Stock H3 handles `euler_ancestral` cleanly, so OUR per-row machinery is the cause. A same-branch,
same-prompt sampler swap isolates it: `euler` shows NEITHER artifact; `euler_ancestral` shows BOTH.

⚠ **"euler clean" qualifier:** this was observed on a 24-f fade. Long-fade runs (≥91 frames) also
show a deterministic noise floor in both modalities (both samplers) — see
[c2-rho-fix-paths/residual-accounting.md](c2-rho-fix-paths/residual-accounting.md).
The "euler clean" claim holds for the two specific artifacts listed below, not for all fade noise.

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
| Anchor fix fe0343a+91078cc (audio-anchor-scale.md) | FALSIFIED for peak (round 9 GPU) — introduces muffling; retained as model-scale-correct K/V fix |
| C2 carry-compression durable port (c2-durable-port.md) | GPU CONFIRMED; residual localized low-m band; anchor falsified — root cause = PLANT_AXIS |
| Plant-over-noise (PLANT_AXIS revert to "row") | GPU CONFIRMED 2026-09-02 — muffling fixed; static persists 0.25–1.5 s band |
| δ re-injection noise-carry fix (commit a28a62b) | GPU CONFIRMED PARTIAL — low-m corr collapsed; mid-m ANCESTRAL-SPECIFIC; mechanism = denoiser noise-leak; see [mid-m-denoiser-leak.md](euler-ancestral-per-row-fix/mid-m-denoiser-leak.md) |

## Child docs

- [plant-axis.md](euler-ancestral-per-row-fix/plant-axis.md) — plant-axis fix record + GPU FALSIFICATION + bypass sub-theory refutation + Bug B refinement
- [content-axis.md](euler-ancestral-per-row-fix/content-axis.md) — content-axis observer fix (GPU PARTIAL 2026-09-01; residual stochastic-only)
- [c2-durable-port.md](euler-ancestral-per-row-fix/c2-durable-port.md) — C2 carry-compression durable port: Fable round-5 verdict, mechanism, decision, spec (GPU CONFIRMED; residual localized low-m band)
- [audio-anchor-scale.md](euler-ancestral-per-row-fix/audio-anchor-scale.md) — round-8: hot audio band anchor theory, impl (fe0343a+91078cc); FALSIFIED for peak (round 9)
- [plant-over-noise.md](euler-ancestral-per-row-fix/plant-over-noise.md) — round-9: PLANT_AXIS "v" untruthful under C2; root cause of 0.75–1.0 s peak; fix = revert to "row"; GPU CONFIRMED (muffling fixed, static persists)
- [delta-reinjection.md](euler-ancestral-per-row-fix/delta-reinjection.md) — round-10: δ-as-RESIDUAL CONFIRMED (Branch 1, ε̂ under-cancellation + 1/σ_c); noise-carry fix IMPLEMENTED commit a28a62b (CPU-verified, GPU pending)
- [noise-carry-gpu-result.md](euler-ancestral-per-row-fix/noise-carry-gpu-result.md) — round-11: low-m δ leak CONFIRMED FIXED; Round-11b: mid-m ANCESTRAL-SPECIFIC; mechanism in mid-m-denoiser-leak.md
- [mid-m-denoiser-leak.md](euler-ancestral-per-row-fix/mid-m-denoiser-leak.md) — CSV-reconciled mid-m static mechanism + fix candidates; clamp RETRACTED (minor, k_d 6–9); cause = denoiser noise-leak; CPU-probe gate

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
