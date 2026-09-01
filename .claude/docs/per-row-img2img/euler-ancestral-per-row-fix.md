<!-- provenance: theory (UNVERIFIED — CPU-designed, GPU pending) -->
<!-- verified: 2026-09-01 (branch fix-euler-ancestral-per-row-renoise) · design + source read on main @36cef34; GPU pending -->
# euler_ancestral per-row fix design — combined clean-K/V wiring + σ_v-axis integration

Design for the ONE combined fix that clears both `euler_ancestral` per-row artifacts at once.
Neither half alone is sufficient — that is why every prior single-sided attempt failed. This is a
CPU-designed proposal, NOT yet implemented or GPU-verified.

Read alongside [bugs.md](bugs.md) Bug C (axis) + Bug F (clean-K/V gap) and
[audio-axis-verdict.md](audio-axis-verdict.md).

## Two artifacts under euler_ancestral (both on main; both absent under euler)

Stock H3 handles `euler_ancestral` cleanly, so OUR per-row machinery is the cause. A same-branch,
same-prompt sampler swap isolates it: `euler` shows NEITHER artifact; `euler_ancestral` shows BOTH.

1. **VIDEO ghost (Bug F).** `_euler_ancestral_rf_step` (sampler.py:412) calls `ctx.model(...)`
   directly at sampler.py:467 with NO observer/frac gate. `_euler_step` instead routes through the
   clean-K/V splice `_single_forward_denoised` (sampler.py:391). The loop arms observer/frac_mask
   state unconditionally, but the ancestral step never fires it → fractional rows (video AND audio)
   receive the ghost-contaminated denoised.
2. **AUDIO noise (Bug C).** NOT a release-schedule problem — that hypothesis is REFUTED: the per-row
   ancestral terms already track each row's schedule, and `renoise_coeff` vanishes branch-free at
   the terminal step. Real cause = AXIS INCOHERENCE: the ancestral integration uses `ctx.sig_row`,
   which for audio rows sits on the σ_a-shifted schedule, while stock steps the whole packed latent
   on the σ_v trajectory. Renoise is injected mis-scaled for the wrong sigma level every step.

## THE FIX = both halves together

Apply BOTH; neither alone clears both artifacts.

1. **Wire the clean-K/V gate into `_euler_ancestral_rf_step`.** Add the same observer/frac gate that
   `_euler_step` uses so the forward routes through `_single_forward_denoised`. The ancestral renoise
   still runs AFTER, unchanged.
2. **Re-apply the σ_v-axis integration (PR #31's change).** Add `sig_row_v`/`sig_row_v_next` to
   `_StepContext` and a `row_sigma_v` closure = per-row sigma on the RAW video schedule for ALL rows.
   Use σ_v-axis tensors for `denoised_r` and every ancestral term (`downstep_ratio`, `sigma_down`,
   `alpha`, `renoise_coeff`, `ratio`).

**σ_a stays load-bearing for the LABEL only** — per-row `w`, pooled conds, observer-split labels.
The σ_a-label proof is untouched ([audio-axis-verdict/sigma-a-label-proof.md](audio-axis-verdict/sigma-a-label-proof.md)).

**Do NOT apply the `/sig_g` velocity change from the #76 thread** — that was the abandoned
σ_a-coherent formulation and conflicts with σ_v coherence.

## Attribution — the σ_v axis was NECESSARY, not wrong

The σ_v axis is a NECESSARY half, GPU-validated for m=1 free audio 2026-08-28. PR #31 failed on
FRACTIONAL audio only because the splice bypass (half 1) still fed it a contaminated denoised. So
the axis was NOT "the wrong cause" — the earlier falsification framing is corrected to **necessary
but insufficient alone**. The audio noise needs BOTH the clean-K/V wiring AND the σ_v axis.

## GPU verification pending

- 0.3 fractional-video ghost cleared.
- Fractional-audio noise cleared.
- m=1 free audio still clean.
- `euler` unchanged.

## Fallback (diagnostic only)

If fractional-audio noise persists after both halves, try elementwise eta-gating: `eta_row =
eta·(m>=1)` so fractional rows take the deterministic RF sub-path. Diagnostic, not the intended fix.

## Stale-thread reconciliation — PR #20 was a PHANTOM code fix

PR #20 (commit b87535c, "Fix #76 audio hiss: divide velocity recovery by per-modality sig_g")
claims a code fix in its message, but its diff is **WIKI-ONLY** — 4 doc files, ZERO code changed
(verified via `git show --stat b87535c`). The advertised `/sig_g` velocity-recovery code change was
NEVER landed. That is fine: the `/sig_g` (σ_a-coherent) formulation is the abandoned direction this
fix explicitly rejects. Flagged so the #76 thread is not mistaken for a shipped code fix.
