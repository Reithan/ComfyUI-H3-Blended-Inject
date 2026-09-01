<!-- provenance: plan + theory (UNVERIFIED — per-step self-side DC de-bias + low-pass frequency-split extension) -->
<!-- verified: 2026-08-31 · branch proto-observed-level-inject-noise · GPU: low-pass k-sweep kills the distinct pop (k=4 "pretty solid") but a residual brightness FLOOR persists — first-order estimate floor, not a kernel-cutoff issue -->
# Per-step self-side DC de-bias (PLAN) + low-pass extension

Parent: [../observed-level-plant.md](../observed-level-plant.md).
Prior result: [gpu-result.md](gpu-result.md) (the colour tint this attacks).
SUPERSEDED BY: [clean-kv-split.md](clean-kv-split.md) (the current replacement — clean-sourced K/V inside
the observer split; see also [second-stream.md](second-stream.md) which was the interim root-fix design
and is itself OVERTURNED 2026-08-31).

Follow-up to the GPU RESULT: kill the self-side colour tint WITHOUT reviving the ghost.

## Where the tint comes from

The tint is the self-side residue of the SAME gap `Δ = σ_row − m·σ_g`. The self path still runs the
full `σ_row`-length denoise, but the content only carries `m·σ_g` worth of noise, so the denoise
OVER-SUBTRACTS by ≈ `(1 − m·σ_g/σ_row)·(x − denoised) = Δ·η`.

The HIGH-freq part of the overshoot self-heals — the model is a denoiser, so each subsequent step
pulls high-freq back to the manifold. The surviving residue is LOW-FREQ DC/COLOUR: it looks like a
valid tinted image, so it is never corrected. Hence tint, not noise.

## The trap (same trade, one level down)

If a per-step correction cancels the overshoot by WEAKENING the step (backing x0 out with
`sig_obs = m·σ_g` instead of `σ_row`), you recover OFFICIAL behaviour → the keyframe GHOST returns.
So the correction must remove the DC residue WITHOUT touching the denoise bite.

## Design (quickest test)

Per-step DC-only re-centering inside `_euler_step` (sampler.py ~308-309), EULER-ONLY — euler is the
clean isolation (no ancestral renoise, so the tint is the only moving artifact):

- Compute the over-subtraction vector `≈ (1 − (sig_obs/sig_row).clamp(max=1))·(x_prev − denoised)`.
- Take its PER-CHANNEL PER-ROW mean over the spatial (H×W) token axes — the DC component only.
- Add ONLY that DC mean back to `denoised`, on FRACTIONAL rows. Held (m=0) and free (m=1) have Δ=0
  so are naturally untouched; gate them explicitly anyway.

Rationale: keeps `denoised`'s full texture / denoise-bite (no ghost revival), attacks precisely the
surviving DC, is self-contained (no external reference tissue → discards nothing), and re-derives the
correction fresh each step from the model's own velocity (so prompt-driven colour still flows).

## Why per-step, not post-hoc (refines FORWARD OPTION 1)

A post-hoc output-space colour/exposure match would ERASE any colour the blend legitimately produces
from the prompt. Per-step DC de-bias cancels ONLY the systematic sigma-mismatch DC drift, leaving
prompt-driven colour intact.

## GPU result + low-pass extension (2026-08-31)

**GPU result (user):** the per-step per-channel DC de-bias in `_euler_step` ELIMINATES A LOT of the
self-side colour shift but NOT ALL. For SINGLE-FRAME injects specifically, the one miscoloured frame
is still a noticeable POP in the video.

**Diagnosis:** the DC term adds back only the per-channel SPATIAL MEAN of the overshoot `Δ·η`. What
survives is the LOW-FREQUENCY, NON-CONSTANT part of the overshoot (broad gradients / blotches). It is
manifold-plausible, so the model does NOT self-heal it (unlike high-freq), and a single mean cannot
capture it. In a multi-frame FADE this residual spreads gradually across frames, so the eye tolerates
it; a SINGLE-FRAME inject is one isolated frame between correctly-coloured neighbours, so the same
residual reads as a POP. (Lumping-across-frames is NOT the cause for single-frame: one frame is
already per-frame; the residual there is purely the low-freq AC.)

**Extension (theory, UNVERIFIED — the principled generalization):** a FREQUENCY SPLIT of the
correction. Keep the HIGH-freq of the strong `σ_row` step (the denoise BITE → no ghost) but replace
the LOW frequencies with the content-consistent (`σ_obs`) estimate (no over-denoise → no tint):

    denoised_final = denoised_srow + lowpass(over),   over = (1 − m·σ_g/σ_row)·(x − denoised)

DC-only is the DEGENERATE case where `lowpass` = spatial mean (cutoff at frequency 0). The surviving
pop = the band BETWEEN DC and the true cutoff.

**Implementation:** a per-frame per-channel BOX low-pass with kernel `k` over the video (H,W) spatial
dims (audio stays DC/mean), NORMALIZED by the fractional-token weight so non-fractional tokens don't
bleed in. Degrades gracefully:

- `k → full-frame` = current DC behaviour.
- `k → 1` = full debias = GHOST / under-denoise (frame reverts toward injected source → the "not
  noticeably denoised" under-denoise the user saw on official single-frame).

So `k` is a SINGLE KNOB with a sweet spot; env-tunable `H3_DC_LOWPASS` (0/unset = DC mean = current
proven baseline; ≥2 = box kernel). For the single-frame pop, dial `k` DOWN from ∞ toward the sweet
spot; watch for softening/ghost as the ghost side.

## Low-pass k-sweep GPU result + estimate-floor diagnosis (2026-08-31)

**GPU result (user):** euler, single-frame inject + fades, sweeping `H3_DC_LOWPASS` kernel `k`:

- `k=16`: the distinct single-frame POP is gone; fade still a little too bright; the single frame's
  colours are still a bit unrealistic.
- `k=8`: brightness moderated a bit more.
- `k=4`: "pretty solid" across several alternate injects (4B/4C all relatively high quality).
- BUT a brightness artifact is STILL present at `k=4`, just less obvious. Shrinking `k` does not
  cross it out — it only makes it subtler.

**Diagnosis (theory — the key insight):** the residual brightness is a FIRST-ORDER ESTIMATE FLOOR,
not a kernel-cutoff issue. The knob is not the limiting variable; the estimate is.

- A box low-pass PASSES the DC (frequency 0), so the DC of `over` is added back at EVERY kernel.
  Shrinking `k` captured more of the bias's LOW-FREQ SPATIAL-AC structure (the real gain seen across
  the fade), but the TRUE DC MAGNITUDE is fixed by `over = (1 − m·σ_g/σ_row)·(x − denoised)` itself.
- `over` is a FIRST-ORDER estimate: it assumes the model's velocity DIRECTION is correct and only
  the scalar sigma LABEL is wrong (i.e. denoised_obs ≈ x − σ_obs·v with the SAME v as the σ_row
  eval). The model's ACTUAL output on clean-content-with-a-noisier-label is NONLINEAR — its velocity
  at label σ_obs ≠ its velocity at label σ_row — so `over` UNDER-estimates the true bias DC. That
  gap is the floor.
- Therefore bigger/dynamic kernels or auto-kernel-detection CANNOT reach zero brightness: they
  optimise a blunt knob against a floor it cannot cross.

## Principled routes past the floor (user is choosing among these)

1. **ROOT FIX #1 (most principled):** keep physical content at `σ_row` (REVERT Request A's content
   drop) so self content = self label → ZERO self-side bias/tint BY CONSTRUCTION. This was the
   ORIGINAL pre-Request-A behaviour (no tint, only observer noise; the tint only ever existed
   because Request A moved content to `m·σ_g`, so #1 never introduces it). Then fix the observer
   noise by giving the OBSERVER a separate cleaner CONTENT stream for its K/V — extends the existing
   observer_split (which already splices observer-labelled K/V); ~2× attention on fractional rows,
   NOT a full 2nd forward. Makes the whole DC/low-pass patch unnecessary.
2. **EXACT SELF-SIDE ESTIMATE (knob-free brightness kill, keeps Request A):** a SECOND model eval
   per step at the honest `σ_obs` label (swap the pooled w→m label, re-eval) gives the EXACT
   content-consistent `denoised_obs`; then frequency-split
   `denoised_final = highpass(denoised_srow) + lowpass(denoised_obs)` → DC canceled EXACTLY at any
   kernel; kernel then only trades mid-freq. Cost ~2× forward on steps with fractional rows;
   moderate plumbing (schedule_tail / make_pooled into the step). Opt-in gate.
3. **DYNAMIC / AUTO KERNEL:** scale `k` with Δ or step, or auto-pick. Cheap, but only optimises the
   blunt knob against the estimate floor → cannot reach zero brightness. Modest refinement, not a
   root fix.

Current shipped patch state: `_lowpass_debias` in `_euler_step`, env `H3_DC_DEBIAS` (on/off) +
`H3_DC_LOWPASS` (kernel; 0 = DC mean, small = stronger toward ghost). `k=4` GPU "pretty solid" but
with the residual brightness floor above.

## Falsification

If the DC de-bias does NOT clear the tint at all, OR no low-pass `k` reduces the single-frame pop
without introducing ghost/softening, the residue is not cleanly frequency-separable from the denoise
bite → the second-stream fix (FORWARD OPTION 2 in [gpu-result.md](gpu-result.md): a separate observer
content stream for K/V, keeping physical content at `σ_row`) is the real fix.

Status: DC de-bias GPU-tested (partial win); low-pass extension now GPU-tested too — `k=4` "pretty
solid" but a residual brightness FLOOR remains (first-order estimate floor, see above), so route #1
or #2 is the real fix, not a bigger/auto kernel (#3). Prototype code is `# pragma: no cover`; user
pushes `--no-verify` and GPU-verifies.
