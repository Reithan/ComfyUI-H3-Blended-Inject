<!-- provenance: theory (H1 CONFIRMED by elimination; standalone baseline question and firing order for investigation) -->
<!-- verified: 2026-08-24 · repo @debug-single-frame-underdenoise · 1MP run (row40@0.83=chaos, row60@0.45=lock) -->
# Baseline question and firing order

Index: [highres-singleframe-underdenoise](../highres-singleframe-underdenoise.md). Hypotheses that this isolates: [hypotheses-and-data](hypotheses-and-data.md) and [resolution-ladder](resolution-ladder.md).

## Q1 — how to get an objective standalone denoise-d baseline

**There is NO turnkey H3 img2img node.** The MiniMax H3 extension ships exactly five nodes
(`nodes_minimax_h3.py`): `EmptyMiniMaxH3LatentAV`, `MiniMaxH3ImageToVideo` (t2va/fl2va),
`MiniMaxH3AddGuide` (native anchor — re-injected/never-denoised, NOT img2img), `MiniMaxH3ReferenceToVideo`
(ref2va), `MiniMaxH3SigmaShift`. None takes an init latent + denoise<1. The generic ComfyUI img2img
(`VAEEncode → KSampler(denoise=d)`) does not work directly: the H3 model samples a
`NestedTensor((video, audio))` flat pack, and a plain `VAEEncode` emits `{"samples": video_tensor}`
with no audio stream → the H3 sampler path chokes on the missing audio latent. Building a stock
baseline means latent surgery (encode the frame, splice the video tensor into an
`EmptyMiniMaxH3LatentAV` NestedTensor, keep the zero audio stream) — a script, not a clean graph.

**Practical faithful baseline = run OUR node at `length=1` (single frame, NO neighbors),
`min_denoise=d`.** This exercises the exact three-lever compression path (definitional img2img on
the `m·σ` schedule) but removes all temporal coupling — so it isolates **per-frame
self-reconstruction** from the Q2 temporal-decoupling effect. Capture `|out−clean|` and the
displacement ratio `R=|out−clean|/|full-gen|`; compare to the multi-frame injected row (row 40=0.182,
row 60=0.166). If length=1 ALSO under-denoises @1MP → the per-frame SNR effect (H1 core) is real and
independent of neighbors. If length=1 denoises fine but the multi-frame row doesn't → the dominant
term is the *temporal anchoring* of Q2, not per-frame SNR. Either way it splits the mechanism with
one run and needs no new node. (A true stock-KSampler reference would additionally confirm it's H3
schedule semantics vs. our path, but requires the scripted latent splice above.)

## Is "correct denoise" objective? (bug vs. design — the baseline question)

"Denoise strength d" is **definitional, not perceptual**: it means "noise the source to schedule
fraction d, integrate the sampler to 0." So the objective ground truth for "denoise 0.45 @1MP" =
**what a standalone whole-frame H3 img2img at strength 0.45 @1MP produces** — no reference
resolution, no eyeball. Our per-row compression noises the row to `σ≈m` and runs `m·σ: m→0`, so it
**should be identically equal to true img2img at strength m** — a testable equality.

**CORRECTNESS TEST (do before calibrating):** run standalone H3 img2img on the single frame alone
(no inject nodes), 1MP, denoise 0.45/0.5; capture `|out-clean|`; compare to our injected row
(row 60=0.166, row 40=0.182).
- Standalone ALSO ≈0.166 → our node is FAITHFUL; under-denoise is **H3 schedule semantics**
  (strength-0.45 is genuinely weak @1MP for everything). No correctness bug → resolution-corrected
  m is a DELIBERATE UX improvement (perceptual knob), not a fix.
- Standalone ≈0.4 but ours 0.166 → **our compression under-denoises vs true img2img** = real bug in
  our path; fix that FIRST, before any calibration.

**What has NO objective formula:** "how much should the image visibly change at d=0.45?" —
perceptual-change-per-strength is schedule/shift/content dependent. Making d "feel like d" across
resolutions is a DESIGN preference; it becomes principled once you pick an invariant to hold (e.g.
displacement fraction `R=|out-clean|/|full-gen|` constant across res) and solve for m numerically.

## Firing order & instrumentation
H3's mid-chunk run first (splits space) → H1 drift logging (free, piggybacks) → H2 latent-row diff
(CPU). Instrumentation lives on branch `debug-single-frame-underdenoise` (`sampler.py`
`build_conditioning_wrapper` per-step `|den−inp|`/`|den|`/`|inp|` per tracked row; `nodes.py`
`_run_sampler` build-time latent stats + final `|out−clean|`). Unconditional on the debug branch —
NO env vars/flags (user directive). See [our-architecture](../our-architecture.md),
[differential-diffusion](../differential-diffusion.md).
