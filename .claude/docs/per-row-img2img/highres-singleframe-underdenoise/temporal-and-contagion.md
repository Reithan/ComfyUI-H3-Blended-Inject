<!-- provenance: theory (H1 CONFIRMED by elimination; Q2 temporal decoupling explained; cross-inject contagion GPU-observed 2026-08-24) -->
<!-- verified: 2026-08-24 · repo @debug-single-frame-underdenoise · 1MP run (row40@0.83=chaos, row60@0.45=lock) -->
# Temporal decoupling and cross-inject contagion

Index: [highres-singleframe-underdenoise](../highres-singleframe-underdenoise.md). Resolution/mechanism findings: [resolution-ladder](resolution-ladder.md). Baseline and firing order: [baseline-question](baseline-question.md).

## Q2 — "the keyframe sits OUTSIDE the process" (temporal decoupling — enriches H1)

User @1MP, watching the preview per step: neighbors (n−1, n+2) start blurry and resolve
**together, matching each other**, using the keyframe as guidance; the keyframe (n) starts ≈source
and **stays** (at 0.5) or is "slightly smeared, resolves slower/if at all" (at 0.7). It looks like
"n−1 and n+2 use n as guidance but only match each other, while n sits outside the process." At
0.2MP the keyframe looks noisy early and its blur resolves **jointly** with the neighbors.

**This is H1 seen in the temporal domain — but it is NOT the native-keyframe mechanism** (user
correctly distinguished the two). The distinction is the crux:

- **Native FL2VA keyframe** (`minimax_keyframes`): an EXTRA clean row **appended to the attention
  sequence** (`model.py:340-355`) at the target frame's exact RoPE coord
  (`cond_t = cursor + FRAME_RESCALE·resolved_frame_index`), injected near-clean
  (`aug≈VISUAL_COND_TIMESTEP≈0.999`, `_cond_video_rows` model.py:499-511), flagged
  **`img_update=zeros → update=False → never denoised**. The sampled latent row at that frame is
  still noise and DOES denoise, but it + neighbors attend to that co-located clean token and are
  **pulled toward it → strong ATTRACTOR → neighbors blend INTO the keyframe → smooth transition.**
  FL2VA keyframes are conditioning context, NOT written into the sampled latent (the latent stays
  empty — `nodes_minimax_h3.py:137,159`).
- **Our inject:** we write the frame **into the denoised latent row itself** (init-lerp,
  update=True), with **NO co-located clean cond token**. At 1 MP the row self-reconstructs to source
  and goes **inert — but it is NOT an attractor**: nothing pulls the neighbors toward it, so they
  denoise their own coherent trajectory and match *each other*, ignoring the inert frame → it "sits
  outside the process" and pops. At 0.2 MP the row carries real per-band noise, participates in the
  joint denoise, and resolves *with* the neighbors (the "resolve together" the user sees).

So a native keyframe **doesn't denoise but ATTRACTS**; our high-SNR frame **doesn't denoise and
doesn't attract** — same "frame doesn't move," opposite blend effect. That asymmetry is exactly why
it "looks different than FL2VA." **Neighbor bleed (fact D)** here is NOT neighbors-pulled-to-anchor
(there's no attractor); it's residual + VAE decode-overlap (H2) skewing ±2 frames @1MP only.

**Alt-fix lead (speculative, unverified):** a HYBRID inject — keep the latent lerp for the strength
knob AND add a *faint* native `minimax_keyframes` cond-row attractor at the same frame so neighbors
actually blend toward it. A full-strength cond row = native FL2VA (≡ min_denoise 0, not img2img-at-d),
so it'd need to be weak/scaled — untested whether a fractional attractor + fractional lerp compose
cleanly. See [conditioning-row-inject.md](../conditioning-row-inject.md) (currently "different tool";
this reopens it as a possible complement, not a substitute). Resolution-corrected-m (H1 fix) remains
the primary path; this is a fallback if raising effective-m alone doesn't restore the blend.
- **0.7 undershoots.** "Slightly smeared, resolves slower/if at all" at the α=√5 map means the row
  is *near* but not *past* the self-reconstruction threshold — it still partly anchors. Signal that
  **α=√5 is a LOWER bound**; the true resolution correction is steeper (self-reconstruction is a
  soft threshold in SNR, not the √(token-ratio) power law alone). Recalibrate α from the kill-shot's
  displacement ratio `R`, don't assume √5.
- **REBOUND (0.7 run, live, row 40 `|inp|`):** fell through steps ~20-30 then rebounded to 0.7058 by
  step 38 — the same ~0.70 source-structure norm as the 0.5 run's 0.697. Fully deterministic: the
  sampler is plain `euler` (no per-step noise; only the ONE-TIME initial lerp at σ≈m·σ_max). The
  dip-then-rebound is the deterministic ODE trajectory of H3's rectified-flow field: early/mid the
  field carries the latent AWAY from source; as σ→0 the RF field collapses to the nearest
  data-manifold point, which for a barely-perturbed structured latent IS the source → trajectory
  curves back. Even at 0.7 the frame is **recaptured** → confirms **0.7 is below threshold and
  α=√5 undershoots.** The fix must push m past the point where mid-trajectory divergence escapes
  the source basin before σ→0 re-captures it.

## Cross-inject temporal contagion (user, 0.95 run — injects are NOT independent)

At 1MP, f40 (row40) raised to 0.95, f204 (row60) **left unchanged** at its lower d. Observed: because
f40's neighbors participated in its (strong) denoise, they **stayed noised longer**, and that fluid
region **propagated down the row axis and pulled f204 into blending too** — f204 blended better than
its own unchanged d would give in isolation (row40→row60 is ~20 latent rows / 68 pixel frames apart).

**Mechanism (user, VISUALLY OBSERVED — watch the preview across steps):** the timeline does NOT
denoise as one coherent block. a denoising **FRONT travels through the timeline**, resolving OUTWARD
from the points the model anchors/spends attention on. **Not necessarily a SINGLE front (user
clarification):** each early-attended/anchored section propagates its OWN front, and the fronts
INTERFERE like waves in a pool before the whole timeline settles — 1 anchor/1 wave (FF2VA/LF2VA),
2 waves (FL2VA), or more. Row coupling is those fronts' propagation:
- A frame **held UNRESOLVED longer (high denoise)** blocks the front from resolving *through* it,
  **holding the surrounding region open** longer → neighbors keep resolving in reference to it →
  blend propagates outward, and can reach a distant inject (f204, ~20 rows away) and hold it open
  long enough to participate too — even at f204's unchanged low d.
- A frame that **resolves EARLY (under-denoised / self-reconstructs to source at high SNR)** lets the
  front pass straight through; it never holds neighbors open, so they resolve on their own trajectory
  and it **pops** (no blend). ← THIS is the bug, restated in front terms.
So "escape the basin" ≙ "stay unresolved long enough for the front to have to resolve through you,"
which is *also* what makes neighbors blend in. Same row-coupling as fact D and the FL2VA-attractor
contrast. (Earlier "barrier that freezes neighbors" wording was backwards — corrected here.)

**Implications:**
- **Calibration is confounded when injects share a gen and differ in d.** Row 60's R in the α=√5
  kill-shot (both raised together) and in this run (only f40 raised) are NOT comparable. To get a
  CLEAN per-row R, isolate: **one inject per gen**, or hold ALL injects at the same d across compared
  runs. The `length=1` standalone (Q1) is the cleanest isolation.
- **Fix design:** a single escaped anchor can carry a region, so per-inject escape may be
  over-sufficient in dense-inject clips — but that's positional and fragile. The robust rule stays
  "push each inject past its own basin"; contagion is a bonus, not something to rely on.
- The 0.2MP TARGET-R run should therefore keep BOTH injects at their original d (0.5/0.45) exactly as
  the validated-correct baseline did — matching what "looked right," contagion included.
