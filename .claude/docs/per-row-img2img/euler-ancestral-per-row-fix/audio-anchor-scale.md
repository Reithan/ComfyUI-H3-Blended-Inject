<!-- provenance: theory (GPU-FALSIFIED for peak — anchor introduces muffling; fix retained as model-scale-correct; root cause in plant-over-noise.md) -->
<!-- verified: 2026-09-02 (branch fix-euler-ancestral-per-row-renoise) · round-9 GPU runs falsify anchor as peak cause; anchor fix fe0343a+91078cc retained (K/V content scale correct) -->
# Audio observer band: clean anchor scale bug — round 8

**Round-9 verdict:** anchor fix FALSIFIED for the 0.75–1.0 s peak; it introduced muffling by
removing compensation for an over-noised row. Root cause identified as `PLANT_AXIS = "v"` making
i=0 plant untruthful under C2's σ_c bookkeeping. Fix = revert PLANT_AXIS to "row".
See [plant-over-noise.md](plant-over-noise.md) for the full round-9 record.
The anchor correction itself is RETAINED (K/V content amplitude is model-scale-correct;
the over-noise is a separate, more fundamental error).



Continuation of the C2 durable port residual investigation.
Context: [c2-durable-port.md](c2-durable-port.md) (C2 port, crackle residual corrected localization).

## Round-8 GPU datum (CONFIRMED, 2026-09-01)

Envelope changed from 0/0/49/73 (25-frame fade-out, ≈1.04 s @24 fps) to 0/0/0/90
(91-frame fade-out, keyframe at frame 0 only), same seed, euler_ancestral, C2 port +
content-axis ratio in place, audio splice ON, curve ease_in_out, min_denoise=0.

Result: crackle "still relatively short, but not single-frame"; contained ≈0.75–1.00 s
on the timeline. Frames 18–24 of the 91-f fade → fade_t ≈ 0.21–0.27 → ease_in_out
m ≈ 0.11–0.18.

Duration scales ≈ proportionally with fade length (1 tick @25-f fade → ≈6 frames @91-f
fade). Residual is a BAND of m (Δfade_t ≈ 0.06) at LOW m, not a point/grid mechanism,
and not m≈0.5. The earlier "mid-fade point" localization was too coarse — 1 s fade at 24 fps
cannot resolve a 6-frame-wide band.

## Fable round-8 verdict (theory, GPU-UNVERIFIED as of writing)

Analysis ruled out all C2 packed-axis singularities at m≈0.1–0.2:
- `sig_row < sig_g` gate is strict for every i<steps (no boundary flip).
- `_observer_timestep` pin t_pin_a=1.0 never reached for m>0.
- `_embed_ratio` never saturates: m=0.15 runs 0.61→0.92.
- 3-dense-step tail: smooth resample; σ_c'<σ_c and r_ret<σ_c' everywhere.
- ε̂ sensitivity a/σ_c=0.47 at m=0.15 self-cancels; coefficient ≈0.016.

**Top candidate: audio observer band clean anchor `h_clean` is S× too hot.**

The one-time embed capture (`sampler.py` embed_capture block) feeds
`model(clean, sigmas[0]·s_in0)` the PACKED clean where audio = audio_scale·A = 4A.
The DiT forward multiplies audio by carry = σ_a/σ_v (`comfy/ldm/minimax/model.py`:527-538),
which is 1.0 at σ_v=1, so `h_clean` on the audio band = embed(4A) while the network's
clean audio scale is A. Native `scale_latent_inpaint` (`model_base.py`:2257-2265) corrects
this with factor (σ_v/σ_a)/audio_scale — the same correction the v6 never-row fix used
(GPU-confirmed, never ported to durable).

The band blend `r·h_main + (1−r)·h_clean` (`observer_split.py` `_blend_hidden`) gives
clean coefficient (1−r)·S + r(1−s) in model space — off the RF manifold by (1−r)(S−1):

- i=0, m=0.15: off-manifold factor ≈1.17; band clean ≈2.3× target.
- i=0, m=0.50: off-manifold factor ≈0.40; band clean ≈1.5× target.
- i=0, m=0.90: off-manifold factor ≈0.05; band clean ≈1.24× target.

Band-local (→ time-localized), per-step persistent, largest at low m, vanishing at m→0
only because those rows carry almost no ancestral noise budget. The product (heat)×(noise
budget) peaks at m≈0.1–0.2, extent ∝ fade length. Deterministic — euler sees the same
K/V (same objection v6 faced; v6 still improved ancestral only).

## Candidate ranking (post round 8)

1. **Hot audio band anchor** (NEW) — band-local, per-step, low-m weighted, ∝ fade length.
2. **Observer residue 1a** — relative gap σ_row/mg 2.3× at m=0.15 vs 1.5× mid; also
   low-m-weighted but on-manifold.
3. **v6 never-row heat** — not band-local; weakest in a 91-f run (≈2 never rows); port later.
4. **Estimation floor 1b** — weakest; only after 1–3 cleared.

Splice opt-out is no longer the sharpest first run — it removes (1) and (2) together and
costs the audio splice.

## Implemented (commits fe0343a + 91078cc, 2026-09-01, durable)

`sampler.py`: new `_clean_at_model_scale(clean, audio_mask, sigma_v, sigma_a, audio_scale)`
returns `where(audio_mask, clean·((σ_v/σ_a)/audio_scale), clean)`; identity when no audio
or audio_scale==1. Embed capture now calls
`model(_clean_at_model_scale(clean, audio_mask, sig_v[0], sig_a[0], shift_v/shift_a), sigmas[0]·s_in0)`.
At sigmas[0]=1 the factor is exactly 1/S. Video slice untouched.
Tests: `tests/test_sampler.py::TestCleanAtModelScale` (4 tests). Suite 662 pass.

`observer_split.py`: `H3BI_SPLICE_AUDIO=0` env opt-out (`SPLICE_AUDIO_ENV`,
`_splice_audio_enabled()`, gated in `install_observer_split`).
Tests: `tests/test_observer_split.py::TestSpliceAudioEnv`. Suite 662 pass.

## GPU plan (Fable, round 8)

**Run 1:** 0/0/0/90, same seed, euler_ancestral, splice ON, anchor fix applied.
Prediction: crackle at 0.75–1.0 s GONE; possible slight fade-audio timbre/level change
(band no longer over-confident); video NOT guaranteed byte-identical (band K/V flows through
joint attention).

**Run 2 (if Run 1 fails, or as confirmation):** `H3BI_SPLICE_AUDIO=0`, same settings.
- Gone in Run 2 but not Run 1 → observer residue 1a; revert `_audio_observer_ratio` to
  native m·σ_a/σ_row_a (exact with corrected anchor).
- Persists in both → main-stream: port v6 `x_model` for never rows; run steps 40
  (floor 1b shrinks with step count, heat does not).

**Optional cheap datum:** curve=linear, same run. Every candidate is a function of m, so the
band should shift earlier (m 0.11–0.18 → frames 9–16 → 0.4–0.7 s). Stays at 0.75–1.0 s
→ timeline/grid-driven → all candidates wrong.

Deferred follow-ups unchanged: `PLANT_AXIS` revert to "row" (principled cleanup under C2);
v6 never-row port as separate PR.
