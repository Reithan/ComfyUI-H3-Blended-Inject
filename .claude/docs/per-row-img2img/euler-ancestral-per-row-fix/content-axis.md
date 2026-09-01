<!-- provenance: theory (GPU PARTIAL 2026-09-01 — ring narrowed toward mid-m; euler UNCHANGED; residual stochastic-only; C2 durable port is next step per c2-durable-port.md) -->
<!-- verified: 2026-09-01 (branch fix-euler-ancestral-per-row-renoise) · GPU PARTIAL — ring narrowed mid-m; euler unchanged; C2 port next -->
# Content-axis observer fix — PR #32 revised (PARTIAL result; C2 durable port next)

Fable round-4 verdict and shipped fix for the residual fade-audio hiss that PERSISTED after the
plant-axis fix (GPU falsification 2026-09-01). Supersedes plant-axis.md as the primary candidate.

GPU PARTIAL (2026-09-01): ring narrowed toward mid-m rows, both ends clean; hiss shorter +
slightly quieter; euler fade UNCHANGED → no regression from shared prime_side_stream. Residual
is stochastic-only. See [c2-durable-port.md](c2-durable-port.md) — C2 correction is next step.

## Diagnosis: per-step K/V content mis-axis (Fable round-4)

The residual defect is NOT yet irreducible v̂-error territory. Fable's verdict: a PER-STEP
coherence defect, refreshed every sampler step. The observer side-stream (`prime_side_stream`,
sampler.py ~706-751) primes audio band K/V CONTENT at every step. That content was built on the
σ_a axis: ratio = m·σ_a read on the σ_a grid.

But post-plant-fix the audio x_prev content sits on the σ_v integration axis (matching stock
`sample_euler_ancestral_RF`). The trained audio contract (Fix A, m=1 GPU-validated) pairs label
`1 − s` with content at `shift⁻¹(s)` on σ_v. The native observer LABEL is `1 − m·σ_a`
(comfy-ref minimax/model.py ~585-610: `rows_t = (1 − m·sigma_a).clamp(max=t_pin_a)`) —
this MUST NOT change. The coherent CONTENT level is `shift⁻¹(m·σ_a)` on σ_v — the Möbius
shift inverse: `time_shift_sigma(m·σ_a, shift_a, shift_v)`.

**Why single-frame clean:** a single-frame inject at m=1 or m=0 never enters the fractional band;
K/V content is either full or frozen. The hiss is m-dependent, peaks mid-m, vanishes at both
endpoints → matches a content-level error proportional to `|shift⁻¹(m·σ_a) − m·σ_a|`.

## The fix: _audio_observer_ratio (PR #32 revised, commits 4644fcf+e4a9940)

New pure helper `_audio_observer_ratio` computes the audio observer band embed-blend ratio on σ_v:

1. `sig_row_band_v = _stream_row_sigma(mrow, i, steps_n, dense_v, sig_v, n_sig)` — row sigma on
   the σ_v grid.
2. `target = _shift_schedule(mrow*σ_a, shift_a, shift_v)` — Möbius shift inverse of m·σ_a.
3. `ratio = _embed_ratio(target, sig_row_band_v)` — interpolation ratio on the σ_v grid.

`prime_side_stream` audio branch calls `_audio_observer_ratio`; video branch unchanged.
LABEL unchanged (still `1 − m·σ_a` via `_observer_timestep`).

**Boundary correctness:** reduces to the old σ_a-axis ratio at m=0 (→0 freeze) and m=1 (→1);
diverges only mid-m where the hiss peaks. Fully stochastic-preserving — touches NO eta/noise math.

## Regression tests (CPU, PR #32 revised)

`tests/test_sampler.py::TestAudioObserverContentRatio` — 4 CPU tests:

1. m=1 → ratio = 1 (stock limit).
2. m=0 → ratio = 0 (freeze limit).
3. mid-m → ratio DIVERGES from old σ_a-axis ratio (confirms fix activates mid-fade).
4. content-target is shift-inverse round-trip (Möbius coherence).

93 sampler tests pass, ruff clean, diff-coverage gate passed.

## GPU result (2026-09-01) — PARTIAL

Content-axis fix is RETAINED (partial improvement, not the sole cause of the residual):

- Ring NARROWED toward mid-m rows; both ends clean. Hiss shorter + slightly quieter.
- Deterministic euler fade: UNCHANGED / CLEAN → no regression from shared prime_side_stream.
- Residual is stochastic-path-specific (ancestral only). NOT a v̂ floor (Fable round-5).

Residual root cause: C2 carry-compression error in ancestral x0 recovery (`denoised_r`,
sampler.py ~539-540). Durable branch carries the prototype ladder's BASELINE — no correction.
Full spec + decision: [c2-durable-port.md](c2-durable-port.md).

Remaining fallback chain if C2 port fails:
1. Audio-band splice opt-out: skip "audio" stream in install/prime; one-line discriminator.
2. If unchanged: declare v̂ floor.
(eta-gating remains REJECTED.)

## Euler regression — CONFIRMED CLEAN (GPU 2026-09-01)

Deterministic euler fade run same-seed after the content-axis A/B: UNCHANGED / CLEAN.
`prime_side_stream` shared path introduced no regression.
