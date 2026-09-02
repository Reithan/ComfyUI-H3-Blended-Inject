<!-- provenance: bug (FALSIFIED as fade-audio cause GPU 2026-09-01; SUPERSEDED round 9 — σ_v plant untruthful under C2, being reverted; historical record preserved, do not delete) -->
<!-- verified: 2026-09-02 · round-9: PLANT_AXIS "v" is untruthful under C2 σ_c bookkeeping and is being reverted to "row" -->
# Plant-axis fix — shipped record & GPU falsification

**⚠ SUPERSEDED (round 9, 2026-09-02):** `PLANT_AXIS = "v"` is untruthful under C2's σ_c
bookkeeping. The σ_v plant was self-consistent with the old per-step σ_v update; it became
an error when C2 moved bookkeeping to σ_c. PLANT_AXIS is being reverted to "row".
See [plant-over-noise.md](plant-over-noise.md) for the full root-cause analysis and fix.

Shipped fix (PR #32 original, commits 3e82dba + e3ec742): planting fractional rows on the σ_v
INTEGRATION axis for the ancestral init-step. FALSIFIED as the fade-audio hiss cause by GPU
2026-09-01 (single-frame clean; fade-region hiss PERSISTS). Historical record preserved as a
confirmed negative result.

Read alongside [content-axis.md](content-axis.md) — the current leading candidate that supersedes
this for fade-audio.

## GPU falsification (2026-09-01)

Single-frame injects: VIDEO ghost FIXED, audio co-location FIXED (consistent with Bug F attribution).
Fade-region audio hiss PERSISTS with at most minor change. Onset smooth, not abrupt. Loudest
MID-FADE; vanishes as m→0 and m=1. Pattern is m-dependent → CONFIRMS the defect lives in a path
ACTIVE only during fractional steps (fade band, 0<m<1).

**Why "minor change" is expected:** the plant fix is a ONE-TIME i=0 init. The fade-audio hiss is a
PER-STEP coherence defect, refreshed every sampler step → a one-time plant cannot touch it.

## Bypass sub-theory REFUTED (source-read, 2026-09-01)

A plausible worry: does the observer side-stream band K/V even reach the band's OUTPUT, or does the
side-stream only prime K/V that the band never reads?

Source-read of `observer_split.py` in full REFUTES bypass:
- The band's denoised OUTPUT comes from the MAIN stream: `x = _mod_gate(h, gmsa, main_out, segs)`
  at observer_split.py:449; `main_out = attn.out_proj(out[:s])` at :383.
- The side stream writes only band K/V: `k[pos]=k_m; v[pos]=v_m` at observer_split.py:343-344.
- The band's QUERY attends to those K/V → the content DOES reach band output via attention.
- Therefore the plant DID reach fade audio content; its falsification stands on per-step-vs-one-time
  grounds, NOT bypass.

## Shipped plant-axis fix details

In `sampler.py`, the init-plant at loop `i==0` (`x_cur = w_plant·x_cur + (1−w_plant)·clean`) uses
`w_plant = row_sigma_v(0)/sig_v[0]` for the ancestral step. Model LABEL/pooled `w` stays σ_a.

Keyed per step-fn: `DEFAULT_PLANT_AXIS = "row"` (module const) +
`_euler_ancestral_rf_step.PLANT_AXIS = "v"`. `euler` keeps the σ_a-ratio plant (GPU-validated).

CPU-verified properties (`tests/test_sampler.py::TestAncestralPlantAxis`):
- Video rows byte-identical (sig_row_v == sig_row for video).
- m=1 → w_plant=1 → no-op → stock bit-identity + noise-draw sequence preserved.
- m=0 → w_plant=0 → exact-preserve intact.
- Terminal flush unchanged.

## Root cause: init-plant axis incoherence (closed-form — now SECONDARY)

Trained contract (Fix A GPU result at m=1): packed audio CONTENT carries noise on the σ_v axis;
LABEL is σ_a = shift(σ_v). Post-Fix-A, label and integration are coherent — but the plant used
the σ_a ratio `w = sig_row/sig_g`, dropping fractional audio content at RF level
σ_row_a ≈ σ_row_v/4 (shift compresses ~4× for small σ; gap peaks mid-grid).
From step 0 the model is told (via label) to expect σ_v-sized noise in a row carrying σ_a-sized
noise → systematic v̂ mis-estimate on fade-band audio ticks, re-excited each step.

GPU falsification shows ONE-TIME plant fix produced at most minor change → the PER-STEP content-axis
defect dominates. See [content-axis.md](content-axis.md).

## Bug B mechanism refinement (2026-09-01)

The per-row ancestral algebra (`renoise_coeff`) is EXACT and level-preserving given accurate v̂ —
**not a coeff defect**. Retention = VELOCITY-ESTIMATION ERROR (`x0̂ = x0 + σ_row·(v − v̂)`)
re-excited each step by fresh ancestral injection. Deterministic euler makes the same error but
never re-excites it. For fractional AUDIO specifically, the systematic v̂ error source is the
content-axis wiring defect (not the init-plant), re-excited each step.

## Rejected alternatives

(a) Rescaling `renoise_coeff` — provably inexact: current coeff is the unique level-preserving
value; smaller under-noises, larger over-noises.
(b) Reprojecting noise onto carrier axis — injects carrier-sized noise into a σ_row-level row.
(c) Eta-gating fractional rows to deterministic — user-rejected (disables stochasticity).
(d) Explicit-noise bookkeeping (subtract known injected ε) — denatures ancestral sampling.
