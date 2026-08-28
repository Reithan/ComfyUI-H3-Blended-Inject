<!-- provenance: reference (experiment pointer table — RES, DATA, HYP, early HOLD; child of experiments-run.md) -->
<!-- verified: 2026-08-25 · cross-checked against home docs @proto-latent-hold-release -->
# Experiment Run — Early Series (RES, DATA, HYP, HOLD-2 through HOLD-14)

Child of [experiments-run.md](../experiments-run.md). Results live in the linked home docs.

| ID | config (res · anchor · d/m · hold · sampler) | one-line result | home |
|---|---|---|---|
| RES-1 | 1MP · r40/r60 · d=0.5/0.45 · — · euler/10-step | POP; source-identical self-reconstruction (also popped @40 steps → step-independent) | [resolution-ladder.md](../highres-singleframe-underdenoise/resolution-ladder.md) |
| RES-2 | 0.1MP · r40/r60 · d=0.5/0.45 · — · euler/20-step | POP; chaotic (incoherent departure, opposite failure mode from RES-1) | [resolution-ladder.md](../highres-singleframe-underdenoise/resolution-ladder.md) |
| RES-3 | 0.2MP · r40/r60 · d=0.5/0.45 · — · euler/20-step | SMOOTH ✓; goldilocks; R non-discriminating across all three | [resolution-ladder.md](../highres-singleframe-underdenoise/resolution-ladder.md) |
| DATA-1 | 0.2MP · r40/r60 · d=0.75/0.70 · — · euler/20-step | r40 CLEAN; r60 SMEAR; window top-edge ~0.72, content-fuzzy | [data-runs.md](../highres-underdenoise-model/data-runs.md) |
| DATA-2 | 0.5MP · r40/r60 · d=0.50/0.45 · — · euler/20-step | Both seams COHERENT; anchors MESSY; resolution-ordered dissociation confirmed | [data-runs.md](../highres-underdenoise-model/data-runs.md) |
| DATA-3 | 1MP · r40/r60 · d=0.75/0.78 (+ lock walls 0.45/0.60/0.68; chaos 0.83) · — · euler/20-step | Window CLOSED @1MP; smear at 0.75/0.78; lock ≤0.68; chaos @0.83 | [data-runs.md](../highres-underdenoise-model/data-runs.md) |
| HYP-1 | 1MP · r40/r60 · d=0.70/0.66 (α=√5 map) · — · euler/40-step | Undershot (R≈0.28); R convex; baseline d=0.5 R=0.18; basin not escaped | [hypotheses-and-data.md](../highres-singleframe-underdenoise/hypotheses-and-data.md) |
| HYP-2 | 1MP · r40 · d=0.95 · — · euler | Neighbors FOLLOW f40; basin escape; slightly too high → sweet spot [0.8, 0.9] | [hypotheses-and-data.md](../highres-singleframe-underdenoise/hypotheses-and-data.md) |
| HYP-3 | 1MP · r40 · d=0.70 · — · euler (per-step trace) | `\|inp\|` dips then rebounds to 0.7058; self-reconstruction fixed point, deterministic | [temporal-and-contagion.md](../highres-singleframe-underdenoise/temporal-and-contagion.md) |
| HYP-4 | 1MP · f40@d=0.95 / f204 unchanged · — · euler | f204 blended better than its own d; cross-inject contagion confirmed | [temporal-and-contagion.md](../highres-singleframe-underdenoise/temporal-and-contagion.md) |
| HOLD-2 | proto · r40 · m=0.5 · hold=0.6 · euler/linear_quadratic | Residency CONFIRMED: anchor bit-identical clean through all 12 hold steps | [mechanism-and-early-findings.md](../latent-hold-release/mechanism-and-early-findings.md) |
| HOLD-4 | proto · r40 · m=0.5 · hold 0.0 vs 0.6 A/B · euler | Hold OFF → blends; Hold ON → hard cut; attraction is baseline, no hold needed | [attraction-and-envelope.md](../latent-hold-release/attraction-and-envelope.md) |
| HOLD-5 | proto · r40 · md=0.0 / none-interp · — · euler | Frozen clean m=0 anchor DOES attract; OBS-B valid (no +0.5 trap under none-interp) | [attraction-and-envelope.md](../latent-hold-release/attraction-and-envelope.md) |
| HOLD-9 | proto · r40 · md=0.0 · hold=0.0 vs 0.5 A/B · euler/20-step | hold=0.5 armed 1.08M elems (opening fade-out, not keyframes) → no attraction | [hold-mechanism-and-confounds.md](../latent-hold-release/hold-mechanism-and-confounds.md) |
| HOLD-11 | 0.5MP · r40+r60 · md=0.5 · hold=0.5, provenance fix · euler/20-step | Armed 1.08M→97,920; r40 blends; r40 under-denoises (Finding 12) | [hold-mechanism-and-confounds.md](../latent-hold-release/hold-mechanism-and-confounds.md) |
| HOLD-13 | 0.5MP · r40 · m=0.99 · hold=0.5 · euler/20-step | r40 "looks like ~0.5 denoise"; neighbor contagion shapes realized output | [anchor-denoise-m-vs-res.md](../latent-hold-release/anchor-denoise-m-vs-res.md) |
| HOLD-14 | 0.5MP · r40 still-repeat + fade · m=0.5 · hold=0.5 · euler | Seam fixed; anchor smears own artifacts; model reads as FREEZE-FRAME → not viable | [anchor-denoise-m-vs-res.md](../latent-hold-release/anchor-denoise-m-vs-res.md) |
