<!-- provenance: status (C2 ρ fix — history: baseline, prototype B superseded, input-side pre-comp @63b291e GPU-FALSIFIED + root-caused) -->
<!-- verified: 2026-08-29 · input-side σ_c pre-comp @63b291e GPU-FALSIFIED (buzz LOUDER) + ROOT-CAUSED (Fable, comfy-ref model.py:535-550 / model_sampling.py:90-92): FALSE-REFERENCE in the input perturbation. -->
# C2 ρ fix — history and falsifications

Index: [index.md](index.md). Current fix chain: [current-fix.md](current-fix.md).

## B. `denoised_r`/ρ hack — SUPERSEDED
Correcting the audio slice of `denoised_r` by `1/ρ` per-row was a first-order approximation; on the
σ_v axis it only *reduced* the GPU buzz (commit 9877350). Superseded by v3/v4.

## FALSIFIED + ROOT-CAUSED — input-side σ_c pre-comp (commit 63b291e)
GPU 2026-08-29 (`0/0/49/73`, euler_ancestral): buzz got LOUDER — a REGRESSION vs prototype B
(9877350). It built `u = x_prev + coeff·clean` (piece 1) and inverted forward's output transform
exactly (piece 2, `denoised_r = S·(k·u − sig_row·v_dit)`). Fable source-derivation (comfy-ref
`model.py:535-550`, `model_sampling.py:90-92`): piece 2 was bit-exact; the fault is piece 1's
**FALSE-REFERENCE** input perturbation — `clean_packed = S·A` is the state's clean content ONLY at
step 0, so during the fade-out ramp the static subtraction injects an incoherent residual
`Δ_in = coeff·S·(A−A_est)` into the DiT input every step, amplified by the `S·k≈4` gain + ancestral
renoise (positive feedback) and leaked into video via JOINT A/V attention. **Input-side
pre-compensation is fundamentally unsound** — it needs the model's own evolving estimate
(chicken-and-egg). REMOVED.

## Why v5's init composite is NOT this failure mode
63b291e perturbed the MODEL INPUT every step against a step-0 static reference. v5
([current-fix.md](current-fix.md)) corrects the i==0 init composite ONCE, using only known init
quantities — the `clean` tensor the composite already uses — with no evolving-estimate dependency
and no per-step input perturbation.
