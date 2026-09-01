<!-- provenance: status (C2 ρ fix paths — INDEX; input-side @63b291e falsified; v3 σ_c÷ρ_true @f06a84a
     + v4 exact-residual @12ea3b6 GPU-CONFIRMED = CURRENT BEST; v5 init-composite @7436165 GPU-FALSIFIED
     + REVERTED @8c8cb90; v6 m=0 audio-context heat @02fee22 IMPLEMENTED, GPU-pending, native-confirmed,
     NOT video-byte-identical; DURABLE PORT DECIDED 2026-09-01 PR #32) -->
<!-- verified: 2026-08-29 · split from c2-rho-fix-paths.md (over budget). Children hold history/falsifications and the current fix chain. -->
# Consequence-2 (ρ) fix paths — index

Parent: [audio-carry-identity.md](../audio-carry-identity.md) (the ρ error and its σ_c-axis
correction). This folder holds the fix-path detail; GPU results are noted per commit.

## Through-line
The C2 ρ error is a per-row carry-compression scale error on audio rows, audible only under
euler_ancestral amplification. The fix converged in stages after the input-side pre-comp was
falsified (63b291e): σ_c-axis projection ÷ρ_true (v3) and exact-residual cancel (v4, `denoised_r`-only,
CURRENT BEST). The C2 ρ/residual work is COMPLETE; a faint remainder persists. v5 (init-composite
bootstrap) was GPU-FALSIFIED + REVERTED → residual is NOT init-driven. v6 (m=0 audio-context heat,
model-input-only, native-confirmed) is IMPLEMENTED + GPU-pending; unlike v3–v5 it is NOT
video-byte-identical (corrected m=0 audio context flows to video via joint A/V attention by design).

## The σ_c axis (unifies all paths)
The `σ_c` axis ([audio-carry-identity.md](../audio-carry-identity.md#consequence-2-per-row-compression-breaks-the-carry-0--m--1))
UNIFIES the fix: both the renoise integration sigma and the clean correction sit on
`σ_c = sig_row·carrier/sig_g`, not on σ_v.

## No held phase / no release event (reframe)
The schedule-tail remap has NO held phase and NO release discontinuity. `row_sigma` reads the
dense-grid index `k_d·(steps−i)+i·steps`, so a fractional row starts at `σ(k_d)` at STEP 0 and
integrates its stretched tail over ALL steps (module docstring). The only truly frozen rows are m=0
(renoise_coeff/ratio → 0; verified zero spurious ancestral noise). Any earlier "held-then-released"
wording is wrong — there is no release discontinuity to chase.

## GATE (resolved 2026-08-29)
The earlier "do NOT build until a pure-euler artifact appears" gate was TOO STRICT — C2 is
real/present in `denoised_r` yet only *audible* under ancestral amplification, so the work was
(correctly) built against the euler_ancestral ramp artifact.

## Option axes (context)
- **A. σ̃ / `sig_row_c`:** recompute the ancestral renoise terms on a carry-consistent integration
  sigma for audio rows. Blast radius = fractional-audio-ancestral only; m=1 bit-exact, video
  byte-identical. May under-fix if the dominant error is in `denoised_r`'s clean coefficient.
- **B. `denoised_r`/ρ hack — SUPERSEDED** by v3/v4 (see history child).
- **eta=0 on fractional audio rows (nuclear fallback):** kills ancestral stochasticity there;
  guaranteed clean but coarse (texture seam vs surrounding free audio).

## Children
- [history-and-falsifications.md](history-and-falsifications.md) — baseline, prototype B, and the
  FALSIFIED + root-caused input-side pre-comp (63b291e).
- [current-fix.md](current-fix.md) — the v3 σ_c fix, v4 exact-residual cancel (current best), v5
  init-composite bootstrap (falsified), and v6 m=0 audio-context heat fix (implemented, GPU-pending).
- [residual-accounting.md](residual-accounting.md) — full accounting of the very-quiet hum left after
  v6, ranked remaining candidates (ρ-drift v7 / eta=0 / DiT floor), and the 2×2 discriminator plan.
- [observed-level-plant.md](observed-level-plant.md) — **theory (UNVERIFIED):** close Δ on the
  CONTENT side (plant injected noise at observed `m·σ_g`, keep self-evolution at `σ_row`); DUAL of
  #81; discriminates the decoupling-residue vs r-lerp fork.
- [stock-mask-remap-port.md](stock-mask-remap-port.md) — **status + confirmed:** `H3RescaleNoiseMask`
  node + least-squares helper that rescale a `noise_mask` so the STOCK sampler reproduces our curved
  `σ_row`; discriminator (d) in residual-accounting.md (NOT a decoupling test).

Native precedent for a wrapper-side variant: `scale_latent_inpaint` pre-divides injected clean audio
by `(σ_v/σ_a)/S` for the same reason.

## Durable port (2026-09-01)

GPU PARTIAL result on content-axis A/B (PR #32 commits 4644fcf/e4a9940) revealed residual is
the C2 carry-compression error (Fable round-5). User decided to port the exact generalized
correction into the durable `_euler_ancestral_rf_step`, folded into PR #32. This reverses the
"C2/ancestral experiments stay proto-only" policy from the durable-base cut. Full spec:
[../euler-ancestral-per-row-fix/c2-durable-port.md](../euler-ancestral-per-row-fix/c2-durable-port.md).
