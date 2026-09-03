<!-- provenance: status (candidate ranking after GPU 2026-08-31 overturn; PRIMARY re-ranked above
     ancestral ρ-drift; SECONDARY is former #1). -->
<!-- verified: PRIMARY isolation = GPU 5-step our-node vs official; SECONDARY = analytical;
     RULED OUT grounds = official-clean-our-noisy delta, not euler-clean claim. -->
# Remaining candidates (re-ranked after the overturn)

Parent: [../residual-accounting.md](../residual-accounting.md)

**PRIMARY (new #1) — deterministic per-row injection/remap error, BOTH modalities.**
Present in our euler @5 steps, absent in official euler @5. This is the raw per-row injection on
fractional rows (video + audio), unnaturalized. `_euler_step` applies zero corrections, so this is
the fully-exposed injection path. This is now the primary source — promoted ABOVE the ancestral
ρ-drift. The unified carry correction ported into `_euler_step` (Fable re-derivation, in progress)
is the intended fix.

**SECONDARY (was #1) — cross-interval ρ-drift in the ancestral write.**
The v4 update (sampler.py:479-492) uses stock RF bookkeeping (`alpha=1−σ`,
`ratio=sigma_down/σ_c`); the truthful carried frac-audio trajectory has clean coeff
`(1−σ_c)/ρ_true` with `ρ_true` STEP-DEPENDENT (≈S early → 1 late), so a step slightly
UNDER-plants clean, the model re-corrects, and ancestral renoises the difference — a persistent
per-step error. Candidate **"v7"**: exact `ρ_eff` cross-interval correction (`ρ_eff≡1` at m=1,
finite at terminal, same frac-audio gate). Now SECONDARY: it is an ancestral-only amplifier on
top of the deterministic error, not the root.

**DIAGNOSTIC — eta=0 on fractional audio rows.**
Kills ancestral stochasticity on the ramp; would silence anything renoise-amplified, isolating the
deterministic component, at cost of a texture seam.

**INHERENT FLOOR — DiT/ancestral estimate floor.**
Each eval's `A_hat` on ramp rows carries irreducible error; not sampler-correctable without giving
up stochasticity. Consider only after the deterministic injection error is corrected.

**STILL RULED OUT (on new grounds).**
VAE decode overlap + held/ramp seam and Bug E remain ruled out NOT by "euler clean" but because
the OFFICIAL euler node @5 steps is clean while OURS is not — the artifact is OUR-node-specific,
so any node-independent decode/seam mechanism cannot be the cause.
