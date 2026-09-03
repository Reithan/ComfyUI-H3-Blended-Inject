<!-- provenance: confirmed (CPU design-level — pooled content-blind leak-removal; GPU listening test pending) -->
<!-- verified: 2026-09-02 (branch fix-euler-ancestral-per-row-renoise) · CPU probes mid_m_leak_probe.py + mid_m_pooled_probe.py -->
# CPU-probe results (2026-09-02) — pooled leak-removal is the fix

Parent: [mid-m-denoiser-leak.md](../mid-m-denoiser-leak.md) (mechanism + fix candidates).
Probes: `.claude/tmp/mid_m_leak_probe.py` (single-row) and
`.claude/tmp/mid_m_pooled_probe.py` (group-pooled, 3 seeds).

**Per-row instantaneous projection is DEAD.**
Estimating λ from a single audio row's `⟨Ĉ,εc⟩` (real N=64 = 32ch×2spec per tick)
is content-biased and stuck: CPU probe at N=64 gives residual ~0.30 regardless of row count,
content-corr −0.3..−0.4. Retired as a candidate (supersedes Fix A/B's per-row projection).

**Pooled (group-averaged) projection is content-blind and works.**
Pool `⟨Ĉ,εc⟩` across R independent rows sharing an (s,g) bin:
the content cross-terms average toward zero as 1/√(N·R) while the common λ survives.
`mid_m_pooled_probe.py` recovers the euler floor (~0.012) once N_eff ≳ 4k:
R=64→~0.037, R=256→~0.016, R=1024→~0.008, content-corr→0.
This DE-SCOPES the fix: it is group-AVERAGING —
NOT a least-squares parametric machine, NOT a two-run calibration.

**Real bin occupancy** (round-11 log `h3bi_c2_debug-normal-3.csv`):
logged `n` counts are exact multiples of 64, so n/64 = audio ticks (rows) per k_d bin per step.
Most k_d bins are rich (6–13 rows, 384–832 elements → N_eff 6k–13k → pooled lands at floor).
BUT the sparse mid bins are thin: the WORST-static bin k_d=13 (m=0.34) has only 1 row (64 elems),
k_d=8 has 2. A single-exact-k_d pool therefore cannot fix exactly the worst bin.

**Fix form (simplest landing).**
λ is smooth in σ_c, so pool over an s-NEIGHBORHOOD (kernel-weighted average over σ_c across
adjacent k_d) so the sparse worst bin borrows rows from neighbors.
Fix = at each step compute one content-blind kernel-smoothed λ̂(σ_c) from pooled `⟨Ĉ,εc⟩`,
subtract `λ̂·σ_c·εc` from Ĉ before recomposition.
Live single run, self-calibrating (removal-on ⇒ no accumulation ⇒ the live projection IS the
instantaneous λ). fresh term / η / r_ret untouched; m=1 bit-exact via the `frac_audio` gate.
~15 lines at the `_c2_audio_ancestral_update` site.

**Decision (2026-09-02):** user approved building a MINIMAL prototype-style toggle for this
pooled-removal, to be GPU-verified in one run (flag on vs off, listen for mid-fade static).
Durable regression tests only AFTER a real run earns it.
