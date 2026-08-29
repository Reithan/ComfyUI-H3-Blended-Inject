<!-- provenance: bug (open — genuine correctness defect, NOT the Bug E gate; present in both ERROR and CLEAN configs; source-verified 2026-08-28) -->
<!-- verified: 2026-08-28 · verified against sampler.py / observer_split.py source this session; fix design only, no code yet -->
# Bug E side-finding — "ghost-observer" boundary mismatch (open defect)

Parent: [long-fade-grid-beat.md](../long-fade-grid-beat.md).

A genuine correctness bug surfaced during the exhaustive CPU audit. It is NOT the Bug E binary gate
(present in BOTH error and clean configs), but it is a real defect worth fixing once the primary
gate is resolved.

## The mismatch

- **Sampler side (sampler.py:496–497):** `k_d = round(steps_n*(1-m)).clamp(0, steps_n)`; then
  `never = k_d >= steps_n`. The sampler hard-restores never-rows to clean at the final
  `where(never, clean, x_cur)`. For steps_n=20 this fires whenever m ≤ 0.025 (round(20*(1−m))≥20
  ⟺ m ≤ 0.5/steps_n).
- **Observer side (observer_split.py:69):** `_fractional_rows` uses `frac = (rows>1e-6) &
  (rows<0.999)`, which INCLUDES rows with m∈(1e-6, 0.025].

So a row the sampler freezes fully clean is simultaneously advertised to its DiT neighbors as a
fractional-denoise K/V observer under a near-preserve label. Neighbors receive incorrect K/V
context for a row that is actually sampler-preserved.

## Fix candidate (design only, no code yet)

Align the observer fractional test's low threshold to the sampler's `never` boundary: exclude
m ≤ 0.5/steps_n (the rows where k_d ≥ steps_n), so never-rows are treated as preserved observers,
not fractional.

## Confirmed boundary instance (2026-08-28)

The defect now has a concrete firing point. In the executed ekf=39→40 diff, row 11 flips observer
membership exactly at this boundary: at ekf=39 its q=1/256 (>1e-6) INCLUDES it in `_fractional_rows`
while the sampler freezes it clean (k_d=20, never=True); at ekf=40 q=0.0 EXCLUDES it. The ERROR
config thus exposes one extra barely-noisy observer row to attention. This is a candidate Bug E gate
(gate ①) — whether it, or the row-13 k_d step, drives the artifact is the open question. See
[ekf-39-40-input-diff.md](ekf-39-40-input-diff.md).

**Status:** record as an open defect; defer the fix until the primary Bug E gate (see
[first-frac-row.md](first-frac-row.md)) is resolved.
