<!-- provenance: theory (UNVERIFIED — GPU-narrowed; revised hypothesis analytical/UNVERIFIED; prior grid-cycle mechanism GPU-FALSIFIED 2026-08-28) -->
<!-- verified: 2026-08-28 · GPU test matrix falsifies grid-cycle count prediction; revised held+ramp theory analytical only; tests on main (euler, 0.2MP) -->
# Bug E: long-fade video interference — held+ramp mid-schedule disparity

Bug E symptom: moiré / streamers / ribbons / electric-like patterns in video latent with long
fades; sampler-independent; audio tracks via joint attention.
Established in [bugs.md](bugs.md) and [audio-axis-verdict.md](audio-axis-verdict.md).

Note: σ̃/H2 and Fix A do NOT address Bug E — present on main before either fix, sampler-independent.

## GPU falsification — 2026-08-28 test matrix (euler, 0.2MP, single video fade at f0, no guides)

Notation: a/b/c/d = fade_in/start_keyframe/end_keyframe/fade_out.
HELD (frozen preserved block) = [skf, ekf]. RAMP = fade span.

| Config    | Held | Ramp            | Result                                           |
|-----------|------|-----------------|--------------------------------------------------|
| 0/0/0/85  | ~0f  | 85f (5 cycles)  | CLEAN                                            |
| 0/0/0/17  | ~0f  | 17f (1 cycle)   | CLEAN                                            |
| 0/0/0/69  | ~0f  | 69f             | CLEAN                                            |
| 0/0/0/73  | ~0f  | 73f             | CLEAN                                            |
| 0/0/0/75  | ~0f  | 75f             | CLEAN                                            |
| 0/0/7/75  | 7f   | 68f (4 cycles)  | CLEAN                                            |
| 0/0/30/90 | 30f  | 60f             | NOISY — streamers/ribbons/electric; audio tracks |
| 0/0/60/90 | 60f  | 30f             | CLEAN                                            |

## Grid-cycle-count theory — REJECTED (GPU falsified 2026-08-28)

Prior hypothesis: artifact onset tracks how many 17-frame grid cycles the ramp spans
(predicted onset ~≥3–4 cycles); the (1,4,4,4,4) sampling-periodicity / stuck-pair mechanism
was the proposed driver.

**Falsified by:** 0/0/0/85 (5 cycles, CLEAN), 0/0/0/75 (CLEAN), 0/0/7/75 (4 cycles, CLEAN).
Long ramps spanning 4–5 grid cycles are clean when the held block is small.
The grid-cycle count does NOT predict the artifact.
The sampling-periodicity / stuck-pair / k_d-quantization mechanism is REJECTED as the driver.

## Revised hypothesis — held + long ramp mid-schedule disparity (UNVERIFIED)

The artifact requires BOTH:
(a) a substantial FROZEN preserved/held block, AND
(b) a LONG graded ramp adjacent to it.

Neither alone triggers it:

- **Long ramp alone is clean:** 0/0/7/75 has a LONGER ramp (68f) than the failing case (60f)
  but only 7f held → CLEAN.
- **Large held alone is clean:** 0/0/60/90 has a LARGER held block (60f) than the failing
  case but only 30f ramp → CLEAN.
- **Not the product:** 0/0/30/90 and 0/0/60/90 have identical held×ramp = 1800 but opposite
  outcomes.

Only 0/0/30/90 combines both a substantial held (30f) and a long ramp (60f) → NOISY.

## Mechanism (theory)

This is a DYNAMIC, not a static geometry. User observation from per-step previews: the error
noise is GENERATED mid-schedule (~1/4 to 1/2 through the steps); later low-sigma steps either
HEAL it or crystallize it into visible artifacts (streamers/bubbles/electricity).

This matches where the schedule-tail remap produces MAXIMUM cross-row sigma disparity:
fully-released ramp rows (low k_d → low sigma, near-clean) sit in H3's full joint attention
([dit-forward.md](native-h3-mechanism/dit-forward.md)) next to still-frozen held rows
(m≈0, k_d≈steps → high sigma). A large frozen block adjacent to a long ramp SUSTAINS this
noisy-vs-clean attention boundary across many rows AND many mid-steps → structured noise that
needs enough low-sigma runway to heal.

Explains: present on main, sampler-independent, video-primary, audio-couples via joint attention.

Corollary (user's read): more cases may carry SOME transient noise that heals when step count
is adequate or the compounding-frame count is too low.

## Falsifiable predictions — next GPU tests

1. **STEPS SWEEP on 0/0/30/90 (steps 20→30→40→60):** if artifact diminishes as steps rise,
   CONFIRMS the compound-vs-heal dynamic and localizes the fix to release schedule / tail runway.
   (Highest-value test.)
2. **Held-threshold bisection — 0/0/15/75** (held 15f, ramp 60f): narrows the held trigger
   between 7 (clean) and 30 (noisy).
3. **Long-ramp-required check — 0/0/30/60** (held 30f, ramp 30f): predict CLEAN; if noisy,
   held≥30 alone triggers and the hypothesis is wrong.
4. **Optional — 0/0/30/90 with held min_denoise=0.05:** tests whether the exact-preserve
   "never-release" path seeds the boundary vs. a near-zero path.

## Fix direction (speculative — confirm mechanism first)

If the steps-sweep confirms heal-vs-compound: smooth the mid-schedule release disparity
(e.g. gentler/continuous release across the held→ramp boundary, or ensure enough tail steps)
rather than any grid-alignment change. Do not implement until GPU-confirmed.
