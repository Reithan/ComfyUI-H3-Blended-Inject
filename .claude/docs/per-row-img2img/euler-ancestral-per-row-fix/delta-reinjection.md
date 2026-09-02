<!-- provenance: theory (UNVERIFIED — analytical, round-10 GPU result recorded; δ not yet established primary) -->
<!-- verified: 2026-09-02 (branch fix-euler-ancestral-per-row-renoise) · T1 GPU round-10 recorded; δ theory UNVERIFIED -->
# δ re-injection theory — round-10 canonical record

Parent: [euler-ancestral-per-row-fix.md](../euler-ancestral-per-row-fix.md).
Competing primary: [../c2-rho-fix-paths/residual-accounting.md](../c2-rho-fix-paths/residual-accounting.md).

## T1 GPU result (round 10)

Config: PLANT_AXIS="row" revert, 0/0/0/90 ease_in_out euler_ancestral, same seed, no env var.

| Timeline | Description |
|---|---|
| 0.0–0.25 s | quiet/normal (m≈0, near-frozen) |
| 0.25–1.5 s | LOUD static |
| 1.5–2.0 s | very similar/identical to original audio |
| 2.0 s–end | new audio |

Fade-OUT: early timeline = low m (preserved), late = high m (new).

## Round-9 verdict (plant-over-noise.md)

- **Muffling CONFIRMED:** PLANT_AXIS="v" over-noised rows → heavy posterior mean → muffling.
  The "row" revert removed it (upper band now clean). Revert was the correct fix for muffling.
- **Static FALSIFIED as plant-caused:** over-noise did NOT cause the static — it was MASKING it
  (over-smoothing converts static energy → muffling). Truthful "row" plant leaves static raw:
  louder and broader (0.75–1.0 s peak → 0.25–1.5 s band). Static and muffling are the SAME
  energy redistributed by how hard the row is denoised; the plant axis chooses the split, neither
  addresses the source.

## δ re-injection theory (UNVERIFIED)

In `_c2_audio_ancestral_update`, ε̂ = (y − a·ĉ)/σ_c is recovered by inverting through the
network clean estimate ĉ. The analytic 1/σ_c cancels ONLY if ĉ is exact. The network carries
error δ = ĉ − C_true.

Tracking δ: y − a·ĉ = σ_c·ε_true − a·δ. The total δ-carried coefficient in the next state
x' = a'·ĉ + r_ret·ε̂ is (a' − (r_ret/σ_c)·a)·δ. At η=1 the C2 downstep gives r_ret ≈ σ_c'²/σ_c,
so this is (a' − (σ_c'/σ_c)²·a)·δ.

At low m (a ≈ a' ≈ 1) this equals (1 − (σ_c'/σ_c)²)·δ > 0 and GROWS as m→0 (the σ_c step-ratio
departs from 1, so the two clean-coefficient reconstructions fail to cancel).

Result: an m-graded, content-correlated δ leak, largest at low m → structured static
concentrated in the low-m band.

Confirmed indirectly by:
- Linear Run 3 (low-m compressed to t=0 → static at 0.00 s)
- Ease_in_out T1 (low-m stretched → static 0.25–1.5 s)

### c_fresh correction (claude-opus-4-8 sanity check, 2026-09-02)

Fable earlier claimed c_fresh = √(σ_c'² − r_ret²) ≡ 0 (identically zero) at η=1 for all m, so
there was NO fresh C2 noise on the audio ramp. **FALSIFIED — it conflated η=1 with η=0.** The C2
downstep is sd = σ_c'·(1 + (σ_c'/σ_c − 1)·η):

- η=0 → sd = σ_c', r_ret = σ_c', c_fresh = 0. (This is the zero case Fable mistook for η=1.)
- η=1 → sd = σ_c'²/σ_c ≠ σ_c', so r_ret = σ_c'²/σ_c and c_fresh = √(σ_c'² − r_ret²) > 0.
  Fresh stochastic noise IS injected on the C2 audio ramp at η=1.

Consequences:

1. The δ-leak (retained-noise) coefficient at low m is 1 − (σ_c'/σ_c)² (from r_ret ≈ σ_c'²/σ_c),
   NOT Fable's 1 − σ_c'/σ_c.
2. The "fresh noise washes δ out as m rises" clause is REINSTATED — its retraction rested on the
   bad algebra. c_fresh → 0 only as m → 0 (since σ_c' → 0).
3. **Core δ-reinjection conclusion SURVIVES unchanged:** δ dominates at low m (c_fresh → 0 there,
   δ-leak coefficient stays O(1)); fresh noise washes it out as m rises.

Empirical (H3BI_C2_DEBUG logger sanity run, claude-opus-4-8, 2026-09-02): fresh_rms NONZERO
(0.0355, 0.134) and leak_to_signal rising as m falls (0.088 @ m=0.6 → 0.194 @ m=0.3) — both
consistent with the corrected picture.

## Competing explanation that outranks δ

residual-accounting.md records a **deterministic per-row injection error, both modalities,
our-node-specific, present in euler@5 steps.** That error exists in VIDEO (S=1 → C2 ≡ 0) and in
euler (no C2 branch at all). δ lives entirely inside the C2 audio branch and CANNOT explain
video noise or euler noise.

Therefore: either there are TWO separate errors, or the mode-independent injection error is
PRIMARY and δ is an audio-only rider on top. Occam favors the latter. δ is NOT established as
primary.

## Unexplained seam (open)

T1: "1.5–2.0 s sounds like the ORIGINAL." Neither δ nor the mode-independent error predicts a
return-to-source mid-ramp (both predict only REDUCED static as m rises). Either the ear is reading
low static as "original," or there is a real null at m≈0.5 where a≈a' and σ_c'/σ_c≈1
simultaneously. Needs a spectrogram to settle.

## Locked discriminating test (pre-registered)

**Test A (free, no render):** inspect the VISUAL fade of the existing T1 render at ≈0.25–1.5 s
at 20 steps; then a 5-step re-render (5 steps exposes raw injection error pre-naturalization).

**Test B (one-line env toggle):** H3BI_DISABLE_C2=1 forces audio_scale→1.0, routing audio
through the σ_v path like video (disables the entire C2 branch); same 0/0/0/90 seed.

Pre-registered outcomes:

| Observation | Conclusion | Action |
|---|---|---|
| Video clean + DISABLE_C2 static drops markedly | δ = audio driver | build stored-ε̂ fix |
| Video comparably noisy OR DISABLE_C2 static unchanged | mode-independent error PRIMARY; δ rider | pivot to unified carry/remap fix in _euler_step + ancestral |
| Video clean @20 but noisy @5 | mode-independent error naturalized in video | same pivot |
| Neither shows static @5 | falsifies both as step-persistent | re-open ρ-drift amplifier |

## Candidate fix (if δ survives Test B)

"Persist ε̂ as sampler state; never re-derive it by inverting ĉ within the step."

Book x' = a'·ĉ + σ_c'·ε̂_stored so the δ terms cancel (a'·δ − a'·δ = 0). Stochasticity
untouched (η still blends fresh noise when η<1). Respects the hard constraint: do NOT
disable/eta-gate stochastic noise.

**Status: PROPOSED, UNVERIFIED. Awaiting Test A + Test B results.**
