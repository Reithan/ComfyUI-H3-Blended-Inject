<!-- provenance: theory (δ overall UNVERIFIED; δ-as-C2-GENERATOR sub-claim FALSIFIED by Test B round-10; δ-as-RESIDUAL open) -->
<!-- verified: 2026-09-02 (branch fix-euler-ancestral-per-row-renoise) · Test B GPU: static WORSE with C2 off → C2 net-corrective; δ-as-generator falsified, δ-as-residual open -->
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
Round-9 verdict (plant-over-noise.md): muffling CONFIRMED fixed by the "row" revert; static
FALSIFIED as plant-caused (over-noise MASKED it as muffling; truthful plant leaves it raw, louder,
0.25–1.5 s). The plant axis only redistributes the energy; neither setting addresses the source.

## δ re-injection theory (UNVERIFIED)

In `_c2_audio_ancestral_update`, ε̂ = (y − a·ĉ)/σ_c is recovered by inverting through the network
clean estimate ĉ. The 1/σ_c cancels ONLY if ĉ is exact; the network carries error δ = ĉ − C_true.
Tracking δ through x' = a'·ĉ + r_ret·ε̂: the δ-carried coefficient is (a' − (r_ret/σ_c)·a)·δ. At
η=1, r_ret ≈ σ_c'²/σ_c, so at low m (a≈a'≈1) this is (1 − (σ_c'/σ_c)²)·δ > 0 and GROWS as m→0.
Result: an m-graded, content-correlated δ leak, largest at low m → structured static in the low-m band.

Indirect support: Linear Run 3 (low-m compressed to t=0 → static at 0.00 s); Ease_in_out T1
(low-m stretched → static 0.25–1.5 s).

### c_fresh correction (claude-opus-4-8, 2026-09-02)

Fable's claim c_fresh = √(σ_c'² − r_ret²) ≡ 0 at η=1 conflated η=1 with η=0. C2 downstep
sd = σ_c'·(1 + (σ_c'/σ_c − 1)·η): η=0 → sd=σ_c', c_fresh=0 (the case Fable mistook for η=1);
η=1 → sd=σ_c'²/σ_c, c_fresh>0. Fresh stochastic noise IS injected at η=1. Consequences: the δ-leak
coefficient at low m is 1 − (σ_c'/σ_c)² (not 1 − σ_c'/σ_c); "fresh noise washes δ out as m rises"
REINSTATED (c_fresh → 0 only as m → 0); core δ conclusion survives — δ dominates at low m.

## Round-10 continued: Test B + logger (claude-opus-4-8 + user GPU, 2026-09-02)

**Test B (`H3BI_DISABLE_C2=1`):** static got LOUDER and LONGER — persisted past 2 s, obscuring ALL
original dialogue; worse than the C2-on run. No CSV produced (the logger only fires inside the C2 path).

**Overturns the pre-registered table.** The pre-registered options were "static drops→δ /
persists→residual floor"; the actual "static WORSE" outcome was in neither branch. Conclusion:
**C2 is net-CORRECTIVE, not the static's generator.**
- **δ-reinjection as a C2-internal GENERATOR is FALSIFIED.** If C2 generated the static, disabling
  it would remove it; instead it amplifies it.
- What survives: error-retention through the ancestral ε̂ is a GENERAL per-row audio-ancestral
  mechanism; C2 MITIGATES it by compressing the σ_a/σ_v carry (plant-over-noise.md: without
  carry-compression every fractional audio row re-enters over-noised → bigger retained error →
  louder static). The residual heard WITH C2 on is the portion C2's compression doesn't reach. δ
  operating WITHIN the (necessary) C2 update — the ĉ→ε̂ re-inversion re-injecting network error,
  amplified at low m by the 1/σ_c division — remains the leading candidate for that RESIDUAL. Test B
  did NOT falsify δ-as-residual; only δ-as-wholesale-generator.

**Normal-run CSV (C2 on, 20 steps):** retained_rms dominates fresh_rms through steps 0–9 (0.79 vs
0.27 at step 0, ~2.9×), crossing over ~step 9–10; sig_c decays 0.69→0.09. Consistent with
schedule-level error retention.

**Logger bug found + fixed:** total_steps was read as `ctx.state.get("total_steps")` but it lives
one level deeper at `ctx.state["schedule_tail"]["total_steps"]`, so it came through as 0 →
k_d = round(steps_n·(1−m)) collapsed to 0 for every row → all 7680 fractional rows pooled into ONE
m-bin. The first CSV thus measured leak-vs-STEP, not leak-vs-m; the δ prediction (leak concentrates
at low m) was untested. Fixed to `max(1, len(ctx.sigmas)−1)` (self-contained; sigmas hold steps_n+1
entries).

**New discriminating column `ret_clean_corr`:** the prior `leak_to_signal = retained_rms/clean_rms`
is just ancestral SNR (naturally high at low m) and cannot isolate the artifact. The new column is
the Pearson corr of the SIGNED retained vs clean terms across each bin's rows. Since
retained = (r_ret/σ_c)·y − (r_ret·a/σ_c)·ĉ, the re-inverted content component anti-correlates
retained with clean = a'·ĉ. **δ predicts ret_clean_corr strongly NEGATIVE and growing toward m→0
(retained "noise" = re-injected content ĉ); ≈0 ⇒ white legitimate ancestral noise, δ absent as a
residual source.**

### Pre-registered outcomes (next fixed-logger run)

1. ret_clean_corr strongly negative AND |corr| grows as k_d→steps_n (m→0): δ confirmed as the
   RESIDUAL, concentrated at the low-m / preserved-keyframe timeline (where static is loudest) →
   build the "persist ε̂, avoid ĉ→ε̂ re-inversion" fix targeted at low-m rows.
2. ret_clean_corr ≈0 across bins: retained term is white noise, δ NOT the residual → pivot to
   observer/content-axis or plant residual.
3. ret_clean_corr negative but FLAT or mid-m-peaked (not growing toward m→0): content re-injection
   is uniform/mid-m, matching the content-axis observer ring → redirect to the observer content path.

## Competing explanation

residual-accounting.md records a deterministic per-row injection error in BOTH modalities,
our-node-specific, present in euler@5 steps. It exists in VIDEO (S=1 → C2≡0) and in euler (no C2
branch). δ lives inside the C2 audio branch and cannot explain video/euler noise. So either there
are two errors, or the mode-independent error is PRIMARY and δ is an audio-only residual rider on top.

## Unexplained seam (open)

T1 "1.5–2.0 s sounds like the ORIGINAL." Neither δ nor the mode-independent error predicts a
return-to-source mid-ramp. Either the ear reads low static as "original," or there is a real null at
m≈0.5 (a≈a', σ_c'/σ_c≈1). Needs a spectrogram to settle.

## Candidate fix (if δ survives as residual)

"Persist ε̂ as sampler state; never re-derive it by inverting ĉ within the step." Book
x' = a'·ĉ + σ_c'·ε̂_stored so the δ terms cancel. Stochasticity untouched (η still blends fresh noise
when η<1). Respects the hard constraint: do NOT disable/eta-gate stochastic noise. **PROPOSED,
UNVERIFIED — awaiting the fixed-logger ret_clean_corr run.**
