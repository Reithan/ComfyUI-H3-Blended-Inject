<!-- provenance: status + confirmed (residual-hum accounting; "euler CLEAN / ancestral-specific"
     is OVERTURNED by GPU 2026-08-31 — a DETERMINISTIC injection noise in BOTH modalities, isolated
     to OUR node; candidate ranking re-ordered). -->
<!-- verified: 2026-08-31 · branch proto-c2-rho-denoised-r-comp · overturn = user GPU
     (euler-not-clean, silence-prompt, 5-step, official-vs-ours); code fact = read of
     sampler.py:305-335; surviving ladder claims tie to prior GPU rungs + comfy-ref. -->
# Residual accounting: the euler-not-clean overturn + candidate re-rank

Index: [index.md](index.md). Current fix chain: [current-fix.md](current-fix.md).

## Purpose
Account for the residual fade-region noise and route between a CORRECTABLE fix and an INHERENT
floor. As of 2026-08-31 the routing basis CHANGED: the artifact is no longer believed
ancestral-specific — see the overturn below.
Config throughout: inject f0, fade `0/0/49/73` (held `[0,49)` m=0, fade-out ramp `[49,73)` fractional),
ease_in_out, audio_mode=fade, min_denoise=0, fixed seed.

Timbre is treated as LOW-INFORMATION (per user): the model reshapes early-gen interference
downstream, so the final artifact's timbre ≠ the raw source. Only the amplitude ladder and the
source math carry information below.

## OVERTURNED (GPU, 2026-08-31): NOT ancestral-specific — deterministic, in OUR node
The prior "euler CLEAN, so the artifact is ancestral-specific" premise (which had ruled out every
sampler-INDEPENDENT mechanism) is FALSE. Three user GPU tests overturn it:
- Re-listening the deterministic **euler** render at the fade: sound IS present (ambiguous alone —
  reads as SCUBA/bubbles).
- Forcing silence by prompt (`overall_soundscape: N/A`, even a blank prompt) STILL yields diegetic
  sound in the fade region.
- Cutting steps **20→5** (so the model can't naturalize noise into plausible sound), original
  prompt, OUR inject node, euler: VISUAL noise becomes VISIBLE and unrequested diegetic AUDIO
  remains. The **OFFICIAL/stock** sampler at the SAME 5 steps (no injection): NO visual noise, NO
  audio noise, NO diegetic sound.

CONCLUSION (user, moderate certainty): a DETERMINISTIC injection-path noise exists in BOTH the
audio fade and the visual fade. The many-step model NATURALIZES it into diegetic sound (why it read
"clean" at 20 steps); low steps expose it raw. It is isolated to OUR node (present in our euler @5,
absent in official euler @5). Ancestral renoise does NOT create the artifact — it merely AMPLIFIES
this deterministic injection error.

**Code fact (verified, sampler.py:305-335).** `_euler_step` — the deterministic per-row euler
registered for `sample_euler` — applies NONE of the C2 corrections: it takes a FULL euler step on
the carrier axis (`d=(x−denoised)/σ_i`, `x_base=x+d·(σ_{i+1}−σ_i)`) then lerps by the legacy r-scale
`r = clamp((sig_row−sig_row_next)/(sig_g−sig_g_next), min=0)`. No σ_c projection, no ρ_true, no
residual cancel, no m=0 context rescale, no per-row sigma integration — ALL of v3/v4/v6 live ONLY in
`_euler_ancestral_rf_step`. So the euler tests exercised the FULLY-UNCORRECTED injection path; the
noise is the raw carry-compression error (audio) + raw per-row injection on fractional rows,
unnaturalized.

**Video-noise pointer (≠ C2).** Video rows have no audio carry (S=1, C2 does not apply), so the
VISUAL fade noise is NOT the C2 error — it points to a per-row-remap/injection noise common to BOTH
modalities, on top of which C2 is an audio-only EXTRA. Endgame re-derivation (unified carry
correction ported to `_euler_step`, video-noise mechanism, next discriminator) is IN PROGRESS on a
separate thread (Fable) — do NOT derive it here.

## Surviving PROVEN claims (still hold)
These tie to GPU rungs untouched by the overturn:
- **Monotone same-seed descent B→v3→v4 in the ancestral path.** The C2 carry-compression clean-coeff
  correction on active fractional audio rows removed the large majority of the ancestral buzz
  (loud→faint). Forward carries all audio by global `k=σ_a/σ_v` (model.py:529-541), so a row labeled
  `sig_row` sits at packed coeff `σ_c=sig_row·carrier/sig_g`; v4 subtracts the exact residual
  `(S−1)(sig_row−sig_g)·x_prev` (model.py:549-550 + CONST `denoised=x−σ·v`, model_sampling.py:90-92).
- **NOT init-state-driven.** v5 corrected the i==0 ~S× over-plant exactly; GPU showed no change ⇒ the
  DiT self-corrects one-shot state errors early. COROLLARY: only PERSISTENT, re-applied-every-step
  errors can sustain the residual.
- **m=0 audio-context heat is a real contributor.** v6 presents frozen audio rows at
  `clean/(1+(S−1)·sig_g)` instead of `S·A` (native `scale_latent_inpaint` `1/(S·k)`,
  model_base.py:2262-2266); GPU quieter than v4.
- **Input-side static-reference correction is unsound.** 63b291e was LOUDER (`S·A` truthful only at
  step 0; subtracting from a regenerating state injects garbage). Rules out any simple fixed
  x-space clean offset.

## Remaining candidates (RE-RANKED after the overturn)
**PRIMARY (new #1) — deterministic per-row injection/remap error, BOTH modalities.** Present in our
euler @5 steps, absent in official euler @5. This is the raw per-row injection on fractional rows
(video + audio), unnaturalized. `_euler_step` applies zero corrections, so this is the fully-exposed
injection path. This is now the primary source — promoted ABOVE the ancestral ρ-drift. The unified
carry correction ported into `_euler_step` (Fable re-derivation, in progress) is the intended fix.

**SECONDARY (was #1) — cross-interval ρ-drift in the ancestral write.** The v4 update
(sampler.py:479-492) uses stock RF bookkeeping (`alpha=1−σ`, `ratio=sigma_down/σ_c`); the truthful
carried frac-audio trajectory has clean coeff `(1−σ_c)/ρ_true` with `ρ_true` STEP-DEPENDENT (≈S
early → 1 late), so a step slightly UNDER-plants clean, the model re-corrects, and ancestral
renoises the difference — a persistent per-step error. Candidate **"v7"**: exact `ρ_eff`
cross-interval correction (`ρ_eff≡1` at m=1, finite at terminal, same frac-audio gate). Now
SECONDARY: it is an ancestral-only amplifier on top of the deterministic error, not the root.

**DIAGNOSTIC — eta=0 on fractional audio rows.** Kills ancestral stochasticity on the ramp; would
silence anything renoise-amplified, isolating the deterministic component, at cost of a texture seam.

**INHERENT FLOOR — DiT/ancestral estimate floor.** Each eval's `A_hat` on ramp rows carries
irreducible error; not sampler-correctable without giving up stochasticity. Consider only after the
deterministic injection error is corrected.

**STILL RULED OUT (on new grounds).** VAE decode overlap + held/ramp seam and Bug E remain ruled out
NOT by "euler clean" but because the OFFICIAL euler node @5 steps is clean while OURS is not — the
artifact is OUR-node-specific, so any node-independent decode/seam mechanism cannot be the cause.

## Reconciliation (2026-08-31): two-layer stack of the fade noise
Reconciles the PRIOR observer-split / label-vs-attention decoupling residue with the CURRENT
audio-fade (C2) findings. The Fable endgame re-derivation owns final mechanism attribution — this
extends its ranking, does not replace it. THROUGH-LINE: fade noise = accepted DECOUPLING residue
(both modalities, FLOOR) + C2 carry-compression (audio-only, CORRECTABLE) + ancestral ρ-drift (AMPLIFIER).

**Layer 1 — decoupling Δ residue (FLOOR, both modalities). theory (UNVERIFIED).** The linear-vs-curved
observer-split gap `Δ = σ_row − m·σ_g` ([../long-fade-grid-beat/kv-observer-mismatch.md](../long-fade-grid-beat/kv-observer-mismatch.md),
REFRAME section) is the ONLY already-identified wiki mechanism that explains the VIDEO-side fade
noise — video rows have S=1, so C2 ≡ 0 for video. Per the reframe it is analytically IRREDUCIBLE
(not zeroable by any single label choice without reviving the keyframe ghost) — an accepted design
trade-off. The Δ math is CPU/source-confirmed; the "this residue IS the artifact / irreducible"
claim is analytical only. Discriminators (#81 kill-switch, stock-linear long-fade) are PENDING.

**Layer 2 — C2 carry-compression (AUDIO-ONLY, CORRECTABLE).** Exact math + fixes v3/v4/v6, applied
ONLY in `_euler_ancestral_rf_step`, ABSENT from `_euler_step` (sampler.py:305-335). Porting corrects
a correctable layer, NOT the floor. **Layer 3 — ancestral ρ-drift (v7): AMPLIFIER**, as ranked above.

**PRIMARY candidate = two rivals, not yet distinguishable (both theory, UNVERIFIED).**
- Δ / observer-split decoupling residue (Layer 1) — leading IDENTIFIED mechanism; attention-label driven.
- Raw `_euler_step` r-lerp arithmetic — legacy r-lerp on the carrier axis, NO C2/σ_c corrections; a
  distinct, attention-label-INNOCENT candidate. The wiki cannot yet tell these apart.

**Discriminating tests (SUPERSEDE the plain H3_FORCE_ETA=0 plan; eta=0 DEMOTED to a cheap tie-breaker
that mostly re-confirms the euler overturn).**
- (a) Port C2 v3/v4/v6 into `_euler_step`, 5-step euler: audio drops + video persists ⇒ two-layer model confirmed.
- (b) #81 kill-switch (broadcast curved `σ_row` to K/V), 5-step euler: fade noise vanishes (ghost
  returns) ⇒ noise IS the decoupling residue; persists ⇒ r-lerp arithmetic is the source, decoupling innocent.
- (c) content-side Δ closure — plant injected noise at observed `m·σ_g`, keep self-evolution at
  `σ_row` (DUAL of #81, attacks decoupling-residue from the content side). **RUN (GPU 2026-08-31):**
  content-side closure REMOVES the observer-side fade noise + Bug-E break and REVEALS a self-side
  colour tint — the noise and the tint are the SAME Δ seen from opposite sides (confirms Layer 1 is
  real + two-sided). See [observed-level-plant.md](observed-level-plant.md).
- (d) stock-mask remap port — `H3RescaleNoiseMask` rescales a `noise_mask` (least-squares scalar
  `m_new = Σσ_g·σ_row / Σσ_g²`) so the STOCK sampler reproduces our curved `σ_row` MAGNITUDE without
  our per-row step fn. A/B stock+m_new vs our node isolates whether the σ_row value is the whole
  story. NOT a decoupling test (stock applies the linear observer label natively — reproduces
  magnitude, not the curved-self/linear-obs split). See
  [stock-mask-remap-port.md](stock-mask-remap-port.md).

**Gate-mismatch tension.** Config `0/0/49/73` has ramp=24 ≪ Bug E's GPU-confirmed ~51 threshold, so
Bug E's OWN gate predicts CLEAN — yet 5-step deterministic noise is present. Same-mechanism
identification requires "sub-threshold residue everywhere, runaway only in the gate" — plausible but
UNPROVEN. Bug E stays STILL RULED OUT as the direct cause; this records the tension, not a reversal.

**Naturalization caveat.** "The model absorbs the injected noise" was NOT in the wiki before today's
overturn — it is today's naturalization finding (high steps naturalize the deterministic error into
diegetic sound), not prior doctrine.

## Confidence + next action
Overturn + code fact: HIGH (direct GPU + source read). Attribution of the deterministic error's
mechanism: the Fable endgame re-derivation (separate thread) owns it — this doc records the finding
only. Next action = that re-derivation (unified carry correction in `_euler_step` + video-noise
mechanism + the right next discriminator); do NOT build v7 as primary.
