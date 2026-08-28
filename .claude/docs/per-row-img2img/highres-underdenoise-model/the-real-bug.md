<!-- provenance: theory (Fable analytical model + 1MP GPU validation 2026-08-24; d_content vs d_blend diverge with resolution; window CLOSED @1MP confirmed by data) -->
<!-- verified: 2026-08-24 · 1MP GPU runs (0.45/0.60/0.68 lock, 0.75/0.78 smear=window closed, 0.83 chaos) + Fable adjudication · data in highres-singleframe-underdenoise.md -->
# THE REAL BUG — d_content vs d_blend diverge with resolution

Index: [highres-underdenoise-model](../highres-underdenoise-model.md). Data for these runs lives in [data-runs](data-runs.md). Experiments and fixes: [experiments](experiments.md) / [fix-strategies](fix-strategies.md).

## THE REAL BUG — d_content vs d_blend DIVERGE with resolution (user, 2026-08-24, IN-FLIGHT 0.75-0.78)

Watching the 0.75-0.78 preview, the user surfaced the deepest statement of the bug yet. The single
`min_denoise` knob has TWO user-facing jobs we'd conflated:
- **d_content** — the img2img regen the ANCHOR FRAME ITSELF needs: enough to resolve it into the model,
  but NOT so much it loses the injected structure (the gen-error correction the inject EXISTS to make).
  User-tuned ≈**0.45** *at 0.2MP*.
  ⚠ **CORRECTION 2026-08-24 (user-falsified; FABLE-ADJUDICATED):** "d_content is resolution-independent"
  is **WRONG**. Same nominal 0.45 → ~0.45 regen @0.2MP but ~0.05-0.1 @1MP. Resolution: d_content and
  d_blend are **two faces of ONE mechanism (basin sharpening / T_N(d) steepening — see CURRENT MODEL
  above)**, NOT two independent knobs. The requirements stay two, but the res-invariant target is *realized*
  regen (≈ the 0.2MP value), not nominal 0.45. "Fix d_content, decouple blend" survives ONLY when restated
  in realized units + the fingerprint gate. Anything that leaned on nominal res-independence is retracted.
- **d_blend** — the value at which NEIGHBORS coherently follow the anchor (seam low). RISES with res:
  ≈0.45 @0.2MP; @1MP there is **NO coherent d — window CLOSED** (data below): 0.68 locks, 0.75/0.78 smear.

**At LOW res they COINCIDE** (0.45 does both) → 0.2MP "just works." **At HIGH res they SEPARATE** → no
single d satisfies both: 0.45 preserves content but LOCKS (pop); 0.78 blends better but **over-cooks the
anchor — the gen errors the inject was placed to correct come BACK** (user, live). **The GAP between
d_content and d_blend IS the high-res pop.** This is the DEMAND-side twin of the t_row crux (supply side:
architecture couples denoise-rate↔attention-trust; demand side: the user needs them at DIFFERENT values).
Same coupling — now proven necessary to break from *requirements*, not just mechanism.

Consequences (these REVISE the plan below):
1. **d-sweeping is dead twice over.** In-flight 0.75-0.78 still pops r40 AND r60 (not the magic bullet →
   window likely CLOSED @1MP), AND even a found d\* would be the WRONG content-denoise. Golden-section on
   seam-alone optimizes the wrong axis.
2. **New target spec:** deliver seam-coherence **AT the anchor's content-correct denoise (≈0.45-equiv)**,
   not at whatever d minimizes seam. FIX d_content; use a DECOUPLER to drive seam down.
3. **Routes 1 (anchor-release) & 3 (attention boost) are now REQUIRED, not just elegant:** set the
   anchor's OWN denoise = d_content (true t_row / release-m), set blend via the independent trust lever
   (hold / attention-logit boost). The two-pass ORACLE must likewise hit seam-coherence at fixed d≈0.45.
4. **Experiment success criterion updated:** seam-z is NECESSARY but NOT SUFFICIENT — pair it with an
   anchor-content-fidelity check (user visual, or the `‖x_final[anchor]−clean‖` / ρ_ret proxy). A method
   "wins" only if seam drops WHILE the anchor's own regen stays at the intended ≈0.45 level.

## Hold-prototype corroboration (2026-08-25) — nominal m is res-compressed, hold-independently

Three route-1 hold-prototype runs on r40 (single-frame keyframe) give NEW evidence for this exact model:
- **m=0.5, hold=0.5** → "looks like ~0.1 denoise" (realized ≪ nominal).
- **m=0.99, hold=0.5** → "looks like ~0.5 denoise" (realized ≈0.5 from nominal 0.99).
- **m=0.5, NO hold** → also "~0.1", i.e. **identical to the hold run** ⇒ the compression is hold-INDEPENDENT.

This matches d_content res-compression (nominal 0.45 → realized 0.05–0.1 @1MP) and reads as a monotonic
nominal→realized curve. It ARGUES AGAINST reading the runs as "hold is a stronger img2img lever than m":
holding m fixed and raising it 0.5→0.99 is what moved the result; hold=0.5 vs none did not. It also argues
AGAINST collapsing m↔hold into one knob — the model (consequence 3) needs them SEPARATE: m = anchor
content-denoise (res-compensate the nominal value), hold = neighbor seam/trust lever.

**Cheap discriminator (untested):** run **m=0.5, hold=0.5 at 0.2MP**. Model predicts it now "looks ~0.5"
(low res: d_content/d_blend coincide). If it still looks ~0.1, resolution is NOT the cause and the
contagion/other tracks reopen. Pair with the `‖x_final[anchor]−clean‖` proxy instead of eyeballing.
Run resolution (user, 2026-08-25): **all r40 runs were 0.5MP** — a proven **1.0MP proxy** (same error
modes, slightly milder, ~1/4 the runtime). So they are in the high-res / window-may-be-closed regime.
Hold-thread context: [../latent-hold-release/anchor-denoise-after-clean-fix](../latent-hold-release/anchor-denoise-after-clean-fix.md).

## Competing MECHANISM for the res-dependence — attention dilution vs basin-sharpening (user, 2026-08-25)

The "basin sharpening / T_N(d) steepening" framing is a **per-frame intrinsic** mechanism (a solo frame
would underdenoise at high res). The user's leading alternative is **attention dilution**: r40's own denoise
math is res-INVARIANT; the res effect is entirely **temporal contagion** — neighbors "hold" the single-frame
inject HARDER at high res / longer timelines. If true, res-compression and contagion are ONE mechanism.
Not yet disproven; supporting experiments exist.

Mechanism refined to **raw-count** (user, 2026-08-25). An earlier caveat ("raw-count has the same scaling
hole as ratio") is **RETRACTED** — `(N-1)/N` invariance is about the *share* of attention, not attention
*quality*; quality genuinely degrades with raw token count (softmax entropy over more keys, RoPE-range
stretch). Open: **sign/pathway** — P1 (neighbors pin r40 — wrong sign under dilution) vs P2 (r40's signal
too diluted to assert — right sign); spectral-detail + RoPE-locality also on the table. Timeline length is
a separate clean axis.

**Strongest evidence favoring the attention family (already in hand): the fade observation.** Faded video
injects denoise well; single-frame / no-fade injects underdenoise worst. A pure per-frame basin would
res-compress EACH faded row too → it does NOT predict "fades work"; the attention family does (graded
neighbors + inject self-anchoring). Confound: video injects differ from a still by fade envelope AND motion.

**Discriminators.** **still-repeat-with-fade on r40 is ALREADY RUN** (result: fade fixes the SEAM but NOT
the anchor's own artifacts, which smear across the span, AND the model reads a static repeat as a
FREEZE-FRAME → not viable; confirms d_content vs d_blend are separate axes). Remaining cheap: **0.2MP**
(res-confirm) + **r60** (free 2nd single-frame point). Cleanest mechanism split = a **timeline-length sweep
at fixed res & m** (dilution ⇒ length-dependent; basin ⇒ length-invariant) but **DEFERRED** (a coherent
multi-inject timeline isn't free). If dilution wins, fix = route-3 (anchor attention-logit boost) — **but
CAUTION: route-3 risks the same freeze-frame read** (neighbors conforming to a boosted lone anchor). The
surviving lever is the **single frame's own effective denoise** (high / res-compensated m), NOT temporal
replication. Hold-thread detail:
[../latent-hold-release/anchor-denoise-m-vs-res](../latent-hold-release/anchor-denoise-m-vs-res.md).
