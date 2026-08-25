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
