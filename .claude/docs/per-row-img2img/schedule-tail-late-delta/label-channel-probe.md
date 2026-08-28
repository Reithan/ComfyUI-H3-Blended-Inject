<!-- provenance: design (RESULT: FALSIFIED/BROKEN — OFFLABEL-1 GPU 2026-08-28; label load-bearing for own velocity prediction; label-lie family CLOSED) -->
<!-- verified: 2026-08-27 · sampler.py @34a5925 (official_labels plumbed, tests 576 pass, lint clean, uncommitted on proto-schedule-tail-release) -->
# Schedule-tail: label-channel probe (official_labels toggle)

Parent: [schedule-tail-late-delta](../schedule-tail-late-delta.md).
See also: [data-and-hypotheses](data-and-hypotheses.md) — H1 hypothesis and the confound note.

## H1 label-confound correction (2026-08-27)

The d=0.05 test (content σ≈0.39, truthful label w≈0.61) was used to claim "content dominates over
label." **WITHDRAWN (user challenge, accepted):** truthful labels track the same convex curve as
content — every data point moves both channels together. The label channel's independent anchoring
weight is UNTESTED. "Content dominates" does not follow from confounded data.

What the test DOES show: the content+label combination at d=0.05 is sufficient to anchor strongly.
Whether content alone, label alone, or both together are necessary is unknown until the channels are
cleanly separated.

## The experiment: official_labels toggle

**Hypothesis being tested:** does the label channel carry independent temporal-anchoring weight,
beyond what the content channel provides?

**Design:** set the DiT label progression to the exact trained form while the content keeps the
remapped dense-grid tail trajectory.

- **Label** (feeds only `make_pooled` → adaLN): `w_label = m_dev` where `m = 1 − d` (the exact
  trained form `t_row = 1 − m·σ_g(i)`, held constant = `m` across the row's steps). This is what
  the model was trained to see.
- **Content** (what neighbors see in Q·K): continues on the rescheduled dense-grid tail — truthful
  `w = σ_row/σ_g`.
- Composite weights and the per-row r-lerp (step-scaling in sampler.py) still use truthful
  `w = σ_row/σ_g`.
- Step-size bookkeeping is label-independent; the row still integrates along the remapped Δσ_row.
  The remapped RATE is guaranteed correct regardless of the label value.

**What this isolates:** for the first time, label and content move on different trajectories.
Existing data always tracked them together; this is the first clean separation.

## Mechanics and caveats

Label feeds only `make_pooled` → adaLN conditioning, not the attention Q·K content channel.
The label also changes the model's per-step velocity prediction: it predicts as-if the row is at
`m·σ_g` while the content sits at σ_row.

**CAVEAT:** the mismatch is largest at step 0 (e.g. d=0.2: label-noise ≈0.2 vs actual ≈0.75);
it decays to 0 by the end of the row's steps since both curves terminate at σ=0. The remapped
step RATE is guaranteed; prediction quality under content-noisier-than-labeled is what the test
measures.

## Implementation

Built 2026-08-27, tests 576 pass, lint clean, uncommitted on `proto-schedule-tail-release`.

- `official_labels` BOOLEAN input on the sampler node (default False; applies only to remap modes
  'both' and 'rescheduled').
- Plumbed: `nodes.py` → `schedule_tail_cfg` → sampler loop.
- `w_label = m_dev if (remap and official_labels) else w` feeds `make_pooled` only.
- Startup log prints the flag.

## OFFLABEL-1 GPU result (2026-08-28) — TOTALLY BROKEN

**Result:** abstract/psychedelic patterns in injected frames — completely non-functional.

**Mechanism (user diagnosis, matches pre-flagged caveat, now confirmed):** the model computed
denoising under the official label (as-if noise = m·σ_g) while the latent sat at the far noisier
remapped σ_row. It treated the excess noise pattern as image content and refined it. The r-lerp
integrated those corrupted predictions correctly — bookkeeping was fine; estimates were garbage.

**Conclusion 1: label-lie family CLOSED.** The row's mask label is load-bearing for its own
velocity prediction; any lie big enough to matter for neighbor-view corrupts the row's own
denoising by the same mismatch. Structural, not tunable.

**Conclusion 2: H1 confound permanent.** The label/content confound on H1 is unresolvable via
output-row label manipulation. It stands as a permanent caveat, not a testable question
(latent-side). No experiment can cleanly isolate the label channel without corrupting the row.

## Predicted outcomes (recorded pre-GPU; see result above)

**If blend improves in the mid-band with in-frame intact:** label channel carries real anchoring
weight independent of content; H1 needs a label term — "content plus trained label" is the correct
framing, not "content alone."

**If nothing changes:** content-legibility account survives the confound — content alone explains all
anchoring; the label channel's contribution is negligible relative to content depth.

**What actually occurred — prediction-corruption (OFFLABEL-1, 2026-08-28):** the label mismatch
(m·σ_g vs actual σ_row) corrupted the velocity predictions so severely it overrode any anchoring
signal; psychedelic/abstract patterns throughout. The label is load-bearing for the row's own
denoising — not just for neighbor anchoring. Label-lie family CLOSED.
