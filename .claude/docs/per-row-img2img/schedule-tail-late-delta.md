<!-- provenance: status (observer-split SHIPPED; H1 sole primary; H2/label-cliff FALSIFIED; label-confound PERMANENT; v1 withdrawn; v2 DISFAVORED; route-3/renoise-release/label-lie/ramp-join REJECTED/DEAD — OFFLABEL-1 GPU 2026-08-28) -->
<!-- verified: 2026-08-27 · sampler.py @34a5925 lines ~404/487-516 (k_d, w-label, held composite); attention.py @b78cec87 lines 171-204 (wrap_attn, optimized_attention_override) -->
# Schedule-tail blend fight: H1 legibility / H2 late-delta / H3 locality

Cross-links: [keyframe-two-views-and-knobs](keyframe-two-views-and-knobs.md) (knob-B coupling;
route-3 promoted here then REJECTED) · [highres-underdenoise-model](highres-underdenoise-model.md)
(H1, σ-shift convexity sister case) · [native-h3-mechanism](native-h3-mechanism.md)
(time_shift_sigma) · [status-and-open-paths](status-and-open-paths.md).

**Through-line:** The mid-band blend problem (d≈0.05–0.3) is explained by H1 — σ-shift convexity
makes content noise deeper than the label, submerging neighbors' legibility of the inject. H2
(late-delta) and the label-category cliff are FALSIFIED as independent drivers. All time-split
approaches (hold + renoise-release), the label-lie (`official_labels`) family, and now
**ramp-join are REJECTED or DEAD.** Ramp-join rejected pre-build (2026-08-28): official-level hold
still skips the σ range between start values; compressed tail cannot recover that work — same
under-denoised-inject shape as every hold variant. Hold family closed; ramp-join joins it.
**OFFLABEL-1 (2026-08-28):** label-lie totally broken — label is load-bearing for the row's
own velocity prediction; label-lie family CLOSED. **H1 confound now PERMANENT:** no latent-side
label manipulation can cleanly isolate the label channel (structural property of the model).
**Knob-B crux confirmed:** with label≡content consistency forced, a single latent row has one dof;
blend and in-frame strength both ride it; true decoupling requires a second channel or second pass.
**Fork RESOLVED — observer-label K/V split SHIPPED to production** (branch
`implement-inject-schedule-remap`, off `main`) as the sole per-row mechanism, kv-only, audio port
completed: inject row emits K+V under official label d while own prediction runs on rescheduled
label; pins observed differential at d:1; H4-vs-H1 discriminated (H4 CONFIRMED on GPU); see
[label-ratio-and-observer-split](schedule-tail-late-delta/label-ratio-and-observer-split.md). The
two other candidates were DROPPED: (2) route-2 two-pass (oracle-correct, 2× cost); (3) accept the
single-trajectory trade-off and tune d per shot.

## Children (read only the one your task needs)

- [data-and-hypotheses](schedule-tail-late-delta/data-and-hypotheses.md): GPU data ladder, source
  facts, σ-shift convexity, label-category cliff (FALSIFIED), surviving hypotheses with H1 confound
  note, falsification matrix. H4 label-ratio hypothesis entry points here.
- [design-history](schedule-tail-late-delta/design-history.md): v1 withdrawn, v2 DISFAVORED for
  in-frame, route-3 attention-logit boost REJECTED, structure-window hold + renoise-release REJECTED.
- [label-channel-probe](schedule-tail-late-delta/label-channel-probe.md): H1 label-confound
  correction, official_labels label/content-split experiment design, mechanics, implementation, and
  result. (OFFLABEL-1 GPU 2026-08-28: DEAD: label load-bearing, label-lie family CLOSED)
- [ramp-join-schedule](schedule-tail-late-delta/ramp-join-schedule.md): **DESIGN (REJECTED
  pre-build, 2026-08-28):** analytical falsification: official-level hold skips σ range between
  start values; compressed tail cannot recover; same under-denoised shape as all hold variants.
- [label-ratio-and-observer-split](schedule-tail-late-delta/label-ratio-and-observer-split.md):
  **SHIPPED (production, `implement-inject-schedule-remap`):** H4 label-ratio hypothesis +
  observer-label K/V split; H4 CONFIRMED on GPU; sole production mechanism, kv-only, audio port
  completed; source grounding, design, ablations.
