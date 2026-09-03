<!-- provenance: confirmed+proof (σ_a LOAD-BEARING for the model LABEL — source+model-contract proof; Fix B REJECTED) -->
<!-- verified: source proof from comfy model.py 604-605 + sampler.py labels; independent of the axis fix, remains valid -->
# σ_a is load-bearing for the LABEL — proof STILL VALID (independent of the axis fix)

Carved out of [../audio-axis-verdict.md](../audio-axis-verdict.md) (char/line budget). This
model-contract proof is independent of the ancestral-axis fix and remains valid.

**VERDICT: σ_a is removable for the ancestral INTEGRATION but LOAD-BEARING for the LABEL.**
Fix B (full-unification: σ_v for BOTH label AND integration) remains **REJECTED — by
model-contract proof, not assumption.**

**PROOF (independently verified from source + model, not circular):** our sampler passes the model
a fraction `w = sig_row/sig_g` (sampler.py:755,758); the model computes `t_row = 1 − w·σ_g`
(comment sampler.py:754; model `_forward` comfy model.py:604-605 `rows_t = 1 − m·σ_a`). For
audio, `sig_g = sig_a[i]` which EQUALS the model's own internally-derived σ_a (both from
time_shift_sigma(σ_v)). So current `w = sig_row_a/sig_a[i]` → model yields `1 − sig_row_a`
(truthful label). If the σ_v fraction were passed instead (`w = sig_row_v/sig_v[i]`), the model
still multiplies by ITS σ_a → `1 − sig_row_v·(σ_a/σ_v)`. Since time_shift_sigma is nonlinear
(σ_a/σ_v ≈ 0.27→1.0 across the schedule,
[../native-h3-mechanism/dit-forward.md](../native-h3-mechanism/dit-forward.md)), this MISLABELS
audio, worst at early/high-σ steps. The σ_a denominator is required for a truthful label.

**σ_a IS LOAD-BEARING IN THREE SITES** (only the ancestral integration axis was ever questioned;
Fix A moves ONLY that integration, not the label):

1. Per-row label denominator `w = sig_row/sig_g` (sampler.py:557,755) — proven above.
2. Observer-label K/V split — observer labels `t_obs = 1−m·σ` use shifted σ_a for audio
   ([../schedule-tail-late-delta/label-ratio-and-observer-split.md](../schedule-tail-late-delta/label-ratio-and-observer-split.md):95).
3. Deterministic r-scaling `r = (sig_row−sig_row_next)/(sig_g−sig_g_next)` in `_euler_step`
   (:305) and `_fallback_step` (:272) — unifying to σ_v would perturb a path with zero benefit.

**PROVENANCE — deliberate design:** σ_a axis was intentional, introduced in commit 41c488d
("Ship schedule-tail remap + observer-label K/V split": "complete the audio port: audio rows run
the remap on the sigma-shifted audio schedule via time_shift_sigma"). The schedule-tail remap idx
(`k_d`, `_stream_row_sigma` sampler.py:496-499) is axis-INDEPENDENT; the 17n+5 A/V tail join
controls layout, not sigma axis — so σ_a affects only sigma VALUES, not tail alignment.
