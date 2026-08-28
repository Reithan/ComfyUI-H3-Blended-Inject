<!-- provenance: reference (external-code notes: LanPaint corrector math read from source @master 2026-08-24; verdicts analytical, no experiment run) -->
<!-- verified: 2026-08-24 · github.com/scraed/LanPaint master (src/LanPaint/lanpaint.py, nodes.py) + arXiv 2502.03491 (TMLR) -->
# LanPaint (arXiv 2502.03491) — Langevin corrector, read for cross-pollination

Investigated 2026-08-24 as a candidate mechanism for our fractional-anchor problem. **Verdict:
NOT a solution to fractional min_denoise (it binarizes masks and hard-preserves the known region
= the m=0 case we already solve), but TWO transplantable ideas: (1) per-σ inner-loop
equilibration as a 4th decoupling route, (2) the BiG score's model-consistency counterweight for
a principled continuous spring.** Also: LanPaint upstream already ships MiniMax H3 A/V inpaint
nodes (m=0 temporal masks) — an existing tool for the case we don't need to build.

## Mechanism (from src/LanPaint/lanpaint.py)

Replaces `KSamplerX0Inpaint`. Per outer sampler step at fixed σ (converted to VP: abt =
(1−t)²/((1−t)²+t²) for flow models):

1. **Replace step (line 94):** `x = x·(1−mask) + noise_scaling(σ, noise, y_o)·mask` — known
   region hard-reset to σ-noised source each outer step, using the STORED init noise field
   (correlated, not fresh).
2. **N inner Langevin iterations** (N = `LanPaint_NumSteps`, 2–10; each = 1 full model eval).
   Overdamped OU substep solved exactly (lines 232–254): `x ← e^{−A·dt}·x + k·C + √(D²k₂)·ε`,
   from `dx = −A·x·dt + C·dt + D·dW`, with `C = (√abt·x0 − x)/(1−abt) + A·x`, `x0 = model(x)`
   refreshed mid-step (midpoint scheme). Stationary law per iteration = **N(√abt·x0, (1−abt)·I)**
   — exactly the forward marginal at that σ. `dt = 2·StepSize·(1−abt)`; Friction only enters the
   (disabled) 2nd-order scheme.
   - **Unknown region** score: `score_x = −(x − x0)` → pure model-guided resampling; the corrector
     never copies known content into the unknown — conditioning flows THROUGH the DiT (attention).
   - **Known region** BiG score (line 183): `score_y = −(1+λ)(x − y) + λ(x − x0_BIG)` — stiff
     spring to source with the model's own pull SUBTRACTED (λ = `LanPaint_Lambda` ≈ 4–10): the
     known latent at mid-t is source *adjusted toward joint consistency*, not the raw noised source.
3. Output: one more model eval; `out = out·(1−mask) + y_o·mask` (hard composite, line 154);
   `input_x.copy_(x)` hands the equilibrated x back to the outer sampler.
4. **EarlyStop** (nodes.py:295): inner iterations DISABLED for the last few outer steps — their
   own empirical "no corrector during render", independently corroborating our ghost principle
   (λ must die before the render phase).

Cost ≈ (N+1)× model evals per step. Paper's "asymptotically exact conditional sampling" = as
inner Langevin time →∞ at each t, joint (x_t,y_t) → target with error O(√(1−abt)) → exact at t=0.

## The decisive facts for us

- **Masks are BINARIZED**: `denoise_mask = (denoise_mask > 0.5).float()` (nodes.py:281). The
  framework is a hard partition z=(x,y); "known" always means preserved exactly. **No fractional
  anchor exists in it.** A naive fractional score-blend `score_x·(1−m)+score_y·m` = a weak spring
  toward CLEAN source persisting to σ=0 = MC's ghost mechanism. They binarize because the blend
  has no clean semantics — same conclusion we reached.
- **Attraction vs clamp:** clamp on the KNOWN region (replace + composite per step, like MC),
  genuine attraction on the UNKNOWN region — but achieved by ITERATION (N model evals letting
  neighbors re-equilibrate to p(unknown|anchor) at each σ), not by trust labels. LanPaint never
  touches per-row t or attention logits; on H3 it presents the anchor at the SAME uniform σ as
  everyone. So it does NOT decouple our t_row crux — it makes the crux matter less by giving each
  σ enough inner steps to converge. Call it **route 4: per-σ equilibration** (vs route 1 time /
  2 duplication / 3 logit-boost).
- **In λ(σ)-spring terms:** known region = λ=∞ pulse each step to the end (minus EarlyStop) →
  ghost-shaped if applied fractionally; unknown region = λ≡0 plus a second time axis (inner
  iterations per σ), which the single-pass spring family cannot express.
- **Stochasticity is safe w.r.t. Bug B:** corrector noise lives at FIXED σ between outer steps
  (stationary variance matched to the marginal); it never touches the outer step's renoise, so
  outer deterministic euler and our σ→mσ scale-invariance are untouched. Output becomes
  non-deterministic, though. On fractional rows the corrector would need per-row abt(m·σ) — and
  its per-element-sigma path ALREADY exists (built for H3 audio: `audio_indicator`,
  `current_times_audio`, elementwise VE/abt blend, lanpaint.py:67–74).
- **Independent confirmation of our audio math:** their `audio_correction`
  `c = σ_a/(σ_v·slope_a)` on audio rows (nodes.py:266–275) is the same flat-grid audio
  velocity-overshoot we fixed with the ×S carry — derived independently. They also use
  `time_shift_sigma` identically.

## Transplant candidates (ranked)

1. **BiG-style consistency counterweight on Fable's continuous spring (route ② upgrade):** anchor
   score `−(1+λ(σ))(x−x_ref) + λ(σ)(x−D(x))` with λ(σ)→0 before render. The −λ·model-pull term
   fights exactly the OOD clean-content/high-t mismatch we flagged as the plausible chaos mode.
   Cheap (no extra evals; D already computed).
2. **Early-step inner equilibration:** for k < k_comp only, run 2–3 inner re-evals per step with
   the anchor held (re-noised clean) and m=1 rows Langevin-relaxed to N(√abt·D,(1−abt)) —
   attacks composition-lock by convergence instead of by hold length. ~+30–50% evals on early
   steps only.
3. Not transplantable: their known-region handling for fractional anchors (it IS the ghost).

**Kill-risk for #2 (the adversarial caveat):** if neighbor conditioning on the anchor is gated by
t_row trust LABELS rather than content (our crux hypothesis), extra iterations converge to the
same conditional a single deterministic pass already approximates — equilibration is then
orthogonal to the crux and buys nothing. Discriminating experiment: early-step equilibration on
the known-lock case (d≈0.6 @1MP) — if seam z-score minimum widens vs d, iteration matters; if
unchanged, the label-gating story wins and route 1 / observer-split stay primary.
(Note: route 3 attention-logit boost was REJECTED 2026-08-27 — latent-side mandate + SDPA perf.)

## Pointers

- Paper: arXiv 2502.03491 (TMLR). Repo: github.com/scraed/LanPaint — `src/LanPaint/lanpaint.py`
  (corrector), `nodes.py` (KSamplerX0Inpaint override, H3 AV nodes, mask binarize),
  `LanPaint_MaskBlend` = pixel-space Gaussian-boundary composite (post-decode QoL, not sampling).
- Related: [highres-underdenoise-model.md](highres-underdenoise-model.md) (crux, routes, spring
  family), [motion-context-comparison.md](motion-context-comparison.md) (the clamp lineage).
