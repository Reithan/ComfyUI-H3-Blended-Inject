<!-- provenance: theory (Fable source-spring unification 2026-08-24; step-starvation analysis; anchor-then-release build params confirmed) -->
<!-- verified: 2026-08-24 · 1MP GPU runs (0.45/0.60/0.68 lock, 0.75/0.78 smear=window closed, 0.83 chaos) + Fable adjudication · data in highres-singleframe-underdenoise.md -->
# Fable's verdict and step-starvation analysis

Index: [highres-underdenoise-model](../highres-underdenoise-model.md). Crux and mechanism: [crux-and-mechanism](crux-and-mechanism.md). Experiments and build order: [experiments](experiments.md) / [fix-strategies](fix-strategies.md).

### Fable's verdict — one source-spring family; anchor-then-release = build-first (2026-08-24)

Fable unified ALL candidates (MC, ours, order-inversion, anchor-then-release, continuous decay) as ONE
object: a source-spring term on the RF ODE — **dx/dσ = v_eff + λ(σ)·(x_ref(σ) − x)**. Every mechanism is
just a choice of λ(σ):
- **MC** = large ~constant λ persisting to σ=0 → source re-injected during the LATE render phase → GHOST.
- **Ours** = λ≡0 → no ghost, but no attraction → bimodal.
- **Middle (continuous)** = small λ that DECAYS to 0 before the render phase → structurally ghost-free.
- **anchor-then-release** = the bang-bang limit: λ=∞ for k<k_sw (hard hold), then λ=0. Ghost is caused
  by *any* λ>0 during late render; killing λ before render is the ghost-free condition, general.
- **Order-inversion = RED HERRING** — it only deletes MC's literal final-step composite; the real crumb
  in it is that a per-step retain for target d over K steps is m_k = d^(1/K) (a small-λ spring), not MC's m.

**Q2 CONFIRMED (attraction only via low-noise-EARLY anchor).** DiT tokens interact solely through
attention → a row-local latent op cannot move neighbors directly, only change how the anchor's keys/values
read. Neighbors attract iff the anchor presents as trustworthy = **low t_row AND content consistent with
that low-noise claim**. It's a TRAINED behavior (H3 ships native per-row denoise_mask / diffusion-forcing:
"propagate from resolved rows into unresolved"). Two edges fall out: (a) **consistency** — clean content
MUST pair with low t_row; a clean-content/high-t mismatch is OOD and plausibly IS the chaos mode; (b)
**attention-logit boost** is a real non-latent second route (back pocket, tests the dilution hypothesis).

**Concrete anchor-then-release params (Fable):**
- **Hold** (k<k_sw): x_row=clean, t_row=low, override D_row=clean (zero velocity — no correction needed
  while held).
- **k_sw = measured k_comp**, NOT tuned: first k where cos(lowpass(D_nbr(k)), lowpass(D_nbr(final))) > 0.9.
  This AUTO-CALIBRATES against resolution/shift (the whole point). **Now instrumented** — see `k_comp` in
  Reading-the-instrumentation below; the debug build prints the suggested k_sw per run.
- **m′ ≈ 0.3–0.5 of the REMAINING schedule**, with a two-sided σ constraint: m′·σ_sw must sit BELOW
  composition-visibility σ (can't re-decide layout) and ABOVE render-commitment σ (texture/lighting can
  still fuse).
  ⚠ **CORRECTION 2026-08-24 (Fable, self-flagged):** m′'s "wide tolerance" is UNPROVEN — the release phase
  is itself an img2img from a clean-held state, so **T_N applies to m′ too** (a nominal m′ ≠ a realized
  regen; same cliff risk). Calibrate m′ in REALIZED units against the fingerprint, don't assume tolerance.
  Structural counter-good (also verify): during release the neighbors are already committed AROUND the
  anchor, so its posterior is pulled by a consistent attention context rather than left to self-recover ⇒
  the basin it must escape is SHALLOWER than at init, so T-at-release is likely LESS cliff-like. Expect,
  don't assume.
- **Re-noise at release = CORRELATED** (shared base-noise field), not fresh:
  x_row(k_sw) = (1−m′σ_sw)·clean_held + m′σ_sw·n_base[row]. Fresh noise only as an ablation.

**Basin WIDENS (genuinely, not a needle-move).** d-only is narrow because one scalar must satisfy two
opposing constraints across the WHOLE trajectory; anchor-then-release DECOUPLES them in time → trade the
narrow scalar window for a RECTANGLE in (k_sw, m′) where k_sw is measured (not tuned) and m′ has a wide
window with gentle failure modes (too low → mild rigid texture; too high → mild drift; neither a pop).
Resolution-invariance comes free via k_comp auto-calibration.

**Build order (Fable):** 1. **anchor-then-release** (build FIRST — even if 0.75 is coherent, a ≤0.15
window that migrates with N/shift isn't shippable); 2. continuous small-λ spring (if bang-bang shows a
seam-in-time artifact); 3. attention-logit boost (dilution diagnostic / back pocket); 4. order-inversion
(red herring). Still run the 0.75-0.78 probe to complete the γ fit + calibrate the seam gate + learn
whether the denoise-only window is even open.

### Step-starvation failure mode (user, 2026-08-24) — and why it decomposes

Concern: anchor-then-release splits K steps into hold `[0,k_sw)` + release `[k_sw,K)`. At extreme
denoise + low steps (e.g. d=0.1 or 0.9, K=10) one side is starved — "all the denoise in 1 step, or all
the attention propagation in 1 step." Real, but it splits cleanly by régime:

- **The hold and a HEAVY-denoise release compete for the SAME budget — the high-σ early steps.**
  Attraction needs the anchor resolved during composition-lock (high σ, early). But a released row can
  only re-noise to `m′·σ_sw`, and σ_sw is already mid/low (we're past k_comp). So the hold *consumes* the
  high-σ region → the achievable release denoise is intrinsically capped at ~(1−k_comp/K). You cannot
  both hold a frame as an anchor AND regenerate 90% of it — those goals **conflict conceptually**, not
  just in code ("keep this as an anchor" vs "replace most of it").
- **Low d (0.1), low steps = the EASY case, not fraught.** Release only needs a tiny nudge → 1 step is
  plenty; hold gets the rest, propagation is well-fed. The user's 0.1/10-step worry inverts: release is
  comfortable, not starved.
- **High d (0.9) = attraction is nearly moot** (little identity left to attract *to*; this is régime 4,
  inject-lost, which already works). Escape: at high d, **shrink/skip the hold** → fall toward plain
  per-row img2img. So the régime where release starves is the régime where you shouldn't anchor anyway.
- **The genuine hard zone = mid-high d (~0.7) + low steps** — want real attraction AND substantial
  regen, few steps to split. Exactly the (0.68,0.83) bracket. Mitigations, in order:
  1. **Couple hold length to target d**: derive k_sw AND m′ together so hold shrinks as d rises (trade
     off along the axis that actually matters). Auto-manages the starvation.
  2. **Continuous small-λ spring (Fable step ②)** DISSOLVES the partition — a little attraction + a little
     denoise every step, no step purely starved. The bang-bang is the MVP; the spring is the
     starvation-robust form. This is *why* it's next in the build order.
  3. **Clamp min hold/min release steps** + runtime detect: if measured k_comp ≥ K−2, warn/fall back.
- **Softener:** attention propagates across ALL rows in ONE forward pass — a single hold step is already
  a full broadcast; extra hold steps buy neighbor *commitment*, which is concentrated in the high-σ steps
  regardless. So hold-starvation is milder than "propagation in 1 step" implies.
- **Reframe the bar:** extreme-d + 10-step is step-starved for *any* method (MC included). Don't hold
  anchor-then-release to a standard nothing meets; the target is mid-d, reasonable-K, where it shines.
