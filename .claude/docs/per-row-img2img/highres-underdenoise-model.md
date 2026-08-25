<!-- provenance: theory (Fable analytical model + 1MP GPU validation; CURRENT MODEL = T_N(d) basin-sharpening, γ SUPERSEDED) -->
<!-- verified: 2026-08-24 · 1MP GPU runs (0.45/0.60/0.68 lock, 0.75/0.78 smear=window closed, 0.83 chaos) + Fable adjudication · data in highres-singleframe-underdenoise.md -->
# Analytical model — resolution-dependent effective denoise (Fable, 2026-08-24)

Companion to [highres-singleframe-underdenoise.md](highres-singleframe-underdenoise.md) (the data +
mechanism). This is the math model derived from that data.

> **GPU UPDATE 2026-08-24 (1MP run, f136→row40 @0.83, f204→row60 @0.45):** the α=ρ (odds-linear,
> γ=2) up-map is **FALSIFIED** — 0.83 is the CHAOS wall (smeared/incoherent), 0.45 is the LOCK wall
> (source-identical). Bimodality confirmed in one run. Refit to a single exponent across all points
> (0.2MP smooth @0.45-0.50, 0.1MP chaos @0.50, 1MP lock ≤0.70, 1MP chaos @0.83) gives **γ≈1.6 →
> d\*≈0.75-0.78 @1MP** (between ρ and √ρ; both single-factor laws fail). Ψ and p-cross-1
> discriminators are DEAD; replaced by the **seam z-score** gate. See the GPU-UPDATE blocks in the
> sections below.

## CURRENT MODEL (2026-08-24) — the T_N(d) transfer function / basin sharpening (SUPERSEDES γ)

Fable's post-falsification adjudication unified everything into ONE object. Define **T_N(d) = realized
anchor displacement at nominal denoise d, resolution N** (measure it as `‖x_final[anchor]−clean‖`,
normalized). Data points now: 0.2MP 0.45→~0.45 (near-identity); 1MP 0.45→~0.05-0.1, 1MP 0.75→~0.38.
⇒ **T_N steepens from ~identity @0.2MP into a near-STEP function @1MP, cliff at d≈0.70-0.75.** This one
curve generates every observation:
- **Low-d shelf = LOCK** — posterior collapse to the source basin (per-band SNR ∝ N → denoised≈clean →
  ODE barely moves). The anchor's suppressed *self*-regen (the falsification) and the lock are the SAME
  fact — one read as displacement, one as seam.
- **The cliff = window CLOSURE** — at high N no nominal d gives *intermediate, coherent* realized
  displacement. A deterministic ODE either returns to source (lock) or escapes; escape happens only near
  the cliff, hence LATE (PSI=+3), after neighbors composed → lands mismatched (smear).
- **Lock and chaos are the two SHELVES of one step function; the vanished coherent window is its CLIFF.**

Precise bug restatement: *the knob sets NOMINAL noise-odds; the model's posterior contraction toward the
source scales with N; the user's two requirements — content-correct realized regen (≈the 0.2MP value) AND
in-phase commitment with the front — sit on OPPOSITE shelves of T_N at high N.* The requirements are
genuinely two; the control is one scalar (t_row) feeding both channels (the crux stands); but the deeper
cause of the narrowness is **T's steepening with N**, not the coupling alone. **T_N(d)'s cliff
location/steepness IS the model now — it replaces the γ exponent.** Log a T point every run; its
release-phase counterpart is the m′ calibration curve.

**Two mechanisms, cleanly separated by proof status.** (A) row-local self-denoise suppression (basin
collapse) — **PROVEN** by the falsification. (B) attention-dilution of neighbor-following at high N —
**UNPROVEN**: A does not establish whether neighbors would follow a *resolved* anchor at 1MP. The
isolated-anchor anchor-then-release run is exactly what discriminates B (hold the row resolved; if
neighbors still don't compose around it, dilution/OOD is real → route 3 rises).

**Success gate for any fix (normalized, not nominal):** match the 0.2MP/d=0.45 coherent **fingerprint** —
`(T_realized, p̂∞, amp)` — WITHIN tolerance AND `seam z < 2`. A norm target ALONE is insufficient (a
0.45-sized *smear* passes it — realized-displacement inherits distance≠destination); the p̂∞ direction
term + seam are what separate regen from smear. User visual is the audit, not the loop.

## Detail docs

- [the-real-bug](highres-underdenoise-model/the-real-bug.md) — d_content vs d_blend diverge with resolution; the GAP between them IS the high-res pop; consequences that revise the experiment plan
- [data-runs](highres-underdenoise-model/data-runs.md) — three 1MP/0.5MP/0.2MP GPU runs: 0.75/0.78 smear (window closed), 0.5MP dissociation (anchor messy + neighbors coherent), 0.2MP high-d top-edge
- [experiments](highres-underdenoise-model/experiments.md) — ordered experiment plan post-closure; the FL2VA existence-proof framing; other axes (anchor↔ambient / anchor↔spacing confounds)
- [fix-strategies](highres-underdenoise-model/fix-strategies.md) — anchor-then-release fix; is-a-stable-fix-possible strategy (FL2VA / one-root / MC re-read); loose ends (noise identity, t_pin, audio, β)
- [history-superseded](highres-underdenoise-model/history-superseded.md) — γ≈1.6 GPU-refit, α=ρ calibration law, why √ρ undershoots, bimodality sliding-window model — ALL SUPERSEDED by T_N(d)
- [metrics-detectors](highres-underdenoise-model/metrics-detectors.md) — seam z-score (PRIMARY gate), lock detector (step-count invariance), Ψ/p-cross-1 killed, full instrumentation readout guide
- [crux-and-mechanism](highres-underdenoise-model/crux-and-mechanism.md) — THE CRUX (t_row coupling), three decoupling routes, MC re-read, mechanism space, switched-mode analysis
- [crux-and-mechanism-2](highres-underdenoise-model/crux-and-mechanism-2.md) — Fable's source-spring unification, concrete anchor-then-release params (k_sw/m′/re-noise), basin-widening argument, step-starvation decomposition
