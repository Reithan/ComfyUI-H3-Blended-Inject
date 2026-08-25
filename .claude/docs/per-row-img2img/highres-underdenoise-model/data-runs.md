<!-- provenance: theory (1MP GPU validation 2026-08-24; three DATA runs bracketing lock/coherent/chaos walls + 0.5MP dissociation) -->
<!-- verified: 2026-08-24 · 1MP GPU runs (0.45/0.60/0.68 lock, 0.75/0.78 smear=window closed, 0.83 chaos) + Fable adjudication -->
# DATA runs — 0.75/0.78, 0.5MP crossover, and 0.2MP high-d

Index: [highres-underdenoise-model](../highres-underdenoise-model.md). The bug framing these runs test: [the-real-bug](the-real-bug.md). Metrics from these runs: [metrics-detectors](metrics-detectors.md).

### DATA — 0.75/0.78 @1MP run (new instrument, 20-step euler, 2026-08-24)
r40 [d=0.75] |out−clean|=0.306 |out|=0.797; r60 [d=0.78] |out−clean|=0.329 |out|=0.815 (realized anchor
displacement ~0.38-0.40 relative — under-denoise persists even at high nominal d). Both = smeary/smear
mess (user visual). Hard reads:
- **WINDOW CLOSED @1MP — now DATA, not prediction.** 0.68 locks, 0.75/0.78 smear → lock→chaos boundary
  ~0.70-0.75 with NO coherent middle. `seam zmax`: r40 **+4.84 [POP]**, r60 **+4.82 [POP]** (gen-pair
  mu=0.350 sd=0.080 n=38). The seam gate fired cleanly and correctly — it's the reliable discriminator.
- **MECHANISM — the anchor commits LAST, at EVERY d (structural impossibility result).** neighbors
  t_c=13/19 (σ@tc=0.882); anchor t_c=16/19 (σ@tc=0.618) → **PSI=+3**. BUT (Fable sharpening): the anchor
  also committed late at the LOCK wall (Ψ=+3.25/+4.25). So PSI=+3 is NOT chaos-specific — it says
  **under d-only control the anchor is last-to-commit for EVERY d in (0,1)**: lock returns late, chaos
  lands late. No knob setting ever lets the anchor commit WITH or BEFORE the front. This is the
  architectural indictment of d-sweeping in one number, and it makes **route 1 (anchor committed from
  step 0) the only mechanism on the table that fixes it.** (Ψ stays DEAD as a wall-discriminator; it's a
  structural argument, not a gate.) → Timed-removal via knob C (cond token) is a route-1 instance that
  bypasses the knob-B coupling; this result backs it — see
  [isolated-frame-attention-support](../isolated-frame-attention-support.md).
- **Tellers under-performed; trust the seam.** rho_ret 0.73/0.78 (not ≪1, not ~1 — ambiguous at smear);
  phi_bar +0.14/+0.12 (weakly POSITIVE despite chaos — did NOT flag it); p̂ cos(D,clean) 0.88-0.91 high,
  amp→1.04. The ρ_ret/φ̄ tellers did not discriminate at these points; seam z-score did. Re-weight toward
  seam as THE gate; treat ρ_ret/φ̄ as soft context only.

### DATA — 0.5MP crossover run (f40@0.50, f60@0.45, 20-step, 2026-08-24) — THE DISSOCIATION
r40 |out−clean|=0.186 |out|=0.775; r60 |out−clean|=0.170 |out|=0.789. **Both seams COHERENT**: r40
zmax=+0.97, r60 zmax=+1.23 (gen-pair mu=0.386 sd=0.100 n=38) — UNDER the 1.5 gate. Neighbors blend
"almost as good as 0.2MP" (user visual). BUT the anchors themselves are **messy — polygonal edges, raw
texture patches** (user visual). PSI=+4.50 both; p̂∞ 0.96/0.97; amp→1.10; rho_ret 0.39/0.34; phi_bar
+0.30/+0.22.
- **THE HEADLINE — a resolution-ORDERED dissociation of the two halves.** 0.2MP: anchor clean + neighbors
  clean. **0.5MP: anchor MESSY + neighbors COHERENT.** 1MP: anchor locked/smeared + neighbors smear
  (closed). ⇒ **the fractional anchor's OWN self-denoise degrades FIRST (lower N); neighbor-attention
  degrades SLOWER (higher N).** User-predicted, now data.
- **Refines (does NOT contradict) Fable's one-mechanism adjudication.** Single cause (basin sharpening /
  rising per-band SNR) still holds, but it manifests on **TWO axes with SEPARATED N-thresholds**, and the
  anchor-self-denoise axis crosses first. "Can't set them as two independent d-knobs" stands; "they
  degrade as one monolith" does NOT. This is the clean separation the isolated-anchor AtR run was meant to
  find — 0.5MP handed it to us for free.
- **Direct support for the cond-strength plan.** Neighbor-attention is the ROBUST half → routing the
  anchor's rendition through ATTENTION (cond-strength) may inherit that robustness where the
  latent-TRAJECTORY route (mechanism A) goes messy. Elevates exp #1 (fractional cond-strength solo).
- **INSTRUMENTATION GAP (important).** NONE of our metrics caught the messy anchor: |out−clean|, p̂∞, amp,
  seam z all read "fine" while the frame is visually raw. Every metric pools SPATIALLY → within-frame
  texture incoherence is INVISIBLE. **seam-z-as-gate is necessary but NOT sufficient**; we lack an
  anchor-quality / high-freq-coherence signal. A 0.5MP "coherent" verdict can still be a bad frame.

### DATA — 0.2MP high-d run (f40@0.75, f60@0.70, 20-step, 2026-08-24) — LOCATES the 0.2MP window top-edge
r40 [d=0.75] |out−clean|=0.294 CLEAN + perfect blend + proper regen (user visual); r60 [d=0.70]
|out−clean|=0.269 but retains POP/SMEAR (user visual). seam: r40 zmax=+2.06, r60 zmax=+2.73 (gen-pair
mu=0.331 sd=0.079 n=38). PSI r40=+1.75 r60=+3.00; φ̄ r40=+0.698 r60=+0.527; ρ_ret 0.575/0.464; p̂∞
0.89/0.93; amp→1.03.
- **The window at 0.2MP is WIDE but NOT infinite — top-edge ~0.70-0.75, CONTENT-dependent.** r40 clean at
  the HIGHER d (0.75) while r60 smears at the LOWER d (0.70) ⇒ the edge is not a pure d-ceiling; it's
  frame/content-fuzzy (the content-confound, now visible AT the edge). Confirms the ESSENCE of the user's
  "arbitrary-d @0.2MP" memory (window genuinely wide; 0.75 works where 1MP closes) with ONE correction:
  not truly arbitrary — a fuzzy chaos edge sits ~0.72 and slides DOWN to ~0.45 @1MP as the basin sharpens.
  The window narrows from the TOP with N; it does not appear from nowhere. Consistent w/ T_N steepening.
- **seam-z POP threshold is too tight at 0.2MP/high-d.** r40 z=+2.06 looks PERFECT; r60 z=+2.73 is truly
  smeary. Ordering correct + r60 flag true, but z≈2.0 is NOT a hard fail here → treat 2.0-2.7 as a WATCH
  band, anchor texture by eye is the arbiter. (Norm: r40 CLEAN @0.294 vs 1MP SMEAR @0.306 — near-identical
  magnitude, opposite quality. Norm ≠ quality, again.)
- **PSI & φ̄ REGAINED discriminating power at 0.2MP.** r40(good) PSI+1.75/φ̄+0.698 vs r60(bad)
  PSI+3.00/φ̄+0.527 cleanly separate — unlike the 1MP smear where both went flat. ⇒ teller reliability is
  RESOLUTION-DEPENDENT (alive @0.2MP, dead @1MP-smear); ρ_ret stays ambiguous. Don't retire PSI/φ̄
  wholesale — they work in the wide-window régime, fail in the collapsed one.
