<!-- provenance: theory (experiment order user-sequenced 2026-08-24; proto nodes built on debug branch; FALSIFIED per-frame lever 1) -->
<!-- verified: 2026-08-24 · mc @d299ea5 · comfy-ref @b78cec87 (+ GPU: MC keyframes blend r40/r60 @1MP, anchors m=0-wrong; drag-vs-freeze discussion 2026-08-24) -->
# Experiment order and proto nodes

Index: [conditioning-row-inject](../conditioning-row-inject.md). Aug falsifications that drive this order: [aug-mechanism](aug-mechanism.md). Fade and decoupler options: [fade-and-decoupler](fade-and-decoupler.md).

## FRACTIONAL denoise vs FRACTIONAL strength — the decisive comparison (user framing, 2026-08-24)

Endpoints COINCIDE: strength→1 (`aug≈0.999`) ≡ denoise→0 = near-perfect keyframe copy; strength→0 ≡
denoise→1 = free gen, no keyframe. **The MIDDLE is the open question — and the two paths may diverge
at HIGH RES in the way that matters:**
- Latent fractional-`m` routes through the DENOISE TRAJECTORY = exactly what `T_N` collapses @1MP
  (our bug). Result KIND: keyframe with errors partially sanded off (STRUCTURE-PRESERVING).
- Cond fractional-strength routes through ATTENTION = GPU-confirmed high-res-ROBUST (MC test). Result
  KIND: a FRESH coherent frame LEANING toward the keyframe (may render "correctly" but DRIFT identity;
  no structure-retention guarantee — output starts from noise, not the keyframe).
**Hypothesis (UNVERIFIED, high-value):** cond-strength may be a high-res-robust fractional dial
precisely where latent-`m` fails — because it uses the attention route (robust) not the trajectory
route (collapsed). Test = fixed frame, sweep latent-`m` vs cond-strength at 1MP, compare realized
regen + keyframe-identity retention + seam z.

## EXPERIMENT ORDER (user-sequenced 2026-08-24 — do NOT jump to hybrid)

1. **Fractional cond-strength, SOLO** (per-keyframe lever 1, sweep). ⚠ **UNACHIEVABLE per-frame — see ⚠
   FALSIFIED.** Per-frame lever 1 is broken (no per-cond-row timestep); only GLOBAL `aug` gives coupled
   fractional-strength, and it hits all refs. The attention-route fractional question can still be probed with
   GLOBAL `aug` on a single-inject graph, but there is no native per-frame cond-strength knob.
   [historical intent:] Does attention-route fractional
   survive @1MP where latent-`m` doesn't? Faithful AND fixed → likely DONE for single-frame (accept
   the goal-compromise: guidance stays in conditioning; but a LONE anchor = 1-2 cond tokens = CHEAP on
   attention — the budget worry was many-refs/long-fades, and latent stays primary for video injects).
   Drifts keyframe identity → need latent structural anchoring →
2. **Live-cond-mirror** (user variant #4): each step, broadcast the anchor's per-step DENOISED latent
   state (NOT the stale clean keyframe) into its cond row. = a HIGH-TRUST MEGAPHONE for the anchor's
   LIVE content: neighbors attend to the anchor's actual evolving (45%-regen) content WITHOUT the
   trust-discount its low `t_row=0.45` label imposes = the crux decoupled, sourced from the live latent
   (fixes the stale-keyframe mini-seam of the §hybrid above). CAVEATS: (a) does NOT fix the anchor's
   OWN under-denoise — still routes through the collapsed latent trajectory (mechanism A); (b) POSITIVE-
   FEEDBACK RISK — anchor attends to its own broadcast (`mask=None`, can't exclude self, model.py:195)
   → may self-reinforce into a fixpoint and LOCK HARDER. Right structure for the neighbor/trust half;
   leaves the anchor-self-denoise half = back to AtR/route-1.
   **Drag-vs-freeze sharpening (discussion, 2026-08-24):** static clean cond pulls toward a FIXED target
   every step (confirmed freeze). Mirror pulls toward the anchor's own step-(k-1) estimate — a MOVING
   target, i.e. a DRAG/lag force. Plausible outcome: slower effective denoise (compensable by raising
   nominal d). The lock-harder fixpoint occurs only if drag beats denoise drive per step — that's
   empirical; one run decides. Mirror content is always the model's own current clean estimate ⇒ the
   clean label stays TRUTHFUL at every step ⇒ no contagion channel; neighbor-view is matched to the
   anchor's final state. Caveat (a) STANDS — mirror does not fix the anchor's own resolution; it is a
   component (reference source inside timed-removal or schedule builds), not a standalone fix.
   See [timed-cond-removal-prototype](../timed-cond-removal-prototype.md) for the build-first mechanism.
Goal-compromise note: a cond-only fix does NOT advance the repo goal of moving guidance OUT of
attention-starved conditioning INTO the latent — but "workable beats elegant-but-broken," and for the
single-frame-inject case (the exact case that's broken) the attention cost is negligible.

## PROTO NODES BUILT (debug branch, 2026-08-24 — no tests, prototypes)

- **`H3SetCondAug`** ("H3 Set Cond Aug (proto, global)"): writes the native
  `minimax_visual_cond_noise_aug` scalar via `conditioning_set_values`. EXACT native mechanism but
  GLOBAL — hits every keyframe cond row AND every ref2va reference image on the stream. Fine for a
  single-inject probe; WRONG when the graph also uses image refs (user hit this 2026-08-24).
- **`H3SetKeyframeStrength`** ("...per-frame"): lever-1 per-frame control. ⚠ **BROKEN — GPU-FALSIFIED
  2026-08-24** (see the ⚠ FALSIFIED section). Blends the target keyframe latent toward noise but does NOT
  relabel its timestep (impossible per-row for cond rows) → DiT reads noise as clean → persistent static in
  the output. Not fixable within the cond path. Leave in the codebase as a documented dead-end prototype; use
  **global `H3SetCondAug`** instead when no conflicting refs, or move fractional control to the LATENT path.

Related: [motion-context-comparison](../motion-context-comparison.md) ·
[native-h3-mechanism](../native-h3-mechanism.md) · [our-architecture](../our-architecture.md).
