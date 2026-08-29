<!-- provenance: confirmed (GPU two-stage FORMATION+HEALING + 4 configs incl. 0/0/29/80 inject_at=17→ERROR; ramp monotone) + confirmed-numeric (KV-observer curvature seed) + theory (seam-attention SNR-trough; position/free-head single-point) -->
<!-- verified: 2026-08-28 · GPU 0/0/39/107, 0/0/38/89, 0/0/29/80 +inject_at=17→ERROR (243f, no truncation) + KV-observer numeric seed (kv-observer-mismatch.md), branch fix-audio-ancestral-axis-mismatch -->
# Bug E — two-stage refinement: FORMATION (ramp) ∧ NOT-HEALED (held)

Parent: [../long-fade-grid-beat.md](../long-fade-grid-beat.md).
Sibling: [ramp-length-decouple.md](ramp-length-decouple.md) (the M-B decoupling matrix this refines).

This REFINES, does not overturn, M-B. Every M-B numeric prediction is kept: "error output"
⟺ held ≥ ~28 AND ramp ≥ 51 on all configs. The refinement DECOMPOSES that single rule into two
mechanistically distinct stages, so "error output" = FORMATION ∧ NOT-HEALED.

Define (as in the decouple matrix): **held = ekf − skf**, **ramp = efo − ekf**, **tail = clip − efo**.

## Three new GPU configs (2026-08-28)

| Config | clip | held | ramp | tail | result |
|--------|------|------|------|------|--------|
| 0/0/39/107 | 107 | 39 | 68 | 0 | ERROR |
| 0/0/38/89 | 90 | 38 | 51 | 1 | ERROR |
| 0/0/29/80 | 90 | 29 | 51 | 10 | MIXED |

MIXED (`0/0/29/80`) is the key observation: the interference pattern FORMS mid-fade (~step 5, around
the m≈0.5 rows) and is VISIBLE in the preview, matching the error cases — then HEALS / denoises back
to a clean final output. Formation and survival are therefore separable events, not one gate.

## Stage 1 — FORMATION (the ramp term)

`ramp = efo − ekf ≥ 51` → a mid-fade interference pattern forms near the fade midpoint (m≈0.5 rows,
~step 5). This is **MONOTONE in ramp with NO UPPER EDGE** found through ramp = 68
(`0/0/39/107`: ramp 68, held 39 → ERROR, tail broken to 0 by clip=107).

**Correction of prior belief:** the old apparent "ramp band 51–60/64" UPPER EDGE was an ARTIFACT of
the clip=90 weld, where held = 90 − ramp. Pushing ramp past ~60 on clip=90 only looked "clean"
because it dropped held below the healing threshold — not because formation stopped. Breaking the
weld (clip=107, held decoupled from ramp) shows ramp = 68 still forms and errors. There is no ramp
formation ceiling in the tested range.

Consequence: the **standing-wave / resonant-band** framing is RETIRED to a tombstone (see
[ramp-length-decouple.md](ramp-length-decouple.md)). A resonance requires a tuned/banded ramp
length; formation is monotone with no upper edge through ramp 68, so no resonant band exists.

**FORMATION seed (CONFIRMED — numeric/source): the KV/observer curvature mismatch.** Every
fractional inject row broadcasts a K/V observer label `t_obs = 1 − m·σ_g` while its OWN content sits
at the schedule-tail remapped sigma `σ_row`. The gap `Δ = σ_row − m·σ_g` is a schedule-CURVATURE
artifact (`Δ ≡ 0` for a linear schedule; pinned to 0 at m=0 and m=1). Numerically (steps=20, shift-12
concave schedule, linear-control-validated) `Δ > 0` for EVERY fade row — content is NOISIER than
attending tokens are told — and the ratio `σ_row/(m·σ_g)` peaks on the HELD-SEAM SIDE (up to ~7.7×
at m≈0.05; absolute Δ peaks at m≈0.18), NOT at the midpoint (the LIE is held-side; the DAMAGE peaks
at the midpoint via receiver-mutability, Δ×m — see kv-observer-mismatch.md). First fade rows past the
held block masquerade as near-clean anchors while secretly ~8× noisier: the seed.
Full derivation + Δ×m table: [kv-observer-mismatch.md](kv-observer-mismatch.md).

**Propagation half (theory — UNVERIFIED): seam-attention reach vs a mid-ramp SNR trough.** The seed
is held-side, but the artifact MANIFESTS mid-ramp — bridging the two needs attention propagation.
The fade ramp is bounded by two high-confidence seams: hold→fade (near m=0, clean-anchored) and
fade→end (near m=1, generation-anchored). Attention from these anchors attenuates inward with
distance. The ramp MIDPOINT is simultaneously (a) lowest-SNR (m≈0.5, half-noised) and (b) furthest
in row/token distance from both anchors. A longer ramp pushes the low-SNR center beyond the reach of
seam-anchored attention → the model cannot resolve it → noise/moiré. This yields monotone-in-length
(longer = worse), a midpoint-localized MANIFESTATION, and a binary threshold (seam-attention reach vs
the SNR trough) — all WITHOUT a resonance. NOTE: the SNR trough is where the artifact SHOWS; it is
NOT where the label mismatch is largest (that is held-side, per the seed above).

Honest nuance: this mechanism predicts the DENSITY of rows near m≈0.5 should matter, yet interp
ease_in_out→linear did not change the outcome. Not fatal — both variants sit over-threshold at
ramp 51, and a binary result hides sub-threshold differences. A clean discriminating test would
vary interp right at the ramp 50↔51 boundary.

Unified survival mechanism (attention-share dilution) + the held-vs-position confound:
[survival-model.md](survival-model.md).

## Stage 2 — SURVIVAL / HEALING (the held term)

Whether the FORMED pattern persists to the final output is governed by **held** (the frozen-prefix
size), a SOFT boundary, not M-B's hard cliff:

- held ≥ ~30 → does NOT heal → error survives to output.
- held < ~28 → heals → clean output.
- held ≈ 29 → ON the healing boundary → PARTIAL heal → MIXED (`0/0/29/80`).

The earlier 0.3MP "clean" resolution result is now reinterpreted as **REAL HEALING** (more tokens
to denoise the formed pattern away), NOT mere dilution hiding the signal. Steps and surrounding-frame
context plausibly also modulate healing capacity (untested).

## Survival/healing driver — NOT yet isolated (RETRACTS "the healer is held")

The earlier claim that survival is governed by held and NOT the free tail was OVERCLAIMED — it
rested on confounded data (held, absolute fade position, free-head, and free-tail all covaried).
RETRACTED. The survival/healing driver is NOT yet isolated: held-size, absolute fade position,
free-head, and free-tail all remain confounded.

What IS established, all at **held = 39** (a robustly non-healing value):
- free tail 0→17 (`0/0/39/90` on clip 107) → ERROR;
- free head 0→17 AND a +17 absolute-position shift (`inject_at=17` on `0/0/39/90`) → ERROR.

So at held=39, neither a free tail, nor a free head, nor a +17 position shift heals. But held-size
is NOT the only survival factor — see the new held=29 datum below.

## Position / free-head degrades healing (NEW GPU datum, 2026-08-28)

`0/0/29/80` (held29, ramp51) — the MIXED knife-edge that HEALED at `inject_at=0` — went to FULL
ERROR at `inject_at=17`. This is a CLEAN position/free-head result, NOT an anchor-truncation
artifact: the output generation is 243 frames (10s), so `inject_at=17` does NOT truncate
(last_latent = 97, well inside; free head [0,17) and free tail [107,243) both intact). An earlier
in-chat truncation hypothesis was RETRACTED once the 243-frame output length was confirmed — do not
reintroduce a truncation confound here.

Reading (TENTATIVE — single knife-edge point, theory/UNVERIFIED): shifting the fade later / adding a
free head DEGRADES healing. Likely mechanism: at `inject_at=0` the fade head sits at frame 0 with no
upstream content, so no noisy signal can propagate in from before it. A free head [0,17) is
freely-generating (noisy-early) content sitting immediately upstream of the hold→ramp seam, and it
can feed mislabeled signal into the fade. So CLEAN anchors (held keyframes, settled downstream
generation) heal; a NOISY free head hurts.

NEEDS REPLICATION — pending a multi-offset test: `0/0/39/90` and `0/0/25/80` at `inject_at ∈ {0,17,34}`.

**Survival driver status (UPDATED):** held-size is still NOT isolated, and position/free-head is now
shown to matter near the boundary. They are CO-FACTORS, not either/or — held-size and
position/free-head both modulate survival.

inject_at semantics (source-verified) make `inject_at=17` a valid free-head / position probe:
sfi/skf/ekf/efo are CLIP-FRAME indices within the injected clip's own content; inject_at offsets
the whole envelope into the target timeline — clip frame k → latent frame inject_at+k
(envelope.py:139-141, 208, 215-216). evaluate_envelope only returns rows inside
[inject_at+sfi−1, inject_at+efo] (envelope.py:215-220); rows before inject_at get no entry and are
left FREE (m=1). So inject_at is a position-shift + free-head probe, NOT an sfi/skf reshape — held
and ramp (clip-frame) are unchanged by inject_at.

**Priority experiment (PARTLY ANSWERED):** shifting a healing config now HAS a first datapoint —
`0/0/29/80` (held29, healed at inject_at=0) ERRORed at `inject_at=17`, so absolute
position / free-head DOES matter (it is not pure held-size). Remaining work is replication and
isolation: run `0/0/25/80` (held 25, CLEAN) and `0/0/39/90` at `inject_at ∈ {0,17,34}` to map how
position and held-size trade off. This SUPERSEDES the earlier-proposed 10/10/49/100 fade-in test.

## Open: held-size vs fade absolute-position — STILL CONFOUNDED

The two levers are now both shown ACTIVE but not yet separated: position/free-head matters
(`0/0/29/80` healed at inject_at=0, ERRORed at inject_at=17) AND reducing held heals. What is
unseparated is how they TRADE OFF. The held=39 `inject_at=17` run only shows a +17 shift does not
heal AT held=39; the held=29 run shows the same shift BREAKS a config that otherwise healed. The
decisive separator is the priority
experiment above: shift a HEALING config (`0/0/25/80`, held 25) via `inject_at=17` and check whether
it stays CLEAN (held drives survival) or errors (position matters).

Position-invariance at FIXED held+ramp is partially shown: `0/0/38/89` (whole fade shifted −1 frame
vs `0/0/39/90`, ramp held at 51) is still ERROR.

## Status of the earlier open question

`0/0/39/107` answers the previously-open "band vs monotone / ramp upper edge" question: the upper
edge is **RESOLVED as monotone** (no edge through ramp 68), superseding the "upper edge untested / do
not assume monotone" caveats in the sibling and index docs.
