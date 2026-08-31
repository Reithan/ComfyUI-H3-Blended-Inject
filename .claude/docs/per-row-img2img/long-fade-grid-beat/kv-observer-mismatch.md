<!-- provenance: confirmed (numeric/source-derived — closed-form + CPU trace, linear-control-validated) + REFRAME (split correct-and-required; artifact = linear-vs-curved residue) + OOD-shape DISFAVORED (pending stock-linear test) -->
<!-- verified: 2026-08-28 · CPU trace (steps=20, shift-12 concave schedule) + source (observer_split.py:54, model.py:604-605), branch fix-audio-ancestral-axis-mismatch -->
# Bug E — the KV/observer curvature mismatch: a quantified noise-HIDING seed

Parent: [../long-fade-grid-beat.md](../long-fade-grid-beat.md).
Sibling: [ramp-length-decouple.md](ramp-length-decouple.md) (seam-attention SNR-trough — the propagation half).
Refines the FORMATION seed in [two-stage-heal.md](two-stage-heal.md).

## The two labels a fractional inject row carries

A fractional inject row (0<m<1) is described to the DiT by TWO different sigma-labels:

- **KV / observer broadcast label** `t_obs = 1 − m·σ_g` (observer_split.py:54, the video branch;
  `σ_g` = the global step sigma). This is what OTHER rows READ of it via the spliced K/V.
- **The row's OWN content noise** `σ_row` = the schedule-tail remapped per-row sigma
  (sampler.py `_stream_row_sigma`, dense path: `idx = round(k_d·(steps−i) + i·steps)`,
  `k_d = round(steps·(1−m))`, sampler.py:496,521). Q, gates, and the MLP path see the truthful
  `t_row = 1 − σ_row`; only the K/V that neighbours attend to is relabeled to `t_obs`.

So a row's neighbours are told its state is `t_obs` while its actual latent carries noise `σ_row`.

## Closed form and the curvature identity

With the remap, the row's schedule POSITION is `p_row(τ) = 1 − m(1−τ)`, where `τ = i/steps`.
Let `s(p)` be the sigma schedule as a function of normalized position. The label GAP is:

    Δ = σ_row − m·σ_g = s(1 − m(1−τ)) − m·s(τ)

**Curvature identity.** `Δ ≡ 0` for a LINEAR schedule `s(p) = c·p` (pure curvature effect); and `Δ`
is pinned to 0 at `m=0` and `m=1` for ANY schedule (both fade endpoints). The mismatch is therefore
a schedule-CURVATURE artifact, extremal in the fade interior — not an endpoint or seam artifact.

## Numeric result (CONFIRMED — control-validated)

Trace parameters: steps=20, shift-12 concave video schedule `s(p) = 12p/(1 + 11p)`. A LINEAR
control gave `Δ ≈ 0` everywhere, confirming the method is sound and the effect is curvature.

Result: **Δ > 0 for EVERY fade row** — the row's real content sits at HIGHER sigma than its
broadcast KV label. Content is NOISIER than attending tokens are told. No sign flip anywhere
(except the linear control). Near-invariant across step `i` (the peak shifts slightly toward larger
`m` at later `i`).

- Ratio `σ_row / (m·σ_g)` reaches **7.74×** at m=0.05.
- Absolute `Δ` peaks at **m ≈ 0.18** (`Δ ≈ 0.575` at i=0); `Δ → 0` as `m → 1`; ratio DIVERGES as `m → 0`.

| m | ratio σ_row/(m·σ_g) |
|------|---------------------|
| 0.05 | 7.74 |
| 0.2  | 3.75 |
| 0.5  | 1.85 |
| 0.8  | 1.22 |
| 0.95 | 1.05 |

## Interpretation — worst on the HELD-SEAM SIDE, not the midpoint

The mismatch is WORST on the held-seam side of the ramp (light fades, m ≈ 0.1–0.2), NOT at the
midpoint. The first fade rows just past the held block masquerade as near-clean keyframe anchors
(`t_obs ≈ 1`) while secretly carrying up to ~8× the labeled noise; downstream rows attend to them as
if clean. That is the artifact SEED.

## Δ×mutability — the LIE is held-side, the DAMAGE peaks at the MIDPOINT

The mismatch is BROADCAST-ONLY and the source frame is SELF-IMMUNE. A fractional row's own
Q/gates/MLP use the truthful label `t_row = 1 − σ_row`; only its K/V — what OTHER rows read — carry
`t_obs = 1 − m·σ` (observer_split.py docstring: "the row's own velocity prediction never sees the
observer label"). So a held-side high-Δ row denoises ITSELF correctly. It is a SOURCE of poison,
never a victim; damage lands only on OTHER rows that attend to it.

Damage to a RECEIVER scales by the receiver's own mutability. A row at denoise `m` has visible
deviation-from-clean ∝ `m`, via the composite blend `m·denoised + (1−m)·x` (sampler.py ~line 600).
So effective damage ∝ **Δ(source) × m(receiver)**.

Weighting the numeric Δ trace by `m`:

| m | Δ·m |
|------|-------|
| 0.05 | 0.017 |
| 0.20 | 0.110 |
| 0.40 | 0.196 |
| 0.50 | 0.212 |
| 0.60 | 0.208 |
| 0.80 | 0.144 |
| 0.95 | 0.044 |

`Δ·m` PEAKS at m≈0.5 — the ramp MIDPOINT.

**CONCLUSION:** raw Δ = where the lie is LOUDEST (held seam, m≈0.18); Δ·m = where the lie does
DAMAGE (midpoint, m≈0.5). The observed artifact origin (rows 17–18, m≈0.33–0.55 for 0/0/39/90) sits
on the DAMAGE peak, not the lie peak. The interior/midpoint peak is ROBUST: Δ is pinned to 0 at m=0
and m=1, so any mutability weight that vanishes at m=0 and rises in `m` drags the product's maximum
toward the center regardless of the weight's exact shape.

Tag: confirmed-numeric (Δ trace) + source-derived (self-immunity from observer_split.py); first-order
on the mutability weight.

## Convention reconciliation — the two labels are the SAME quantity (confirmed)

Stock ComfyUI masked-denoise presents a soft-masked token at noise `σ·(1−m_keep)`, where its `m`
is the KEEP fraction. Our observer label is `σ·m_denoise` (observer_split.py:54, video branch).
Since `1 − keep = denoise`, these are the SAME quantity — their `(1−m)` = our `m`. NO sign
discrepancy; the conventions just name the mask complement differently.

The DiT natively computes the per-token label `1 − m·σ` (model.py:604-605; observer_split.py:54
docstring "mirrors the model's own label formula") — LINEAR in the denoise fraction. Our remap
instead sends the frame's SELF label to the TRUE CURVED `σ_row` (the `steps·m`/`k_d`
compressed-schedule sigma), NONLINEAR in the denoise fraction. The gap `Δ = σ_row − m·σ` (closed
form above) is exactly this linear-vs-curved difference — this section names WHY they differ.

## The REFRAME — the split is correct-and-REQUIRED; the artifact is its residue

The observer split is NOT a defect. It is DELIBERATE and REQUIRED:

- Observation (the K/V other rows READ) MUST carry the stock LINEAR label `1 − m·σ` — that is the
  label the model's trained temporal/neighbor-inference was built around.
- Self-evolution (Q / gate / MLP / residual) MUST carry the TRUE CURVED `σ_row`, or the keyframe
  ghost returns.

The mid-fade artifact is therefore the IRREDUCIBLE RESIDUE of broadcasting a linear label over a
curved truth — not a bug in our decoupling. It is loudest held-side (raw Δ, m≈0.18) but most
damaging mid-ramp (Δ·m, m≈0.5). The "damage" is the model's trained local inference painting
neighbor content — labeled cleaner than it truly is — in as real structure (→ moiré / streamers).

Frame it as: the split does not CAUSE the artifact so much as it TRADES the keyframe ghost for it.
The linear-vs-curved gap CANNOT be zeroed by any single label choice — pick linear and neighbors
read a lie; pick curved and self-evolution ghosts. The residue is what remains.

## 2026-08-31 overturn — residue generalizes beyond Bug E's gate. theory (UNVERIFIED).

A GPU overturn (see [../c2-rho-fix-paths/residual-accounting.md](../c2-rho-fix-paths/residual-accounting.md))
generalizes this residue BEYOND Bug E's ramp≥51 gate: deterministic fade noise is apparently present
at ramp=24 (config 0/0/49/73), EXPOSED at 5 steps and NATURALIZED into diegetic sound at 20 steps.
The confirmed Δ math below is unchanged; this only widens WHERE the residue is claimed to act.

## OOD-of-the-fade-SHAPE — DISFAVORED (pending the discriminator test below)

An earlier note recorded a per-frame m-gradient OOD as an OPEN co-cause. New reasoning DISFAVORS it:
stock/MC carry the SAME per-frame m-gradient (the fade SHAPE) yet work on fades WITHOUT moiré. The
shape is thus present in the WORKING config too, so it is not the driver — the artifact isolates to
the self≠observation DECOUPLING (curved self vs linear obs label), i.e. the curvature Δ.

Epistemic nuance: "the model handles fade observation" proves it is CALIBRATED to the LINEAR obs
label (the break under un-decoupled remap shows this); it does NOT prove the exact shape was in
TRAINING (per-token training + generalization explains it too) — moot for root-cause.

Residual caveat (PENDING the discriminator test, not UNVERIFIED-open): the refutation assumes
"stock/linear long fade does NOT moiré," not yet directly confirmed.

## #81 prediction under the reframe — a TRADE, not a vanish

The observer-split kill-switch (#81: broadcast the true curved `σ_row` to K/V too) predicts a
TRADE, not a clean win: moiré should VANISH **and** the keyframe GHOST should RETURN. If moiré
vanishes with NO ghost return, the linear-broadcast story is INCOMPLETE — reviving a regime/OOD
co-cause. So #81 must watch BOTH symptoms (streamers AND ghost), not only the streamers.

## Discriminator test — stock-linear long fade (DISTINCT from #81)

NEW test, distinct from #81: disable the remap AND the split (pure composite / MC-equivalent = ROW 1
of the three-config table in [survival-model.md](survival-model.md): linear self + linear obs) on
0/0/39/90 (ramp 51); check for moiré.

- Predicted moiré-FREE-but-ghosty → shape is fine, curvature Δ is the whole story (OOD dead). If it
  ALSO moirés → the long-fade regime contributes independent of the decoupling (OOD/regime survives).
- ROW 1 of the table; #81 tests ROW 2 (curved self + curved obs) — NOT the same thing.

## Cross-link — seed here, manifestation elsewhere

The artifact MANIFESTS mid-ramp (rows 17–18), NOT at this held-side seed location. Bridging
held-side-seed → midpoint-manifestation still REQUIRES the attention-propagation / seam-attention
argument in [ramp-length-decouple.md](ramp-length-decouple.md). Curvature supplies the SEED and its
SIGN; attention supplies the WHERE and the length dependence.
