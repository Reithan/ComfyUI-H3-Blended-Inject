<!-- provenance: status + confirmed (residual-hum accounting; "euler CLEAN / ancestral-specific"
     is OVERTURNED by GPU 2026-08-31 — a DETERMINISTIC injection noise in BOTH modalities, isolated
     to OUR node; candidate ranking re-ordered). -->
<!-- verified: 2026-08-31 · branch proto-c2-rho-denoised-r-comp · overturn = user GPU
     (euler-not-clean, silence-prompt, 5-step, official-vs-ours); code fact = read of
     sampler.py:305-335; surviving ladder claims tie to prior GPU rungs + comfy-ref. -->
# Residual accounting: the euler-not-clean overturn + candidate re-rank

Index: [index.md](index.md). Current fix chain: [current-fix.md](current-fix.md).

## Purpose

Account for the residual fade-region noise and route between a CORRECTABLE fix and an INHERENT
floor. As of 2026-08-31 the routing basis CHANGED: the artifact is no longer believed
ancestral-specific — a DETERMINISTIC injection-path noise exists in BOTH modalities, isolated to
OUR node (present in our euler @5 steps, absent in official euler @5).

Config throughout: inject f0, fade `0/0/49/73` (held `[0,49)` m=0, fade-out ramp `[49,73)` fractional),
ease_in_out, audio_mode=fade, min_denoise=0, fixed seed.

Timbre is LOW-INFORMATION (per user): the model reshapes early-gen interference downstream, so the
final artifact's timbre ≠ the raw source. Only the amplitude ladder and the source math carry
information below.

## Detail documents

| File | Content |
|---|---|
| [residual-accounting/overturn.md](residual-accounting/overturn.md) | GPU overturn (2026-08-31): euler-not-clean proof; code fact (`_euler_step` zero C2 corrections); video-noise pointer |
| [residual-accounting/surviving-claims.md](residual-accounting/surviving-claims.md) | PROVEN claims still holding: monotone descent B→v3→v4, NOT init-state-driven (+ round-9 boundary), m=0 heat, input-side static ruled out |
| [residual-accounting/candidates.md](residual-accounting/candidates.md) | Candidates re-ranked: PRIMARY deterministic injection (both modalities), SECONDARY ρ-drift (amplifier), DIAGNOSTIC eta=0, INHERENT FLOOR, RULED OUT |
| [residual-accounting/reconciliation.md](residual-accounting/reconciliation.md) | Two-layer stack + discriminating tests (a)/(b)/(c)/(d) + gate-mismatch tension + naturalization caveat + round-10 δ |

## Confidence + next action

Overturn: HIGH (direct GPU + source read). Mechanism attribution: Fable re-derivation (separate
thread). Next = unified carry correction in `_euler_step` + video-noise mechanism; do NOT build
v7 as primary.

**Round-11b (2026-09-02):** mid-m band (k_d 6–14) is NOT a residual-accounting instance —
euler-deterministic control on identical config is clean (flatness ratio 1.10 vs ancestral 2.83).
Band is ANCESTRAL-SPECIFIC; round-11 PRIMARY attribution is FALSIFIED.
Detail: [../euler-ancestral-per-row-fix/noise-carry-gpu-result.md §Round-11b](../euler-ancestral-per-row-fix/noise-carry-gpu-result.md).
