<!-- provenance: reference (pointer index of RUN experiments; results live in the linked home docs) -->
<!-- verified: 2026-08-25 · cross-checked against home docs @proto-latent-hold-release -->
# Experiment Run Index

Before proposing an experiment, grep this file first. If an ID exists the run is done — check its
home doc for data and conclusions. This file is a pointer table only; results live in the links.

Experiment rows are split into two child files (original exceeded the 12,000-char ceiling at 40 rows):

- [early-series.md](experiments-run/early-series.md) — RES, DATA, HYP, early HOLD (HOLD-2 through HOLD-14)
- [hold-continued.md](experiments-run/hold-continued.md) — HOLD-15 through HOLD-25 (current), STILL, MC, AUG, AUD, BUG-B, DD, VER

## DEFERRED / NOT-RUN

- **m-sweep at 1MP** (m=0.8/0.9 hold=0.5 — does the high-m amount band stay res-invariant, or only near-full m=0.99?) — [resolution-invariance-goal.md](latent-hold-release/resolution-invariance-goal.md)
- **Isolated single 1MP frame at SDEdit partial strength** (no hold/neighbors/correction — clean or smeared?
  discriminates base-model deficiency vs our mechanism; RES-1 already suggests smeared=base) —
  [amount-floor-and-step0-redesign.md](latent-hold-release/amount-floor-and-step0-redesign.md)
- **Low-m at BOTH res** (m=0.7, +0.6 if cheap, hold=0.5, 0.5MP & 1MP — floor-vs-late-release + low-end res-invariance + native-m-vs-contagion split in one) — [knob-design-open-questions.md](latent-hold-release/knob-design-open-questions.md)
- **Hold vs no-hold at matched level** (Test B — is the step-0 clean hold needed vs plain re-noise+tail?) — [held-keyframe-m-vs-sdedit.md](latent-hold-release/held-keyframe-m-vs-sdedit.md)
- **Timeline-length sweep at fixed res & m** (cleanest basin-vs-dilution discriminator; DEFERRED by user) — [anchor-denoise-m-vs-res.md](latent-hold-release/anchor-denoise-m-vs-res.md), [the-real-bug.md](highres-underdenoise-model/the-real-bug.md)
- **0.2MP re-confirm on hold setup** (m=0.5, hold=0.5 → should look ~0.5 if res is the cause) — [the-real-bug.md](highres-underdenoise-model/the-real-bug.md)
- **Standalone length=1 correctness baseline** (isolates per-frame vs temporal; requires latent splice) — [baseline-question.md](highres-singleframe-underdenoise/baseline-question.md)
- **Two-pass oracle** (anchor at content-correct d; measure seam ceiling at 1MP) — [experiments.md](highres-underdenoise-model/experiments.md)
- **0.2MP denoise-width sweep (exp1b)** (confirm window width at low N; load-bearing T_N prediction) — [experiments.md](highres-underdenoise-model/experiments.md)
- **Isolated single-anchor anchor-then-release @1MP (exp#0)** — [experiments.md](highres-underdenoise-model/experiments.md)

## UNVERIFIED (seed claim, home not found)

- **0.1MP odds-linear probe** (seed: d≈0.30 @0.1MP testing α=ρ law; home `history-superseded.md`). Home doc lists this as a
  proposed KEY TEST only. The 0.1MP at d=0.50 IS confirmed as RES-2; that is likely the entry the seed meant.
