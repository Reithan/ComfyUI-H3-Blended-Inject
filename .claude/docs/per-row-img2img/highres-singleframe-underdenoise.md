<!-- provenance: theory (H1 CONFIRMED by elimination; bimodality GPU-confirmed 2026-08-24 — lock+chaos walls bracketed @1MP; d* refit + fix pending) -->
<!-- verified: 2026-08-24 · repo @debug-single-frame-underdenoise · 1MP run (row40@0.83=chaos, row60@0.45=lock) -->

> **VERDICT (2026-08-23): H1 — sigma-shift is NOT auto-scaled with token count → resolution-
> dependent effective denoise.** (`shift_video` is a USER config ~10–16+, `shift_audio`~3, via
> ModelSamplingMiniMaxH3 — NOT a hardcoded constant. Unlike SD3/Flux, H3 does not raise shift with
> sequence length, so at a fixed user shift a given `min_denoise` weakens as resolution/tokens grow.)
> Confirmed by elimination across two full 1MP runs (EasyCache on AND off). The keyframe row
> under-denoises to `|out−clean|≈0.18` (≈21% of a full gen 0.86; a true m=0.5 blend ≈0.4),
> **stable across the EasyCache toggle** (0.182 cached / 0.187 uncached — cache is a ~1% suppressor,
> not the cause; uncached run had no skip line + 1507s vs 1119s runtime). Row denoises fully
> (`|inp|` 0.56→0.70, not frozen) but its predicted x0 tracks `clean`=source: at ~1032 tokens/frame
> the m=0.5 half-noise lerp carries enough per-band SNR for exact source reconstruction.
> **All alternatives REFUTED with data:** freeze, init-lerp/m-misalign, VAE-encode, VAE-special-case,
> EasyCache. Kitchen-Attn (Sage-attn) still technically on for both runs but low prior (clean `den`
> convergence; H1 fully accounts) — a default-attn run would close it if ever doubted.
> **FIX = resolution-corrected effective-m** (= "inject more noise", auto-scaled by token count) —
> DIRECTION confirmed (0.7 & 0.95 @1MP progressively restore the blend), but the **α=√5 / target-R
> CALIBRATION is REFUTED** by the resolution ladder (see "RESOLUTION LADDER" below): displacement R
> does NOT discriminate pop from smooth (0.2MP-smooth R can be LOWER than 1MP-pop R). The failure is
> **bimodal** — too-high res pops to SOURCE (under-denoise), too-low res (0.1MP) pops CHAOTICALLY;
> **0.2MP is a goldilocks** where the frame departs source COHERENTLY. Real discriminator = coherence
> / phase with the denoising front, NOT magnitude. Recalibrate the effective-m map on coherence
> proxies: raise d @1MP until `|out|₄₀`→~0.735 and `|inp|` rebound→~0.67 (the 0.2MP-smooth values),
> then confirm the blend is coherent (not 0.1-style chaotic) VISUALLY. Still preferred over raising
> `shift_video` (TARGETED to inject rows vs globally reshaping motion/detail + `audio_scale`). Open
> design Q: n_ref (0.2MP is the confirmed intuitive baseline — 99% seamless at d=0.5/0.45).
> **UPDATE 2026-08-24 (1MP GPU run — α=ρ FALSIFIED; see
> [highres-underdenoise-model.md](highres-underdenoise-model.md)):** injecting f136→row40 @**0.83**
> (the ρ up-map) came out **CHAOS/smeared (incoherent)**, while f204→row60 @**0.45** stayed **LOCK
> (source-identical)** — the two failure WALLS bracketed in one run. So ρ (γ=2) overshoots into chaos;
> refit to a single exponent **γ≈1.6 → d\*≈0.75-0.78 @1MP** (between ρ and √ρ). **FOUR régimes**:
> lock→coherent→chaos→generic-gen (d=0.95 was régime 4 = inject lost, not a blend). Discriminators
> **Ψ=0 and p-cross-1 are DEAD** (lock was excursion-and-return, not an early basin lock) → replaced
> by the **seam z-score** gate (U-shaped in d; min locates d\*) + ρ_ret/φ̄ mode-tellers. AND `d` is not
> the only knob: **anchor↔ambient distance** and **anchor↔anchor spacing** modulate the window (see
> model doc "Other axes"). Next: probe d≈0.75 @1MP; if it also fails → window closed → anchor-then-release.
# High-res single-frame under-denoise ("pop") — investigation

**Symptom (user, GPU).** SINGLE-FRAME stills injected at fractional `min_denoise` (f136 @ 0.5,
f204 @ 0.45) come out **near source-identical** — Photopea retouch artifacts (hard polygon thumb
edges, distorted texture, wrong lighting) preserved — at **1.0 MP / 40-step euler**, but denoise
**cleanly at 0.2 MP** with identical seed/steps. VIDEO (multi-frame) injects blend fine at both
resolutions. **Resolution is the isolated sole variable** (user-confirmed same run/seed/steps).
Also at 1MP only: the ~2 frames each side of the keyframe artifact too.

Four facts to explain: **A** single-frame source-identical @1MP · **B** clean @0.2MP same seed ·
**C** video injects fine at both · **D** neighbor bleed @1MP only.

**Exonerated (summary):** levers' arithmetic, mask construction/pool, grid chunking, cond-row
conditioning, sigma schedule variation — all ruled out with data. Key nuance: x-space invariance
proofs are correct but don't imply perceptual-corruption invariance. Full list:
[hypotheses-and-data](highres-singleframe-underdenoise/hypotheses-and-data.md).

## Detail docs

- [hypotheses-and-data](highres-singleframe-underdenoise/hypotheses-and-data.md) — Exonerated list, measured trajectory data (1MP debug), kill-shot result (α=√5 undershoots), H1–H5 ranked
- [resolution-ladder](highres-singleframe-underdenoise/resolution-ladder.md) — FINDING 1/2/3: R REFUTED as discriminator; bimodal failure (lock @1MP, chaos @0.1MP, goldilocks @0.2MP); coherence/timing is the real signal; H1–H5 continued
- [temporal-and-contagion](highres-singleframe-underdenoise/temporal-and-contagion.md) — Q2 temporal decoupling (keyframe "sits outside the process"); FL2VA vs our inject contrast; cross-inject contagion GPU data + front mechanism
- [baseline-question](highres-singleframe-underdenoise/baseline-question.md) — Q1 standalone baseline (length=1 isolation); correctness test vs true img2img; firing order + instrumentation
