<!-- provenance: status (GPU RESULT — first pooled-fix run, 2026-09-02) -->
<!-- verified: 2026-09-02 (branch fix-euler-ancestral-per-row-renoise) · first GPU run of the pooled content-blind leak-removal fix -->
# Pooled content-blind leak-removal — first GPU result

Parent: [mid-m-denoiser-leak.md](mid-m-denoiser-leak.md)

First GPU run of the prototype pooled leak-removal. Mechanism + lever validated
directionally; two open forks remain (residual bands, garbled dialogue).

## Run config

- Toggle `H3BI_C2_POOL_LEAK=32` — content-blind σ_c-space kernel-regression
  leak-removal in `_c2_audio_ancestral_update`.
- Round-11 config: `euler_ancestral`, η=1, 20 steps, `min_denoise=0`,
  full fade 0/0/0/90.
- Output spectrogram saved as `.claude/tmp/spectrogram3.txt`.

## Perceptual result (user listen-test)

- Static AMPLITUDE substantially reduced vs the pre-fix run — loud static →
  "very quiet noise" at ~0.25–0.5 s.
- The injected character's VOICE now emerges ~0.5–1.5 s, starting with the
  CORRECT first word "My", then word-sounding GIBBERISH (resembles the model
  hallucinating dialogue when none is prompted).
- A second static band returns ~1.5–2.5 s, then audio resolves to normal.

## Spectral flatness (amplitude-independent — now less informative)

- Static window flatness 0.041→0.115; ref 0.014→0.030; static/ref ratio
  2.83→3.87.
- Flatness ROSE even though perceived amplitude DROPPED: flatness is
  amplitude-normalized and the static window is now a voice+static MIX.
- Key signal — the REF window (2.0 s+) changed at all: the 0/0/0/90 full fade
  leaves NO m=1 region, so the fix also touches high-m low-leak rows (mild
  over-correction there).

## Verdict

Mechanism + lever VALIDATED directionally — static amplitude down, injected
content emerges. Two OPEN items:

- (a) Two residual static bands (~0.25–0.5 s quiet, ~1.5–2.5 s louder) — likely
  fade-edge / sparse-bin residual and/or high-m over-correction.
- (b) Garbled dialogue — RESOLVED (see fork-(b) section below): the listen-test
  comparison shows it is fix-induced over-subtraction along εc, not pre-existing
  gibberish. Full removal is too aggressive → replaced-share/strength knob.

## Fork (b) RESOLVED — full removal garbles the voice

The user's intelligibility comparison landed: in ALL prior runs the injected dialogue
was INTELLIGIBLE whenever audible (static could cover it, muting could soften it, but
words were never garbled). This FULL-removal run is the FIRST garbled result. So the
garble is fix-induced over-subtraction, not pre-existing gibberish — full removal strips
the coherent voice substrate. Reinterpretation + resurrected replaced-share fix:
[mid-m-denoiser-leak/replaced-share-revision.md](mid-m-denoiser-leak/replaced-share-revision.md).

## Strength knob (built this thread, CONFIRMED on CPU)

Tunable env `H3BI_C2_POOL_STRENGTH` on `_c2_audio_ancestral_update`:

- unset / "1" = FULL removal (default, the garbling run above).
- float in [0,1] = that FRACTION of the pooled removal.
- "replaced" = fresh-decorrelated share ONLY (euler-invariant replaced-share form).

Removal moved to AFTER `r_ret` is computed. CPU-verified: "replaced" is euler-invariant
(η=0 → pool-on == pool-off bit-exact); half-strength lies between off and full; 117/117
sampler tests green; toggle-off still bit-exact. Commits on branch
`fix-euler-ancestral-per-row-renoise`.

## Next GPU sweep (user)

strength ∈ {"replaced" (primary hypothesis), 0.5, 0.75} at
`H3BI_C2_POOL_LEAK` ∈ {16, 32}. SUCCESS = injected voice stays INTELLIGIBLE (like plain
ancestral) AND static amplitude drops (toward the full-removal run). If "replaced"
preserves voice but leaves too much static, the float strengths map the bias/variance knee.

Still open independently: gate the removal to the high-leak mid-σ_c band so near-clean
high-m rows are untouched (ref-region over-correction + fade-edge static band).
