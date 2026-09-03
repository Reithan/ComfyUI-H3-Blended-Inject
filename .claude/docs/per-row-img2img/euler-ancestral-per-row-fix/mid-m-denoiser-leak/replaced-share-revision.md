<!-- provenance: theory (UNVERIFIED that the replaced-share fix resolves the garble) — listen-test comparison + euler-invariance/tests are CONFIRMED -->
<!-- verified: 2026-09-02 (branch fix-euler-ancestral-per-row-renoise) · GPU-driven revision of the full-removal conclusion -->
# Replaced-share revision — coherent leak is the voice substrate

Parent: [mid-m-denoiser-leak.md](../mid-m-denoiser-leak.md) (mechanism + fix candidates).

Supersedes the parent/[cpu-probe-results.md](cpu-probe-results.md) conclusion that FULL
pooled leak-removal is the correct lever. GPU listen-test says full removal is too aggressive.

## Listen-test comparison (CONFIRMED — decisive)

In ALL prior runs the injected dialogue was INTELLIGIBLE whenever audible — static could
COVER it, or a MUTING effect could soften it, but the words were never garbled.

The full-strength pooled-removal run (`H3BI_C2_POOL_STRENGTH` unset = full) is the FIRST
garbled result: "My" + word-shaped gibberish.

## Reinterpretation (theory, GPU-motivated, UNVERIFIED that the fix below resolves it)

The εc-aligned "leak" in the clean estimate ĉ is NOT pure garbage. Its COHERENT part is the
substrate the network naturalizes into speech — intelligible under plain ancestral, just
buried under added static (matches the euler-clean run being "content-shaped").

Ancestral only turns the FRESH-DECORRELATED √(1−ρ²) share of the leak into broadband static.

FULL removal strips the WHOLE leak → removes the coherent voice substrate too → garbled
voice. So full removal is too aggressive.

## Correction to the earlier CPU-probe conclusion

The probe that "required" full removal (and "falsified" the replaced-share fix at ~15%
static-removal) optimized ONLY for static-removal; it never modeled coherent-leak →
naturalized-voice. It therefore missed the content-preservation constraint the GPU just
exposed.

GPU is the arbiter. The principled target is to remove ONLY the fresh-decorrelated share —
the euler-invariant REPLACED-SHARE form (parent Fix B) × (1 − r_ret/σ_c′). This resurrects
Fable's original replaced-share proposal.

## Where the knob + sweep live

The tunable `H3BI_C2_POOL_STRENGTH` (built this thread) and the next GPU strength/leak sweep
are recorded in [../pooled-fix-gpu-result.md](../pooled-fix-gpu-result.md).
