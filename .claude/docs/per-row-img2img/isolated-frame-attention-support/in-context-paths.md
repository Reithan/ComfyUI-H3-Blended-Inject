<!-- provenance: theory (mixed — refuted paths are GPU/logic-CONFIRMED dead; surviving paths UNVERIFIED) -->
<!-- verified: 2026-08-24 · user GPU runs 0.5MP + analytical arguments -->
# Anchor denoise must be IN-CONTEXT — refuted and surviving paths

Parent: [isolated-frame-attention-support](../isolated-frame-attention-support.md).

**Implication — the cond channel alone can't solve it; DECOUPLE. Anchor partial denoise MUST be
IN-CONTEXT** (user, 2026-08-24) — the keyframe generates its own temporal context, so pre-denoise
is paradoxical:
- **⚠ Bake-beforehand REFUTED (paradox).** You can't partially-denoise the keyframe against a video that
  doesn't exist yet — the keyframe is what generates that video. A pre-denoised still is denoised against
  nothing (no temporal neighbors), so it has no timeline blend and is just a different clean keyframe.
  The img2img crop/upscale story was an SDXL illustration; **H3 has NO single-frame img2img mode.**
  Cyclic/paradoxical — dead end.

The remaining paths keep the clean cond for BLEND and denoise the anchor IN-CONTEXT:
- **⚠ Single-pass decouple FALSIFIED (GPU, 2026-08-24):** clean cond + OUR fractional latent+mask on the
  SAME row, together, was TESTED — great in/out blend but the anchor came out **non-denoised-looking AND its
  hard edges contaminated neighbors**. Mechanism NOT settled — likely a row-pinned ATTRACTOR (the latent DOES
  denoise but the cond token re-pulls it + neighbors toward the raw inject each step; favored by `|inp|` moving
  while x0 tracks source), possibly suppression. Either way dead as a single-pass. Data:
  [aug-mechanism](../conditioning-row-inject/aug-mechanism.md) ("SINGLE-PASS DECOUPLE FALSIFIED").
- **Route-2 two-pass:** pass A the anchor denoises IN-CONTEXT (fractional, real gen) → capture its
  realized content; pass B re-inject that as a CLEAN anchor → clean neighbor blend, no contagion, no
  re-freeze. Denoises against the real (pass-A) video → NOT paradoxical. Caveat: if pass A under-denoises
  via T_N, pass B just re-injects an under-denoised frame; gain only if pass A's in-context denoise is
  adequate (r60 hints context helps).
- **Cross-resolution two-pass (workaround, back pocket):** pass A at 0.2MP (window open) → extract
  realized keyframe → upscale → re-inject CLEAN into the 1MP run. Avoids the paradox and T_N; cost ≈
  few %. USER VERDICT (2026-08-24): viable but workaround family; back pocket while a direct 1MP fix
  is pursued. Risks: upscale softness; realized frame must read compatible in the fresh 1MP context.
- **Timed removal / anchor-then-release:** clean reference EARLY so neighbors compose around it, release
  to the latent+mask fractional denoise in the tail. Single-pass; truncated-tail denoise + a release
  transition to manage. Cond-channel form GPU-CONFIRMED viable @0.5MP and building first as H3AddGuide
  ([timed-cond-removal-prototype](../timed-cond-removal-prototype.md)); a LATENT-resident
  hold-and-release remains an active parallel goal (user, 2026-08-24) — see
  [status-and-open-paths](../status-and-open-paths.md) path 1.
