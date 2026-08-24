<!-- provenance: confirmed (all function names, formulas, and wiring verified against committed code; GPU sampling of the current build unverified — see status-and-open-paths) -->
<!-- verified: 2026-08-23 · repo @72b61c6 -->

# Our architecture — synthesized per-row img2img (the three levers)

Read this when working on our sampler code (`sampler.py`, `composite.py`,
`nodes.py::_run_sampler`). Since H3 has no free per-row img2img (see
[differential-diffusion](differential-diffusion.md)), we SYNTHESIZE it: run per-row *schedule
truncation* inside a sampler that only truncates globally. Design memory:
`per-row-img2img-architecture`.

Code: `sampler.py`, `composite.py`, `nodes.py::_run_sampler` (GPU-only, `# pragma: no cover`).

## The three levers

1. **Per-row init noise = a lerp** (`sampler.py::per_row_init_lerp`, used in
   `build_per_row_sampler_function`). After comfy's global `noise_scaling`
   `x_global = σ_max·ε + (1−σ_max)·clean`, the img2img start for row r is
   `x_r = m_r·x_global + (1−m_r)·clean`. m=1→full gen, m=0→clean, 0<m<1→exact img2img start.
   Applied at the top of the wrapped `sampler_function` — a model_function_wrapper CANNOT set the
   initial x (the sampler steps its own outer x).

2. **Per-row DiT conditioning** (`build_conditioning_wrapper`, wired in `_run_sampler`). Feed the
   fractional `denoise_mask` so the DiT's `_forward` row-label computation compresses the
   network's timestep embedding to `m_r·t` (H3's native path; `RowSchedule.denoise` IS m_r).
   Pooled via `model._denoise_mask_values(m_packed, latent_shapes)`. See
   [dit-forward](native-h3-mechanism/dit-forward.md).
   **Quantization consistency:** `_token_grid_masks` snaps the mask to a 1/256 grid
   (`ceil(m·256)/256`), so `_run_sampler` pre-quantizes `m_packed` with
   `sampler.py::quantize_denoise` BEFORE any lever uses it — otherwise levers 1/3 run at raw m
   while the DiT runs at quantized m (up to 1/256 mismatch in the lever-3 identity).

3. **Denoised correction (REQUIRED)** (`build_conditioning_wrapper`, `m_packed` arg; commit
   78e4c87). Lever 2 alone is a bug: `process_timestep` compresses only the embedding but
   `calculate_denoised(sigma, v, x) = x − sigma·v` uses the OUTER sigma ⇒ sampler computes
   `d = (x−denoised)/sigma = v` and integrates each row over the FULL interval ⇒ low-m rows
   stepped 1/m too far ⇒ off-distribution (pixelation/static on euler). FIX:
   `corrected = m·denoised + (1−m)·input` ⇒ `d = m·v` ⇒ each row integrated over its compressed
   m·σ interval. Affine in denoised ⇒ commutes with CFG. m=1 unchanged; m=0 frozen.

## Plus

- **`noise_mask=None`** passed to `sample_custom` ⇒ `KSamplerX0Inpaint` never runs its PRE/POST
  composite ⇒ no ghost (see [sampler-loop](native-h3-mechanism/sampler-loop.md)). Exact m==0 rows
  / audio-preserve ticks restored by
  `composite.py::post_composite_preserve` AFTER sampling (binary, no compounding).
- `clean` reference (`composite.py::build_clean_reference`) = target latent with ALL inject
  video/audio content composited in (every covered row/tick, not just d==0), so fractional rows
  img2img FROM inject content; m==1 rows ignore it.

## Scale-invariance premise (memory `per-row-sampler-scale-invariance`)

Running the GLOBAL sampler on the global schedule with per-row-corrected x0 + per-row init noise
reproduces EXACTLY what row r would get on its own compressed schedule m_r·σ — IFF the per-step
update is invariant under scaling all sigmas by m_r, holding the CORRECTED denoised fixed. Holds
for all deterministic samplers. The ONLY non-invariant piece is the stochastic renoise (see
[bugs](bugs.md#bug-b)).

**Semantic note — rescaled grid, not truncated tail.** Our synthesized img2img runs row r over
the RESCALED schedule `m_r·σ_i` (same step count, every step shrunk by m_r) — not classic img2img
"skip the first steps, then run the σ-tail" (a truncated grid). Same endpoints, different
interior grid; both are valid img2img discretizations, but comparisons against a stock img2img
run at denoise=m will differ slightly by construction. This also interacts with the
chaining/leftover-noise limitation in [status-and-open-paths](status-and-open-paths.md).

**Sampler-surface limitations (current build).** `KSAMPLER.sample`'s final
`inverse_noise_scaling(σ_end, x)` divides ALL rows by `(1−σ_end)`, but a fractional row ends at
`m·σ_end` — so `return_with_leftover_noise=enable` (σ_end>0) mis-scales fractional rows by
`(1−m·σ_end)/(1−σ_end)`; and resuming (`add_noise=disable`, `start_at_step>0`) feeds the init
lerp an already-noised x it wasn't derived for. Only `add_noise=enable / start=0 / end=full /
leftover=disable` is per-row-correct. RESOLVED @72b61c6: the four chaining widgets are HIDDEN
(removed from `INPUT_TYPES`, correct values hardcoded internally); revisit post-prototype
([status-and-open-paths](status-and-open-paths.md)).

**Stochastic warning.** `_run_sampler` warns (no hard gate, prototype) when
`sampler.py::sampler_is_stochastic` detects an ancestral/SDE sampler with fractional rows
present — detection is signature-based (`eta` param defaulting >0), no hardcoded sampler list;
blind spot: ddpm/lcm/er_sde (noise without an eta knob).
