<!-- provenance: status (C2 ρ fix chain: v3 σ_c÷ρ_true @f06a84a GPU quieter; v4 exact-residual cancel
     @12ea3b6 GPU much-quieter = CURRENT BEST; v5 init-composite @7436165 GPU-FALSIFIED + REVERTED
     @8c8cb90; v6 m=0 audio-context heat @02fee22 GPU-CONFIRMED improvement (very-quiet metallic
     hum, character shifted buzz→metallic), native-confirmed) -->
<!-- verified: 2026-08-29 · v3/v4 GPU-confirmed; v5 (7436165) FALSIFIED + REVERTED @8c8cb90; v4
     (12ea3b6) current best; v6 (02fee22) m=0-context fix IMPLEMENTED (620/620 CPU), GPU-pending —
     NOT video-byte-identical by design (context flows through joint A/V attention). -->
# C2 ρ — the current fix chain (v3 → v4 → v5 → v6)

Index: [index.md](index.md). History/falsifications: [history-and-falsifications.md](history-and-falsifications.md).

## v3 — σ_c projection ÷ρ_true (commit f06a84a, GPU-confirmed quieter than B)
`denoised_r`-ONLY on the σ_c axis: NO input perturbation, NO static reference. Keep the state on the
truthful carried trajectory `y_true = σ_c·ε + c_star·(S·A)`, with `σ_c = sig_row·carrier/sig_g`,
`c_star = (1−sig_row)/(S·k) = (1−σ_c)/ρ_true`, `ρ_true = S·k·(1−σ_c)/(1−sig_row)`, `k = sig_g/carrier`.
Because there is no input perturbation, the UNPERTURBED input `x_prev` is truthful at every eval.
For `frac_audio` rows: `denoised_r = (x_prev − σ_c·v)/ρ_true` — a **σ_c PROJECTION** (not σ_v) — and
ancestral renoise integrates on `σ_c/σ_c_next`. `ρ_true ≡ 1` at m=1 (carry identity) and → 1 at the
terminal step; the gate keeps m=1 bit-exact and video byte-identical.

**σ_v nuance.** Under a truthful state the σ_v projection overshoots by exactly `ρ_old = k/k_row`
(`k_row = sig_row/sig_row_v`). "B with `ρ_true` swapped in" (σ_v ÷ ρ_true) still lands `ρ_old·c_star`
≈2.5× over; only **σ_c-projection ÷ ρ_true** is self-consistent. **GPU (2026-08-29, euler_ancestral
`0/0/49/73`, same seed as B/63b291e):** v3 quieter than B (reduced, not eliminated), VALIDATING the σ_c
convention; no over-damp → `ρ_old·ρ_true` fallback NOT indicated. Superseded as best by v4.

## v4 — exact-residual cancel (commit 12ea3b6, GPU-confirmed much-quieter) — CURRENT BEST
The residual is known EXACTLY, no linear-model assumption. From forward's carry (`carry = k =
sig_g/carrier`, comfy-ref `model.py:535-538`) + output transform (`model.py:549-550`) and the exact
shift identity `S·k ≡ 1+(S−1)·sig_g`, the σ_c projection expands for ANY network output to:

    x_prev − σ_c·v = S·A_hat + (S−1)·(sig_row − sig_g)·x_prev

The clean estimate `S·A_hat` carries coefficient EXACTLY 1 (the `S·k` identity) — why v3's σ_c
convention is correct. v4 subtracts the leftover `(S−1)(sig_row−sig_g)·x_prev` exactly (multiply the
KNOWN state `x_prev` by an exact scalar; a pure re-expression of the model's own output — no input
change, no static reference). For frac-audio rows:

    denoised_r_frac = (x_prev − σ_c·v − (S−1)(sig_row−sig_g)·x_prev) / ρ_true = S·A_hat / ρ_true

Guard rails (Fable-verified): `resid_coeff=0` at m=1 AND the `frac_audio` gate excludes m=1 ⇒
bit-exact; correction ⊂ audio_mask ⇒ video byte-identical; terminal finite. 620/620 CPU tests pass.

**GPU (2026-08-29, euler_ancestral `0/0/49/73`, ease_in_out, audio:fade, min_denoise=0, same seed).**
v4 MUCH QUIETER than v3 — buzz dropped from clearly audible to FAINT, not zero. Ordering worst→best:
baseline > 63b291e (louder) > B (reduced) > v3 (quieter) > v4 (much quieter). The C2 ρ/residual line
is COMPLETE. **v4 remains the current best until v6 is GPU-verified.**

## v5 — init-composite bootstrap fix (commit 7436165) — GPU-FALSIFIED, REVERTED @8c8cb90
Hypothesis: v3/v4's per-step algebra is exact GIVEN the state, so only an INIT-STATE error could
remain — the i==0 composite plants the packed audio clean coeff over by ≈S≈4× on fractional-audio
rows. v5 corrected ONLY that init coeff to `(1−sig_row)/(S·k0)` (known init quantities, not a stale
reference). **GPU (2026-08-29, same `0/0/49/73` config): NO CHANGE vs v4.** FALSIFIED + REVERTED
(`8c8cb90`). The faint residual is NOT init-state-driven — the DiT self-corrects the init state
within the first few steps, ruling out the bootstrap hypothesis.

## v6 — m=0 audio-context heat fix (commit 02fee22) — GPU-CONFIRMED improvement
The last per-step lever v5's falsification pointed at. Frozen (never-preserve, m=0) audio rows are
held at `clean = S·A` all run, but the model's global audio carry (comfy-ref `model.py:536-538`
multiplies audio by `k = σ_a/σ_v`) means the network SEES `S·k·A = (1+(S−1)·sig_g)·A` — up to S≈4×
too hot at early steps, → A late. v6 presents those rows to the model at the truthful value
`clean/(S·k) = clean/(1+(S−1)·sig_g)` (exact shift identity `S·k ≡ 1+(S−1)·sig_g`; denom ∈[1,S],
finite, no `1/carrier`). Endpoints: i=0 (`k0=1`) → `clean/S = A` (reconciles with v5's c_star at
sig_row=0); carrier→0 → `clean = S·A`.

**NATIVE CONFIRMATION (strongest source backing of any candidate this chase).** comfy
`scale_latent_inpaint` (`model_base.py:2262-2266`) rescales preserved audio by `(σ_v/σ_a)/S = 1/(S·k)`
with the source comment "rescale for the model to see it clean"; since `cleans[1]=S·A`, native
presents preserved audio context at exactly `clean/(S·k)` — the IDENTICAL per-step σ-dependent value
v6 uses. Our sampler was off-contract (feeding `S·A`) on every eval.

**Implementation (option a, model-input-only).** New `_StepContext` field `never_mask` (packed bool,
the loop's `never = k_d >= steps`), passed at ctx construction. In `_euler_ancestral_rf_step`, before
the model call: `x_model = where(audio_mask & never_mask, clean_packed/(1+(S−1)·sig_g), x_prev)`;
`denoised = model(x_model, ...)`. `x_prev` itself is NOT modified — frozen rows' OUTPUT stays
`clean = S·A` (ratio=0 on never rows + final `where(never, clean, x_cur)`), nothing persisted.
620/620 CPU tests pass.

**Guard-rail caveat (state plainly).** Unlike v3/v4/v5, v6 CANNOT be video-byte-identical to v4 — by
design the corrected m=0 audio context flows into video (and m=1) outputs through joint A/V attention
(that attention channel IS the lever under test). The provable rail is only "no DIRECT video/x_prev
perturbation" (gate = `audio_mask & never_mask`); the achievable acceptance check is "video visually
indistinguishable / no regression." A real departure from every prior fix's byte-identical rail.

**Why not 63b291e (the falsified input-side fix).** 63b291e perturbed ACTIVE fractional rows with a
STATIC step-0 reference (wrong at every i>0 vs an evolving state — ~3× incoherent garbage,
amplified+renoised). v6 perturbs FROZEN rows (state is exactly `clean = S·A` by construction every
step, so clean is never stale), uses the per-step σ-dependent-truthful value, and those rows' outputs
are discarded (ratio=0) — no self-feedback loop; the only channel is attention context.

**Discriminator (one GPU run).** Same config — inject frame 0, fade `0/0/49/73`, ease_in_out,
audio:fade, min_denoise=0, euler_ancestral, same fixed seed — vs v4 (12ea3b6). PASS = fade-out ramp
`[49,73)` buzz quieter than v4. Do NOT expect byte-identical video (context flows through joint
attention by design); acceptance = video visually indistinguishable / no regression, held region
`[0,49)` and free audio unchanged in character.

**GPU RESULT (2026-08-31, same discriminator config).** v6 is a further IMPROVEMENT over v4 (NOT
falsified): the residual is now a VERY VERY QUIET metallic hum in the background — quieter than v4's
faint buzz. The LEVEL reduction (very-very-quiet) is the RELIABLE signal that v6 helped. The
CHARACTER also changed, from broadband buzz to a narrower metallic/tonal hum, but that timbre shift
is LOW-INFORMATION / likely a RED HERRING: interference/visual-noise generation in this model peaks
in roughly the first ~1/3 of the gen and is then reshaped downstream by the generation itself, so
the final artifact's timbre reflects downstream reshaping, NOT the raw noise source. A character
shift therefore does NOT imply the residual is a different mechanism than v3/v4/v6 addressed.
Residual source accounting + 2×2 plan: [residual-accounting.md](residual-accounting.md).

## Confidence (Fable)
v4-derivation ~95% (both trajectory endpoints checked vs forward's transforms). v6 ~45% that it
audibly reduces the faint residual vs v4. FOR: native source proves the model's contract expects the
cool value (strongest backing yet); error is up-to-4×, persistent across all evals, on ~49
audio-context rows. AGAINST: v5 showed the DiT self-corrects large state errors; the residual is
already faint; euler sees the same hot context yet is clean. If v6 fails → declare the DiT/ancestral
floor and ship v4.
