<!-- provenance: status (code-quality audit findings, 2026-08-23; findings unactioned unless noted) -->

# Code-Quality Audit — 2026-08-23

Seven parallel review agents audited every source module + the test suite as the rapid-prototype
phase winds down. Ruff lint/format: fully clean. Tests: 535 passed in 7.5s. No high-severity
*bugs* found; findings are engineering-quality items to burn down during stabilization.

**STATUS:** Cleanup work (tasks #49–#60) completed on branch `cleanup-audit-tail-tasks-49-60`.
All file:line refs here and in the original tasks are as of commit `72b61c6` on branch
`rework-sampler-to-per-row-img2img` — lines will drift as tasks land; symbol names are the
stable anchors.

## Grades

| Module | Grade | Headline |
|---|---|---|
| guidance.py | A | Exemplary docstring citing comfy source; clean branches |
| sampler.py | A- | Pure/testable; dead noise shim wired as live API |
| schedule.py | A- | Clean data model; string enums unchecked |
| envelope.py | A- | Crisp semantics; parked `crossfade=True` dead path |
| composite.py | A- | Clean; minor per-row dict recompute |
| nodes.py | B+ | Great errors/lazy imports; step-numbering drift, dup unpack |
| sanitize.py | B+ | Zero dead code; misleading `snap_inject_at_audio_tick` name |
| constants.py→grid.py | B+ | ✓ Renamed to grid.py; time_shift_sigma moved; TODO resolved |
| mask.py | B- | Dead `derive_mask`/`apply_derived_mask`; asserts as validation |
| tests/ | B+ | Fast, green, good property tests; no conftest.py (fixture dup) |

## Top findings (by priority)

1. **mask.py:45,247 dead pair** — ✓ ADDRESSED. `derive_mask` + `apply_derived_mask` deleted;
   module docstring rewritten to fractional-mask focus; loops vectorized.
2. **sanitize.py:23-24 constant drift hazard** — ✓ ADDRESSED. `_FPS`/`_AUDIO_HZ` now imported
   from grid.py.
3. **sanitize.py:72 misleading API** — ✓ ADDRESSED. `snap_inject_at_audio_tick` renamed to
   `warn_audio_tick_alignment` returning `None`.
4. **sampler.py:170-205 dead shim as live API** — ✓ ADDRESSED. `make_per_row_noise_sampler`
   made private as `_make_per_row_noise_sampler`; marked deferred (Bug B).
5. **nodes.py step numbering** — ✓ ADDRESSED. Step labels fixed; `_unpack_av` helper extracted;
   math.prod used; reviewer-speak removed from INPUT_TYPES.
6. **tests/: no conftest.py** — ✓ ADDRESSED (in progress by tests-agent). Consolidation underway;
   test_constants.py→test_grid.py rename queued.
7. **grid.py identity** — ✓ ADDRESSED. File renamed from constants.py; `time_shift_sigma` moved
   to sampler.py; `VISUAL/AUDIO_COND_TIMESTEP` TODO resolved (values confirmed from comfy source).
8. **mask.py asserts** — ✓ ADDRESSED. Assertions replaced with `raise ValueError`.

Low-severity leftovers now addressed: ✓ Literal types in schedule.py + keyword args + eq=False
comment; ✓ dict[str, Any] in guidance.py + de-brittled comfy refs; ✓ scale_packed_audio dual
contract documented; ✓ build_conditioning_wrapper docstring trimmed; ✓ nodes tooltips (stochastic
warning + exclusive-end); ✓ envelope dead still_inject_denoise removed; ✓ composite inject_row_map
cached; ✓ over-broad TypeError narrowed via inspect.signature; ✓ requirements.txt stub added; ✓
RELEASING.md PublisherId/Icon placeholder noted. See cleanup-audit-tail-tasks-49-60 branch.

## Project structure review (same date): A-

Dependency graph is fully acyclic and cleanly layered: grid → {envelope, sanitize, guidance}
→ schedule → {mask, composite} + sampler (torch layer, no internal deps) → nodes. The lazy-import
pattern inside `_run_sampler` is load-bearing — it's what keeps every module importable and
CPU-testable without live ComfyUI. Well above ComfyUI-extension norm: registry-ready `[tool.comfy]`
skeleton, 3.10–3.12 CI matrix with ruff + 90% diff-branch-coverage gate, pre-commit, Hypothesis
tests, LICENSE/README/RELEASING.md/uv.lock all present.

Structure findings: ✓ (med) `grid.py` rename complete — grid math now obvious from filename; ✓
(med) `sampler.py` docstring now notes "imported lazily inside nodes._run_sampler"; ✓ (med)
`requirements.txt` stub added; ✓ (low) RELEASING.md notes PublisherId/Icon placeholders; (low)
tests/conftest.py consolidation — in progress by tests-agent.

## Strengths (consistent across reviewers)

Docstring discipline (numpydoc everywhere, comfy internals explained in place); user-actionable
error messages with nearest-valid suggestions; systematic lazy `comfy.*` imports keeping every
module CPU-testable; honest `# pragma: no cover` on GPU paths; AST signature tests; property
tests on numeric boundaries; clean ComfyUI registration conventions.
