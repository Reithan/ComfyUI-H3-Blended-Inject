# Contributing to ComfyUI-H3-Blended-Inject

Thank you for your interest in contributing! This document covers setting up a development
environment and the workflow for landing changes.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Git

### Initial Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Reithan/ComfyUI-H3-Blended-Inject.git
   cd ComfyUI-H3-Blended-Inject
   ```

2. **Create the environment and install dev tools**:

   ```bash
   uv sync --group dev
   ```

   The dev group installs `torch`, `numpy`, and `hypothesis` so the pure-logic modules
   (envelope, schedule, sanitization, derived mask) can be tested CPU-side without a running
   ComfyUI.

3. **Install the git hooks** (run once per clone):

   ```bash
   uv run pre-commit install
   ```

   This installs both the `pre-commit` and `pre-push` hook types (the config sets
   `default_install_hook_types`), so a single command wires up everything.

4. **Verify the setup**:

   ```bash
   uv run pre-commit run --all-files   # run all hooks manually
   uv run ruff check .                 # lint
   uv run pytest                       # run the test suite
   ```

## Git Workflow

### Protected Branches

- **Direct commits to `main` are blocked** by a git hook (`forbid-main-commit`).
- **Direct pushes to `main` are blocked** by a git hook (`forbid-main-push`).
- All changes go through feature branches and pull requests.

### Recommended Workflow

1. **Create a feature branch** off `main` with a descriptive, action-oriented name:

   ```bash
   git checkout -b fix-nsfw-blur-timing
   ```

2. **Make your changes and commit**. The `pre-commit` hook runs `ruff` lint (with autofix)
   and `ruff format`, and blocks commits on `main`. If the linter auto-fixes files, re-stage
   and commit again.

   ```bash
   git add <files>
   git commit -m "fix broken envelope clamp in schedule.py"
   ```

3. **Push your branch**. The `pre-push` hook runs the full test suite with branch coverage
   (mirroring CI) and blocks pushes on `main`.

   ```bash
   git push -u origin fix-nsfw-blur-timing
   ```

4. **Open a pull request** on GitHub and merge after review.

### Bypassing Hooks (Emergency Only)

```bash
git commit --no-verify    # skip pre-commit hooks
git push --no-verify      # skip pre-push hooks
```

Only use `--no-verify` when you must — it can introduce lint issues, break CI, or push
untested code.

## Development Commands

```bash
# Lint
uv run ruff check .            # check
uv run ruff check --fix .      # check and auto-fix
uv run ruff format .           # format

# Test
uv run pytest                  # run all tests
uv run pytest -v               # verbose
uv run pytest tests/test_nodes.py::TestClassMappings   # a single test

# Reproduce the CI coverage gate locally on a feature branch
uv run pytest --cov=. --cov-branch --cov-report=xml
uv run diff-cover coverage.xml --compare-branch=origin/main --branch-coverage --fail-under=90
```

## Coverage Gate

CI enforces **≥ 90% branch coverage on changed code only** (via
[diff-cover](https://github.com/Bachmann1234/diff_cover)) — new and modified lines must have
both arms of their branches exercised. It is branch coverage, not line/statement coverage,
and it applies only to the diff, so pre-existing gaps do not block unrelated PRs. The
pure-logic modules are kept importable without a running ComfyUI so they can be tested
CPU-side with a mock model.

## Code Style

This project uses **ruff** for linting and formatting:

- **Line length**: 100 characters
- **Target Python version**: 3.10+
- **Enabled checks**: pycodestyle (E/W), pyflakes (F), isort (I), pyupgrade (UP), bugbear (B)

The `INPUT_TYPES` classmethod is exempted from the naming rule (`# noqa: N802`) because
ComfyUI's node API requires that exact name.

## Commit Messages

Write each message as a short, imperative statement:

- ✓ `fix broken logger call in nodes.py`
- ✗ `Fixed the logging bug.`

## Pull Request Guidelines

1. **Keep PRs focused** — one feature or fix per PR.
2. **Update documentation** — if you change behavior, update `README.md` and the developer
   wiki under [`.claude/docs/`](.claude/docs/PER_ROW_IMG2IMG_NOTES.md).
3. **Test your changes** — the pack targets MiniMax H3; GPU-side behavior should be verified
   on a real H3 run where relevant.
4. **Run the hooks** — make sure lint, format, tests, and the coverage gate all pass.

## Project Structure

```
ComfyUI-H3-Blended-Inject/
├── comfyui_h3_blended_inject/
│   ├── nodes.py             # ComfyUI node classes + registration
│   ├── sampler.py           # per-row img2img sampler + native step registry
│   ├── schedule.py          # inject list, per-row schedule merge
│   ├── envelope.py          # denoise fade envelope
│   ├── composite.py         # clean-reference compositing
│   ├── mask.py              # derived per-row denoise mask
│   ├── grid.py              # H3 latent frame/row grid
│   ├── sanitize.py          # input validation + snap warnings
│   ├── guides.py            # native H3 guide entries
│   ├── guidance.py          # conditioning helpers
│   └── observer_split.py    # observer-label K/V split
├── tests/
├── .claude/docs/            # developer wiki (design notes, findings)
├── __init__.py              # ComfyUI node exports
├── pyproject.toml
├── .pre-commit-config.yaml
├── README.md
├── RELEASING.md             # Comfy Registry publishing checklist
├── LICENSE
└── CONTRIBUTING.md          # this file
```

## License

By contributing, you agree that your contributions will be licensed under the project's
[GPL-3.0](LICENSE) license.
