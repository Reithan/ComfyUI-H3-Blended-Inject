# ComfyUI-H3-Blended-Inject

ComfyUI node pack for injecting video, image, and audio content into MiniMax H3
generation with per-row scheduled denoise blends.

> **Status:** early development. The repository scaffold is in place; the nodes
> described below are not implemented yet.

## What it does

Blend injected content into an H3 audio-visual latent with per-frame control over
how much the model may redraw it. Every latent row carries a denoise value `d` in
`[0, 1]`: `d = 0` preserves the source exactly, `d = 1` generates freely, and
fractional values let the model redraw that fraction of the frame. Two use cases
share one mechanism:

- **Video inject** — a clip written into the timeline with a fade-in, a hold at a
  minimum denoise, and a fade-out.
- **Still inject** — a single image at one position with a denoise value
  controlling how closely the original is retained.

Fractional denoise is realized outside ComfyUI's noise mask, via a hold-and-release
wrapper on the sampler; exact `d = 0` spans route through the trained mask
preservation path. See [`.claude/plans/plan.md`](.claude/plans/plan.md) for the
full design.

### Planned nodes

- **H3 Add Inject** — appends one inject (images and/or audio) to an
  `INJECT_LIST`, with a denoise envelope and audio mode. Chainable.
- **H3 Inject Sampler** — a KSampler Advanced clone that encodes the inject list,
  builds the per-row schedule, derives the preservation mask, and runs the sampler
  once with hold-and-release.

## Installation

Clone into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Reithan/ComfyUI-H3-Blended-Inject.git
```

Restart ComfyUI. This pack targets MiniMax H3 and relies on ComfyUI's bundled
`torch`; no extra runtime dependencies are required today.

## Development

Tooling is managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev        # create the environment and install dev tools
uv run pre-commit install  # install the git hooks (run once per clone)
uv run pytest              # run the test suite
uv run ruff check .        # lint
uv run ruff format .       # format
```

### Git hooks

Managed with [pre-commit](https://pre-commit.com/). After `uv run pre-commit install`:

- **pre-commit** — blocks direct commits on `main`, then runs `ruff` lint (with
  autofix) and `ruff format`.
- **pre-push** — blocks direct pushes on `main`, then runs the full test suite
  (the same checks CI runs).

### Coverage gate

CI enforces **≥ 90% branch coverage on changed code only** (via
[diff-cover](https://github.com/Bachmann1234/diff_cover)) — new and modified
lines must have both arms of their branches exercised. It is branch coverage, not
line/statement coverage, and it applies only to the diff, so existing gaps do not
block unrelated PRs. To reproduce the gate locally on a feature branch:

```bash
uv run pytest --cov=. --cov-branch --cov-report=xml
uv run diff-cover coverage.xml --compare-branch=origin/main --branch-coverage --fail-under=90
```

Pure-logic modules (envelope, schedule, sanitization, derived mask) are kept
importable without a running ComfyUI so they can be tested CPU-side with a mock
model.

## Releasing

Publishing to the Comfy Registry is not wired up yet. See
[`RELEASING.md`](RELEASING.md) for the checklist to enable it (registry secret,
icon, `[tool.comfy]`, and the publish workflow). Once active, releases are cut by
bumping `version` in `pyproject.toml` and merging to `main`.

## License

[GPL-3.0](LICENSE).
