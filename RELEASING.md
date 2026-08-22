# Releasing to the Comfy Registry

Publishing is not wired up yet. This document is the checklist for turning it on
when the pack is ready for its first release. It mirrors the setup used in
[`negative_rejection_steering`](https://github.com/Reithan/negative_rejection_steering).

Once active, the release flow is: **bump `version` in `pyproject.toml`, merge to
`main`, and the pack auto-publishes.** No tags or GitHub releases required.

## One-time setup

1. **Add the registry token.** In repo Settings → Secrets and variables → Actions,
   add a secret named `COMFY_REGISTRY_KEY` containing a Comfy Registry API key for
   the `reithan` publisher.

2. **Add an icon.** Drop an `icon.png` in the repo root (square, e.g. 256x256).

3. **Fill in `[tool.comfy]`** in `pyproject.toml` (currently stubbed):

   ```toml
   [tool.comfy]
   PublisherId = "reithan"
   DisplayName = "H3 Blended Inject"
   Icon = "https://raw.githubusercontent.com/Reithan/ComfyUI-H3-Blended-Inject/main/icon.png"
   ```

   Optionally add authorship to `[project]`:

   ```toml
   authors = [{ name = "Bryan O'Malley", email = "bo122081@hotmail.com" }]
   ```

4. **Add the publish workflow** at `.github/workflows/publish.yml` (see below).

No build backend is required. `Comfy-Org/publish-node-action` zips the source and
reads `[tool.comfy]` / `[project]`; it does not pip-build the package, so this
repo stays backend-free (it is loaded from `custom_nodes`, not pip-installed).

## Cutting a release

1. Bump `version` in `pyproject.toml` (must strictly increase; downgrades are
   rejected by the workflow).
2. Open a PR, let CI pass, and merge to `main`.
3. The push to `main` touching `pyproject.toml` triggers the publish workflow,
   which confirms the version incremented and publishes to the registry. You can
   also run it manually from the Actions tab (`workflow_dispatch`).

## The publish workflow

Save this as `.github/workflows/publish.yml` during step 4 above.

```yaml
name: Publish to Comfy registry
on:
  workflow_dispatch:
  push:
    branches:
      - main
    paths:
      - "pyproject.toml"

permissions:
  issues: write

jobs:
  publish-node:
    name: Publish Custom Node to registry
    runs-on: ubuntu-latest
    if: ${{ github.repository_owner == 'Reithan' }}
    env:
      FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true
    steps:
      - name: Check out code
        uses: actions/checkout@v5
        with:
          submodules: true
          fetch-depth: 2  # Need HEAD and HEAD~1 for version comparison
      - name: Check version increment
        id: version_check
        run: |
          python3 << 'EOF'
          import sys
          import os
          import subprocess
          from packaging.version import parse as parse_version

          def extract_version(file_content):
              for line in file_content.split('\n'):
                  if line.strip().startswith('version = '):
                      return line.split('=', 1)[1].strip().strip('"').strip("'")
              return None

          def get_version_at_commit(commit_ref):
              try:
                  result = subprocess.run(
                      ['git', 'show', f'{commit_ref}:pyproject.toml'],
                      capture_output=True, text=True, check=True)
                  return extract_version(result.stdout)
              except subprocess.CalledProcessError:
                  return None

          current = get_version_at_commit('HEAD')
          if not current:
              print("::error::Could not extract current version from pyproject.toml")
              sys.exit(1)

          previous = get_version_at_commit('HEAD~1')
          if not previous:
              print(f"::notice::First version commit detected: {current}")
              with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                  f.write("should_publish=true\n")
              sys.exit(0)

          try:
              cur, prev = parse_version(current), parse_version(previous)
          except Exception as e:
              print(f"::error::Invalid version format - {e}")
              sys.exit(1)

          if cur > prev:
              print(f"::notice::Version increment detected: {previous} -> {current}")
              with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                  f.write("should_publish=true\n")
          elif cur == prev:
              print(f"::notice::Version unchanged: {current} - skipping publish")
              with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                  f.write("should_publish=false\n")
          else:
              print(f"::error::Version downgrade detected: {previous} -> {current}")
              sys.exit(1)
          EOF
      - name: Publish Custom Node
        if: steps.version_check.outputs.should_publish == 'true'
        uses: Comfy-Org/publish-node-action@main
        with:
          personal_access_token: ${{ secrets.COMFY_REGISTRY_KEY }}
```
