#!/usr/bin/env bash
# Block direct commits/pushes on the main branch. Invoked by pre-commit hooks.
# Usage: forbid-main.sh <commit|push>
set -euo pipefail

action="${1:-commit}"
branch="$(git symbolic-ref --short HEAD 2>/dev/null || echo '')"

if [ "$branch" = "main" ]; then
  echo "✖ Direct $action to 'main' is blocked." >&2
  echo "  Create a feature branch and open a PR instead:" >&2
  echo "    git switch -c my-feature-branch" >&2
  exit 1
fi
