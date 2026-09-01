#!/usr/bin/env bash
# Run the test suite with branch coverage and enforce the same diff-coverage gate
# that CI applies on pull requests.  Called by the pre-push pre-commit hook.
#
# Compare-branch: origin/main mirrors the PR base.
# (CI uses github.base_ref; locally we pin the same base branch so the gate is
# equivalent.  Update this if the PR base changes.)
set -euo pipefail

COMPARE_BRANCH="origin/main"

# Mirror CI's lint/format jobs over the WHOLE tree. The commit-time ruff-pre-commit
# hook only sees [python, pyi, jupyter], so ruff's Markdown code-block formatting
# (which CI runs via `ruff format --check .`) slips past commit and fails only in CI.
# Running the exact CI commands here blocks the push before it can fail remotely.
echo "--- ruff lint (mirrors CI) ---"
uv run ruff check .

echo "--- ruff format check (mirrors CI; includes Markdown code blocks) ---"
uv run ruff format --check .

echo "--- running tests with branch coverage ---"
uv run pytest --cov=. --cov-branch --cov-report=xml

echo "--- enforcing 90% branch coverage on changed lines (diff vs ${COMPARE_BRANCH}) ---"
uv run diff-cover coverage.xml \
    --compare-branch="${COMPARE_BRANCH}" \
    --branch-coverage \
    --fail-under=90
