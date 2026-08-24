#!/usr/bin/env bash
# Run the test suite with branch coverage and enforce the same diff-coverage gate
# that CI applies on pull requests.  Called by the pre-push pre-commit hook.
#
# Compare-branch: origin/rework-sampler-to-per-row-img2img mirrors the PR base.
# (CI uses github.base_ref; locally we pin the same base branch so the gate is
# equivalent.  Update this if the PR base changes.)
set -euo pipefail

COMPARE_BRANCH="origin/rework-sampler-to-per-row-img2img"

echo "--- running tests with branch coverage ---"
uv run pytest --cov=. --cov-branch --cov-report=xml

echo "--- enforcing 90% branch coverage on changed lines (diff vs ${COMPARE_BRANCH}) ---"
uv run diff-cover coverage.xml \
    --compare-branch="${COMPARE_BRANCH}" \
    --branch-coverage \
    --fail-under=90
