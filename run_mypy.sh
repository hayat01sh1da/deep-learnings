#!/bin/sh
# Run mypy for a single volume, from the repository root so the root
# pyproject.toml [tool.mypy] config applies.
#
# vol3 is split into independent per-chapter packages that intentionally
# reuse module basenames (src/template.py, test/test_template.py); checking
# the whole tree at once triggers mypy "Duplicate module" errors (the same
# reason pytest is configured with --import-mode=importlib), so vol3 is
# type-checked one chapter at a time.
set -u
vol="$1"

if [ "$vol" = "vol3" ]; then
  status=0
  for chapter in vol3/*/; do
    echo "===== mypy ${chapter} ====="
    mypy "${chapter}" || status=1
  done
  exit "${status}"
fi

mypy "${vol}"
