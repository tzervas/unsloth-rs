#!/usr/bin/env bash
# Versioned hooks: git uses .githooks/ instead of a one-shot copy into .git/hooks.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"
chmod +x .githooks/pre-commit scripts/gitleaks-staged.sh
git config core.hooksPath .githooks
echo "core.hooksPath=.githooks (gitleaks protect --staged on every commit)"
if ! command -v gitleaks >/dev/null 2>&1; then
  echo "gitleaks is not on PATH — commits will fail until it is installed (v8.30.1)." >&2
  exit 1
fi
