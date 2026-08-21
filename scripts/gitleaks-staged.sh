#!/usr/bin/env bash
# Scan staged files for secrets. Fail closed.
#
# CI gitleaks is too late: a finding in git history means rotate the
# credential (un-committing does not un-leak a pushed secret).
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"
cfg="$root/.gitleaks.toml"

if [[ ! -f "$cfg" ]]; then
  echo "gitleaks-staged: missing .gitleaks.toml" >&2
  exit 1
fi

if ! command -v gitleaks >/dev/null 2>&1; then
  echo "gitleaks-staged: gitleaks is not on PATH — refusing to commit." >&2
  echo "Install v8.30.1 (Debian/Ubuntu apt 8.16 is a different scanner)." >&2
  echo "  https://github.com/gitleaks/gitleaks/releases/tag/v8.30.1" >&2
  echo "SHA256 551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb" >&2
  exit 1
fi

if ! gitleaks protect --staged --source "$root" --config "$cfg" --redact --verbose --exit-code 1; then
  echo "gitleaks-staged: secret in staged files. Do not commit." >&2
  echo "Remove it from the index. If it was a real credential, rotate it." >&2
  echo "Un-staging does not un-leak a secret that already hit a remote." >&2
  exit 1
fi
