#!/usr/bin/env bash
# Resolve the target path for a handoff document and ensure its directory exists.
# Writes nothing but the directory; prints the path for the caller to write to.
#
#   new-handoff.sh [slug]
#
# Honors CLIO_CODER_HANDOFF_DIR; defaults to <repo-root>/.clio-coder/handoffs.
# The default .gitignore policy ignores this machine-local context directory.
set -euo pipefail

slug="${1:-}"
date_str="$(date +%F)"

# Prefer the git repo root; fall back to the current directory.
if root="$(git rev-parse --show-toplevel 2>/dev/null)"; then :; else root="$PWD"; fi

dir="${CLIO_CODER_HANDOFF_DIR:-$root/.clio-coder/handoffs}"
mkdir -p "$dir"

if [ -n "$slug" ]; then
	# Normalize the slug: lowercase, non-alphanumerics to hyphens, trim.
	slug="$(printf '%s' "$slug" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/^-//;s/-$//')"
	printf '%s/handoff-%s-%s.md\n' "$dir" "$date_str" "$slug"
else
	printf '%s/handoff-%s.md\n' "$dir" "$date_str"
fi
