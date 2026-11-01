#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

git config core.hooksPath .githooks

echo "Git hooks enabled for this clone."
echo "pre-commit: cargo fmt --all"
echo "CI still runs cargo fmt --all -- --check as the final guard."
