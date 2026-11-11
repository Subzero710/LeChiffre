#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"
git config core.hooksPath .githooks

echo "Git hooks enabled for this clone."
echo "pre-commit: staged whitespace check (Git only; no Docker, Rust or sudo)"
echo "pre-push: cargo fmt --all -- --check and cargo clippy; tests run in CI"
