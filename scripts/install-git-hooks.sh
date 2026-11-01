#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

git config core.hooksPath .githooks

echo "Git hooks enabled for this clone."
echo "pre-commit: cargo fmt --all"
echo "pre-push: cargo fmt --all -- --check"
echo "pre-push: cargo clippy --locked --all-targets --all-features -- -D warnings"
echo "pre-push: cargo test --locked --all-targets --all-features"
echo "pre-push: cargo test --locked --doc --all-features"
echo "CI keeps the same quality gate as the final remote guard."
