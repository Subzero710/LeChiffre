#!/bin/sh
set -eu
cd /workspace

echo 'Checking root formatting'
cargo fmt --all -- --check

echo 'Running root Clippy'
cargo clippy --locked --all-targets --all-features -- -D warnings

echo 'Running root tests'
cargo test --locked --all-targets --all-features

echo 'Running root documentation tests'
cargo test --locked --doc --all-features

echo 'Running vendored rs-poker tests (including the rsp CLI feature)'
cargo test --locked --manifest-path vendor/rs-poker/Cargo.toml --features rsp --lib --tests

echo 'Running vendored rs-poker documentation tests'
cargo test --locked --manifest-path vendor/rs-poker/Cargo.toml --features rsp --doc
