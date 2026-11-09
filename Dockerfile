# syntax=docker/dockerfile:1.7

# All compiler and dependency installation happens in containers.
FROM rust:1.99.1-bookworm AS toolchain
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config perl \
    && rm -rf /var/lib/apt/lists/*
RUN rustup component add rustfmt clippy

# Do not inherit the vendor crate's x86-64-v3 CPU requirement: containers
# must run on baseline x86-64 machines. Keep the deep CFR process stack budget.
ENV RUSTFLAGS="-C link-arg=-Wl,-zstack-size=47185920 -C force-frame-pointers=yes" \
    CARGO_TARGET_DIR=/cargo-target
WORKDIR /workspace

FROM toolchain AS build
# Copy only application sources and their required local dependencies.
# Never copy host datasets, checkpoints, .env files or unrelated experiments.
COPY Cargo.toml Cargo.lock /workspace/
COPY src /workspace/src
COPY vendor/rs-poker /workspace/vendor/rs-poker
COPY configs /workspace/configs
COPY agents /workspace/agents
RUN --mount=type=cache,id=lechiffre-cargo-registry,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=lechiffre-build-target,target=/cargo-target,sharing=locked \
    set -eu; \
    cargo build --release --locked --bin LeChiffre; \
    cargo build --release --locked --manifest-path vendor/rs-poker/Cargo.toml --features rsp --bin rsp; \
    mkdir -p /dist; \
    install -m 0755 /cargo-target/release/LeChiffre /dist/lechiffre; \
    install -m 0755 /cargo-target/release/rsp /dist/rsp

FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgcc-s1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
ENV HOME=/tmp
USER 65532:65532
COPY --from=build /dist/lechiffre /usr/local/bin/lechiffre
COPY --from=build /dist/rsp /usr/local/bin/rsp
COPY agents /workspace/agents
COPY configs /workspace/configs
# Production images contain no Rust sources or experimental crates.
# No user data or host directories are mounted unless requested explicitly.
