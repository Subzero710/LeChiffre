# Docker — LeChiffre (application principale uniquement)

This deployment is for **LeChiffre**, its existing **rs-poker** engine and
**agents**, and the **hierarchical Behavioral Engine Z0/Z1/Z2**. All Rust
compilation, CLI execution, source import, Parquet conversion, training,
evaluation, checkpoint resume and quality checks run in Docker. The separate
`crates/lechiffre-sdcfr/` and `crates/lechiffre-nlhe/` experiments are excluded
from the build context and images; the source files remain in the repository.

## Host prerequisites

Git, Docker Engine (or Docker Desktop with a Linux/WSL shell) and Docker Compose
V2. **Do not install Rust, Cargo, Clippy or rustfmt on the host.** Network access
is needed on first build for the base images and crates.io dependencies. Docker
Engine access is an OS-level requirement, not something the project modifies.
On a host where `docker compose` requires `sudo`, only explicit manually-run
Docker commands need that privilege; Git hooks and CI never call `sudo` or
silently grant access to the daemon.

```sh
git clone https://github.com/Subzero710/LeChiffre.git
cd LeChiffre
sh docker/init.sh
docker compose build
```

`docker/init.sh` creates an `.env` containing the **actual** numeric host
UID/GID, plus `inputs/`, `data/`, `models/` and `runs/`. If `.env` already exists,
the script validates it and does not overwrite it. `LOCAL_UID` and `LOCAL_GID`
are mandatory; Compose rejects missing values. Do not simply assume that the
example UID/GID are correct for your machine.

The initialization script also adds local exclusion rules to `.git/info/exclude`
for `.env`, `inputs/`, `data/`, `models/`, and `runs/`, without modifying
your checkout's `.gitignore`. This protects the local clone; repeat the
initialization step in every new clone.

The root and vendored `Cargo.lock` files are required by `--locked` builds.
The Docker build uses Rust 1.99.1 and a portable Linux CPU baseline (instead
of the vendored crate's x86-64-v3 optimization). Supported Docker platforms
must support the upstream Rust toolchain and Debian bookworm images.

## One-off jobs

These are CLI jobs; no permanently running server or exposed port is needed.
The host directories are bind-mounted, so removing a container never removes
files written to `data/`, `models/` or `runs/`. Runtime jobs start in the
writable `/workspace/runs` directory; provide absolute paths for inputs.

```sh
# LeChiffre's existing random simulation.
docker compose run --rm app --simulate-random

# Existing rs-poker agents; configs are read from ./agents in the host checkout.
docker compose run --rm poker arena compare /workspace/agents -n 1000 -p 2

# Other supported rs-poker CLI subcommands are passed through unchanged.
docker compose run --rm poker holdem rank AcKdQhJsTs
```

## Behavioral Engine — real datasets

Place original CSV/JSON source sessions under `inputs/` (read-only inside
containers), and write your real training JSON under `configs/`. The checked-in
`hierarchical.example.json` is a **syntax example**, not a valid configuration
for arbitrary datasets. Each `source-file session-id` pair must be provided
consistently for import, training, validation and resume, because exact replay
legality is reconstructed from those original session files.

```sh
# 1. Import the source and build the Parquet dataset (new output directory).
docker compose run --rm behavioural \
  --export-poker-with-the-boys-parquet /workspace/data/parquet \
  /workspace/inputs/source.csv session-id

# 2. Inspect the resulting dataset.
docker compose run --rm behavioural \
  --behaviour-dataset-stats /workspace/data/parquet

# 3. Train the hierarchical Behavioral Engine. Use a real matching config.
docker compose run --rm behavioural \
  --train-behavioural /workspace/configs/train.json \
  /workspace/data/parquet /workspace/models/model-v1.json \
  /workspace/inputs/source.csv session-id

# 4. Evaluate the checkpoint (prospective split remains unchanged).
docker compose run --rm behavioural \
  --evaluate-behavioural /workspace/models/model-v1.json \
  /workspace/data/parquet /workspace/inputs/source.csv session-id

# 5. Resume to a NEW checkpoint, never overwrite the input.
docker compose run --rm behavioural \
  --resume-behavioural /workspace/models/model-v1.json \
  /workspace/data/parquet /workspace/models/model-v2.json \
  /workspace/configs/hierarchical-resume.example.json \
  /workspace/inputs/source.csv session-id
```

Additional source sessions are appended as `file session-id` pairs. The
Behavioral Engine validates exact `rs-poker` replay legality, global split,
dataset fingerprint and model/checkpoint compatibility. **Docker does not
modify, bypass or relax these checks**. Old A/G checkpoints are refused by the
application; no automatic migration or training fallback is introduced.

### Individual Z1 baselines

```sh
# First version; the output path must not exist.
docker compose run --rm behavioural \
  --create-player-baseline /workspace/models/model-v1.json \
  /workspace/data/parquet 42 800 /workspace/models/player42-v1.json \
  /workspace/inputs/source.csv session-id

# Only process new hands before index 1000, writing a separate checkpoint.
docker compose run --rm behavioural \
  --advance-player-baseline /workspace/models/model-v1.json \
  /workspace/data/parquet 42 1000 \
  /workspace/models/player42-v1.json /workspace/models/player42-v2.json \
  /workspace/inputs/source.csv session-id
```

The checkpoint paths in these examples are illustrative. Select subject IDs,
indices and source sessions from your actual dataset. The format and strict
compatibility conditions are documented in `docs/BEHAVIOURAL_ENGINE.md`.

## Development, testing and CI

```sh
# All root quality checks, plus explicit rs-poker tests and doc tests.
docker compose run --rm test

# Format the root Rust source tree on the host through a Docker toolchain.
docker compose run --rm format

# Run the exact root formatting check without changing any files.
docker compose run --rm format -- --check
```

The `test` service mounts source **read-only** and uses persistent Docker named
volumes `cargo-registry` and `cargo-target` for Cargo dependencies and build
artifacts. The `format` service mounts the checkout writable and runs with the
host UID/GID, so formatted source files are not owned by root. Git hooks never invoke Docker, Rust, `sudo` or change host permissions.
The optional `pre-commit` hook only runs `git diff --cached --check`; there is
no `pre-push` hook. All Rust quality checks and smoke tests run in GitHub
Actions, using Docker as the hosted runner user without `sudo`. CI uses
`docker compose run --rm -T` because workflows have no interactive terminal.
A self-hosted runner must provide unprivileged Docker access in advance; the
workflow checks this and fails rather than escalating privileges.

`docker compose run --rm test` verifies build/test correctness, not predictive
quality on private real-world poker histories. To validate the Behavioral
Engine scientifically, import a real dataset and run the training/evaluation
commands above with the appropriate chronological split.

## Data and security

- `inputs/` is mounted read-only: raw, private source histories.
- `configs/` and `agents/` are mounted read-only: runtime configuration.
- `data/`, `models/`, `runs/` are persistent writable bind mounts.
- `.env`, source histories, data, models and runs are ignored locally through
  `.git/info/exclude` (set by `docker/init.sh`) and excluded from the Docker
  build context; **no real poker dataset is baked into an image**.
- Runtime containers use the numeric host UID/GID and do not require root.
- All services are CLI jobs with no published ports.

On failure the process exits with a non-zero status; there is no silent
checkpoint migration, automatic restart or alternate trainer. Keep model
checkpoints with their matching Parquet data and original sources for resume.
