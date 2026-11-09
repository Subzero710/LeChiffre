# LeChiffre — Docker

Docker **builds and runs** the existing LeChiffre CLI, the vendored `rs-poker`
engine and its built-in agents, and the Behavioral Engine Z0/Z1/Z2. The
experimental `crates/lechiffre-sdcfr` and `crates/lechiffre-nlhe` are not built.
This setup does **not** change the application source or project layout.

## Invariants

- No `docker/init.sh`, `.env` generation, UID/GID discovery or prestart scripts.
- No implicit `mkdir`, `chown`, host permission changes, or Git configuration
  changes. Neither `build` nor the default `run` commands create host folders.
- Compose defines **no persistent volumes** or writable project bind mounts.
- Runtime processes run with an unprivileged numeric UID/GID (65532:65532).
- No network ports or long-running service dependencies; these are CLI jobs.
- The only automatic runtime output is stdout/stderr. Results not explicitly
  mounted remain inside a disposable container and are discarded with `--rm`.
- Docker Engine access is a host prerequisite. This configuration never uses
  `sudo` internally and never changes access to the Docker daemon. If this
  machine requires `sudo docker`, use it explicitly for Docker commands only.

## Run the existing programs

From the repository root, with Docker Engine and Compose V2 installed:

```sh
# Build the runtime image. To build the test toolchain too, enable the CI profile.
sudo docker compose build

# A single LeChiffre random-agent simulation.
sudo docker compose run --rm app --simulate-random

# Existing rs-poker arena agents, including CFR Basic and CFR Simple.
sudo docker compose run --rm poker arena compare /workspace/agents -n 100 -p 2

# rs-poker CLI help and hand evaluation.
sudo docker compose run --rm poker --help
sudo docker compose run --rm poker holdem rank AcKdQhJsTs
```

`docker compose up` executes the default one-shot CLI jobs (`app` and `poker`),
then exits. It does not start a long-running application server. Use
`docker compose run --rm` for a specific command, especially the Arena.

The `agents/` and `configs/` already in the repository are copied into the
image. No host directory is initialized or created for them. You must rebuild
an image to incorporate changes made to these files.

## Explicit dataset access for the Behavioral Engine

The default services have **no** host data mounts. If you want to import,
train, evaluate or resume with private data, choose the source and output paths
and opt in to exposing an **already existing** host directory. This example
mounts the existing current repository directory at `/host` and uses the
**calling user's** UID/GID for any deliberately requested output files:

```sh
# For example only: replace file paths and session id with your own real inputs.
# The mounted source directory ($PWD) already exists; no new host folder is
# created by Compose. Output files/dirs are created only by the requested CLI.
sudo docker compose run --rm -T \
  --user "$(id -u):$(id -g)" \
  --volume "$PWD:/host" \
  behavioural \
  --export-poker-with-the-boys-parquet /host/chemin/vers/sortie-parquet \
  /host/chemin/vers/source.csv id-session
```

For subsequent training, use the same *explicit* `--user` and `--volume` options,
with the existing CLI's exact arguments. For example:

```sh
sudo docker compose run --rm -T \
  --user "$(id -u):$(id -g)" --volume "$PWD:/host" \
  behavioural \
  --train-behavioural /host/chemin/vers/config.json \
  /host/chemin/vers/parquet-existant /host/chemin/vers/nouveau-modele.json \
  /host/chemin/vers/source.csv id-session
```

The paths above are **examples**, not paths asserted to exist. Pass real,
existing input paths; outputs are created by the program only when requested. The CLI retains its strict exact-legality, chronological split,
checkpoint and resume checks. See `docs/BEHAVIOURAL_ENGINE.md`.

The explicit `--volume` form can create an absent host **source** path with
some Compose versions; therefore mount **only a path you have checked already
exists** (`$PWD` is an existing directory when run from the project root).
Never mount missing paths implicitly from `compose.yaml`.

## Tests and CI

```sh
# Same checks as GitHub Actions: no local files written by the test container.
sudo docker compose --profile ci build
sudo docker compose run --rm -T test
```

`test` mounts the **existing** checkout read-only. Build products and temporary
Cargo files are inside the ephemeral test container. The CI runs Docker as the
GitHub runner user and does not require `sudo`; user Git hooks do not run Docker.
The build and test processes may create internal image layers and transient
container filesystems; they do not create project directories on the host.

Removing obsolete files left by an older dockerization (`.env`, `inputs/`,
`data/`, `models/`, `runs/`, or `.git/info/exclude` entries) is **not performed**.
Review these manually if they already exist; this patch never deletes user data.
