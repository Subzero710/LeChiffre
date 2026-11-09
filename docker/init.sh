#!/bin/sh
set -eu

# Execute at the repository root, regardless of the caller's current directory.
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

if [ ! -e .env ]; then
    umask 077
    {
        printf 'LOCAL_UID=%s\n' "$(id -u)"
        printf 'LOCAL_GID=%s\n' "$(id -g)"
    } > .env
    echo "Created .env with the current host UID/GID."
fi

if ! grep -Eq '^LOCAL_UID=[0-9]+$' .env || ! grep -Eq '^LOCAL_GID=[0-9]+$' .env; then
    echo 'Invalid .env: LOCAL_UID and LOCAL_GID must be explicitly set to numeric values.' >&2
    exit 1
fi

mkdir -p inputs data models runs

# Keep private datasets and checkpoints out of Git without modifying the
# checkout's .gitignore: branches and developers can have different versions.
# Git's info/exclude is local to this clone and supports the same patterns.
exclude_path="$(git rev-parse --git-path info/exclude)"
mkdir -p "$(dirname "$exclude_path")"
touch "$exclude_path"
for pattern in '/.env' '/inputs/' '/data/' '/models/' '/runs/'; do
    if ! grep -Fxq "$pattern" "$exclude_path"; then
        printf '%s\n' "$pattern" >> "$exclude_path"
    fi
done
