#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 1 ] && [ "$#" -ne 3 ]; then
    echo "Usage: $0 <mnist|cifar> [--resume <checkpoint_path>]" >&2
    exit 1
fi

dataset="$1"
resume_args=()
if [ "$#" -eq 3 ]; then
    if [ "$2" != "--resume" ]; then
        echo "Usage: $0 <mnist|cifar> [--resume <checkpoint_path>]" >&2
        exit 1
    fi
    resume_args=(--resume "$3")
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -d "/work3/$USER" ]; then
    default_scratch_base="/work3/$USER"
elif [ -d "/work1/$USER" ]; then
    default_scratch_base="/work1/$USER"
else
    default_scratch_base="$HOME/scratch"
fi
scratch_root="${SCRATCH_ROOT:-$default_scratch_base/sandbox}"
cache_root="$scratch_root/cache"
tmp_root="$scratch_root/tmp"

mkdir -p "$cache_root" "$tmp_root" "$repo_root/jobs/logs"

export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$cache_root/xdg}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$cache_root/uv}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$cache_root/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$cache_root/torch}"
export TMPDIR="${TMPDIR:-$tmp_root}"

cd "$repo_root"

echo "Repository: $repo_root"
echo "Commit: $(git rev-parse HEAD)"
echo "Dataset: $dataset"
echo "Scratch root: $scratch_root"
if [ "${#resume_args[@]}" -gt 0 ]; then
    echo "Resume checkpoint: ${resume_args[1]}"
fi

uv sync --frozen
uv run python -m src.engine.train "$dataset" "${resume_args[@]}"
