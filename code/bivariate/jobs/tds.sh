#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -d "/work3/$USER" ]; then
    default_scratch_base="/work3/$USER"
elif [ -d "/work1/$USER" ]; then
    default_scratch_base="/work1/$USER"
else
    default_scratch_base="$HOME/scratch"
fi

scratch_root="${SCRATCH_ROOT:-$default_scratch_base/bivariate}"
cache_root="$scratch_root/cache"
tmp_root="$scratch_root/tmp"
venv_root="$scratch_root/venv"

mkdir -p "$cache_root" "$tmp_root" "$venv_root" "$repo_root/jobs/logs"

export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$cache_root/xdg}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$cache_root/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$cache_root/pip}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$cache_root/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$cache_root/torch}"
export TMPDIR="${TMPDIR:-$tmp_root}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$venv_root}"

cd "$repo_root"

echo "Repository: $repo_root"
echo "Commit: $(git rev-parse HEAD)"
echo "Scratch root: $scratch_root"
echo "Python env: $UV_PROJECT_ENVIRONMENT"
nvidia-smi || true

uv sync --frozen
uv run python -m src.engine.tds_sample "$@"
