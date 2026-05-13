#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_CMD="${PYTHON_CMD:-uv run python}"
HYDRA_OVERRIDES=("$@")
SAMPLES_DIR="runs/samples"

for override in "${HYDRA_OVERRIDES[@]}"; do
  case "$override" in
    sampling.output_dir=*)
      SAMPLES_DIR="${override#sampling.output_dir=}"
      ;;
  esac
done

latest_model_sample() {
  find "$SAMPLES_DIR" -maxdepth 1 -type f -name "*.pt" \
    ! -name "*exact_jeffrey*" \
    ! -name "*importance_resampling*" \
    ! -name "*naive_guidance*" \
    ! -name "*reverse_guided*" \
    ! -name "*jeffrey*" \
    -print0 |
    xargs -0 ls -t |
    head -n 1
}

echo "==> Model sampling"
if ((${#HYDRA_OVERRIDES[@]})); then
  printf '==> Hydra overrides:'
  printf ' %q' "${HYDRA_OVERRIDES[@]}"
  printf '\n'
fi

$PYTHON_CMD -m src.engine.sample "${HYDRA_OVERRIDES[@]}"
MODEL_SAMPLE_PATH="$(latest_model_sample)"
echo "==> Model sample source: $MODEL_SAMPLE_PATH"

echo "==> Eval model sampling"
$PYTHON_CMD -m src.engine.eval "${HYDRA_OVERRIDES[@]}"

echo "==> Exact Jeffrey sampling"
$PYTHON_CMD -m src.engine.exact_jeffrey_sample "${HYDRA_OVERRIDES[@]}"

echo "==> Eval exact Jeffrey sampling"
$PYTHON_CMD -m src.engine.eval "${HYDRA_OVERRIDES[@]}"

echo "==> Importance resampling"
$PYTHON_CMD -m src.engine.importance_resampling \
  "${HYDRA_OVERRIDES[@]}" \
  jeffrey.source_sample_path="$MODEL_SAMPLE_PATH"

echo "==> Eval importance resampling"
$PYTHON_CMD -m src.engine.eval "${HYDRA_OVERRIDES[@]}"

echo "==> Naive guidance sampling"
$PYTHON_CMD -m src.engine.guided_sample "${HYDRA_OVERRIDES[@]}"

echo "==> Eval naive guidance sampling"
$PYTHON_CMD -m src.engine.eval "${HYDRA_OVERRIDES[@]}"

echo "==> Done"
