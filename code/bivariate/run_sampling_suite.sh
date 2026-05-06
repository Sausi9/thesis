#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_CMD="${PYTHON_CMD:-uv run python}"

latest_model_sample() {
  find runs/samples -maxdepth 1 -type f -name "*.pt" \
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
$PYTHON_CMD -m src.engine.sample
MODEL_SAMPLE_PATH="$(latest_model_sample)"
echo "==> Model sample source: $MODEL_SAMPLE_PATH"

echo "==> Eval model sampling"
$PYTHON_CMD -m src.engine.eval

echo "==> Exact Jeffrey sampling"
$PYTHON_CMD -m src.engine.exact_jeffrey_sample

echo "==> Eval exact Jeffrey sampling"
$PYTHON_CMD -m src.engine.eval

echo "==> Importance resampling"
$PYTHON_CMD -m src.engine.importance_resampling \
  jeffrey.source_sample_path="$MODEL_SAMPLE_PATH"

echo "==> Eval importance resampling"
$PYTHON_CMD -m src.engine.eval

echo "==> Naive guidance sampling"
$PYTHON_CMD -m src.engine.guided_sample

echo "==> Eval naive guidance sampling"
$PYTHON_CMD -m src.engine.eval

echo "==> Done"
