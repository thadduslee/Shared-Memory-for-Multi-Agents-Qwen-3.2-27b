#!/usr/bin/env bash
set -euo pipefail

# Stage 1 of 2: agent inference on the local vLLM server (serve_qwen38.sh must
# already be up). No judging happens here -- run rejudge_qwen38.sh afterwards.

source "$HOME/gatemem/.venv-nb/bin/activate"
export HF_HOME="$HOME/.cache/huggingface"
unset HF_HUB_CACHE TRANSFORMERS_CACHE

export LOCAL_API_KEY="dummy"

DOMAIN="${1:-medical}"

cd "$HOME/GateMem"
python scripts/sweep.py \
  --config configs/sweeps/paper_matrix.yaml \
  --domains "$DOMAIN" \
  --models qwen38_27b \
  --continue_on_error
