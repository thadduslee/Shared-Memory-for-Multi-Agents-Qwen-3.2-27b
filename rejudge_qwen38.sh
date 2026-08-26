#!/usr/bin/env bash
set -euo pipefail

# Stage 2 of 2: grade the predictions from run_qwen.sh with GPT-4.1 via
# OpenRouter. Reads predictions.jsonl only -- the GPU server can be shut down.
# Results land in outputs/<run>__judge-gpt41, leaving the raw runs untouched.

source "$HOME/gatemem/.venv-nb/bin/activate"

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY before running}"

DOMAIN="${1:-medical}"

cd "$HOME/GateMem"
python bench/scripts/rejudge.py \
  --out_root outputs \
  --domain "$DOMAIN" \
  --model_key qwen38_27b \
  --all_baselines \
  --judge gpt41 \
  --concurrency 8 \
  --resume
