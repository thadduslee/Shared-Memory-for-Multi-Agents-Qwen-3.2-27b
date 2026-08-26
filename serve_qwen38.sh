#!/usr/bin/env bash
set -euo pipefail

source "$HOME/gatemem/.venv-vllm/bin/activate"
export HF_HOME="$HOME/.cache/huggingface"
unset HF_HUB_CACHE TRANSFORMERS_CACHE

# The two idle L40s. vLLM sees them as local ranks 0 and 1.
export CUDA_VISIBLE_DEVICES=4,5

MODEL="Qwen/Qwen3.8-27B"

# Qwen3.8 has thinking on by default (reasoning_effort xhigh in its chat
# template). Without a reasoning parser vLLM concatenates the chain-of-thought
# into `content`, which corrupts the graded answer and breaks every JSON parse
# in the harness. This splits it into `reasoning_content`, which the benchmark
# ignores, leaving `content` as the answer alone.
vllm serve "$MODEL" \
  --served-model-name "$MODEL" \
  --reasoning-parser qwen3 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key dummy
