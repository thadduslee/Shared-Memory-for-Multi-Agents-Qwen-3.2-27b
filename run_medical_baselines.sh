#!/bin/bash

# 1. Export the dummy key and local base URL directly in the script
export OPENAI_API_KEY="sk-local-dummy-key"
export OPENAI_BASE_URL="http://localhost:8000/v1"

DOMAINS=("medical" "office" "education" "household")
AGENTS=("a_mem" "remem_i" "remem_s" "mem0" "long_context" "rag_naive" "rag_policy")
MODEL_NAME="Qwen/Qwen2.5-32B-Instruct"

for DOMAIN in "${DOMAINS[@]}"; do
for AGENT in "${AGENTS[@]}"; do

# 2. Handle the specific arguments required for ReMeM baselines
if [ "$AGENT" == "remem_i" ]; then
AGENT_ARG="--agent remem --remem_variant iterative"
elif [ "$AGENT" == "remem_s" ]; then
AGENT_ARG="--agent remem --remem_variant single"
else
AGENT_ARG="--agent $AGENT"
fi

echo "====================================================="
echo "Running Agent: $AGENT on Domain: $DOMAIN"
echo "====================================================="

python bench/scripts/run_eval.py \
--data_dir bench/data/$DOMAIN \
$AGENT_ARG \
--llm_provider openai \
--llm_model $MODEL_NAME \
--temperature 0.2 \
--max_output_tokens 4096 \
--use_llm_judge \
--judge_provider openai \
--judge_model $MODEL_NAME \
--judge_concurrency 4

done
done