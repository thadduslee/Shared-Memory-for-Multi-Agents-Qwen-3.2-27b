#!/usr/bin/env bash
# Evaluator and Judge on the LOCAL vLLM; all three agents on OpenRouter.
# Sourced, never executed.
#
#   Architect, Developer, Critic   deepseek-v4-flash-0731   OpenRouter
#   Evaluator, Judge               Qwen/Qwen3.8-27B         localhost:8002
#
#   CUDA_VISIBLE_DEVICES=<n> scripts/serve/serve_agents.sh   # if not already up
#   set -a; source .env; set +a
#   source scripts/serve/env_local_eval.sh
#   cd "Auto Research" && python main.py --real --no-full --dev-checkpoints 579 --max-iterations 20
#
# The benchmark-facing pair is local because it is the metered, high-fan-out
# half -- 579 answers plus 579 verdicts per iteration -- and a dedicated GPU has
# no rate limit to degrade them. The three agents are API-backed because they
# issue a handful of long calls each and want a stronger model than a 27B.

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
set -a; [[ -f "$_REPO_ROOT/.env" ]] && source "$_REPO_ROOT/.env"; set +a
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY missing (expected in $_REPO_ROOT/.env)}"

# ---- agentic nodes: DeepSeek on OpenRouter ----
unset AGENT_BASE_URL AGENT_API_KEY_ENV AGENT_ROUTE AGENT_IS_VLLM DSH_BASE_URL DEEPSEEK_BASE_URL DEEPSEEK_API_KEY
unset CRITIC_BASE_URL CRITIC_API_KEY_ENV CRITIC_ROUTE
unset ARCHITECT_MAX_TOKENS DEVELOPER_MAX_TOKENS
export ARCHITECT_MODEL=deepseek/deepseek-v4-flash-0731
export DEVELOPER_MODEL=deepseek/deepseek-v4-flash-0731
export CRITIC_MODEL=deepseek/deepseek-v4-flash-0731
export AGENT_TRANSPORT=dsh
export DEVELOPER_TRANSPORT=http      # a tool schema has nowhere to go on dsh
# http for the Critic too. Its profile now mounts NO tools, so dsh buys it
# nothing, and the http path is where `temperature` and `max_tokens` are
# actually honoured. See the capabilities note in harness/profiles.py.
export CRITIC_TRANSPORT=http
export OPENROUTER_REASONING_EFFORT="${OPENROUTER_REASONING_EFFORT:-low}"
# Runaway guard, paired with DSH_CRITIC_TIMEOUT_S=1800. Uncapped, the Critic
# inherits the harness default of 256000 and times out every iteration.
export CRITIC_MAX_TOKENS="${CRITIC_MAX_TOKENS:-32768}"

# ---- evaluator + judge: local vLLM, served as Qwen/Qwen3.8-27B ----
# The "Qwen/" prefix is load-bearing: vllm_chat_template_kwargs only emits
# {"enable_thinking": false} for a model matching VLLM_THINKING_MODEL_PREFIXES,
# and a thinking evaluator reasons past its read timeout on every checkpoint.
export VLLM_BASE_URL=http://localhost:8002/v1
export VLLM_API_KEY="${LOCAL_API_KEY:-sk-local-dummy-key}"
export EVALUATOR_MODEL=Qwen/Qwen3.8-27B
export EVAL_TRANSPORT=http
export EVALUATOR_MAX_TOKENS="${EVALUATOR_MAX_TOKENS:-4096}"

export JUDGE_ROUTE=openai
export OPENAI_BASE_URL=http://localhost:8002/v1
export OPENAI_API_KEY="$VLLM_API_KEY"
export JUDGE_MODEL=Qwen/Qwen3.8-27B
export JUDGE_TRANSPORT=http
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-2048}"
export USE_LLM_JUDGE=true
# The openai route infers is_vllm from its base URL; pinned so the judge gets
# the same thinking-disable the evaluator does.
export OPENAI_ROUTE_IS_VLLM=true

# One dedicated GPU, no rate limit: sized for the fan-out rather than for
# politeness. The Evaluator and the Judge run in separate stages, so they do
# not contend.
export VLLM_MAX_CONCURRENCY="${VLLM_MAX_CONCURRENCY:-32}"
export JUDGE_MAX_CONCURRENCY="${JUDGE_MAX_CONCURRENCY:-16}"

# ---- benchmark ----
export GATEMEM_REPO="$_REPO_ROOT"
export MOCK_MODE=false
export SKIP_FULL_STAGE=true        # the dev slice IS all 579 here
export EXPECTED_DEV_CHECKPOINTS=579
# GateMem's own query_prompt.txt / judge_prompt.txt for the two benchmark-facing
# nodes; `local` would restore this project's hand-written pair.
export BENCHMARK_PROMPTS=gatemem

export MAX_WALLCLOCK_S="${MAX_WALLCLOCK_S:-172800}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-100000000}"

unset _REPO_ROOT
