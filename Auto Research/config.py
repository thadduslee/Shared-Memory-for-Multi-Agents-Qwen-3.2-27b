"""Central settings for the GateMem self-improving orchestrator.

Every model id, endpoint, threshold and path used anywhere in the package is
defined here exactly once.  Nothing else in the codebase may hardcode a model
slug or a base URL -- if you find one, it is a bug.

Values are read from the environment (optionally via a `.env` file) so that a
deployment can be reconfigured without touching code.  See `.env.example`.

MODEL SLUG WARNING
------------------
The model ids below are the ones specified in the research brief.  They must be
confirmed against the provider's *current* model list before a real run
(`GET https://openrouter.ai/api/v1/models`, `GET {OPENAI_BASE_URL}/models`), and
`EVALUATOR_MODEL` must match the vLLM cluster's `--served-model-name` exactly or
the OpenAI-compatible endpoint will reject every request with a 404.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------
# .env loading (no hard dependency on python-dotenv)
# --------------------------------------------------------------------------


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: `KEY=value` lines, `#` comments, no interpolation.

    Environment variables that are already set always win, so an operator can
    override a committed .env from the shell.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
_load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_opt_int(key: str, default: int | None) -> int | None:
    """`int` or the explicit absence of one.

    `""`, `"none"`, `"null"`, `"0"` and `"-1"` all mean UNCAPPED -- the caller
    omits the parameter entirely rather than sending a zero, which every
    provider would read as "generate nothing".
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    raw = raw.strip()
    if raw.lower() in {"", "none", "null", "unlimited", "off"}:
        return None
    try:
        value = int(raw)
    except ValueError:
        return default
    return None if value <= 0 else value


# ==========================================================================
# 1. Mock mode
# ==========================================================================

# MOCK_MODE=True is the shipped default: the entire macro-graph runs offline
# with no API keys, no GPUs and no `dsh` runtime.  Flipping this to False swaps
# in the real clients *behind the same interfaces* -- graph topology is
# identical in both modes (see graph.py, which never branches on MOCK_MODE).
MOCK_MODE: bool = _env_bool("MOCK_MODE", True)

# Which deterministic script drives the mock scores.  See mocks/scripted.py for
# the full list; each one forces a different routing path through the graph.
MOCK_SCENARIO: str = _env("MOCK_SCENARIO", "happy_path")


# ==========================================================================
# 2. Models  (brief section 4)
# ==========================================================================

ARCHITECT_MODEL: Final[str] = _env("ARCHITECT_MODEL", "deepseek/deepseek-v4-flash-0731")
DEVELOPER_MODEL: Final[str] = _env("DEVELOPER_MODEL", "deepseek/deepseek-v4-flash-0731")
CRITIC_MODEL: Final[str] = _env("CRITIC_MODEL", "deepseek/deepseek-v4-flash-0731")
# Must equal the vLLM server's --served-model-name.
EVALUATOR_MODEL: Final[str] = _env("EVALUATOR_MODEL", "qwen3.8-27b")
JUDGE_MODEL: Final[str] = _env("JUDGE_MODEL", "gpt-4.1")


# --------------------------------------------------------------------------
# Completion budgets  (per node, per call)
# --------------------------------------------------------------------------
#
# `None` = UNCAPPED: `max_tokens` is omitted from the request body entirely and
# the model generates until it emits a stop token or fills its context window.
# This is the shipped default -- a node is allowed to run until it has actually
# finished an answer, rather than being cut off mid-document at an arbitrary
# ceiling.
#
# WHAT STILL BOUNDS A CALL, because "uncapped" does not mean "unbounded":
#
#   * wall clock -- `nodes/_transport.py` wraps every http call in
#     `asyncio.wait_for(timeout_s)` and the dsh path passes
#     `request_timeout_seconds`. A runaway generation is killed on time, which
#     is the bound that actually holds; a token ceiling never was one.
#   * MAX_TOTAL_TOKENS / MAX_WALLCLOCK_S -- the run-level budget guards in
#     section 7 still halt the loop.
#   * the model's own context window, the hard upper bound in every case.
#   * OPENROUTER_REASONING_EFFORT -- still capped, and MORE important now, not
#     less. See the reasoning-budget note below: on a reasoning model the
#     thinking channel is billed against the same completion budget, so an
#     uncapped call lets an uncapped reasoner think until the context fills.
#     Removing the token ceiling does not remove the need for that cap.
#
# Set any of these to a positive integer to restore a ceiling for one node;
# "", "none" or "0" means uncapped. The previous shipped values, kept here so a
# revert does not need archaeology: architect 12288, developer 16384,
# evaluator 2048, judge 2048, critic 8192.
ARCHITECT_MAX_TOKENS: Final[int | None] = _env_opt_int("ARCHITECT_MAX_TOKENS", None)
DEVELOPER_MAX_TOKENS: Final[int | None] = _env_opt_int("DEVELOPER_MAX_TOKENS", None)
EVALUATOR_MAX_TOKENS: Final[int | None] = _env_opt_int("EVALUATOR_MAX_TOKENS", None)
JUDGE_MAX_TOKENS: Final[int | None] = _env_opt_int("JUDGE_MAX_TOKENS", None)
CRITIC_MAX_TOKENS: Final[int | None] = _env_opt_int("CRITIC_MAX_TOKENS", None)


# ==========================================================================
# 3. Endpoints
# ==========================================================================

OPENROUTER_BASE_URL: Final[str] = _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
VLLM_BASE_URL: Final[str] = _env("VLLM_BASE_URL", "http://localhost:8000/v1")
OPENAI_BASE_URL: Final[str] = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")

OPENROUTER_API_KEY_ENV: Final[str] = "OPENROUTER_API_KEY"
OPENAI_API_KEY_ENV: Final[str] = "OPENAI_API_KEY"
VLLM_API_KEY_ENV: Final[str] = "VLLM_API_KEY"  # usually a dummy for a local cluster

# OpenRouter attribution headers (optional, but they get your app listed on the
# OpenRouter leaderboards and are used for abuse triage).
OPENROUTER_REFERER: Final[str] = _env("OPENROUTER_REFERER", "https://github.com/rzhub/GateMem")
OPENROUTER_TITLE: Final[str] = _env("OPENROUTER_TITLE", "GateMem Self-Improving Orchestrator")

# --------------------------------------------------------------------------
# OpenRouter provider pinning
# --------------------------------------------------------------------------
#
# WHY (measured, runs_iter3, 2026-08-25). An OpenRouter model id is not one
# server: "deepseek/deepseek-v4-flash-0731" was being served by 28 competing
# hosts, and OpenRouter picks one per request. During that run several hosts
# were down or dying -- 30-minute uptimes of 11% (Ambient), 32% (BaseTen),
# 41% (Phala), 58% (Baidu) -- while DeepSeek's own endpoint sat at 99.99%.
# A request landing on a sick host failed in one of the two shapes the logs
# show: an SSE stream that emitted a single token and then closed without
# [DONE] ("architect failed: empty response (error)", which halted the run),
# or a connection that returned no bytes until the transport's wall-clock
# timeout ("http transport exceeded 150s"). Pinning names the hosts we trust
# and refuses the rest.
#
# The pin applies ONLY to models matching OPENROUTER_PIN_MODEL_PREFIXES. The
# Judge can share the openrouter route with "openai/gpt-4.1", which none of
# the DeepSeek hosts serve -- a route-wide pin with fallbacks off would fail
# every Judge call outright. Prefix-gating scopes the pin to the models that
# actually have the problem.
OPENROUTER_PROVIDER_ORDER: Final[tuple[str, ...]] = tuple(
    name.strip()
    for name in _env("OPENROUTER_PROVIDER_ORDER", "DeepSeek,Fireworks,Novita").split(",")
    if name.strip()
)
# With three healthy providers named, refusing the long tail is safer than
# falling back into it -- the long tail is the failure mode being fixed.
OPENROUTER_ALLOW_FALLBACKS: bool = _env_bool("OPENROUTER_ALLOW_FALLBACKS", False)
OPENROUTER_PIN_MODEL_PREFIXES: Final[tuple[str, ...]] = tuple(
    prefix.strip()
    for prefix in _env("OPENROUTER_PIN_MODEL_PREFIXES", "deepseek/").split(",")
    if prefix.strip()
)


def openrouter_provider_preferences(model: str) -> dict[str, object] | None:
    """The OpenRouter `provider` routing block for `model`, or None if unpinned.

    Setting OPENROUTER_PROVIDER_ORDER to the empty string disables pinning
    entirely and restores OpenRouter's own routing.
    """
    if not OPENROUTER_PROVIDER_ORDER:
        return None
    if not any(model.startswith(prefix) for prefix in OPENROUTER_PIN_MODEL_PREFIXES):
        return None
    return {
        "order": list(OPENROUTER_PROVIDER_ORDER),
        "allow_fallbacks": OPENROUTER_ALLOW_FALLBACKS,
    }


# --------------------------------------------------------------------------
# Reasoning budget
# --------------------------------------------------------------------------
#
# WHY (measured, 2026-08-25). DeepSeek's v4-flash is a REASONING model, and on
# OpenRouter its reasoning tokens are billed against `max_tokens` alongside the
# visible answer. Left uncapped on the Architect's open-ended design prompt it
# spends the entire budget thinking and never starts the document: an observed
# call returned `finish_reason="length"` with `reasoning_tokens=12288` -- the
# whole allowance -- and `content=None`. The node sees an empty string, reports
# "empty response", and the run halts at MGS=0 having paid full price for the
# turn. This is the failure behind the runs_iter3 and runs_fresh2 halts, and it
# is NOT the provider roulette the pin above fixes; it reproduces on DeepSeek's
# own 99.99%-uptime endpoint.
#
# Capping the effort makes the model commit. The same prompt with effort="low"
# returned `finish_reason="stop"`, 3932 reasoning tokens and a complete 13k-char
# design document.
#
# Prefix-gated for the same reason the provider pin is: the Judge shares the
# openrouter route with "openai/gpt-4.1", which is not a reasoning model and has
# no business being sent a reasoning block.
OPENROUTER_REASONING_EFFORT: Final[str] = _env("OPENROUTER_REASONING_EFFORT", "low")
OPENROUTER_REASONING_MODEL_PREFIXES: Final[tuple[str, ...]] = tuple(
    prefix.strip()
    for prefix in _env("OPENROUTER_REASONING_MODEL_PREFIXES", "deepseek/").split(",")
    if prefix.strip()
)


def openrouter_reasoning(model: str) -> dict[str, object] | None:
    """The OpenRouter `reasoning` block for `model`, or None to leave it alone.

    Setting OPENROUTER_REASONING_EFFORT to the empty string restores the
    provider's own default reasoning budget.
    """
    if not OPENROUTER_REASONING_EFFORT:
        return None
    if not any(model.startswith(prefix) for prefix in OPENROUTER_REASONING_MODEL_PREFIXES):
        return None
    return {"effort": OPENROUTER_REASONING_EFFORT}


# The same failure, on the other route (measured, 2026-08-29). Qwen3 is a hybrid
# reasoning model and vLLM applies its thinking chat template by default, so the
# Evaluator's answer call reasons before it emits the ```json fence. On the
# medical prompts that runs long enough to exceed HTTP_TIMEOUT_S: an observed
# dev round burned 12 ReadTimeouts (4 attempts x 3 checkpoints, 1440s of a 1460s
# shard) and recorded `answer: ""` for all three -- the only three checkpoints in
# the round whose retrieval had actually succeeded. U was 0.0000 with a working
# retrieval layer, and the Critic then attributed the loss to retrieval.
#
# The fix is not a `max_tokens` ceiling: the reply must carry a COMPLETE fenced
# block, and truncating it produces the same empty answer by a different route
# (see tests/test_completion_budget.py for why every profile ships uncapped).
# Turning the thinking template off is measured at 110 completion tokens in 4.2s
# against 447 in 16.8s, and the non-thinking reply was the more complete of the
# two.
#
# Prefix-gated exactly like the OpenRouter block above: a non-thinking model
# served on the same vLLM route would reject the unknown template kwarg.
VLLM_DISABLE_THINKING: Final[bool] = _env_bool("VLLM_DISABLE_THINKING", True)
VLLM_THINKING_MODEL_PREFIXES: Final[tuple[str, ...]] = tuple(
    prefix.strip()
    for prefix in _env("VLLM_THINKING_MODEL_PREFIXES", "Qwen/,qwen/").split(",")
    if prefix.strip()
)


def vllm_chat_template_kwargs(model: str) -> dict[str, object] | None:
    """vLLM `chat_template_kwargs` for `model`, or None to leave it alone.

    Setting VLLM_DISABLE_THINKING=0 restores the model's own default template,
    which is the escape hatch if a future evaluator model needs to reason.
    """
    if not VLLM_DISABLE_THINKING:
        return None
    if not any(model.startswith(prefix) for prefix in VLLM_THINKING_MODEL_PREFIXES):
        return None
    return {"enable_thinking": False}


# ==========================================================================
# 4. Concurrency / HTTP
# ==========================================================================

# The evaluator cluster is 6 GPUs behind one load balancer.  This semaphore is
# the ONLY thing standing between the fan-out and a thundering herd, so it is
# sized to the cluster, not to the checkpoint count.
VLLM_MAX_CONCURRENCY: int = _env_int("VLLM_MAX_CONCURRENCY", 12)  # ~2 in-flight per GPU
OPENROUTER_MAX_CONCURRENCY: int = _env_int("OPENROUTER_MAX_CONCURRENCY", 4)
JUDGE_MAX_CONCURRENCY: int = _env_int("JUDGE_MAX_CONCURRENCY", 4)

HTTP_MAX_CONNECTIONS: int = _env_int("HTTP_MAX_CONNECTIONS", 64)
HTTP_MAX_KEEPALIVE: int = _env_int("HTTP_MAX_KEEPALIVE", 16)
HTTP_TIMEOUT_S: float = _env_float("HTTP_TIMEOUT_S", 120.0)
HTTP_CONNECT_TIMEOUT_S: float = _env_float("HTTP_CONNECT_TIMEOUT_S", 10.0)
HTTP_MAX_RETRIES: int = _env_int("HTTP_MAX_RETRIES", 4)
HTTP_BACKOFF_BASE_S: float = _env_float("HTTP_BACKOFF_BASE_S", 0.75)
HTTP_BACKOFF_MAX_S: float = _env_float("HTTP_BACKOFF_MAX_S", 30.0)

# The ceiling for a `Retry-After` the provider states explicitly, as opposed to
# a backoff we invented. Separate from HTTP_BACKOFF_MAX_S because the two answer
# different questions: that one is "how long should WE guess", this one is "how
# long is the provider willing to tell us to wait before we give up on it".
#
# OpenRouter's in-flight 402 says `Retry-After: 120`; clamping that to the 30s
# guess-ceiling meant every retry arrived while the same requests were still in
# flight. See `llm/client._is_transient_payment_required`.
HTTP_RETRY_AFTER_MAX_S: float = _env_float("HTTP_RETRY_AFTER_MAX_S", 180.0)


# ==========================================================================
# 5. GateMem dataset  (brief section 6.3)
# ==========================================================================

# Path to a GateMem checkout.  The medical domain lives at
# {GATEMEM_REPO}/bench/data/medical/{episodes,checkpoints}.jsonl
GATEMEM_REPO: Final[Path] = Path(_env("GATEMEM_REPO", str(Path.home() / "GateMem"))).expanduser()
GATEMEM_DOMAIN: Final[str] = "medical"
GATEMEM_DATA_DIR: Final[Path] = GATEMEM_REPO / "bench" / "data" / GATEMEM_DOMAIN

# Configured *expectations*, not authorities.  The real counts are always
# derived from the loaded data; a mismatch is logged as a warning (brief 6.3).
EXPECTED_DEV_CHECKPOINTS: int = _env_int("EXPECTED_DEV_CHECKPOINTS", 50)
EXPECTED_FULL_CHECKPOINTS: int = _env_int("EXPECTED_FULL_CHECKPOINTS", 579)

# Seed for the dev-slice selection.  Fixed so that iteration N and iteration
# N+1 are scored on exactly the same 50 checkpoints and are therefore
# comparable -- this is the whole point of a "dev slice".
DEV_SLICE_SEED: int = _env_int("DEV_SLICE_SEED", 20260824)

# One Send per shard of this many checkpoints.  Smaller shards = finer-grained
# failure isolation and a tighter circuit breaker; larger shards = less
# per-Send overhead.
EVAL_SHARD_SIZE: int = _env_int("EVAL_SHARD_SIZE", 10)


# ==========================================================================
# 6. Thresholds and routing constants  (brief section 7)
# ==========================================================================

# DEV_GATE_MGS is the *scale-up gate*: clear it on the 50-checkpoint dev slice
# and you earn the right to spend the full 579-checkpoint run.
DEV_GATE_MGS: float = _env_float("DEV_GATE_MGS", 0.80)

# MGS_TARGET is the *stop condition*: reach it and the research loop is done.
# Deliberately distinct from (and above) DEV_GATE_MGS -- a system can be worth
# scaling up to full evaluation well before it is actually finished.
MGS_TARGET: float = _env_float("MGS_TARGET", 0.85)

MAX_ITERATIONS: int = _env_int("MAX_ITERATIONS", 10)

# Hard-disable the full 579-checkpoint stage regardless of the gate.
#
# WHY THIS EXISTS: a real run costs two model calls per checkpoint per stage.
# Verifying that OpenRouter, the vLLM cluster and the judge are all reachable
# should cost cents, not the price of a full evaluation. `--smoke` sets this
# together with a tiny dev slice; leave it False for real research runs.
SKIP_FULL_STAGE: bool = _env_bool("SKIP_FULL_STAGE", False)
MAX_DEV_RETRIES: int = _env_int("MAX_DEV_RETRIES", 5)

# ---- Self-correction: never build on top of a measured regression ----
#
# WHY THESE EXIST. Before them, `prepare_workspace` seeded iteration N from
# iteration N-1 unconditionally, so the loop's lineage followed the LAST code
# rather than the BEST code and could only ever drift away from its own high
# water mark. run-8cf58d33b311 went 0.3172 -> 0.2222 -> 0.1830 -> 0.1190 doing
# exactly that, and reported `best MGS=0.1190` at the end. See scoreboard.py.
#
# ROLLBACK_TO_BEST=False restores the old unconditional N-1 lineage. It is a
# knob rather than a deleted branch because "always inherit the previous
# iteration" is a legitimate experiment -- it is just not a safe default.
ROLLBACK_TO_BEST: bool = _env_bool("ROLLBACK_TO_BEST", True)

# ---- Linear lineage: may a FAILED build be a parent? ----
#
# Only consulted when ROLLBACK_TO_BEST is False. Under the champion rule the
# question cannot arise -- an iteration that never reached the Judge has no
# score row, so it is not a champion candidate and can never be inherited.
# Strict N-1 lineage has no such protection: iteration 14 of run-b5d7565ddb4a
# ended `2 failed, 38 passed` and was never judged, and unconditional N-1 would
# have handed that broken tree to iteration 15 as its foundation.
#
# True (the default) seeds iteration N from the most recent iteration that
# actually BUILT and was judged -- 15 <- 13 when 14 failed -- so the lineage
# still compounds but never compounds onto a tree whose tests do not pass. The
# failed iteration is skipped as a PARENT only; everything it taught the loop
# still flows forward, because `dev_failure_report` and `dev_failure_history`
# are separate channels from the workspace and the Architect reads both. It is
# told, in `trend_table`, that this is why its workspace is not N-1's.
#
# False restores genuinely unconditional N-1 lineage, broken parents included.
# There is no good reason to want it outside of reproducing an old run.
LINEAGE_SKIP_FAILED_BUILDS: bool = _env_bool("LINEAGE_SKIP_FAILED_BUILDS", True)

# ---- The Developer's per-episode turn ceiling, and the landing window ----
#
# DEVELOPER_MAX_TURNS bounds one episode's model turns. MAX_DEV_RETRIES only
# counts FAILED observations, so an episode making slow *green* progress --
# read, read, list, read -- is bounded by nothing else except the wall clock.
#
# DEVELOPER_LANDING_TURNS is how many turns before that ceiling the loop starts
# telling the model to STOP EDITING AND RUN THE GATES.
#
# WHY THE SECOND ONE EXISTS. The ceiling used to be a cliff the model could not
# see: the status block reported `retry=0/5` and said nothing at all about
# turns, so an episode could arrive at turn 59 with every file written and no
# idea it was about to be cut off. That is not hypothetical -- it is how
# run-b3275eb7e373 lost two of its five iterations:
#
#   iteration 3: 70 steps, 53 of them `read_file`, all three source files
#                written, ZERO gate tools ever called.
#   iteration 5: 69 steps, 60 of them `read_file`, `run_tests` green at
#                pass_rate 1.0, and `compile_check` and `sql_exec` never run.
#
# Both were two tool calls from a green build. An episode that ends without
# running its gates scores nothing at all and the work it wrote is discarded,
# so the cheapest possible intervention -- telling the model how many turns it
# has left, and when to spend them landing rather than reading -- is worth far
# more than a larger ceiling would be. Note the failing episodes were 76% and
# 87% `read_file`: they were over-reading, not under-working, and a bigger
# budget is more room to over-read.
DEVELOPER_MAX_TURNS: int = _env_int("DEVELOPER_MAX_TURNS", 60)
DEVELOPER_LANDING_TURNS: int = _env_int("DEVELOPER_LANDING_TURNS", 12)

# How much better than the champion an iteration must score to BECOME the
# champion. 0.0 means "strictly greater by any margin". Raise it to require a
# real margin on a noisy slice; a tie is never adopted at any setting.
ROLLBACK_TOLERANCE: float = _env_float("ROLLBACK_TOLERANCE", 0.0)

# How many times one iteration may be re-run when the Developer episode died of
# INFRASTRUCTURE rather than of the design -- transport timeouts, empty replies,
# a turn ceiling reached without a single gate tool ever running. Redesigning in
# response to a flapping endpoint is how run-8cf58d33b311 turned an eight-minute
# OpenRouter outage into a schema change: iteration 2's report said "unmet
# mandatory gates: tests_ok, migration_ok" when the truth was that `run_tests`
# and `sql_exec` had never been called at all. See
# nodes/developer.py::classify_failure.
MAX_INFRA_RETRIES: int = _env_int("MAX_INFRA_RETRIES", 2)

# ---- The answerer's health: when a stage's scores stop meaning anything ----
#
# A render call that fails falls back to the gated record bodies (see
# `nodes/medical_evaluator._render_answer`). For ONE checkpoint that is the
# right trade -- better than scoring a transport blip as a design failure. For a
# whole stage it is a different system being measured: the retrieval and gating
# layer with raw evidence pasted in as the answer, rather than the pipeline
# under test.
#
# RENDER_DEGRADED_THRESHOLD is the fraction of answering checkpoints that may
# fall back before the stage is marked `render_degraded`. A degraded stage is
# still scored and still recorded -- the action-shape half of U is genuinely
# unaffected -- but it is flagged everywhere the score is reported, and it is
# never eligible to become the champion, because comparing a score measured
# without the answerer against one measured with it is a measurement error.
RENDER_DEGRADED_THRESHOLD: float = _env_float("RENDER_DEGRADED_THRESHOLD", 0.5)

# HALT_ON_DEGRADED_EVAL stops the run outright when a stage comes back degraded.
#
# WHY THE DEFAULT IS TRUE. run-c993a6e93050 ran 23 iterations over 3 hours and
# spent 12.5 million tokens with the local vLLM server unreachable: 7,496
# ConnectErrors, every single render falling back, 0 of 357 answering
# predictions written by a model. The loop optimised, rolled back, critiqued and
# reported `best MGS=0.8366` throughout. A dead endpoint does not fix itself, so
# every iteration after the first degraded one is money spent measuring
# something nobody asked about. Set it False to let a run continue on degraded
# numbers -- they are labelled either way.
HALT_ON_DEGRADED_EVAL: bool = _env_bool("HALT_ON_DEGRADED_EVAL", True)

# Circuit breaker: abort the batch once this many results share one normalized
# failure signature.  3 is low on purpose -- if the first three shards all die
# the same way, shard four will too.
FAILFAST_SIGNATURE_K: int = _env_int("FAILFAST_SIGNATURE_K", 3)

# Curriculum: a phase must score at least this on its own slice to advance.
# The curriculum is OFF by default because, on a full-benchmark run, it does not
# do what its name says -- it mislabels.
#
# `phase_score` is the pass rate on the DOMINANT phase, meaning the one with the
# most checkpoints in the round. Evaluate all 579 and that is always
# `standard_retrieval` (210), whose predicate is `query_type == "utility" and not
# attack_type` -- i.e. the utility bucket. So `phase_score` IS `U`, identically,
# every iteration, and the advance rule reduces to "U >= 0.70".
#
# Measured on runs_full579_v4/buckets_told: phase_score equalled U to three
# decimals in all 20 iterations; the phase advanced one step per iteration from
# iteration 2 and parked on `adversarial_injection` at iteration 7. For the
# remaining 13 iterations the Architect was told it was designing for
# adversarial injection -- which was at 0.978 -- while the losses were 23
# checkpoints in standard_retrieval and 9 in cross_principal_leakage. The label
# was not merely inert; it pointed away from the failures.
#
# It was sound when the dev slice was 50 stratified checkpoints and the current
# phase genuinely dominated the sample. It is not sound on the full benchmark.
# Set CURRICULUM_ENABLED=true to restore the advance/halt behaviour; the phase
# BREAKDOWN in judge_report is unaffected either way and stays useful.
CURRICULUM_ENABLED: bool = _env_bool("CURRICULUM_ENABLED", False)

CURRICULUM_PASS_THRESHOLD: float = _env_float("CURRICULUM_PASS_THRESHOLD", 0.70)

# Ordered easy -> hard.  Phase names are the brief's; the mapping from these
# names onto real GateMem (query_type, attack_type) annotations lives in
# gatemem_adapter.CURRICULUM_PREDICATES.
CURRICULUM_PHASES: Final[tuple[str, ...]] = (
    "standard_retrieval",
    "scoped_access_control",
    "cross_principal_leakage",
    "active_forgetting",
    "cryptographic_shredding",
    "adversarial_injection",
)


# Transport for the two high-fan-out nodes.
#
# The brief requires every node to run on the dsh harness, and "dsh" is the
# default here for exactly that reason.  "http" exists as a documented escape
# hatch because the Evaluator and the Judge are single-shot scorers rather than
# tool-using agents, and launching 579 harness subprocesses per stage costs far
# more than 579 chat completions.  Both paths are implemented; the graph
# topology is identical either way.
# Transport for the three agentic nodes. `dsh` is the default per the brief.
# `http` exists because reaching OpenRouter through the harness relies on an
# unverified assumption about DEEPSEEK_BASE_URL; see nodes/_transport.py.
AGENT_TRANSPORT: str = _env("AGENT_TRANSPORT", "dsh")  # dsh | http

# PER-NODE OVERRIDE FOR THE DEVELOPER, and the one place this project departs
# from "every node on dsh". Empty string = inherit AGENT_TRANSPORT.
#
# WHY IT DEFAULTS TO http (measured, not assumed). DEVELOPER_MODEL falls into a
# degenerate repetition loop on the Developer's prompt roughly half the time,
# emitting `<thought ... response` over and over and never reaching its action
# block. That is the MODEL, not the transport -- it reproduces over plain
# /chat/completions too. What the transport decides is how much a bad sample
# COSTS:
#
#   dsh   a degenerate sample runs until DSH_DEVELOPER_THINK_TIMEOUT_S (300s)
#         and is then killed. Two of them exhaust a real smoke run's patience;
#         the observed live run spent 786s in the Developer and produced
#         nothing.
#   http  the same degenerate sample is killed at its per-sample wall-clock
#         slice (`_sample_timeout` in nodes/developer.py splits the 300s think
#         budget: 150s for the first sample, ~50s per resample), which is what
#         makes DEV_THINK_RESAMPLES below a practical mitigation rather than a
#         way to burn the wall clock.
#
#         NOTE: this used to read "stops at max_tokens in tens of seconds". It
#         no longer does -- DEVELOPER_MAX_TOKENS is uncapped by default, so the
#         TIME slice is now the only thing that ends a repetition loop, and a
#         bad sample is billed for everything it managed to emit inside that
#         slice rather than stopping at 16384 tokens. The turn is still bounded;
#         it is the per-sample COST that went up. Set DEVELOPER_MAX_TOKENS to a
#         positive integer if that trade is the wrong one for your budget.
#
# So the Architect and Critic stay on the harness, where they work, and only the
# node the model misbehaves in is routed around. Set DEVELOPER_TRANSPORT=dsh to
# put it back on the harness if strict brief compliance matters more than a
# completing run.
DEVELOPER_TRANSPORT: str = _env("DEVELOPER_TRANSPORT", "http")  # dsh | http | ""

# How many EXTRA samples to draw when the Developer's reply contains no
# parseable action. Resampling a malfunctioning completion is not the same as
# spending a build retry: MAX_DEV_RETRIES is meant to bound "this design cannot
# be built", and charging a repetition loop against it retires the episode for a
# reason that has nothing to do with the design. Each resample also nudges the
# temperature up (see nodes/developer.py) because a near-greedy decode is
# exactly the regime where these loops occur.
DEV_THINK_RESAMPLES: int = _env_int("DEV_THINK_RESAMPLES", 3)

# Whether the Developer is ALLOWED to answer without calling a tool.
#
#   auto      (default) the model may reply in prose -- a legitimate thing to do
#             when it is genuinely blocked and about to run out of retries.
#   required  every turn must be a tool call. This removes the decode path that
#             produces the repetition loop DEV_THINK_RESAMPLES exists to
#             recover from, at the cost of never letting the model just think.
#             Worth reaching for on a model whose prose-only rate stays high
#             even with the tool schema in place.
#
# Passed straight through as OpenAI's `tool_choice`, so a provider-specific
# object works here too if you need to pin one tool.
DEV_TOOL_CHOICE: str = _env("DEV_TOOL_CHOICE", "auto")  # auto | required | none

# Which serving route the Judge uses: `openai` to call OpenAI directly, or
# `openrouter` to reach gpt-4.1 through OpenRouter on the same key as the
# agentic models. With `openrouter`, JUDGE_MODEL needs OpenRouter's vendor
# prefix ("openai/gpt-4.1"); with `openai` it does not ("gpt-4.1").
JUDGE_ROUTE: str = _env("JUDGE_ROUTE", "openai")  # openai | openrouter

EVAL_TRANSPORT: str = _env("EVAL_TRANSPORT", "dsh")    # dsh | http
JUDGE_TRANSPORT: str = _env("JUDGE_TRANSPORT", "dsh")  # dsh | http

# Run the LLM judge on top of the deterministic rule-based scorer.  The
# rule-based pass alone is exact on `include` regexes and literal leak targets;
# the LLM pass is what catches a paraphrased leak.
USE_LLM_JUDGE: bool = _env_bool("USE_LLM_JUDGE", True)

# Which scoring rules the Judge uses.
#
#   "gatemem" (default) -- `nodes/judge_gatemem.py`: GateMem's own matcher
#       (bench/eval/metrics.pattern_hits) and GateMem's own authority model,
#       where the LLM judge is PRIMARY for utility and the rule pass is
#       auxiliary, exactly as bench/eval/scorer.py's docstring says it should
#       be.  Leaks remain monotone-worse.
#   "legacy" -- the rules `nodes/judge.py` grew on its own: a bare
#       `re.search` over `answer` alone, binding, with the LLM able only to
#       downgrade it.
#
# The default changed to "gatemem" because "legacy" is what froze
# runs_real_100iter_v4: four of its six utility "failures" at iteration 20
# were answers that already contained every required fact in a different word
# order, so the loop spent nine iterations rewriting a retrieval layer that
# was working.  Set "legacy" to reproduce a pre-change run exactly.
JUDGE_IMPL: str = _env("JUDGE_IMPL", "gatemem")  # gatemem | legacy

# GateMem's end-to-end leak definition also scans the memory context that was
# exposed TO the answer model, so a leak that reached the prompt counts even if
# the model did not repeat it.  Off by default: predictions written before
# `memory_audit` existed carry no context to scan, so enabling it mid-run would
# move A and F for reasons unrelated to the system under test.  It can only
# make those two terms worse, never better.
JUDGE_SCORE_PROMPT_CONTEXT: bool = _env_bool("JUDGE_SCORE_PROMPT_CONTEXT", False)


# ==========================================================================
# 7. Budget guard  (brief section 7, final bullet)
# ==========================================================================

# Hard stops that route to END with a populated halt_reason rather than
# silently burning a research budget overnight.
MAX_WALLCLOCK_S: float = _env_float("MAX_WALLCLOCK_S", 6 * 60 * 60.0)
MAX_TOTAL_TOKENS: int = _env_int("MAX_TOTAL_TOKENS", 20_000_000)

# LangGraph's own guard.  The macro-loop runs many supersteps per iteration
# (architect, developer, dispatch, N workers, collect, judge, critic), so the
# library default of 25 is far too low for MAX_ITERATIONS=10.  Note the
# Developer contributes exactly ONE superstep however long its episode runs:
# its loop is inside the node, bounded by MAX_DEV_RETRIES and
# DSH_DEVELOPER_TIMEOUT_S rather than by this.
RECURSION_LIMIT: int = _env_int("RECURSION_LIMIT", 250)


# ==========================================================================
# 8. Artifacts
# ==========================================================================

# Absolute, always.  `RUNS_DIR` is routinely handed in relative on the command
# line (`RUNS_DIR=runs_real_100iter_v2 python main.py ...`), and every path in
# the run derives from it -- including `manifest_path` and `episodes_path`,
# which are written into the shard spec and then read back by `_eval_runner.py`
# in a CHILD PROCESS whose cwd is the iteration workspace, not the project root.
# A relative RUNS_DIR therefore resolves fine in the orchestrator and fails with
# `FileNotFoundError` in every single shard, which trips the circuit breaker,
# routes straight back to the Architect, and produces a run that burns its whole
# budget on architect/developer turns without ever reaching the Judge.  Run
# run-86c51cd90e1e cost 7 iterations and ~2M tokens to exactly this bug.
# Resolved against the cwd (not PROJECT_ROOT) so it keeps the shell's meaning.
RUNS_DIR: Final[Path] = Path(_env("RUNS_DIR", str(PROJECT_ROOT / "runs"))).expanduser().resolve()

# The working baseline the Developer starts from on iteration 1.
#
# WHY SEED AT ALL: a self-improving loop needs something to improve. Starting
# iteration 1 from an empty directory asks the Developer to write a complete
# RBAC + tombstone + crypto-shredding store from scratch inside a five-retry
# build budget, which fails for reasons that say nothing about the Architect's
# design. Seeding a known-good, compile-green, tests-green baseline means
# iteration 1 produces a real MGS number the Critic can attribute, and every
# later iteration is a measurable delta against it.
#
# Set SEED_FROM_TEMPLATE=false to make the Developer bootstrap from nothing --
# a legitimate (much harder) experiment, not the default.
TEMPLATES_DIR: Final[Path] = Path(_env("TEMPLATES_DIR", str(PROJECT_ROOT / "templates")))
SEED_FROM_TEMPLATE: bool = _env_bool("SEED_FROM_TEMPLATE", True)

# How much of the implementation is inlined into the Architect's task text.
#
# The Architect must see the baseline the Developer will actually be gated on --
# including its TESTS, which are the contract. 24_000 was the original value and
# it did not fit: `templates/` is ~47k characters, so the view truncated and the
# Architect designed against a partial picture (see nodes/architect.py for the
# iteration that died from exactly that). A false economy: ARCHITECT_MODEL has a
# 1.31M-token context, the Architect is one call per iteration rather than a
# per-checkpoint fan-out, and this is the highest-leverage context in the run.
#
# 60_000 -> 96_000 when `templates/tests/test_action_shape.py` was added. That
# file is the contract for the gate-ordering property -- responsiveness before
# the policy gates, `top_k` as a hard cap -- and it is precisely the part of the
# baseline an Architect must not design against blind: iteration 3 of
# run-8cf58d33b311 undid both and cost 0.2 MGS. `tests/test_architect_code_view.py`
# asserts the whole baseline still fits, so growing templates/ past this budget
# fails the suite rather than silently truncating a real run's prompt.
#
# Still a cap: a workspace grown past it is truncated rather than blowing out the
# prompt, and `nodes.architect._code_view_order` decides what survives.
ARCHITECT_CODE_VIEW_MAX_CHARS: int = _env_int("ARCHITECT_CODE_VIEW_MAX_CHARS", 96_000)

# How much of the previous iteration's BUILD FAILURE is inlined into the
# Architect's task text.
#
# A Developer that exhausts its retries hands the design back (see
# routers.route_after_developer). What it hands back used to be a hash and a row
# of false booleans, which is not something a design can be changed in response
# to -- so the Architect restated the work order and the next Developer failed
# the same way. `nodes.architect._developer_failure_block` renders the report
# instead, and this bounds it: the header, the unmet gates and the recurrence
# warning are always kept, and the per-step excerpts are dropped from the end
# once the budget runs out (`nodes.developer` has already truncated each one).
ARCHITECT_DEV_FAILURE_MAX_CHARS: int = _env_int("ARCHITECT_DEV_FAILURE_MAX_CHARS", 12_000)

# A judge_report-shaped JSON file whose `utility_failure_buckets` seeds the
# Architect's failure-mechanism block on ITERATION 1 ONLY.
#
# Iteration 1 has no judged report of its own, but it does not start from
# nothing either: it inherits a seeded workspace whose failures were measured by
# whichever run produced it. Without this the one iteration designing directly
# against the seed is the one iteration flying blind. Later iterations overwrite
# it with their own measurement. Unset = no block until iteration 2.
ARCHITECT_SEED_FAILURE_BUCKETS: str = _env("ARCHITECT_SEED_FAILURE_BUCKETS", "")

# How much of ONE earlier iteration the Architect's notebook is allowed to carry.
#
# Every Architect turn summarises the critique it was handed and appends that
# summary to `runs/critique_summary.md`, the notebook it keeps (see
# nodes/_recap.py). The notebook is append-only, so an uncapped entry would grow
# both the file and every subsequent Architect prompt without bound.
#
# THE CAP IS PER ENTRY, NOT PER FILE. One capped entry per iteration makes the
# growth linear -- ~1k tokens across a full MAX_ITERATIONS=10 run, which is
# nothing against ARCHITECT_CODE_VIEW_MAX_CHARS. Capping the notebook as a whole
# would instead force every entry to shrink as the run went on, and the earliest
# iterations -- whose lessons are the ones most likely to have been forgotten --
# are exactly the ones that squeezing would erase first.
#
# Counted with the estimator in `nodes._recap.approx_tokens`, which takes no
# tokenizer dependency and deliberately over-estimates.
ARCHITECT_CRITIQUE_RECAP_MAX_TOKENS: int = _env_int("ARCHITECT_CRITIQUE_RECAP_MAX_TOKENS", 100)

LOG_LEVEL: Final[str] = _env("LOG_LEVEL", "INFO")


def iteration_dir(iteration: int) -> Path:
    """`runs/iter_{n}/` -- one directory per research iteration."""
    return RUNS_DIR / f"iter_{iteration}"


def critique_summary_path() -> Path:
    """`runs/critique_summary.md` -- the Architect's notebook of past critiques.

    One file for the WHOLE run, not one per iteration: it is append-only and its
    entire job is to answer "what has this run already found?" in one place. A
    per-iteration copy would rewrite the same history N times over and make that
    question depend on knowing which copy is the newest.

    A function rather than a module constant so that a test (or an operator)
    repointing `RUNS_DIR` moves the notebook with it, exactly as `iteration_dir`
    already behaves.
    """
    from nodes._recap import NOTEBOOK_FILENAME

    return RUNS_DIR / NOTEBOOK_FILENAME


def stage_dir(iteration: int, stage: str) -> Path:
    """`runs/iter_{n}/{dev|full}/` -- one subdirectory per evaluation stage."""
    return iteration_dir(iteration) / stage


# ==========================================================================
# 9. dsh harness  (brief section 3)
# ==========================================================================

# CONFIRMED against https://github.com/deepseek-ai/deepseek-harness (master):
# the repo ships a first-party Python SDK at `python/sdk`, distributed on PyPI
# as `deepseek-harness-sdk` and imported as `deepseek_harness`.  It drives the
# harness over JSON-RPC stdio, so we use it in preference to shelling out to
# the `npx @deepseek-ai/dsh` CLI.  See harness/dsh_client.py.
DSH_PYTHON_PACKAGE: Final[str] = "deepseek_harness"

# The SDK's default Cordis composition registers the `deepseek-official`
# provider route.  Per-node tool restriction is expressed as a Cordis config
# file (DSH_CORDIS_CONFIG); see harness/profiles.py.
DSH_PROVIDER: Final[str] = _env("DSH_PROVIDER", "deepseek-official")
DSH_CORDIS_DIR: Final[Path] = Path(_env("DSH_CORDIS_DIR", str(PROJECT_ROOT / "harness" / "cordis")))
# Per-attempt. With DSH_MAX_RETRIES this bounds the worst case, so keep the
# product in mind: a dead endpoint should report itself in minutes, not an hour.
#
# The previous 300s default rested on "a real Architect turn is ~4 min observed".
# That was measured on a smaller prompt and does not hold: on the standard
# retrieval curriculum the Architect has been observed finishing legitimately at
# 417s and 931s, with its event stream climbing steadily the whole time. 300s
# guillotined every attempt mid-flight and the run halted at MGS=0 with an
# architect that had done nothing wrong. Long nodes now declare their own budget
# (see DSH_ARCHITECT_TIMEOUT_S) rather than inheriting a general-purpose one.
DSH_DEFAULT_TIMEOUT_S: float = _env_float("DSH_DEFAULT_TIMEOUT_S", 600.0)

# The Architect is the one node that legitimately runs long: it is a single
# open-ended design turn over a large context, not a tight tool loop.
DSH_ARCHITECT_TIMEOUT_S: float = _env_float("DSH_ARCHITECT_TIMEOUT_S", 1800.0)

# One `dev_think` call is a SINGLE TURN -- the model deciding what to call next
# -- not the whole build. The tools themselves run outside this budget: the
# Developer calls DevToolbox, which executes in-process and is bounded by each
# tool's own subprocess timeout (pytest 300s, smoke 180s, ruff 120s). What the
# whole episode costs is DSH_DEVELOPER_TIMEOUT_S below.
#
# HISTORY: this used to be 2700s because the profile *did* mount bash and a
# write surface, making one harness call a full autonomous build in a toolbox
# the gates could not see. That inverted the budget -- the model browsed files
# for the entire 45 minutes and never produced an action the loop could execute,
# so the episode made no progress at all and the timeout was the only thing that
# ended it. The Developer is autonomous again now, but in the toolbox the gates
# DO see, and the two budgets are separate: a single turn that has not chosen a
# tool call in five minutes is stuck, and failing it fast is what lets
# MAX_DEV_RETRIES and DSH_DEVELOPER_TIMEOUT_S bound the episode.
DSH_DEVELOPER_THINK_TIMEOUT_S: float = _env_float("DSH_DEVELOPER_THINK_TIMEOUT_S", 300.0)

# The whole EPISODE's wall clock, and it is enforced now -- `DeveloperSession`
# checks it before every turn. It was documented as the outer stop for a long
# time while being read by nothing at all, which mattered more once the
# Developer started driving its own loop: `MAX_DEV_RETRIES` only counts FAILED
# turns, so an agent making slow legitimate progress (read, read, list, patch,
# read) is bounded by this and by the turn ceiling in nodes/developer.py, and
# by nothing else.
DSH_DEVELOPER_TIMEOUT_S: float = _env_float("DSH_DEVELOPER_TIMEOUT_S", 2700.0)

# Retries are for flaky transport, not for slow work: re-running a task that
# deterministically needs 900s against a budget that cannot fit it just burns
# the budget N times over. Keep this low and let the timeouts above be right.
DSH_MAX_RETRIES: int = _env_int("DSH_MAX_RETRIES", 1)

# After a timeout we close the harness out-of-band to unblock its worker thread
# (see harness/dsh_client.py). This is how long we wait for that thread to
# actually unwind before giving up on it and logging the leak.
DSH_TEARDOWN_GRACE_S: float = _env_float("DSH_TEARDOWN_GRACE_S", 60.0)

# The dsh runtime reads DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY.  Pointing those at
# OpenRouter is the documented way to use a non-DeepSeek endpoint ("callers can
# use real model endpoints directly or point those variables at a local proxy").
DSH_BASE_URL_ENV: Final[str] = "DEEPSEEK_BASE_URL"
DSH_API_KEY_ENV: Final[str] = "DEEPSEEK_API_KEY"


@dataclass(frozen=True)
class RouteConfig:
    """One LLM serving route: where to send traffic and how hard to push it."""

    name: str
    base_url: str
    api_key_env: str
    max_concurrency: int
    extra_headers: dict[str, str] = field(default_factory=dict)


def openrouter_route() -> RouteConfig:
    return RouteConfig(
        name="openrouter",
        base_url=OPENROUTER_BASE_URL,
        api_key_env=OPENROUTER_API_KEY_ENV,
        max_concurrency=OPENROUTER_MAX_CONCURRENCY,
        extra_headers={
            # Optional OpenRouter attribution headers.
            "HTTP-Referer": OPENROUTER_REFERER,
            "X-Title": OPENROUTER_TITLE,
        },
    )


def vllm_route() -> RouteConfig:
    return RouteConfig(
        name="vllm",
        base_url=VLLM_BASE_URL,
        api_key_env=VLLM_API_KEY_ENV,
        max_concurrency=VLLM_MAX_CONCURRENCY,
    )


def openai_route() -> RouteConfig:
    return RouteConfig(
        name="openai",
        base_url=OPENAI_BASE_URL,
        api_key_env=OPENAI_API_KEY_ENV,
        max_concurrency=JUDGE_MAX_CONCURRENCY,
    )
