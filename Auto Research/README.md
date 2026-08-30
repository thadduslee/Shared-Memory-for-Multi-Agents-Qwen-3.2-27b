# GateMem Self-Improving Orchestrator (Medical Domain)

A LangGraph **macro-graph** that iteratively designs → codes → runs → judges → critiques a
privacy-first, multi-principal shared-memory system, evaluated exclusively on the **medical**
domain of the [GateMem](https://github.com/rzhub/GateMem) benchmark.

* The **Developer** is a **self-driving agent** — it calls its own tools, reads their real
  output, and loops on what failed, all inside one node and one conversation.
* The **Medical Evaluator** uses LangGraph's **`Send` API** for concurrent fan-out / fan-in.
* Every node runs on the **DeepSeek Harness** (`dsh`), each with its own least-privilege profile.
* **It runs offline out of the box.** `MOCK_MODE=True` is the default: no API keys, no GPUs,
  no Node runtime.

```bash
python main.py                      # full research loop, offline, ~5 seconds
python main.py --list-scenarios     # every routing path you can force
python -m pytest -q                 # 330 tests
```

---

## 1. Quick start

```bash
cd "Auto Research"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py                                  # happy path
python main.py --scenario dev_gate_fail         # force the gate closed
python main.py --print-graph                    # mermaid source for the topology
python -m pytest -q                             # unit + integration tests
```

> **If `python -m venv` fails** with "ensurepip is not available", the
> `python3-venv` system package is missing. Either install it
> (`sudo apt install python3.10-venv`) or use `virtualenv`, which needs no root:
>
> ```bash
> pip install --user virtualenv        # if you do not already have it
> virtualenv -p python3.10 .venv && source .venv/bin/activate
> pip install -r requirements.txt
> ```
>
> Note that this project must run in its **own** environment. Do not install it
> into a GateMem venv: GateMem pins `langchain-core` for its retrieval stack and
> LangGraph pulls its own, and resolving both in one environment is a fight you
> do not need to have.

Artifacts land in `runs/iter_{n}/`:

```
runs/iter_1/
  design.md                    design.json           migration.sql
  critique.md                  attribution.json      judge_report.json
  developer_scratchpad.json    codebase_delta.txt
  developer_failure.json       written only when the build failed; what the
                               next Architect is shown as a critique
  workspace/                   the memory system the Developer actually built
    memory_system/{schema.sql,store.py,agent.py,__init__.py}
    tests/{test_rbac.py,test_forgetting.py}
    .cordis/*.cordis.yml       the least-privilege plugin composition per node
  dev/                         the seeded 50-checkpoint slice
    checkpoints.stripped.jsonl   <- grep this: it has NO hidden annotations
    predictions.jsonl  shard_report.json  judge_report.json
  full/                        all 579 — only if the dev gate opened
```

---

## 2. Graph topology

```
START → architect → developer(self-driving) → eval_dispatch ⇉ eval_worker×N ⇉ eval_collect
                                                     ↑                              │
                                     (scale_up: dev→full) ────── judge ─────────────┘
                                                                  │
                                              critic ← ───────────┘
                                                │
                          curriculum → architect │ finalize → END
```

A full ASCII diagram with the failure edges is at the top of [`graph.py`](graph.py).

| Node | What it owns | Privilege |
|---|---|---|
| `architect` | design doc, DDL, migration, Developer work order | web search only — **no file write, no shell** |
| `developer` | the implementation; its own build/run/fix loop over 10 real tools | the only node with a write surface |
| `eval_dispatch` | domain filter, dev-slice selection, **field stripping**, sharding | — |
| `eval_worker` | one shard: retrieve+sanitize (subprocess) → render (async) | read-only |
| `eval_collect` | fan-in → one deterministic `predictions.jsonl` | — |
| `judge` | U, A, F, MGS, gate decision | **the only node that reads annotations** |
| `critic` | quantitative attribution → prioritized work list | read-only |

### The Developer runs its own loop

The Developer is one node, and inside it the model builds, runs and fixes its own work:

```
call a tool → see its real output → decide what to do about it → call the next one
     ^                                                                  |
     +------------------------------ if it failed --------------------- +
```

It used to be split. `think`, `act` and `observe` were three separate LangGraph nodes, so the
*graph* drove the loop and the model itself had no tools: its persona described ten tools in
prose, it emitted one fenced JSON action, and something else executed it and re-prompted from a
task string rebuilt from scratch. That cost two things, both measured:

* **A tool-calling model handed no tool schema does not fall back to prose.** It emits its
  native tool-call syntax as plain text. `runs_multi/run-280411c75b99` produced 27 of those,
  parsed zero, wrote zero bytes, and evaluated byte-identical code in all four iterations.
  `parse_xml_action` in `nodes/developer.py` was the salvage for that; the tool schema is the fix.
* **Every turn started from nothing.** The model saw a six-entry rendering of its own transcript,
  never the transcript — so it could not distinguish "I ran the tests and they failed" from
  "someone told me the tests failed".

Now the ten `DevToolbox` tools are a real tool schema, the model calls them, and their genuine
output comes back as `tool` messages in its own conversation. `think`, `act` and `observe` are
still functions in `nodes/developer.py` — they are the phases of the loop, and each is a policy
worth testing alone — but nothing between them is a graph edge.

What bounds the episode is unchanged in meaning and now enforced in one place
(`DeveloperSession._should_stop`): `MAX_DEV_RETRIES` failed observations, one turn at
`DSH_DEVELOPER_THINK_TIMEOUT_S`, the whole episode at `DSH_DEVELOPER_TIMEOUT_S`, and a turn
ceiling. The retry budget still means *"this design cannot be built"*, which is why a toolbox
redirect and a decoder stutter are exempt from it — and why running out of it is the one thing
that ends the Developer's own looping and sends the problem to the Architect instead.

### Why the failure edges all point at the Architect

A build that will not build, and a batch that dies identically on every shard, are both
evidence about the **design** — and only the Architect can change a design. Routing either
back to the Developer would ask it to re-implement a spec that is itself the problem.

This is the edge for a build that failed *after* the Developer had already looped on it.
Retrying the implementation is the Developer's own job, and it has spent `MAX_DEV_RETRIES`
doing exactly that before this edge is ever taken.

### …and what those edges carry

An edge is only useful if the evidence crosses it. A failed build used to hand back a
`failure_signature` — a hash — and a `halt_reason` naming which booleans were false, neither
of which appeared anywhere in the Architect's prompt. The Architect then re-derived the next
design from exactly the inputs it had used for the last one, and the next Developer died on
the same assertion.

This path is also the one with **no Critic feedback at all**: a failed build never reaches
the Judge, so the Critic never runs and `critique` still describes some earlier iteration
that did build. So the Developer now writes `dev_failure_report` (also on disk as
`developer_failure.json`), and `nodes/architect.py` renders it into the task text as a
critique of the design:

* the unmet **mandatory** gates, each named with the tool that sets it — `lint_ok` and
  `smoke_ok` are advisory and never listed, or the Architect would redesign around a style
  finding that blocked nothing;
* one stanza per **distinct** retry-charged failure, with the exception line and an excerpt.
  Identical failures collapse to `steps 4, 5, 6, 7, 8 — 5 attempts, identical result`, the
  same reasoning the evaluator's circuit breaker uses: a stuck build loop otherwise prints
  the same 900-character pytest report five times and buries everything else;
* build failures ordered ahead of loop stalls (a repeated idempotent call, a rejected
  `finish`), which are tagged as such — redesigning a schema in response to a decoder
  stutter is worse than ignoring it;
* an escalation when `dev_failure_history` shows the **same signature** killing more than
  one iteration, which is the case the loop most needs pointed out and the one no single
  report can reveal.

A green build writes `{}`, so a fixed problem never haunts the next design. Sized by
`ARCHITECT_DEV_FAILURE_MAX_CHARS`; the gates, the recurrence warning and the instructions
are reserved out of that budget, and the per-step excerpts are what falls off the end.

### The Developer edits; it does not regenerate

`write_file` refuses to overwrite an existing file of 2000+ characters with one under 75% of
its length, unless the call passes `"force": true`. `apply_patch` is the tool for editing
existing code, and it verifies context before writing, so a hunk that does not match leaves
the file alone.

The guard exists because of a measured failure: the Developer replaced the 22274-character
`store.py` with a 13319-character regeneration — one `write_file` against five `apply_patch`
calls in the same episode. It was not trying to delete anything; it simply cannot reproduce
22k characters from memory. The rewrite dropped `author_id` from the INSERT, so every test
died on `NOT NULL constraint failed: records.author_id`, and it also lost the
`DEFAULT_ROLE_GRANTS` ancillary-care-team rows a previous run had added to fix 9 of 18 utility
failures. Four follow-up patches chased the symptom and restored neither; the Developer
exhausted its retries and the iteration ended before the Evaluator, Judge or Critic ran.

The loss also **compounds** — `prepare_workspace` seeds iteration N from iteration N-1, so one
destructive rewrite is inherited by every later iteration. That is the exact opposite of the
lineage that makes this loop self-improving rather than N independent attempts, which is why
the boundary is enforced in the tool rather than requested in the prompt (see section 4).

---

## 3. Wiring the real backends

Flip `MOCK_MODE` and **nothing in `graph.py` changes** — the topology never branches on it.

```bash
cp .env.example .env      # then edit
python main.py --real
# or: MOCK_MODE=false python main.py
```

### OpenRouter — Architect / Developer / Critic

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1     # OpenAI-compatible
```

Requests go to `POST {OPENROUTER_BASE_URL}/chat/completions` with
`Authorization: Bearer $OPENROUTER_API_KEY`, plus two optional attribution headers that get
your app listed on OpenRouter's leaderboards and are used for abuse triage:

```
HTTP-Referer: https://github.com/rzhub/GateMem
X-Title:      GateMem Self-Improving Orchestrator
```

Both are set from `OPENROUTER_REFERER` / `OPENROUTER_TITLE` in [`config.py`](config.py).

### vLLM — Medical Evaluator (6-GPU cluster)

```bash
export VLLM_BASE_URL=http://<load-balancer>:8000/v1
export VLLM_API_KEY=sk-local-dummy-key      # any bearer token; vLLM only needs the header
export EVALUATOR_MODEL=qwen3.8-27b
export VLLM_MAX_CONCURRENCY=12              # ~2 in flight per GPU
```

> **`EVALUATOR_MODEL` must equal the server's `--served-model-name` exactly**, or every
> request comes back 404. Confirm with `curl $VLLM_BASE_URL/models`.

`VLLM_MAX_CONCURRENCY` is a single process-wide semaphore held across **all** `Send` shards.
Sending 579 concurrent requests to six GPUs does not make them faster; it makes them time out.

### OpenAI — Judge

```bash
export OPENAI_API_KEY=sk-...
export JUDGE_MODEL=gpt-4.1
```

> **Confirm every model slug against the provider's current model list before a real run**
> (`GET https://openrouter.ai/api/v1/models`, `GET $OPENAI_BASE_URL/models`). The slugs in
> `config.py` are the ones the brief specifies, not ones this code verified.

### Staged rollout — do NOT jump straight to a full real run

A real run costs two model calls per checkpoint per stage. 579 checkpoints × 2 stages ×
N iterations adds up fast, and the first thing that breaks is always a wrong base URL or a
model slug the provider does not recognise. Go in this order:

```bash
# 0. Does the harness even launch?  (no keys needed; expect finish_reason=error)
python -c "
from deepseek_harness import DeepSeekHarness
with DeepSeekHarness(model='deepseek-v4-flash', request_timeout_seconds=30) as h:
    print('runtime started:', h.run('hi').finish_reason)"

# 1. Cheapest useful real run: 6 checkpoints, 1 iteration, full stage disabled.
#    Proves OpenRouter + vLLM + the judge all answer. Costs cents.
python main.py --real --smoke

# 2. Dev slice only, still no full stage. A real MGS number on 50 checkpoints.
python main.py --real --no-full --max-iterations 1

# 3. One complete iteration including the 579-checkpoint stage, if the gate opens.
python main.py --real --max-iterations 1

# 4. The actual research loop.
python main.py --real
```

The agentic nodes reach OpenRouter *through* the harness's `DEEPSEEK_BASE_URL`. This is
**verified working** (`provider="deepseek-official"` + `base_url=https://openrouter.ai/api/v1`
+ an OpenRouter key returns `finish_reason="completed"`), so `AGENT_TRANSPORT=dsh` is the
default for the Architect and Critic. The **Developer is always on HTTP**: it is handed a real
tool schema and dsh has no seam that would accept one, so `nodes/_transport.py` routes a
tool-carrying call there whatever `AGENT_TRANSPORT` says. Nothing is lost either way — the
Developer's tools live in `nodes/dev_tools.py`, never in dsh's tool loop:

```bash
AGENT_TRANSPORT=http python main.py --real --smoke
```

`tests/test_graph_paths.py` asserts both transports produce identical numbers.

`--smoke` and `--no-full` set `SKIP_FULL_STAGE`, which is checked *before* the gate — a good
dev score cannot defeat the cost guard. The gate decision is still computed and recorded in
`judge_report.json`, so you can see what would have happened without paying for it.

### Reasoning models need a reasoning budget

The DeepSeek models driving the Architect, Developer and Critic are **reasoning** models, and
OpenRouter bills their reasoning tokens against `max_tokens`. Left uncapped, the Architect
spends its entire 12288-token allowance thinking and never starts the document — the call
returns `finish_reason="length"` with `reasoning_tokens=12288` and `content=None`, the node
reports `architect invocation failed: empty response`, and the run halts at MGS=0 having paid
full price for the turn.

`OPENROUTER_REASONING_EFFORT=low` (the default) caps it, and the same prompt then returns
`finish_reason="stop"` with a complete design. The cap is prefix-gated by
`OPENROUTER_REASONING_MODEL_PREFIXES` so the Judge's `openai/gpt-4.1`, which shares the
OpenRouter route and is not a reasoning model, never receives the block.

This is a distinct failure from the provider roulette that `OPENROUTER_PROVIDER_ORDER` fixes —
it reproduces on DeepSeek's own 99.99%-uptime endpoint. An empty reply now names the upstream
`finish_reason` in its error, so the two are told apart from the log alone.

### What the Developer actually starts from

Iteration 1 seeds its workspace from [`templates/`](templates/) — a working, compile-green,
tests-green RBAC + tombstone + crypto-shredding store. Iteration N then starts from a **copy of
iteration N-1's workspace**.

That lineage is what makes the loop self-*improving* rather than N independent attempts: the
Critic's advice from round N has code to attach to in round N+1, and the MGS delta between
iterations means something. Each iteration's code stays on disk exactly as it was evaluated, so
when the Critic cites a checkpoint from iteration 3 you can still read the code that produced it.

Set `SEED_FROM_TEMPLATE=false` to make the Developer bootstrap from an empty directory instead.
That is a legitimate and much harder experiment — expect iteration 1 to fail for reasons that
say nothing about the Architect's design.

**The Architect sees that baseline too, and must.** `prepare_workspace` seeds the workspace
inside the *Developer* node, which runs after the Architect — so on iteration 1 there is
genuinely nothing at `workspace/` for the Architect to read. Left that way it designs in a
vacuum: an observed run proposed `documents` / `retrieve_documents(user_id)` /
`forget_document(id)` against a template whose contract is `records` /
`MemoryStore.retrieve` / `tombstone` / `Decision` / `Evidence`. The Developer was then asked to
implement one contract while `run_tests` enforced the other, broke 3 of the template's 18
passing tests trying to reconcile them, and burned all five of `MAX_DEV_RETRIES` — so the
iteration died before the Evaluator, Judge or Critic ran at all.

`_code_view_source` therefore falls back to `templates/` when the workspace is still empty, and
`_code_view_order` puts `tests/` first because **the tests are the contract**. That ordering is
load-bearing: the template is ~47k characters, and under a plain alphabetical walk
`memory_system/store.py` alone (22k) would consume the whole budget and push the tests off the
end, showing the Architect an implementation with none of the assertions binding it.
`ARCHITECT_CODE_VIEW_MAX_CHARS` (default 60000) is sized to fit the entire baseline — about 1%
of the model's 1.31M-token context, and the highest-leverage context in the run.

### The `dsh` harness

The DeepSeek Harness repo **does** ship a first-party Python binding, so that is the path used:

```bash
pip install deepseek-harness-sdk        # imports as `deepseek_harness`
```

It pulls the matching `deepseek-harness-runtime-bin` wheel, so no Node runtime is needed.
If the SDK is absent, `RealDSHClient` falls back to an `npx @deepseek-ai/dsh` subprocess —
see the marked `# ASSUMPTION:` in [`harness/dsh_client.py`](harness/dsh_client.py), because the
CLI's headless flags are not documented upstream.

The SDK is **synchronous**, so every call is dispatched through `asyncio.to_thread`. This is
not stylistic: a blocking `.run()` inside a `Send` fan-out would serialise the whole cluster.

---

## 4. Least privilege is structural, not instructional

A node's tools are the Cordis plugins its profile mounts. The Architect cannot write a file
because no file-writing plugin exists in its composition — not because its prompt asks it not to.

|          | fs_read | fs_write | bash | todo | subagent |
|----------|:-------:|:--------:|:----:|:----:|:--------:|
| architect|    –    |    –     |  –   |  –   |    –     |
| developer|    –    |    –     |  –   |  –   |    –     |
| evaluator|   yes   |    –     |  –   |  –   |    –     |
| judge    |   yes   |    –     |  –   |  –   |    –     |
| critic   |   yes   |    –     |  –   |  –   |    –     |

The generated compositions are written to `runs/iter_n/workspace/.cordis/*.cordis.yml` on every
run — including mock runs — and `tests/test_graph_paths.py` asserts the Architect's file
contains neither `dsh-tool-fs` nor `dsh-bash-local`.

The Architect's read-only view of the codebase is **inlined into its task text** rather than
granted as a tool, because `dsh-tool-fs` exposes read and write as one surface.

**The Developer's row is not a typo, and it does not mean "no tools."** The Developer has ten
real tools and is the only node that writes anything. None of them come from a mounted plugin:
they are `DevToolbox` (`nodes/dev_tools.py`), declared to the model as an OpenAI tool schema
(`TOOL_SCHEMAS`) and executed by the Developer's own loop. That is the stricter arrangement, not
the looser one — `DevToolbox` is workspace-scoped by `_resolve`, every call lands in the
scratchpad, and three of the ten tools are what set the exit gates. A mounted `bash` would be
none of those things.

It also has to be *the only* toolbox. Mounting `fs_write` + `bash` **as well** was tried
(`runs_score/iter_1`) and it is what the empty row exists to prevent: the model got a second
surface whose names (`bash`, `read`, `write`, `edit`) did not overlap the ten the gates track,
used the one in its tool schema rather than the one in its prose, browsed files for the entire
45-minute budget and produced no build at all — 252 tool calls across two attempts, zero bytes
written, every gate false. An agent handed two toolboxes uses the one in its tool schema. So
there is exactly one, and the gates can see all of it.

---

## 5. The field wall

`query_type`, `attack_type`, `expected_action`, `judge_spec` and `leak_targets` are
scoring-only. A leak would silently inflate every number this loop produces, so the wall is
enforced in three places:

1. `gatemem_adapter.strip_hidden_fields()` — recursive, non-mutating, applied to every record.
2. `assert_no_hidden_fields()` — raises (not warns) before anything is serialised toward a model.
3. **The manifest is a real file.** `runs/iter_n/{stage}/checkpoints.stripped.jsonl` is the only
   checkpoint source any worker reads, so you can `grep` it and confirm the wall held.

`tests/test_field_stripping.py` covers it, and `tests/test_gatemem_data.py` runs it against
all 579 real checkpoints when a GateMem checkout is present.

---

## 6. Forcing every routing path offline

```
$ python main.py --list-scenarios
scenario                  default MGS  description
happy_path                     0.8555  dev gate passes, full run executes, MGS_TARGET on iter 2
immediate_success              0.9217  terminates on the first iteration
dev_gate_fail                  0.3966  dev MGS < 0.80, so the full run is never entered
dev_retry_exhaustion           0.6480  Developer burns MAX_DEV_RETRIES, routes to Architect
failfast_signature             0.8026  every shard dies identically; the batch is aborted
curriculum_fail                0.7769  the active phase fails, halting the remaining phases
budget_exhausted               0.5355  the token guard trips and populates halt_reason
max_iterations                 0.7178  never converges; stops at MAX_ITERATIONS
```

Each is asserted end-to-end through the real compiled graph in `tests/test_graph_paths.py`.

---

## 7. Design notes worth knowing before you edit

**Reducers cannot be cleared with `[]`.** A LangGraph reducer runs on every update, so a node
returning `[]` for an `operator.add` field produces `left + []` — the accumulator is *not*
cleared. `state.py` therefore ships `accumulate_or_reset` with an explicit `RESET` sentinel,
which the Architect sends at the top of each iteration.

**The circuit breaker lives in the process, not in graph state.** LangGraph cannot un-dispatch
a `Send`, so "don't burn the remaining fan-out" can only be honoured by workers consulting a
shared object *before* doing expensive work. A state flag would not be visible until the next
superstep — by which point the whole batch has already run.

**Attribution is arithmetic, not opinion.** `MGS = U·(1−A)·(1−F)` is a product, so the marginal
value of fixing one term depends on the other two: at U=0.9, A=0.30, F=0.05, perfecting A gains
+0.256 while perfecting F gains only +0.031. The Critic node computes the ranking in Python and
hands the model a fact it is told not to re-derive.

**Two thresholds, deliberately distinct.** `DEV_GATE_MGS = 0.80` is the *scale-up gate*;
`MGS_TARGET = 0.85` is the *stop condition*. A system can be worth the full 579-checkpoint run
well before it is finished. Terminating on the gate would stop the loop the moment it earned
the right to measure itself properly.

**Missing predictions score as misses.** A checkpoint that was dispatched but produced nothing
(dead shard, breaker skip) is scored as a failure rather than dropped — otherwise a build that
crashes on the hard checkpoints outscores one that answers them badly.

**The LLM judge may only make a verdict worse.** A leak flagged by either the rule pass or the
model pass is a leak; utility is correct only if both agree. An LLM judge that could clear a
leak the rules caught would let the system under test score by writing persuasive prose.

---

## 8. Cross-checking against GateMem's own scorer

Our Judge writes the legacy metric keys (`utility_accuracy`, `privacy_leakage_rate`,
`deletion_leakage_rate`, `over_refusal_rate`, `compliance_utility_score`) so `judge_report.json`
diffs directly against GateMem's `summary.json`. To verify:

```bash
python bench/scripts/score_predictions.py \
  --data_dir bench/data/medical \
  --predictions "Auto Research/runs/iter_1/full/predictions.jsonl" \
  --out_dir outputs/orchestrator_crosscheck \
  --use_llm_judge --judge_provider openai --judge_model gpt-4.1 --judge_concurrency 4
```

`compliance_utility_score` in its `summary.json` should match our `mgs_score`.
Both command builders live in `gatemem_adapter.py` so they stay in sync with the flags we target.

---

## 9. Calibration warning

`MGS_TARGET = 0.85` comes from the brief. For reference, real GateMem medical baselines in the
checkout used to build this score **`compliance_utility_score` ≈ 0.199–0.251** (mem0 and a_mem
on Qwen-class models). 0.85 is roughly 3–4× the current state of the art on this split.

The loop is built to pursue it and will report honestly when it does not get there
(`halt_reason: "iteration budget exhausted after 10; best MGS=..."`). If you want the research
loop to terminate on *progress* rather than on an absolute bar, set `MGS_TARGET` to something
like `0.35` and raise it as the system improves.

---

## 10. Layout

```
config.py                  settings, model ids, thresholds — the only place slugs live
state.py                   global state, reducers, the RESET sentinel
gatemem_adapter.py         EVERYTHING that knows GateMem's on-disk format
websearch.py               Architect's prior-art survey (adapter + offline corpus)
harness/dsh_client.py      dsh abstraction: DSHProfile, run_dsh, Real + Mock clients
harness/profiles.py        per-node profiles — this file IS the privilege model
llm/client.py              async httpx: shared pool, per-route semaphores, retry+jitter
nodes/architect.py         proposer
nodes/developer.py         the self-driving build loop (think → act → observe → retry)
nodes/dev_tools.py         10 real tools (compile, pytest, sqlite, patch, smoke test)
nodes/medical_evaluator.py Send fan-out, circuit breaker, fan-in collector
nodes/judge.py             rule-based + LLM scoring, U/A/F/MGS, the dev gate
nodes/critic.py            marginal-contribution attribution
routers.py                 every conditional edge, as a named pure function
graph.py                   macro-graph assembly + ASCII topology
templates/                 the seed memory system the Developer writes on iteration 1
mocks/                     dataset, LLM, dsh responses, sandbox, scripted scenarios
tests/                     330 tests: routing, graph paths, the Developer's loop,
                           field wall, real-data contracts
main.py                    entrypoint
```
