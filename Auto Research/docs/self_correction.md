# Self-correction: how the loop stops walking away from its own best answer

This document exists because of one run.

`runs_real5_notebook`, thread `run-8cf58d33b311`, five iterations, 4.9 million
tokens. Its MGS by iteration:

| iter | U | A | F | MGS | Δ |
|---|---|---|---|---|---|
| 1 | 0.4444 | 0.1765 | 0.1333 | **0.3172** | — |
| 2 | — build failed, never evaluated — | | | | |
| 3 | 0.2222 | 0.0000 | 0.0000 | 0.2222 | −0.0950 |
| 4 | 0.2222 | 0.1176 | 0.0667 | 0.1830 | −0.0392 |
| 5 | 0.1667 | 0.1765 | 0.1333 | **0.1190** | −0.0641 |

Every design predicted a rise. Every measurement was a fall. `A` and `F` ended
exactly where they started, so the entire loss is `U`. And the run signed off
with:

```
"halt_reason": "iteration budget exhausted after 5; best MGS=0.1190"
```

That number is the run's *worst* result, reported as its best.

The loop was not slow to improve. It was **structurally unable to keep a good
answer**, and it could not see that it wasn't. Five separate mechanisms had to
be missing at once for that to be true, and this document is about all five.

---

## The five holes

### 1. The lineage followed the last code, not the best code

`prepare_workspace` seeded iteration N from iteration N−1, unconditionally:

```python
previous = config.iteration_dir(iteration - 1) / "workspace"
if previous and previous.is_dir():
    _copy_tree(previous, workspace)
```

So a change that lost MGS became the permanent foundation of everything after
it. Iteration 4 inherited iteration 3, which had just been measured as 0.095
worse than iteration 1; iteration 5 inherited iteration 4, worse again. The loop
had no way back.

The same rule also allowed a **failed build to become a parent**. Iteration 2
hit its 60-turn ceiling having written zero bytes, and iteration 3 was seeded
from its leftover workspace anyway.

**Fixed by:** `scoreboard.champion_iteration` + `prepare_workspace(parent=…)`.
Iteration N inherits the highest-scoring iteration judged so far. On an
improving run that *is* N−1, so the change is invisible; on a regressing run it
is the rollback. An iteration with no score row — a failed build — is not a
candidate at all.

### 2. Nothing remembered the previous score

`OrchestratorState` held exactly one number about quality: `mgs_score`, a scalar
the Judge overwrote every iteration. There was no history, no maximum, nothing.
Four things followed from that single omission, and they are holes 3, 4, 5 and
the halt-reason lie below.

**Fixed by:** `state.score_history`, an append-only list of one row per
`(iteration, stage)` the Judge scored, and `scoreboard.py`, which derives
everything else from it — best-so-far, the champion, the verdict, the trend
table, the term decomposition, the run summary. Everything derives; nothing is
stored twice, so nothing can disagree with anything else.

### 3. The Architect could not see the fall

Its prompt said:

```
## MEASURED PERFORMANCE SO FAR
U (utility_accuracy)        = 0.2222
A (privacy_leakage_rate)    = 0.1176
F (deletion_leakage_rate)   = 0.0667
MGS (compliance_utility_score) = 0.1830   target 0.85
```

Four numbers, true of one iteration, silent about the run. It designed
iterations 3, 4 and 5 without ever being told that its previous three designs
had each lost MGS. The notebook (`critique_summary.md`) carried prose summaries
and no numbers at all.

**Fixed by:** `scoreboard.trend_table`, which replaces those four lines with the
whole trajectory, a delta column, a per-row verdict, a row for each failed build
saying why it has no numbers, and — when there is a regression — an explicit
instruction about what to do with it.

### 4. The Critic had no word for "revert"

Its entire vocabulary was `marginal_contributions`, which ranks the three terms
by what perfecting each would be worth. Under `MGS = U·(1−A)·(1−F)` with small
`A` and `F`, that ranking names `U` on essentially every round **regardless of
what actually changed**. It named `U` four times out of four:

```
iter 1: dominant=U gain=+0.3965
iter 3: dominant=U gain=+0.7778
iter 4: dominant=U gain=+0.6405
iter 5: dominant=U gain=+0.5948
```

All four are arithmetically correct and none of them is a finding. The real
story — "iteration 3's edit halved U; undo it" — was not expressible.

**Fixed by:** `scoreboard.term_deltas`, which asks the complementary question:
which term *moved*, and what did that movement cost? It is placed **above** the
marginal ranking in the Critic's prompt, the ranking is explicitly demoted
("a fact about the algebra of MGS, not about this run"), and a numbered task
step makes `revert` a first-class answer with `keep_and_fix` and
`not_the_cause` as the alternatives. The verdict lands in `attribution.json`.

The deterministic fallback critique carries the regression too — it is computed
in Python, so an unreachable Critic model is no reason for the Architect not to
be told.

### 5. An outage was reported as a design failure

Iteration 2 spent sixteen minutes and 1.22 million tokens on OpenRouter
timeouts:

```
16:19:15 WARNING developer: http transport exceeded 150s; abandoning the call
16:19:15 WARNING developer: no action in reply (attempt 1/4, ok=False, 0 chars)
...
16:32:08 WARNING developer: turn ceiling (60) reached
```

It never called `run_tests`. It never called `sql_exec`. It wrote nothing. What
reached the Architect was:

```
unmet mandatory gates: tests_ok (set by `run_tests`),
                       migration_ok (set by `sql_exec`)
```

…under the heading *"Read this as a critique of the design."* The report's
`n_build_failures` was 0, `n_loop_failures` was 0 and `last_error` was empty —
it contained no evidence at all, and said nothing about the transport.

**A gate that never ran is not a gate that failed.** The three gate booleans
start `False` and are only ever set by their tool completing, so `tests_ok is
False` is ambiguous between two opposite facts, and the report rendered both the
same way.

The Architect duly redesigned. The work order it produced was *"remove the
`if _evaluate(row) == "allow" and len(decision.allowed) >= top_k: break`"* —
and that is the change that cost the run 0.095 MGS the moment iteration 3 built
it.

**Fixed by:** `nodes/developer.py::gate_status` (`passed` / `failed` /
`never_ran` per gate), a `transport_failures` counter kept separately from
`retry_count`, and `classify_failure`, which returns one of:

| classification | means | the graph |
|---|---|---|
| `design` | a mandatory gate ran and failed | → Architect, redesign |
| `infrastructure` | the transport ate the episode | → **re-run the same iteration** |
| `inconclusive` | no gate failed; none ever ran | → Architect, rewrite the *work order* |

`infrastructure` is the only one that retries, bounded by `MAX_INFRA_RETRIES`
per iteration. `inconclusive` deliberately does not: an episode that read files
for sixty turns and wrote nothing has a work-order problem, and re-running the
identical work order reproduces it.

Each classification carries its own framing into the Architect's prompt. The old
one sentence — "read this as a critique of the design" — was wrong for two of
the three cases and is now used only for the one it was right about.

### And the halt reason lied

```python
return f"iteration budget exhausted after {config.MAX_ITERATIONS}; best MGS={mgs:.4f}"
```

`mgs` there is `state["mgs_score"]` — the **last** score. On a run that
regressed, that is the largest possible error the field can contain.

**Fixed by:** `scoreboard.best_of`. The halt reason, the final log line, the run
summary, `inspect_run.py` and `check_learning.py` all report the best and the
final separately, and say plainly which workspace is worth keeping.

---

## The artifact-side bug the loop was chasing

All five holes above are about the *research loop*. There was also a real bug in
the thing being researched, and it is worth stating because it is what the loop
spent five iterations failing to find.

The Judge scores utility as:

```python
verdict["utility_correct"] = action_correct and include_ok
```

The **action label** is half of it. In iteration 5, seven of the ten failing
utility checkpoints contained *every* required string and scored zero purely
because the label was `answer_redacted` instead of `answer`.

The cause was gate ordering in `templates/memory_system/store.py`. The relevance
filter ran **last**, on the way to `allowed`:

```python
# Gate 1: tombstone.   Gate 2: role grant.   Gate 3: relationship + scope.
...
if terms and not (terms & _distinctive_terms(plaintext)):
    continue  # simply not relevant; not a policy denial
```

So a record the requester merely happened not to be cleared for — about a
different appointment, a different clinician, a different week — was denied
*first* and never tested for relevance. It landed in `denied_rbac`,
`Decision.touched_unauthorized` went true, and `sanitize_and_decide` read that
as "there is responsive content this requester may not have" and downgraded a
complete, correct, fully-authorised answer to `answer_redacted`.

**The fix is gate 0.** Responsiveness is now tested *before* any policy gate,
against a term index built at ingest (`record_terms`, keyed-hashed so it is not
a plaintext copy of confidential bodies, and purged on tombstone so
cryptographic shredding stays whole). A record that is not about the query is
skipped entirely and appears in no list — which establishes the invariant every
branch of `sanitize_and_decide` depends on: **a non-empty denied list always
means there IS responsive content the requester must not get.**

Measured with `bench_template.py`, which runs the evaluator's phase 1 with no
model in the loop and is therefore exact for the action label:

|  | dev slice (50) | | full set (579) | |
|---|---|---|---|---|
|  | before | after | before | after |
| U | 0.2778 | **0.7778** | 0.2333 | **0.5095** |
| A | 0.0588 | 0.0588 | 0.0990 | 0.0990 |
| F | 0.1333 | 0.1333 | 0.1864 | 0.1864 |
| MGS | 0.2266 | **0.6344** | 0.1710 | **0.3735** |

**Zero of 579 checkpoints regressed.** `A` and `F` are bit-identical, which is
the property that makes this safe: the change is to which records are
*considered*, not to which records are *released*.

`templates/tests/test_action_shape.py` is the contract for it. `run_tests` is a
mandatory gate, so an iteration that reintroduces the ordering — or that widens
the candidate scan until irrelevant records reach the gates again — fails its
build rather than discovering it 0.4 MGS later.

---

## Postscript: what the first fixed run found

`run-b3275eb7e373` was the first real 5-iteration run with all of the above in
place. It did not regress:

| iter | U | A | F | MGS | |
|---|---|---|---|---|---|
| 1 | 0.4444 | 0.0588 | 0.0000 | 0.4183 | first |
| 2 | 0.6111 | 0.0588 | 0.0000 | **0.5752** | new best |
| 3 | — build failed (`inconclusive`) — | | | | |
| 4 | 0.6111 | 0.0588 | 0.0000 | 0.5752 | tied — not adopted |
| 5 | — build failed (`inconclusive`) — | | | | |

Best 0.5752 against the old run's 0.3172, final 0.5752 against 0.1190, and
`regressed_from_best: false`. The machinery worked on live data: iteration 4 was
seeded from iteration 2 rather than from iteration 3's failed workspace, and
iteration 5 from iteration 2 again after iteration 4 tied.

**But it lost two of five iterations to the turn ceiling**, and both were two
tool calls from a green build:

* iteration 3 — 70 steps, **53 of them `read_file`**, all three source files
  written, and `compile_check` / `run_tests` / `sql_exec` never called once.
* iteration 5 — 69 steps, **60 of them `read_file`**, `run_tests` green at
  `pass_rate 1.0`, two files written, `compile_check` and `sql_exec` never run.

The `inconclusive` classification named it correctly and could not prevent it.
The cause was one omission: the status block the Developer reacts to each turn
reported `retry=0/5` and **said nothing at all about turns**. An episode could
arrive at turn 59 with the work finished and no idea it was about to be cut off.

An episode that ends without running its gates is scored as a failed build — it
is never evaluated and every byte it wrote is discarded — so the fix is to make
the budget visible and, once it is nearly gone, to say plainly that reading is
now the wrong move:

```
retry=0/5
turn=52/60 (8 left)
!! TURN BUDGET NEARLY SPENT. STOP READING AND STOP EDITING.
!! Call compile_check NOW, then `sql_exec`, then `finish`.
!! An episode that ends without running its gates is scored as a FAILED BUILD:
!! it is never evaluated, and every edit you have already made is discarded.
!! Whatever is half-finished, land what works. A green build of less is worth
!! more than a perfect design that never ran.
```

Deliberately **not** a bigger ceiling. Those episodes were 76% and 87%
`read_file` — over-reading, not under-working, and more budget is more room to
over-read. `DEVELOPER_MAX_TURNS` is configurable now (default 60) and
`DEVELOPER_LANDING_TURNS` (default 12) is how early the directive starts.

The Architect gets the other half, because work-order **size** is the lever it
controls. The correlation across that run is clean:

| work order | turns used | outcome |
|---|---|---|
| 5 steps | 17 | built |
| 5 steps | 19 | built |
| 10 steps | 60 | **ceiling** |
| 5 steps | 43 | built |
| 8 steps | 60 | **ceiling** |

So each Architect turn is now told what the last work order actually cost, and
warned at 75% of the ceiling rather than only after the failure:

```
Previous Developer episode used 60/60 turns for a 10-step work order.
!! IT RAN OUT OF TURNS. ... YOUR WORK ORDER WAS TOO BIG.
```

Tested in `tests/test_turn_budget.py`, including an end-to-end assertion that
the directive reaches the model's conversation with turns left to act on it.

---

## Postscript 2: two ways to spend 12.5M tokens on nothing

`run-c993a6e93050` — 100 iterations requested, 23 run, 3 hours, 12.5 million
tokens — failed in two independent ways, and neither was visible while it ran.

### The answerer was never answering

The local vLLM evaluator was unreachable for the entire run: **7,496
`ConnectError`s**. Every render call failed, and every one took
`_render_answer`'s fallback — the gated record bodies, joined, served as the
answer.

```
predictions total:        1000
answering actions:         357
LLM-RENDERED answers:        0      <-- zero
fell back to raw bodies:   357  (100.0%)
```

The fallback is *correct* for one checkpoint: better than scoring a transport
blip as a design failure. As a silent default for a whole run it is a different
experiment. The loop scored, ranked, rolled back, critiqued and reported
`best MGS=0.8366` — an honestly computed number about the retrieval and gating
layer with raw evidence pasted in where the answer should be. `runs_real5_fixed`
rendered 54/54 and `runs_real5_fixed2` 94/94, so the pipeline could always have
told you; it just never did.

**Fixed by** counting the fallbacks (`n_render_degraded` per shard, aggregated
per stage), logging them at ERROR with the endpoint name, carrying
`render_degraded` into the judge report and the score row, and — by default —
**halting before the Judge scores them** (`HALT_ON_DEGRADED_EVAL`). A degraded
row is still recorded and reported, but it is never eligible to be the champion:
with the renderer down the answer contains every string the gated records hold,
so a degraded row tends to score *higher* than a healthy one, and ranking them
together would hand the dead endpoint's workspace to the next iteration. That is
the same measurement error as ranking a 50-checkpoint score against a
579-checkpoint one, which `scoreboard` already refused to do.

### One rate limit ended the run

Iteration 23's Architect call returned a single HTTP 402:

```
"This request would exceed your available credits given your current in-flight
 requests. Retry after in-flight requests settle, or add credits."
 "reason": "in_flight_budget_exhausted"      "Retry-After": "120"
```

The provider said when to come back. Two things stopped anyone listening: 402
was not in `RETRYABLE_STATUS`, so the client returned an error without waiting;
and `route_after_architect` read *"the Architect only fails fatally"* as fatal to
the **run** rather than to the **iteration**. Waiting two minutes would have
saved it.

**Fixed by** distinguishing the two errors that share status 402 — "out of
money" (terminal) from "too many requests in flight" (transient, and always
accompanied by `Retry-After`) — honouring the stated wait in full
(`HTTP_RETRY_AFTER_MAX_S`, since the old 30s guess-ceiling clamped `120` down to
a retry that arrived too early), and adding an `architect_retry` node that sends
the same iteration back to design, bounded by `MAX_INFRA_RETRIES`. That mirrors
`infra_retry` on the Developer side; the two keep separate counters because the
Architect resets the Developer's at the top of its own turn.

### The knobs

| setting | default | what it does |
|---|---|---|
| `RENDER_DEGRADED_THRESHOLD` | `0.5` | fraction of answering checkpoints that may fall back before the stage is marked degraded |
| `HALT_ON_DEGRADED_EVAL` | `True` | stop the run when a stage comes back degraded, rather than scoring it |
| `HTTP_RETRY_AFTER_MAX_S` | `180.0` | ceiling for a wait the provider states explicitly, as opposed to one we invent |

Tested in `tests/test_degraded_evaluator.py` and `tests/test_transient_402.py`.

---

## Using it

**See what a finished run actually did:**

```bash
python check_learning.py runs_real5_notebook
```

The `KEEPING THE BEST ANSWER` section is the new one. On the archived run it
prints four failures; on any run made with these fixes it prints four OKs on the
same numbers.

**Watch the mechanism offline, end to end:**

```bash
python main.py --scenario regression --max-iterations 4
```

`MOCK_SCENARIO=regression` scripts that run's real scores. The loop must roll
back to iteration 1, tell the Architect it did, hand the Critic a revert
decision, and finish by naming iteration 1 as the best.

**Measure a workspace without spending a run:**

```bash
python bench_template.py templates
python bench_template.py templates --compare runs_real5_notebook/iter_5/workspace
```

Three seconds, no model, no tokens. The action-shape half of `U` is fully
determined by phase 1, and that is the half `run-8cf58d33b311` spent 4.9 million
tokens losing.

---

## Knobs

| setting | default | what it does |
|---|---|---|
| `ROLLBACK_TO_BEST` | `True` | seed iteration N from the champion. `False` selects linear N←N−1 lineage — see *When rollback is the wrong tool* below. Best-so-far tracking is unaffected by this flag either way |
| `LINEAGE_SKIP_FAILED_BUILDS` | `True` | under linear lineage, seed from the most recent iteration that actually **built** rather than from N−1 unconditionally. Not consulted under the champion rule, which never had the hole. `False` restores genuinely unconditional N−1 |
| `ROLLBACK_TOLERANCE` | `0.0` | how much better than the champion an iteration must score to *become* the champion. Raise it to demand a margin on a noisy slice; a tie is never adopted at any setting |
| `MAX_INFRA_RETRIES` | `2` | how many times one iteration may be re-run after an `infrastructure` failure. `0` disables the retry edge |

## Why a tie is not adopted

`best_row` breaks ties toward the **earlier** iteration, so code that scores
exactly what its parent scored is not adopted and the next iteration re-inherits
the champion.

That is deliberate. The failure this whole mechanism exists to stop is
unmonitored drift, and a tie is drift with a better cover story: the code
changed, the measurement did not, and adopting it buys nothing while carrying
whatever it broke that a 50-checkpoint slice cannot see. The Architect is *told*
when this happens — the trend table marks the row `TIED -- not adopted` — so it
is visible rather than silent, and `ROLLBACK_TOLERANCE` exists for anyone who
wants to require a real margin instead.

## When rollback is the wrong tool

The tie rule and the rollback compose into a failure mode of their own, and
`run-b5d7565ddb4a` is what it looks like from the inside.

Iteration 4 scored MGS 0.5392. Iterations 8, 10, 11, 13, 15, 17, 18 and 19 all
scored **exactly** 0.5392 — same U (0.6111), same A (0.1176), same nine failing
checkpoints every time. Five of them (4, 8, 10, 11, 17) produced *byte-identical*
`predictions.jsonl`, because the eval runner is deterministic and every one of
those patches was inert with respect to it. No tie is adopted, so the champion
never moved off iteration 4, so iterations 5 through 20 were each seeded from
iteration 4's workspace:

```
workspace iter=18 ROLLED BACK: seeded from iteration 4 (the current champion) instead of iteration 17
```

Iterations 17 and 18 are therefore **siblings, not successors** — each is
iteration 4 plus one patch, and 18 does not contain 17's work at all (it resets
`SCHEMA_VERSION` from 5 back to 4 and drops the synonym table 17 added). The run
made fifteen independent one-patch attempts on one parent and accumulated
nothing; `regression_streak` reached 14.

Rollback did exactly what it was designed to do here. It protected a high-water
mark — and on a plateau, protecting the high-water mark is the same motion as
refusing to explore. `ROLLBACK_TO_BEST=false` gives up that protection in
exchange for a lineage that compounds: 5←4, 6←5, 7←6. The cost is the one this
document opens with, and it is not hypothetical either — a regression becomes
the permanent foundation of everything after it.

Which way to set it is a property of the run, not of the code. A run that is
still climbing wants the rollback. A run that has plateaued — several ties in a
row, `regression_streak` climbing, the Critic returning `not_the_cause` each
round — is being held still by it.

## The hole linear lineage would open, and `LINEAGE_SKIP_FAILED_BUILDS`

The champion rule never had to answer "may a failed build be a parent?", because
an iteration that never reached the Judge has no score row and so is not a
champion candidate at all. Strict N−1 has to answer it, and the honest answer is
no: a failed build still leaves a workspace on disk, full of the code that could
not pass its own gates. Iteration 14 of `run-b5d7565ddb4a` ended
`2 failed, 38 passed`; unconditional N−1 would have made that tree the
foundation of iteration 15 and of every iteration after it.

`LINEAGE_SKIP_FAILED_BUILDS` (default on, consulted only under linear lineage)
seeds iteration N from the most recent iteration that actually built — 15←13
when 14 failed. Degraded rows *count* as built: a degraded row is a bad
measurement, not a bad build, and `best_row`'s reason for excluding them (their
scores cannot honestly be compared) has nothing to do with whether their code
compiles.

### The skip is about the workspace, not the knowledge

Discarding a failed iteration's *code* while also discarding what it taught the
loop would be amnesia with a tidier lineage: iteration 15 would design against
iteration 13's code, and the obvious design against iteration 13's code is the
one iteration 14 just tried. Three separate channels keep that from happening,
and none of them touches the workspace:

* **`dev_failure_report`** — the full report, written by the Developer node on
  any failed episode and rendered by `nodes/architect.py::_developer_failure_block`
  into the next Architect's prompt: classification, the failure signature, which
  gates failed and which never ran, the retry spend, the local pass rate, the
  files that were written, and per-step excerpts. Cleared by the next *green*
  build, so it never describes a problem that has already been fixed.
* **`dev_failure_history`** — the compact entry, appended for the life of the
  run. `_repeat_warning` reads it to escalate when the same signature has now
  broken more than one iteration, and `trend_table` prints a row for the failed
  iteration so a gap in the numbering reads as a build that failed rather than
  as a lost record.
* **The lineage reason itself** — `lineage_parent` returns *why* the parent is
  not N−1, and `trend_table` renders it. The Architect is told its workspace is
  iteration 13's because iteration 14 could not build, and specifically is *not*
  told the champion-rollback sentence, which would have it revert a regression
  that never happened. The same distinction is in the log
  (`SKIPPED A FAILED BUILD` rather than `ROLLED BACK`) and on the Developer span
  (`lineage_reason`), so a post-mortem can tell the two apart too.

Tested end to end through the compiled graph in
`tests/test_linear_lineage_graph.py`, under `MOCK_SCENARIO=failed_build_midrun`.

## Where it lives

```
scoreboard.py                        all of it, as pure functions of a history list
state.py::score_history              the append-only list the Judge writes
nodes/judge.py                       appends the row; logs REGRESSION / TIED / NEW BEST
nodes/developer.py::prepare_workspace inherits the parent scoreboard.lineage_parent chose
nodes/developer.py::classify_failure  design vs infrastructure vs inconclusive
nodes/architect.py                   the trend block; per-classification framing
nodes/critic.py::_regression_block    the finding, above the marginal ranking
routers.py::route_after_developer     the infrastructure-retry edge
graph.py::infra_retry_node            charges the retry, clears the stale report
bench_template.py                    offline U/A/F/MGS for any workspace
```

Tested by `tests/test_scoreboard.py`, `tests/test_champion_lineage.py`,
`tests/test_linear_lineage_graph.py`,
`tests/test_regression_verdict.py`, `tests/test_dev_failure_classification.py`,
`tests/test_regression_recovery_graph.py`, `tests/test_infra_retry_graph.py`,
`tests/test_template_baseline.py` and `templates/tests/test_action_shape.py`.
