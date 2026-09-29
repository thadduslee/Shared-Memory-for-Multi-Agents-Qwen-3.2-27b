# Changes on `evaluator-parity-critic`

This branch is based on `main` at `d3b73fe`. It holds two groups of changes:

1. **Evaluator and judge made comparable to the GateMem baselines.** The answerer model, its prompt and the grader now match what the baselines ran with, so MGS numbers can be compared fairly.
2. **Failure-mechanism table, prompt cleanup and the Critic's role.** The Critic diagnoses, the Architect designs from that diagnosis, and no prompt hands either of them the answer.

The branch is a selective port. The serving and infrastructure work from Sep 17–18 is **not** included: H200 serving scripts, local-agent and hybrid environment files, per-node Critic routing, the judge and evaluator retry loops, and the shard-runner race fix. Where a commit here depended on that work, only the part that stands on its own was kept (see [Differences from the original commits](#differences-from-the-original-commits)).

Test suite (`gatemem-auto` env, `pytest` from `Auto Research/`): **680 passed, 3 failed**. All 3 failures already happen on `main`: `test_completion_budget`, `test_degraded_evaluator` and `test_reasoning_budget::…finish_reason…`.

---

## 1. Evaluator and judge matched to the baselines

| Commit | Change |
|---|---|
| `e93edcf` Adopt GateMem's judge semantics behind `JUDGE_IMPL` | Adds `nodes/judge_gatemem.py`, a port of GateMem's scoring primitives. **Matcher:** literal, whitespace-normalised, word-boundary matching, and it also scans `answer_structured`. **Authority:** the LLM judge has the final say on utility and the regex pass is only auxiliary, which is how GateMem itself works. Before this, our regex result was binding, which froze `runs_real_100iter_v4` on answers that already contained every required fact in a different word order. The old behaviour is still available as `JUDGE_IMPL=legacy`. Also adds `rescore_run.py` to rescore a finished run offline. |
| `6a8e770` Separate role from benchmark in the prompts; score on GateMem's own (point 4) | The Evaluator and Judge now render **GateMem's own prompt files** (`bench/prompts/query_prompt.txt`, `judge_prompt.txt`) through `prompts_gatemem.py`, reusing bench's helpers for the policy block and relationship facts. Our own answerer prompt told the model the evidence was pre-authorised, which inflated U. `BENCHMARK_PROMPTS=local` brings back the old prompt pair. The judge is now also given the checkpoint's visible half. |
| `7fa7542` Allow a local Qwen judge on the openai route | `RouteConfig.is_vllm` is now a declared field instead of a check on the route name, and the openai route infers it from its base URL (`OPENAI_ROUTE_IS_VLLM` overrides). This way a judge on a local vLLM gets `enable_thinking: false`, like the one that scored the baselines. |
| `054a486` Attribute the answerer's evidence; score the baselines on our judge | `format_memory_block` now falls back to `role`. Before, every memory line reached the answerer as `speaker=unknown`, so under GateMem's access policy it refused whenever it could not work out who had said what. Also adds `score_baseline.py`, which runs a stored baseline's predictions through **this project's** judge using the same functions as the live loop. That way our system and the baselines are graded by one instrument. The rescored results are in `outputs_qwen38-27b_judge/` (see [Rescored baselines](#rescored-baselines)). |
| `b4d04cb` Carry the speaker through the evaluation shard, not just the renderer | The previous fix did nothing in a live run, because the shard's `_evidence` rebuilt each record as `{record_id, text}`. It now carries `principal_id` / `speaker` / `author_id` / `role`, and a new test runs the shard's own `_evidence` source end to end. |
| `3a6e036` Let the answerer think, as every baseline's did | Every baseline answered with Qwen3's template default, **thinking on**, while our evaluator forced it off. `vllm_chat_template_kwargs(model, role)` now leaves thinking on for the `evaluator` role (`EVALUATOR_THINKING`, default on); the judge stays non-thinking. On 27 office checkpoints: thinking off answered 3/27, thinking on answered 22/27. The same commit adds:<br>• `EVALUATOR_TIMEOUT_S` (900 s)<br>• `EVALUATOR_MAX_TOKENS` raised from 4096 to 16384 in `env_local_eval.sh`<br>• `score_baseline.py` naming the grader from evidence instead of from the directory name<br>• regenerated `.cordis` personas |

## 2. Failure-mechanism table, prompt cleanup, the Critic's role (Sep 19–20)

| Commit | Change |
|---|---|
| `ad161bc` Show the Architect which MECHANISM loses its utility checkpoints | Adds `judge.failure_buckets`, which groups utility failures by the mechanism that lost them. There are four buckets: `suppressed_by_tombstone`, `wrong_action_label`, `withheld_other` and `content_missing`, and each one points to a different repair. The table is added to `judge_report`. |
| `9b0a070` Route the failure-mechanism table through the Critic; retire the curriculum | The Critic's prompt now includes the buckets and must address the largest one, or give a measured reason not to. The curriculum is **off by default** (`CURRICULUM_ENABLED`): on the full 579 checkpoints, `phase_score` was identical to U, and the phase label pointed away from where the failures were. |
| `03c9c01` Make the Critic the only carrier of the failure-mechanism table | Removes the Architect's direct copy of the table and `ARCHITECT_SEED_FAILURE_BUCKETS`. The findings now reach the Architect only through the critique. This deliberately gives up a measured +0.17 MGS shortcut in order to test the intended design: one node diagnoses, the next designs from its diagnosis. |
| `6a8e770` Separate role from benchmark in the prompts (points 1–3) | The Architect, Developer and Critic personas are now **generic**, and everything benchmark-specific has moved into their tasks. The Critic's persona no longer claims it has file access, no longer ignores aggregate evidence, and no longer proposes DDL or picks the fix. `CRITIC_EVIDENCE_MAX` goes from 12 to 40, and evidence is sampled per failure mechanism. |
| `2f35e13` Take the answers out of the prompts | Removes the conclusions we had written into the prompts before any run: the per-metric suspect list `_COMPONENT_HYPOTHESES`, the label naming `sanitize_and_decide` as *the* fix, the "no retrieval change will help" legend, and the hard-coded source-file list (the files are now discovered from the workspace). `tests/test_prompt_neutrality.py` keeps them from coming back. |
| `9914f07` The Architect curates its notebook instead of appending to it | The Architect returns the **full** notebook each round and the file is replaced, capped at 40 notes of 45 words each and trimmed only by dropping whole notes. `write_notebook` refuses an empty list, and `notebook_history.md` keeps every version. Truncation used to cut off the *decision* clause of 63% of entries. |
| `7813017` Give the Critic no tools at all, and add the local-eval topology | The Critic profile mounts **no capabilities**, so "YOU HAVE NO TOOLS" is true on both transports; the source it needs is inlined by `_source_view`. Adds `scripts/serve/env_local_eval.sh`, which runs the Evaluator and Judge on local Qwen3.8-27B and the agents on OpenRouter. |

---

## Differences from the original commits

These commits were cherry-picked from `judge-gatemem-adopt`. Because the Sep 17–18 prerequisites were left out:

- **No whole-call retries.** The judge (`JUDGE_LLM_ATTEMPTS`) and evaluator (`EVAL_RENDER_ATTEMPTS`) retry loops are not ported. Each call is made once and relies on the HTTP client's own retries, as on `main`. A judge call that fails still falls back to the rule verdict, but on this branch the fallback is not counted in the report.
- **Critic routing.** The Critic stays on the OpenRouter route; per-node `CRITIC_*` routing is not included. As a result, `CRITIC_TRANSPORT=http` in `env_local_eval.sh` has no effect here, and the Critic follows `AGENT_TRANSPORT` instead. That is harmless, since it no longer has tools.
- **`is_vllm`.** The route field was originally introduced by the local-agents commit. Only the field and the client check were brought into `7fa7542`. `env_local_judge.sh` was dropped because it depends on `env_local_agents.sh`.
- **Comments** that referred to the probe gate or the starved-reply retry were trimmed, because neither exists on this branch.

## Rescored baselines

`outputs_qwen38-27b_judge/` holds every stored baseline's predictions rescored by **this project's judge**: Qwen/Qwen3.8-27B with `JUDGE_IMPL=gatemem` and GateMem's `judge_prompt.txt`. Each `summary.json` names its grader, its prompt file and the predictions it came from.

| Directories | What they are |
|---|---|
| `medical_qwen38_27b_*` (7) | Medical baselines with the Qwen3.8-27B answerer |
| `medical_qwen32b_*__judge-gpt41` (7) | Medical baselines with the Qwen3-32B answerer. The suffix is the name of the **source** directory in `outputs/`; the scores here come from our judge, not gpt-4.1. |
| `office_qwen38_27b_*` (7) | Office baselines with the Qwen3.8-27B answerer |
| `smoke` | Smoke test of the rescoring path |
| `scores_table.json`, `rescore.log` | Medical comparison table (our judge vs the published numbers) and the rescoring log |

The outputs are not duplicated. In `outputs/`, each `medical_qwen38_27b_X` has byte-identical predictions to `medical_qwen38_27b_X__judge-gpt41` (the same run, scored by the regex matcher only in one and by gpt-4.1 in the other), so each of those pairs was rescored **once**. The `qwen32b` sets exist only under their `__judge-gpt41` names and are different predictions.

On utility, our judge reproduces gpt-4.1 to a mean absolute difference of 0.0062 across the 15 medical systems. On privacy it is stricter for all 15 (mean +0.151), which is a difference in definition rather than noise. `scores_table.json` covers medical only; the office summaries are the per-directory `summary.json` files.

## Running

```bash
set -a; source .env; set +a
source scripts/serve/env_local_eval.sh
cd "Auto Research" && python main.py --real --no-full --dev-checkpoints 579 --max-iterations 20
```

`env_local_eval.sh` expects a vLLM serving `Qwen/Qwen3.8-27B` on `localhost:8002`. The model name must keep its `Qwen/` prefix, because the thinking policy matches on it.
