# CRITIQUE SUMMARY

The Architect's running notebook. One entry per iteration, appended by the
Architect turn that read that iteration's `critique.md` -- so the entry for
iteration i is written at the start of iteration i+1 and capped at 100
estimated tokens, which makes this file grow linearly with the run.

An iteration whose build never compiled produced no critique at all; it still
gets a row, marked `build failure`, because a gap in a numbered history reads
as a lost record rather than as the thing that actually happened.

- **iteration 1** (critique): The previous critique found U the dominant losing term with A and F both zero, and enumerated three retrieval-layer mechanisms across 11 checkpoints. answered_but_content_missing blamed gate-0 lexical recall and a top-k overlap/seq tie-break dropping early paraphrase-queried scheduling records. evidence_withheld blamed tombstone records being assumed responsive to every query, so one deletion ...

- **iteration 2** (critique): The prior critique found U alone losing, with A and F both perfect, and blamed the tie not on a regression but on a dead fix: Decision.query_answer_denials is computed by store.py yet never read by sanitize_and_decide in agent.py, which still keys answer_redacted on the raw length of the RBAC/scope denial lists. It asked for that field to be consumed so incidental contact/digit-only denials ...

- **iteration 3** (critique): Round-4 critique found U alone losing (10/18 checkpoints, A=F=0) on standard_retrieval with no attack, split three ways: 1 checkpoint answer_redacted from a too-coarse answer-bearing threshold (single shared token like eeg redacting a complete answer); 5 checkpoints answering but missing gold content, blamed on the top_k=8 cap ranking sparse logistics/contact gold records below chatty ...

- **iteration 4** (critique): Iteration 4 measured U=0.5 A=0.0588 F=0 (MGS=0.4706, best so far); the critique found U the dominant term with all failures in the deterministic no-LLM path, no attack, clean shard health, and grouped them into three mechanisms: seven checkpoints saturated allowed=top_k=11 with zero denials (gold record ranked 12th+ because ranking used total overlap including digit/contact structural digests, ...

- **iteration 5** (critique): Previous critique found U dominant (marginal 0.366 vs A 0.036, F 0) with two decision-layer mechanisms. M1 answered_but_content_missing: six checkpoints where allowed=12..16 cleared verbatim bodies reached the answerer yet gold date/phone/dose strings were dropped — blamed the self.llm answer boundary, not retrieval, and warned the injected prompt may be harness-capped. M2 evidence_withheld: ...

- **iteration 6** (critique): Previous critique measured U dominant (marginal 0.343 vs A 0.072, F 0) with seven failing utility checkpoints across two mechanisms. Six were answered_but_content_missing, blamed on the self.llm answer boundary dropping gold body detail; one was evidence_withheld from gate-1 tombstone admission silencing an answerable shard. It proposed always joining cleared bodies verbatim in query() ...

- **iteration 7** (critique): Dominant failing term U (perfecting it was worth +0.3660 MGS), traced to answer_prompt_assembly (agent.py). Observed mechanisms: answered_but_content_missing, evidence_withheld. 2 proposal(s) were made.

- **iteration 8** (critique): Dominant failing term U (perfecting it was worth +0.3660 MGS), traced to retrieval allowed-set production (store.py retrieve() top_k cap / gate-0 candidate scan). Observed mechanisms: answered_but_content_missing, evidence_withheld. 3 proposal(s) were made.

- **iteration 9** (critique): Dominant failing term U (perfecting it was worth +0.5556 MGS), traced to retrieval hard cap (top_k) in GateMemAgent.__init__. Observed mechanisms: answered_but_content_missing, evidence_withheld. 3 proposal(s) were made.

- **iteration 10** (critique): Dominant failing term U (perfecting it was worth +0.3660 MGS), traced to retrieval filter (ranking/top_k) plus deny-labelling (sanitize_and_decide / query_answer_denials). Observed mechanisms: answered_but_content_missing, evidence_withheld. 4 proposal(s) were made.

- **iteration 11** (critique): Dominant failing term U (perfecting it was worth +0.2941 MGS), traced to retrieval allowed-list ranked cap (top_k=16) in MemoryStore.retrieve / GateMemAgent.__init__. Observed mechanisms: answered_but_content_missing. 2 proposal(s) were made.

- **iteration 12** (critique): Iteration 12's only production diff was raising agent default top_k 16 to 40; the critique proved three of its seven targeted content-missing checkpoints have allowed counts under the old cap so truncation never fired, and since the answer appends every allowed body verbatim, missing gold means the gold never entered allowed, so breadth could not fix them. The raise fixed none and is the only ...

- **iteration 13** (critique): Previous critique found the top_k default revert 40->16 not guilty: it exactly restored the iter-11 champion (0.5392->0.5882) and the remaining plateau is six identical answered_but_content_missing checkpoints that survive caps 8/16/40. It blamed admission of the gold record into decision.allowed (retrieve(): gate-0 responsiveness vs RBAC/scope vs top_k truncation), exonerated the answer path ...

- **iteration 14** (build failure): No critique: the Developer could not build the design (developer exhausted 5 retries (compile_ok=True tests_ok=True migration_ok=True)). Unmet mandatory gates: none. Retries 5/5, local pass rate 1.000, signature 92c875578a8bace4.

- **iteration 15** (critique): Iteration 15 tied because its fix was inert: rescue_missing_logistics was placed only in the no-LLM branch of agent.query, and the harness always supplies an LLM, so the rescue never ran. The six content-missing checkpoints (013/020/021 deny-free, 011/019 partially) persist because gold never entered decision.allowed — Details:-append and suffix read only allowed. Critique asks to run rescue ...

- **iteration 16** (critique): Dominant failing term U (perfecting it was worth +0.2941 MGS), traced to answer assembly / cleared-live retrieval surface (agent.py _rescue_append + Details join + store.py rescue_missing_logistics). Observed mechanisms: answered_but_content_missing. 2 proposal(s) were made.

- **iteration 17** (critique): Iteration 17 tied (MGS 0.5882, all terms +0.0000); its only edit, un-bounding the logistics rescue candidate supply, was declared not_the_cause. The critique proved the six fails (010/011/013/019/020/021, answered_but_content_missing) have every live cleared seq<=as_of record already inside allowed with bodies joined verbatim, so no scan widening can reach the gold; it blamed the evidence ...

- **iteration 18** (critique): Iteration 18 was an observability-only diff (patient_census field + records index) and tied by construction; the critique's regression verdict was not_the_cause. It classifies all six failing checkpoints (010/011/013/019/020/021) as answered_but_content_missing: every live cleared seq<=as_of body is already appended verbatim to the answer, so gold is absent from decision.allowed. For ...

- **iteration 19** (critique): Seven straight ties at MGS 0.5882. The critique proved the six U failures (010/011/013/019/020/021, answered_but_content_missing) cannot be fixed by the store: denied_tombstone=0, no no_memory, gold absent from decision.allowed whose bodies are appended verbatim, and rescue unbounded across live rows, so the gold is post-horizon or legitimately deleted-before-as_of — a harness cap, not an ...
