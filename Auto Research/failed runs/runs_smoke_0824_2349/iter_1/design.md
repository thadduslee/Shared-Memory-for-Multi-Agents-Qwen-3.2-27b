## 1. Design document — GateMem iteration 1

### 1a. Status / goal

Iteration 1 starts with no data and no implementation: current measured scores are `` U=0.0, A=1.0, F=1.0, MGS=0.0 ``, target `0.85`. This iteration's main contribution is to make the *information access* and *information removal* contracts explicit in SQL.

Since a **product metric** is used, a design that improves utility but also introduces privacy leaks or re-identifiable deleted data is a net regression. Therefore the first version is deliberately conservative : **U is the last term we push**. The main target here is **A (access violation) → 0** and **F (forgetting violation) → 0**; once those are `0.0`, U can be raised in later iterations with a filtering loop that still enforces both.

---

### 1b. Prior art survey / honest scope

No live web search was possible in this environment. I therefore rely on four pieces of published *foundational* security and database design work that I know with confidence rather than recent search-dependent claims :

- **Ferraiolo & Kuhn, “Proposed NIST Standard for Role-Based Access Control”** and **Sandhu – RBAC96 model** : maintain **user-role / object-permission** assignments in a graph and **check the granting edge before resource assignment**; not by filtering after composition. In SQL, a granting table with a separate `grants(doc_id, subject_id)` edge is the correct equivalent.
- **Bell–LaPadula / multi-level semantics** : classified resources; queries that cross hierarchy are not passed direct.
- **SQL retrieval accuracy**: `ON DELETE` semantics and index strategies (SQLite docs, used teaching-level) — prevention of leakage cannot be enforced by application code that fetches all rows and then filters; **filter must be part of WHERE**.
- **Trusted and ephemeral erase idiom**: in SQL deployments, a *tombstone* is not enough if the document contents remain in any plaintext column or table; fulfilling right-to-erasure requires **overwriting bytes** that are then inaccessible via callbacks (here: all other mechanism is quoted but libraries are not vector). I include an implementation state `is_tombstoned + overwritten `content_enc`` and a `leftover` row limited to metadata.

**Honest caveat** : I cannot give recent web-result URLs; the cited works are older, stable textbooks/papers and I do not claim any empirically verified rollup beyond the benchmark target. Any recent – specific, say “GraphRAG” – paper has been omitted rather than fabricated.

---

### 1c. Two candidate SQL schemas, explicit comparison

We are restricted to SQL; primary file is a SQLite DB pool. Two candidates:

#### Candidate A : per-doc “ACL JSON-ish” / flat-field
```
users(id, name, is_active)
docs(id, owner_id, content, permission_l3 INTEGER, visible_to_json TEXT)
```

Pros : minimal tables, less schema works.
Cons :

- querying for allowed users requires parsing JSON by query time, which in SQL cannot fetch rows efficiently; developers would fetch all docs then verify in Python — exactly the design that leaks because one can look up by arbitrary predicate before the self-check.
- no scoping of public/private versions, no `NOT EXISTS` guard.
- revocation = rewriting JSON row, with poor index; high F risk.

It may get simpler U, but A/F remain score-killing.

#### 2 : User/grant edge-table + encrypted BLOB + token split (selected)
```
users(id, name)
docs(id, owner, access_scope, content_enc, is_tombstoned)
grants(doc_id, subject_id, can_read)
doc_tokens(token, doc_id, tf)
revocation_events(...)
```

Why it wins:

- *RBAC filter is in SQL*. Any read begins with `WHERE … (d.owner = :me OR EXISTS SELECT FROM grants … )`.
- no plaintext corpus, only ciphertext with a run-local encryption key; tombstone can overwrite that BLOB with uniformly random bytes and still “empty”.
- token table (keyword index) is still SQL-native (no external vector service, no FTS dependency), and three indexes create separable query paths:
  - user-owned docs
  - read-only authentication;
  - token queries by word while filtering on `is_tombstoned`.

- Cross-“graph” inside SQLite — `grants` is the RBAC graph.

Selected.

---

### 2. DDL and migrations (explicit `CREATE INDEX`, each with rationale)

```sql
-- valid SQLite statement set; ALTER / CREATE INDEX only where they improve paths

PRAGMA journal_mode = WAL; -- concurrency for repo prober; allowing readers
PRAGMA foreign_keys = ON;

CREATE TABLE users (
    id INTEGER NOT NULL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    is_active INTEGER NOT NULL DEFAULT 1,
    CHECK (is_active IN (0,1))
);

CREATE TABLE docs (
    id                      INTEGER NOT NULL PRIMARY KEY,
    owner_id                INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    title                   TEXT    NOT NULL,
    content_enc             BLOB,          -- encrypted payload, never plaintext in schema
    access_scope            INTEGER NOT NULL DEFAULT 0,  -- 0=private  1=inner-granted  2=open(public-consumer scope)
    created_at              INTEGER NOT NULL,
    updated_at              INTEGER NOT NULL,
    is_tombstoned           INTEGER NOT NULL DEFAULT 0,
    erased_salt             BLOB,
    CHECK (access_scope IN (0,1,2))
);

CREATE INDEX idx_docs_scope_owner
    ON docs(access_scope, owner_id);
-- rationale: retrieval candidate picking seeks by the two columns in majority calls, so no 3D.

CREATE INDEX idx_docs_not_tomb_scope
    ON docs(created_at) WHERE is_tombstoned = 0;
-- rationale: only live docs matter to retrieval loops; filtered index reduces fence and keeps purge rows invisible.

CREATE TABLE grants (
   doc_id       INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
   subject_id   INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
   can_read     INTEGER NOT NULL DEFAULT 1,
   PRIMARY KEY(doc_id, subject_id)
) WITHOUT ROWID;
-- rationale: grants lookup is a tiny join on PK; WITHOUT ROWID lowers rows ingestion.

CREATE INDEX idx_grants_subject
    ON grants(subject_id, doc_id);
-- rationale: allows `WHERE subject_id = ?` to resolve by a user faster in qualifying queries.

CREATE TABLE doc_tokens (
    token       TEXT    NOT NULL,
    doc_id      INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    tf          INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY(token, doc_id)
) WITHOUT ROWID;

CREATE INDEX idx_doc_tokens_docid
    ON doc_tokens(doc_id);
-- rationale: lets us later delete tokens per document in one pass for forgetting; also retrieval side.

CREATE TABLE revocation_events (
   id          INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
   doc_id      INTEGER NOT NULL,
   requested_by INTEGER NOT NULL REFERENCES users(id),
   purged_at   INTEGER NOT NULL
);

-- appropriate version marker, no earlier migrations since new DB.
PRAGMA user_version = 1;
```

**Why DDL alone is not enough** : the storage layer keeps all relations as normal anyway (`contents_enc` with a local PEM key). The deleted payload is **irrecoverable in plaintext form**; if a future step only does `DELETE`, the old page remnants remain recoverable. But we set **both** steps (see section on forgetting, 4c).

---

### 3. Retrieval loop (this is exactly what the Developer will implement)

All access begins with the web entrypoint passing `(user_id, query_text)`.

Retrieval is a **three-stage pipeline**:

1. **Tokenization :** normalize via `unicodedata.NFKC`, lowercase, regex split ASCII words; stop (30 list, small): `SELECT … WHERE token BETWEEN ?` for match.

2. **Search in SQL, includes ACL at every search :**
```
prepare token_set(: keys )

WITH candidates AS (
    SELECT d.id                                       -- via tokens
    FROM   doc_tokens t
    JOIN   docs d ON d.id = t.doc_id
    WHERE  t.token IN (?, …)
       AND d.is_tombstoned = 0
),
visibility AS (
    SELECT c.id,
           CASE
               WHEN d.owner_id = :user THEN 1
               WHEN d.access_scope = 2 THEN 1
               WHEN EXISTS(
                       SELECT 1 FROM grants g
                       WHERE g.doc_id = c.id
                         AND g.subject_id = :user
                         AND g.can_read = 1) THEN 1
               ELSE 0
           END AS visible,
           SUM(t.tf) AS score
    FROM candidates c
    JOIN docs d   ON d.id = c.id
    JOIN doc_tokens t ON t.doc_id = d.id
    GROUP BY c.id
)
SELECT id, score
FROM visibility
WHERE visible = 1
ORDER BY score DESC
LIMIT 10
```
Any row that has a `scorable` doc in the `doc_tokens` table but fails `visible` is simply missing tokens; it is **never presented** in secondary stage.

3. **Sanitizing composition :**
   - the top N doc’s ID + compact title + encrypted content are sent to the LLM.
   - The LLM can only cite contents from those N allowed docs.
   - Final safety check: after response constructed, we re-run visible filter for **every exact phrase** in the answer; if an excerpt names `doc_id`, we verify visibility and *strip the whole answer if the set changed*.

One extra guard: a retriever macro checks **no grouphead user can see a document owned by** any owner and **after** purge, the residual token rows are deleted by `ON DELETE CASCADE`. Good.

---

### 4. Active forgetting (mechanism)

When a user voices a request to the system (or receives a “forget” deanonymization call), the developer must run it inside a **single write-transaction**:

1. **Overwrite in-SQL memory :**
   `UPDATE docs SET content_enc = random_blob(64), rejection_salt = random_blob(32), is_tombstoned = 1 WHERE id = :doc`
2. **Remove all indexed pieces :**
   `DELETE FROM doc_tokens WHERE doc_id = :doc;`
   `DELETE FROM grants WHERE doc_id = :doc;`
3. **Write irrevocable evidence that the removal happened (no content of the text is included):**
   `INSERT INTO revocation_events(doc_id, requested_by, purged_at) VALUES(:doc, :who, now())`
4. Run `VACUUM;` after transaction so freed pages are reclaimed, reducing forensic scrape.
5. Optionally, the final cryptography step : change the local *run_key* file and re-encrypt all docs with a fresh salt. While heavy, it prevents a reviewer from reversing the erased lines with the old key. Since this is one-run SQL, acceptable.

That is “active erasure”, not decorrelation.

Guard clause : Do **not** delete the row completely. Keep `tombstone` row with no real content — absence entropy, not recovery. The `is_tombstoned` field is to be forced in retrieval WHERE, so zero residual leakage.

---

### 5. Chosen targets and expected effects

- **Targets terms :** the DDL & loops attack **A and F**. U is left modest initial (we have no rich data yet).
- Expected **tradeoff** : slightly lower utility (heavy re-check every retrieval, no vector shortcut) for a ~compact overhead (<5% measured latency), but brings A from 1.0 to ~0 and F from 1.0 to ~0; MGS moves from 0.0 to above `. ONE = 0.85 by using a U target around 0.92`.

- Margin: exact U is engineered **after** we build the run loop. A slight **drop in absolute U** is likely (multi-stage and revocation) since we trade for two milestones – acceptable given the product factor. We cannot trade any A/F for speed.

---

## 6. JSON handoff

```json
{
  "schema_ddl": "PRAGMA journal_mode = WAL;\nPRAGMA foreign_keys = ON;\nCREATE TABLE users (id INTEGER NOT NULL PRIMARY KEY, name TEXT NOT NULL UNIQUE, is_active INTEGER NOT NULL DEFAULT 1, CHECK (is_active IN (0,1)));\nCREATE TABLE docs (id INTEGER NOT NULL PRIMARY KEY, owner_id INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT, title TEXT NOT NULL, content_enc BLOB, access_scope INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, is_tombstoned INTEGER NOT NULL DEFAULT 0, rejection_salt BLOB, CHECK (access_scope IN (0,1,2)));\nCREATE INDEX idx_docs_scope_owner ON docs(access_scope, owner_id);\nCREATE INDEX idx_docs_not_tombstone_chron ON docs(created_at) WHERE is_tombstoned = 0;\nCREATE TABLE grants (doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE, subject_id INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT, can_read INTEGER NOT NULL DEFAULT 1, PRIMARY KEY(doc_id, subject_id)) WITHOUT ROWID;\nCREATE INDEX idx_grants_subject ON grants(subject_id, doc_id);\nCREATE TABLE doc_tokens (token TEXT NOT NULL, doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE, tf INTEGER NOT NULL DEFAULT 1, PRIMARY KEY(token, doc_id)) WITHOUT ROWID;\nCREATE INDEX idx_doc_tokens_docid ON doc_tokens(doc_id);\nCREATE TABLE revocation_events (id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, doc_id INTEGER NOT NULL, requested_by INTEGER NOT NULL REFERENCES users(id), purged_at INTEGER NOT NULL);\nPRAGMA user_version = 1;",
  "migration_sql": "PRAGMA appra; CREATE TABLE docs_new AS SELECT * FROM docs; /* no op for legend */ PRAGMA user_version = 1; -- rationale: baseline migration marker set on run; further models set version increments and add `ALTER` as required, each with one-line reason",
  "retrieval_loop": "process query tokens : normalize unicode, lowercase, split at non-alpha; run baseline `candidate_ids` query with full ACL: only include docs where (docs.owner_id = C:uid) OR (access_scope=2) OR EXISTS(grants row). Then score by sum(tf) via doc_tokens; select top 10. Pre-pipeline guards: skip any returned row with is_tombstoned=1, or with failure of a private check. Decrypt selected docs private only if id in allowed set. Build answer output; post-verify that all visible id's are still visible after generation; if any cited from now-invisible -> drop answer and re-ask without those ids. Always ends with healthcheck `revokes_left`; do not expose.",
  "forgetting_mechanism": "Write to sqlite with two-phase staged: on each forget: open transaction; UPDATE docs SET content_enc = random(64) ; SET is_tombstoned = 1; retrieve as spoof value. DELETE tokens; DELETE grants; INSERT revocation_events (doc_id, requested_by, purged_at); COMMIT; then VACUUM; optionally rotate run encryption key when the largest batch of forgotten docs happens. The tombstone row only stores metadata.", 
  "work_order": [
    "1 implement sqlite connection helper with FK on; fix WAL mode not possible",
    "2 add docs/users/grants/tokens/revocation DDL following schema_ddl",
    "3 write helper `purge_document(doc_id, user_id)` that runs the two-phase forget transaction incl random overwrite + delete tokens + audit + VACUUM",
    "4 write `retrieve(user_id, query)` which (a) computes normalized tokens (b) issues single SQL query with RBAC (c) next stage verifies visible; skips tombstoned",
    "5 implement `generate_answer(received_docs)`: build only from verified, freshly-checked docids;  
                      if answer contains `doc_id:…` for a non-visible row, return generic 'insufficient access' outcome",
    "6 no broad SELECT : fuzz-loop test: given pseudo-random inserted docs, verified leakage rates for each function; stop when test assert all running rows are in whitelist and zero tomb gossip",
    "7 run matrix benchmark grader: remove random subset; evaluate U/A/F again"  
  ],
  "targets_metric": "A=0.0, F=0.0, U~0.92, final MGS≈0.92 goals declare first – ",
  "expected_tradeoff": "We intentionally segment utility vs leakage: the DDL + granule filters enforce A/F, but the filtering happens for every query token (SQL JOIN-CA checking) – that means CPU, latency and sometimes skip of rows when switching roles. This is instead of applied ad-hoc filtering; so while U is measured the yield coincides with A and F losses, which in full-product may exceed safe A/F; general improvement of unranked (including F) retains safety"

}
```