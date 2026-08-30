"""Regression tests for the Architect's deliverable extraction.

WHAT BROKE (runs_smoke/iter_1). The Architect's JSON block failed to parse, so
`design.json` was written with every field null and `migration.sql` was written
ZERO BYTES. Nothing logged an error. The Developer was handed an empty work
order and spent its whole budget in a repetition loop.

`extract_json_block` now repairs that block (see tests/test_json_repair.py).
These tests cover the second line of defence: the coercion and salvage that run
when a field is the wrong shape or the block is unrecoverable.
"""

from __future__ import annotations

from nodes.architect import _as_sql, _fence_tags, _sql_from_prose

# ======================================================================
# _as_sql -- models emit DDL as a string OR as a list of statements
# ======================================================================


def test_a_string_passes_through_unchanged() -> None:
    assert _as_sql("CREATE TABLE t (x);") == "CREATE TABLE t (x);"


def test_a_list_of_statements_is_joined_into_sql() -> None:
    """`str(["CREATE ..."])` yields a Python repr, which is not SQL and cannot run."""
    assert _as_sql(["CREATE TABLE t (x)", "CREATE INDEX i ON t(x);"]) == (
        "CREATE TABLE t (x);\nCREATE INDEX i ON t(x);")


def test_missing_and_empty_values_become_the_empty_string() -> None:
    assert _as_sql(None) == ""
    assert _as_sql([]) == ""
    assert _as_sql("") == ""


def test_blank_entries_in_a_list_are_dropped() -> None:
    assert _as_sql(["CREATE TABLE t (x)", "", "   "]) == "CREATE TABLE t (x);"


# ======================================================================
# _sql_from_prose -- last-resort salvage when the block is unrecoverable
# ======================================================================


DESIGN = """# Design

```sql
CREATE TABLE memories (id TEXT PRIMARY KEY, owner_id TEXT);
CREATE INDEX idx_owner ON memories(owner_id);
```

The retrieval loop then runs:

```sql
SELECT * FROM memories WHERE owner_id = ?;
```

```python
def retrieve(): ...
```
"""


def test_ddl_is_salvaged_from_the_prose_fences() -> None:
    salvaged = _sql_from_prose(DESIGN)
    assert "CREATE TABLE memories" in salvaged
    assert "CREATE INDEX idx_owner" in salvaged


def test_illustrative_selects_are_not_salvaged_as_schema() -> None:
    """A design's SQL fences also hold example queries.

    Feeding a SELECT to `sql_exec` as schema fails against tables the DDL has
    not created yet -- turning a recoverable parse failure into a hard gate
    failure the Developer cannot fix.
    """
    assert "SELECT" not in _sql_from_prose(DESIGN)


def test_non_sql_fences_are_ignored() -> None:
    assert "def retrieve" not in _sql_from_prose(DESIGN)


def test_prose_with_no_sql_yields_nothing() -> None:
    assert _sql_from_prose("# Design\n\nNo code here.") == ""
    assert _sql_from_prose("") == ""


# ======================================================================
# _fence_tags -- the diagnostic printed when a design does not parse
# ======================================================================


def test_fence_tags_names_every_fence_language() -> None:
    assert _fence_tags(DESIGN) == "sql,sql,python"


def test_an_untagged_fence_is_labelled() -> None:
    assert _fence_tags("```\nplain\n```") == "(untagged)"


def test_fence_tags_on_empty_input_is_empty() -> None:
    assert _fence_tags("") == ""
