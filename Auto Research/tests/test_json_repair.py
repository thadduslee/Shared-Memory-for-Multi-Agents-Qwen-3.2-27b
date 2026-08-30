"""Regression tests for `extract_json_block`, the loop's structured-output seam.

THE FAILURE THESE PIN (runs_smoke/iter_1). The Architect produced a complete
10,260-character design document whose ```json block contained a JavaScript
expression instead of a JSON value:

    "migration_sql": "-- Initial creation...\\nPRAGMA foreign_keys = ON;\\n" + schema_ddl,

`json.loads` failed at character 1775 and the brace-matching fallback failed
too, so `extract_json_block` returned None. The Architect wrote a 154-byte
design.json with every field null and a ZERO-BYTE migration.sql, then handed the
Developer an empty work order -- which is why that Developer sat in a repetition
loop until its think timeout. Nothing logged an error; the run looked healthy.

The repair pass must be STRING-AWARE. `schema_ddl` and `migration_sql` are SQL
values that legitimately contain braces, commas, quotes, comment markers and the
word `True`; a naive regex sweep would corrupt exactly the payload this loop
exists to produce. Half of the tests below exist to hold that line.
"""

from __future__ import annotations

import json

from harness.dsh_client import extract_json_block

# ======================================================================
# The observed live failure
# ======================================================================


def test_the_architects_javascript_concatenation_is_recovered() -> None:
    """The exact shape that zeroed out migration.sql."""
    reply = """# Design Document

Prose about the schema.

```json
{
  "schema_ddl": "CREATE TABLE memories (id TEXT PRIMARY KEY);",
  "migration_sql": "PRAGMA foreign_keys = ON;\\n" + schema_ddl,
  "work_order": ["step one", "step two"],
  "targets_metric": "A"
}
```
"""
    block = extract_json_block(reply)
    assert block is not None, "the design document was discarded again"
    assert block["migration_sql"] == "PRAGMA foreign_keys = ON;\n"
    assert block["work_order"] == ["step one", "step two"]
    assert block["targets_metric"] == "A"


def test_string_plus_string_concatenation_is_spliced() -> None:
    reply = '```json\n{"a": "left " + "right"}\n```'
    assert extract_json_block(reply) == {"a": "left right"}


# ======================================================================
# The payload must survive intact
# ======================================================================


def test_sql_containing_braces_and_commas_is_not_corrupted() -> None:
    """A well-formed block must come back byte-identical, repair or no repair."""
    ddl = (
        "CREATE TABLE memories (\n"
        "  id TEXT PRIMARY KEY,\n"
        "  meta TEXT CHECK (json_valid(meta)),\n"
        "  body BLOB NOT NULL\n"
        ");\n"
        "CREATE INDEX idx_owner ON memories(owner_id, is_deleted);"
    )
    payload = {"schema_ddl": ddl, "work_order": ["a"]}
    reply = "prose\n\n```json\n" + json.dumps(payload, indent=2) + "\n```\n"
    assert extract_json_block(reply) == payload


def test_sql_comment_markers_inside_a_string_are_preserved() -> None:
    """`--` and `/* */` are SQL comments the value legitimately contains."""
    payload = {"migration_sql": "-- add the index\n/* block */ CREATE INDEX i ON t(c);"}
    assert extract_json_block("```json\n" + json.dumps(payload) + "\n```") == payload


def test_the_word_true_inside_a_string_is_not_rewritten() -> None:
    payload = {"expected_tradeoff": "True positives rise; None are lost."}
    assert extract_json_block("```json\n" + json.dumps(payload) + "\n```") == payload


def test_an_apostrophe_inside_a_string_is_not_read_as_a_quote() -> None:
    payload = {"expected_tradeoff": "the model's recall improves"}
    assert extract_json_block("```json\n" + json.dumps(payload) + "\n```") == payload


def test_a_plus_sign_inside_a_string_is_preserved() -> None:
    payload = {"retrieval_loop": "score = relevance + recency"}
    assert extract_json_block("```json\n" + json.dumps(payload) + "\n```") == payload


# ======================================================================
# The other dialects models emit
# ======================================================================


def test_trailing_commas_are_tolerated() -> None:
    assert extract_json_block('```json\n{"a": 1, "b": [1, 2,],}\n```') == {"a": 1, "b": [1, 2]}


def test_python_literals_are_tolerated() -> None:
    assert extract_json_block('```json\n{"a": True, "b": False, "c": None}\n```') == {
        "a": True, "b": False, "c": None}


def test_line_and_block_comments_are_tolerated() -> None:
    reply = '```json\n{\n  // the metric this targets\n  "targets_metric": "A" /* not U */\n}\n```'
    assert extract_json_block(reply) == {"targets_metric": "A"}


def test_single_quoted_strings_are_tolerated() -> None:
    assert extract_json_block("```json\n{'tool': 'read_file'}\n```") == {"tool": "read_file"}


def test_a_raw_newline_inside_a_string_is_escaped() -> None:
    reply = '```json\n{"migration_sql": "CREATE TABLE a (x);\nCREATE INDEX i ON a(x);"}\n```'
    block = extract_json_block(reply)
    assert block is not None
    assert block["migration_sql"] == "CREATE TABLE a (x);\nCREATE INDEX i ON a(x);"


# ======================================================================
# Candidate selection
# ======================================================================


def test_a_json_fence_wins_over_an_earlier_sql_fence() -> None:
    """A design document opens with DDL fences; the deliverable is the json one."""
    reply = (
        "```sql\nCREATE TABLE t (id TEXT);\n```\n\n"
        "```json\n{\"schema_ddl\": \"CREATE TABLE t (id TEXT);\"}\n```"
    )
    assert extract_json_block(reply) == {"schema_ddl": "CREATE TABLE t (id TEXT);"}


def test_a_later_json_fence_is_tried_when_the_first_is_unsalvageable() -> None:
    reply = (
        "```json\n{{{ totally broken\n```\n\n"
        '```json\n{"tool": "compile_check", "args": {}}\n```'
    )
    assert extract_json_block(reply) == {"tool": "compile_check", "args": {}}


def test_an_untagged_fence_is_used_when_no_json_fence_exists() -> None:
    assert extract_json_block('```\n{"action": "refuse"}\n```') == {"action": "refuse"}


def test_an_unfenced_object_is_still_found() -> None:
    assert extract_json_block('Thought: done.\n{"tool": "finish", "args": {}}') == {
        "tool": "finish", "args": {}}


def test_an_unclosed_fence_still_parses() -> None:
    """A reply truncated at max_tokens loses its closing fence."""
    assert extract_json_block('```json\n{"tool": "list_dir", "args": {"path": "."}}') == {
        "tool": "list_dir", "args": {"path": "."}}


def test_a_brace_inside_a_prose_string_does_not_desynchronise_the_scan() -> None:
    """The unfenced scan is string-aware, so a `{` in SQL text is not a depth bump."""
    reply = 'Note that CHECK (json_valid("{")) is legal.\n\n{"verdict": "ok"}'
    assert extract_json_block(reply) == {"verdict": "ok"}


# ======================================================================
# Degenerate input
# ======================================================================


def test_prose_with_no_object_returns_none() -> None:
    assert extract_json_block("I could not complete the task.") is None


def test_empty_and_whitespace_input_returns_none() -> None:
    assert extract_json_block("") is None
    assert extract_json_block("   \n\n  ") is None


def test_a_json_array_at_top_level_is_not_mistaken_for_the_block() -> None:
    """Every caller expects a mapping; a bare array must not be returned as one."""
    assert extract_json_block('```json\n["a", "b"]\n```') is None


def test_the_repetition_loop_reply_returns_none_without_raising() -> None:
    """The real Developer failure text: 140 repeats of an unclosed tag."""
    assert extract_json_block("<thought\nLet's read the files. response\n" * 140) is None


def test_unterminated_string_does_not_hang_or_raise() -> None:
    assert extract_json_block('```json\n{"a": "never closed') is None


# ======================================================================
# Unquoted words -- the other dialect models reach for
# ======================================================================


def test_unquoted_keys_are_quoted() -> None:
    """`{tool: "x"}` is JavaScript object syntax, not JSON."""
    assert extract_json_block('```json\n{tool: "read_file", args: {path: "s.py"}}\n```') == {
        "tool": "read_file", "args": {"path": "s.py"}}


def test_an_unresolvable_bare_value_becomes_null() -> None:
    """The value it names is not in the document, so there is nothing to recover.

    Nulling one field beats discarding the whole design: the Architect's fields
    all have fallbacks, and a design with one null is repairable by the loop
    while a discarded one silently zeroes the iteration.
    """
    assert extract_json_block('```json\n{"a": undefined_ident, "b": 1}\n```') == {
        "a": None, "b": 1}


def test_an_ellipsis_placeholder_becomes_null() -> None:
    """Models write `...` where they mean "and so on"."""
    assert extract_json_block('```json\n{"work_order": ["a", ...]}\n```') == {
        "work_order": ["a", None]}


def test_a_bare_word_inside_a_string_is_untouched() -> None:
    """The repair is string-aware; SQL text is full of bare words."""
    payload = {"schema_ddl": "CREATE TABLE principals (id TEXT); -- see schema_ddl above"}
    assert extract_json_block("```json\n" + json.dumps(payload) + "\n```") == payload


def test_a_key_containing_a_colon_in_its_value_is_not_confused() -> None:
    payload = {"note": "ratio a:b", "other": 1}
    assert extract_json_block("```json\n" + json.dumps(payload) + "\n```") == payload


def test_json_keywords_are_not_rewritten_as_bare_words() -> None:
    assert extract_json_block('```json\n{"a": true, "b": false, "c": null}\n```') == {
        "a": True, "b": False, "c": None}


def test_a_valid_payload_is_never_altered_by_the_repair_pass() -> None:
    """The guarantee that matters: repair must not corrupt what already parsed.

    `schema_ddl` and `migration_sql` carry SQL containing braces, commas,
    quotes, comment markers, backslashes and the words True/None -- exactly the
    characters a naive repair would mangle.
    """
    payload = {
        "schema_ddl": "CREATE TABLE m (\n  id TEXT,\n  meta TEXT CHECK (json_valid(meta))\n);",
        "migration_sql": "-- add it\n/* block */ CREATE INDEX i ON m(id);",
        "expected_tradeoff": "True positives rise; the model's recall improves. None lost.",
        "retrieval_loop": "score = relevance + recency",
        "work_order": ["a", "b"],
        "nested": {"deep": [1, {"x": None}]},
        "escaped": "a\\backslash and a \"quote\"",
    }
    reply = "# Design\n\n```sql\nSELECT 1;\n```\n\n```json\n" + json.dumps(payload, indent=2) + "\n```"
    assert extract_json_block(reply) == payload
