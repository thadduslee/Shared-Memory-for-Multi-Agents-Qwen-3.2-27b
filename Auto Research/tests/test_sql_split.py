"""Splitting a migration into statements without cutting inside a comment.

WHAT BROKE (runs_verify4, 2026-08-26). The Architect emitted perfectly valid
DDL with a trailing rationale comment on each line:

    ALTER TABLE records ADD COLUMN search_tokens TEXT NOT NULL DEFAULT '';
        -- tokenized routine body for relevance; speeds INSTR scans vs full body

The old splitter dropped only lines that *start* with `--`, so that trailing
comment survived into the text being split on `;` -- and the comment contains a
semicolon. sql_exec was handed the fragment `speeds INSTR scans vs full body
ALTER TABLE records ADD COLUMN content_hash TEXT` and died with
`near "speeds": syntax error`.

The cost is not the error, it is the misdirection: the message accuses the
Architect's DDL, which was fine, and the Developer spent one of MAX_DEV_RETRIES
discovering that. With MAX_DEV_RETRIES=5 that is 20% of the episode's budget
spent on a punctuation bug in the orchestrator.
"""

from __future__ import annotations

from nodes.dev_tools import _split_sql

# ----------------------------------------------------------------------
# the exact regression
# ----------------------------------------------------------------------


def test_a_semicolon_inside_a_trailing_comment_does_not_split() -> None:
    script = (
        "ALTER TABLE records ADD COLUMN search_tokens TEXT NOT NULL DEFAULT '';"
        "  -- tokenized body for relevance; speeds INSTR scans vs full body\n"
        "ALTER TABLE records ADD COLUMN content_hash TEXT;  -- sha256 for audit\n"
    )
    statements = _split_sql(script)
    assert statements == [
        "ALTER TABLE records ADD COLUMN search_tokens TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE records ADD COLUMN content_hash TEXT",
    ]


def test_no_statement_ever_begins_with_comment_prose() -> None:
    """The observed failure shape: a fragment starting mid-sentence."""
    script = "CREATE INDEX i ON t(c);  -- fast; because reasons\nDROP INDEX i;\n"
    for statement in _split_sql(script):
        assert statement.upper().startswith(("CREATE", "DROP", "ALTER", "INSERT")), statement


# ----------------------------------------------------------------------
# comments generally
# ----------------------------------------------------------------------


def test_full_line_comments_are_still_dropped() -> None:
    script = "-- header; with a semicolon\nCREATE TABLE t (a INT);\n"
    assert _split_sql(script) == ["CREATE TABLE t (a INT)"]


def test_block_comments_are_ignored() -> None:
    script = "/* multi; line\n   comment; here */\nCREATE TABLE t (a INT);\n"
    assert _split_sql(script) == ["CREATE TABLE t (a INT)"]


def test_an_unterminated_comment_does_not_hang_or_leak() -> None:
    assert _split_sql("CREATE TABLE t (a INT); -- trailing, no newline") == [
        "CREATE TABLE t (a INT)"
    ]


# ----------------------------------------------------------------------
# string literals
# ----------------------------------------------------------------------


def test_a_semicolon_inside_a_string_literal_does_not_split() -> None:
    script = "INSERT INTO t(v) VALUES ('a;b');\nINSERT INTO t(v) VALUES ('c');\n"
    assert _split_sql(script) == [
        "INSERT INTO t(v) VALUES ('a;b')",
        "INSERT INTO t(v) VALUES ('c')",
    ]


def test_a_double_dash_inside_a_string_literal_is_not_a_comment() -> None:
    script = "INSERT INTO t(v) VALUES ('-- not a comment');\nCREATE TABLE u (a INT);\n"
    assert _split_sql(script) == [
        "INSERT INTO t(v) VALUES ('-- not a comment')",
        "CREATE TABLE u (a INT)",
    ]


def test_an_escaped_quote_inside_a_literal_is_handled() -> None:
    script = "INSERT INTO t(v) VALUES ('it''s; fine');\nCREATE TABLE u (a INT);\n"
    assert _split_sql(script) == [
        "INSERT INTO t(v) VALUES ('it''s; fine')",
        "CREATE TABLE u (a INT)",
    ]


# ----------------------------------------------------------------------
# ordinary shapes must be unchanged
# ----------------------------------------------------------------------


def test_a_plain_script_splits_as_before() -> None:
    assert _split_sql("CREATE TABLE a (x INT);\nCREATE TABLE b (y INT);\n") == [
        "CREATE TABLE a (x INT)",
        "CREATE TABLE b (y INT)",
    ]


def test_a_trailing_statement_without_a_semicolon_is_kept() -> None:
    assert _split_sql("CREATE TABLE a (x INT)") == ["CREATE TABLE a (x INT)"]


def test_empty_and_whitespace_only_scripts_yield_nothing() -> None:
    assert _split_sql("") == []
    assert _split_sql("\n\n  ;;  \n") == []


def test_the_real_failing_migration_yields_only_ddl() -> None:
    """End to end on the shape that actually broke: 8 statements, all DDL."""
    script = """\
-- Iteration 1 migration (apply via store.apply_migration; bump SCHEMA_VERSION)
ALTER TABLE records ADD COLUMN search_tokens TEXT NOT NULL DEFAULT '';  -- tokenized body for relevance; speeds INSTR scans vs full body
ALTER TABLE records ADD COLUMN content_hash TEXT;  -- sha256; for shred audit
ALTER TABLE records ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1;  -- flag
ALTER TABLE tombstones ADD COLUMN content_hash TEXT;  -- retains hash
CREATE INDEX IF NOT EXISTS i1 ON records(patient_id, is_active);  -- gate 0; path
CREATE INDEX IF NOT EXISTS i2 ON records(search_tokens);  -- relevance; routine
CREATE UNIQUE INDEX IF NOT EXISTS i3 ON records(content_hash);  -- guard
CREATE INDEX IF NOT EXISTS i4 ON access_log(requester_id, ts);  -- evidence
"""
    statements = _split_sql(script)
    assert len(statements) == 8
    assert all(s.upper().startswith(("ALTER", "CREATE")) for s in statements), statements
