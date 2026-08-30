"""Regression tests for pulling a node's deliverable out of a dsh run.

These exist because of a real failure. The Architect's first live run emitted
three `assistant/message` events -- 0, 15231 and 494 characters -- and the code
used the SDK's `RunResult.final_response`, documented as "the last committed
root-session assistant text". It returned the 494-character acknowledgement, so
a complete 15KB design document was silently discarded and `migration.sql` was
written empty. The run then fed the Developer a one-sentence work order.

The event fixtures below reproduce the exact shape observed in that session.
"""

from __future__ import annotations

from harness.dsh_client import collect_assistant_text, extract_json_block


def message_event(text: str) -> dict:
    """One `assistant/message` event, in the shape dsh actually emits."""
    return {
        "type": "assistant/message",
        "data": {"message": {"role": "assistant",
                             "content": [{"type": "text", "text": text}]}},
    }


# The observed sequence: an empty opener, the deliverable, a trailing summary
# produced after a tool call.
REAL_SHAPE = [
    message_event(""),
    {"type": "step/start", "data": {}},
    message_event("# Design Document\n\nBody.\n\n```json\n{\"migration_sql\": \"CREATE INDEX i ON t(c);\", \"work_order\": [\"a\", \"b\"]}\n```"),
    {"type": "tool/call", "data": {"name": "todo_write"}},
    message_event("The design document and work order are complete above."),
    {"type": "turn/end", "data": {"reason": {"kind": "completed"}}},
]


def test_the_deliverable_is_not_lost_to_a_trailing_acknowledgement() -> None:
    text = collect_assistant_text(REAL_SHAPE)
    assert "# Design Document" in text, "the deliverable was discarded"
    assert "complete above" in text, "later messages should still be kept"


def test_the_json_block_survives_collection() -> None:
    block = extract_json_block(collect_assistant_text(REAL_SHAPE))
    assert block is not None
    assert block["migration_sql"] == "CREATE INDEX i ON t(c);"
    assert block["work_order"] == ["a", "b"]


def test_last_message_alone_would_have_lost_the_block() -> None:
    """Pins the exact bug: the old behaviour must stay broken in the test."""
    last_only = REAL_SHAPE[-2]["data"]["message"]["content"][0]["text"]
    assert extract_json_block(last_only) is None


def test_empty_messages_are_dropped_not_joined_as_blanks() -> None:
    text = collect_assistant_text([message_event(""), message_event("real"), message_event("   ")])
    assert text == "real"


def test_a_json_block_split_across_two_messages_still_parses() -> None:
    """Messages are joined, so a fence opened in one and closed in the next survives."""
    events = [
        message_event('Prose.\n\n```json\n{"a": 1,'),
        message_event(' "b": 2}\n```'),
    ]
    assert extract_json_block(collect_assistant_text(events)) == {"a": 1, "b": 2}


def test_non_message_events_are_ignored() -> None:
    noise = [
        {"type": "assistant/chunk", "data": {"chunk": {"text": "streaming noise"}}},
        {"type": "tool/result", "data": {"content": "tool noise"}},
        message_event("kept"),
    ]
    assert collect_assistant_text(noise) == "kept"


def test_malformed_events_do_not_raise() -> None:
    """Session logs come off a subprocess; anything can be in them."""
    junk = [None, "string", 42, {}, {"type": "assistant/message"},
            {"type": "assistant/message", "data": None},
            {"type": "assistant/message", "data": {"message": {"content": "not-a-list"}}},
            message_event("survivor")]
    assert collect_assistant_text(junk) == "survivor"


def test_content_blocks_without_the_message_wrapper_are_read() -> None:
    """Some events carry `content` directly on `data` rather than under `message`."""
    event = {"type": "assistant/message",
             "data": {"content": [{"type": "text", "text": "flat shape"}]}}
    assert collect_assistant_text([event]) == "flat shape"


# ==========================================================================
# Tool-call blocks (regression, runs_smoke/iter_1)
#
# The Developer profile mounts no capabilities, so the model gets an agentic
# persona with an empty tool schema -- and answers in its native tool-call
# format regardless. Those blocks matched no mounted tool and used to be
# discarded by `collect_assistant_text`, which kept only `type == "text"`.
# `dev_think` then saw a bare thought with no action and logged "developer
# produced no parseable action". Token accounting proved the loss: two
# Developer turns reported 233 and 44 content tokens and landed 59 and 1
# characters, while the Architect's reconciled exactly.
# ==========================================================================


def blocks_event(*blocks: dict) -> dict:
    """One `assistant/message` event carrying arbitrary content blocks."""
    return {
        "type": "assistant/message",
        "data": {"message": {"role": "assistant", "content": list(blocks)}},
    }


def test_an_anthropic_shaped_tool_call_survives_collection() -> None:
    event = blocks_event(
        {"type": "text", "text": "Thought: read the store first."},
        {"type": "tool_use", "name": "read_file",
         "input": {"path": "memory_system/store.py"}},
    )
    action = extract_json_block(collect_assistant_text([event]))
    assert action == {"tool": "read_file", "args": {"path": "memory_system/store.py"}}


def test_an_openai_shaped_tool_call_survives_collection() -> None:
    """`function_call` nests the payload and passes `arguments` as a JSON string."""
    event = blocks_event(
        {"type": "function_call",
         "function": {"name": "run_tests", "arguments": '{"path": "tests"}'}},
    )
    action = extract_json_block(collect_assistant_text([event]))
    assert action == {"tool": "run_tests", "args": {"path": "tests"}}


def test_the_observed_live_failure_now_parses() -> None:
    """The exact runs_smoke/iter_1 shape: a lone thought plus a dropped call."""
    event = blocks_event(
        {"type": "text", "text": " \n\n<thought>\nNeed to inspect current "
                                 "implementation files.\n\n"},
        {"type": "tool_call", "toolName": "list_dir", "args": {"path": "."}},
    )
    text = collect_assistant_text([event])
    assert "<thought>" in text, "the prose must still be collected"
    assert extract_json_block(text) == {"tool": "list_dir", "args": {"path": "."}}


def test_a_real_deliverable_outranks_a_recovered_call() -> None:
    """Recovered calls are appended last so a node's own fence still wins.

    The Judge and Critic hold `fs_read` and legitimately emit tool calls before
    their verdict. Interleaving the recovered block would put a `read_file`
    envelope ahead of the verdict and `extract_json_block` takes the FIRST
    fence -- silently replacing every verdict in the run.
    """
    event = blocks_event(
        {"type": "tool_use", "name": "read_file", "input": {"path": "preds.json"}},
        {"type": "text", "text": 'Verdict.\n\n```json\n{"score": 0.91}\n```'},
    )
    assert extract_json_block(collect_assistant_text([event])) == {"score": 0.91}


def test_reasoning_blocks_are_not_collected() -> None:
    """A draft in the reasoning trace must not outrank the committed answer.

    Reasoning tokens are accounted separately by `_usage_from_events`; folding
    them into the text would let a rejected draft win the first-fence race.
    """
    event = blocks_event(
        {"type": "thinking", "text": '```json\n{"tool": "finish", "args": {}}\n```'},
        {"type": "text", "text": '```json\n{"tool": "compile_check", "args": {}}\n```'},
    )
    text = collect_assistant_text([event])
    assert "finish" not in text
    assert extract_json_block(text) == {"tool": "compile_check", "args": {}}


def test_a_tool_call_with_undecodable_arguments_keeps_its_name() -> None:
    """Better an unknown-arguments turn than a silent no-op: the name is the
    signal the STATUS block re-anchors on."""
    event = blocks_event(
        {"type": "function_call", "name": "compile_check", "arguments": "{not json"},
    )
    assert extract_json_block(collect_assistant_text([event])) == {
        "tool": "compile_check", "args": {}}


def test_a_nameless_tool_call_block_is_ignored() -> None:
    event = blocks_event(
        {"type": "tool_use", "input": {"path": "x"}},
        {"type": "text", "text": "prose only"},
    )
    assert collect_assistant_text([event]) == "prose only"


def test_unknown_block_types_carrying_text_are_collected() -> None:
    """Adapters spell the prose block `output_text` as often as `text`."""
    event = blocks_event({"type": "output_text", "text": "kept anyway"})
    assert collect_assistant_text([event]) == "kept anyway"


def test_malformed_tool_call_blocks_do_not_raise() -> None:
    event = blocks_event(
        {"type": "tool_use", "name": "", "input": None},
        {"type": "tool_call", "name": "read_file", "args": "not-a-dict"},
        {"type": "function_call", "function": "not-a-dict", "name": "list_dir"},
    )
    text = collect_assistant_text([event])
    assert extract_json_block(text) == {"tool": "read_file", "args": {}}
