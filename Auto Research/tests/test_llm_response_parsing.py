"""Regression tests for reading an OpenAI-compatible chat response.

`content` alone is the wrong field for a reasoning or tool-calling model, and
the mistake is SILENT: it yields an empty string, the caller reports "empty
response", and the turn is spent for nothing. This is the http-transport twin
of the `collect_assistant_text` regression -- the same content loss, a different
wire format.
"""

from __future__ import annotations

from harness.dsh_client import extract_json_block
from llm.client import AsyncLLMClient


def body(message: dict, usage: dict | None = None) -> dict:
    return {"choices": [{"message": message}], "usage": usage or {}}


def parse(message: dict):
    return AsyncLLMClient()._parse(body(message), model="m", route="openrouter")


def test_a_plain_string_content_is_unchanged() -> None:
    """The ordinary case must be byte-identical to the old behaviour."""
    assert parse({"content": "hello"}).text == "hello"


def test_null_content_is_not_an_error() -> None:
    result = parse({"content": None})
    assert result.ok and result.text == ""


def test_list_shaped_content_is_joined() -> None:
    """Gateways return typed parts when the model emitted mixed output."""
    result = parse({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]})
    assert result.text == "a\n\nb"


def test_a_tool_call_is_rendered_as_a_parseable_action() -> None:
    result = parse({
        "content": "",
        "tool_calls": [{
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "store.py"}'},
        }],
    })
    assert extract_json_block(result.text) == {
        "tool": "read_file", "args": {"path": "store.py"}}


def test_prose_outranks_a_tool_call_in_the_same_message() -> None:
    """Appended last, so a real deliverable fence still wins the first-fence race."""
    result = parse({
        "content": 'Verdict.\n\n```json\n{"utility_correct": true}\n```',
        "tool_calls": [{"function": {"name": "read_file", "arguments": "{}"}}],
    })
    assert extract_json_block(result.text) == {"utility_correct": True}


def test_a_tool_call_with_undecodable_arguments_keeps_its_name() -> None:
    result = parse({"tool_calls": [{"function": {"name": "compile_check",
                                                 "arguments": "{not json"}}]})
    assert extract_json_block(result.text) == {"tool": "compile_check", "args": {}}


def test_a_nameless_tool_call_is_ignored() -> None:
    result = parse({"content": "prose", "tool_calls": [{"function": {"arguments": "{}"}}]})
    assert result.text == "prose"


def test_a_malformed_response_shape_is_reported_not_raised() -> None:
    client = AsyncLLMClient()
    assert not client._parse({}, model="m", route="r").ok
    assert not client._parse({"choices": []}, model="m", route="r").ok
    assert not client._parse({"choices": [{"message": "text"}]}, model="m", route="r").ok


def test_usage_is_still_accumulated() -> None:
    client = AsyncLLMClient()
    result = client._parse(
        body({"content": "x"}, {"prompt_tokens": 3, "completion_tokens": 4}),
        model="m", route="openrouter",
    )
    assert result.usage == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}


# ======================================================================
# Cancellation safety (the timeout wrapper in nodes/_transport.py)
# ======================================================================


def test_the_route_semaphore_is_released_when_a_call_is_cancelled() -> None:
    """`agent_call` now abandons a hanging http request via `asyncio.wait_for`.

    That cancellation lands INSIDE `chat`, which holds the per-route semaphore
    across its retry loop. If the semaphore leaked on cancellation, each
    abandoned Developer sample would permanently consume one slot and the route
    would deadlock after `max_concurrency` timeouts -- a failure that would only
    appear in a long run, and would look like the loop mysteriously hanging.
    """
    import asyncio

    client = AsyncLLMClient()
    semaphore = client._semaphores["openrouter"]
    capacity = semaphore._value

    class _HangingTransport:
        async def post(self, *args, **kwargs):
            await asyncio.sleep(60)

    client._client = _HangingTransport()

    async def drive() -> None:
        for _ in range(3):
            try:
                await asyncio.wait_for(
                    client.chat(route="openrouter", model="m",
                                messages=[{"role": "user", "content": "x"}], role="t"),
                    timeout=0.05,
                )
            except asyncio.TimeoutError:
                pass

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(drive())
    assert semaphore._value == capacity, "the semaphore leaked on cancellation"
