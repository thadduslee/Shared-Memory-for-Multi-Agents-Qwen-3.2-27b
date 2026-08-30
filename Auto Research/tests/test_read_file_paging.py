"""`read_file` must be able to show the Developer a whole file.

WHAT BROKE (runs_4iter, 2026-08-29). `read_file` returned `text[:8000]`, with no
way to ask for anything else, and every tool result was then clipped to the LAST
2400 characters before it reached the scratchpad. So a read of the 23150-character
`memory_system/store.py` showed the Developer characters 5600-8000 and nothing
else -- never the imports, never `retrieve()`, never the same window twice.

What it did about that is the point. Iteration 1 wrote pytest modules that sliced
the source into 1500-character files and then `assert False`ed so the runner
would surface them, turning the test runner into a file reader; 66 of its 86
turns were `read_file` calls and every dump run cleared the `tests_ok` gate, so
it exhausted MAX_DEV_RETRIES having written no fix. Iteration 3 inherited the
blindness and sent `apply_patch` a search block whose two lines had a third line
between them in the real file -- four identical failures, retry cap gone, nothing
written. Between them those two episodes burned 983,073 of the run's 1,131,556
tokens and produced no code.

Both are downstream of one thing: the Developer could not read the file it was
employed to edit. These tests pin the window, the footer that says where the
window ends, and the head-preserving truncation -- a file's answer is at the top.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nodes.dev_tools import DevToolbox


@pytest.fixture()
def toolbox(tmp_path: Path) -> DevToolbox:
    return DevToolbox(workspace=tmp_path)


def _read(toolbox: DevToolbox, path: str, **args):
    return asyncio.run(toolbox._t_read_file({"path": path, **args}))


def _numbered(n: int) -> str:
    """A file whose every line names itself, so a window is checkable."""
    return "".join(f"line_{i}\n" for i in range(1, n + 1))


# ----------------------------------------------------------------------
# the window
# ----------------------------------------------------------------------


def test_a_short_file_comes_back_whole(toolbox: DevToolbox) -> None:
    (toolbox.workspace / "small.py").write_text(_numbered(10))

    result = _read(toolbox, "small.py")

    assert result.ok
    assert "line_1\n" in result.stdout
    assert "line_10\n" in result.stdout
    assert result.data["complete"] is True
    assert "End of file." in result.stdout


def test_a_long_file_is_paged_from_the_top(toolbox: DevToolbox) -> None:
    (toolbox.workspace / "store.py").write_text(_numbered(600))

    first = _read(toolbox, "store.py")

    assert first.data["offset"] == 1, "a read with no offset starts at the beginning"
    assert first.data["last_line"] == DevToolbox.READ_FILE_LINES
    assert first.data["complete"] is False
    assert "line_1\n" in first.stdout, "the FIRST line must be in the first window"


def test_offset_reaches_the_part_the_first_window_missed(toolbox: DevToolbox) -> None:
    (toolbox.workspace / "store.py").write_text(_numbered(600))

    result = _read(toolbox, "store.py", offset=400, limit=5)

    assert result.stdout.startswith("line_400\n")
    assert "line_404\n" in result.stdout
    assert "line_405\n" not in result.stdout


def test_the_windows_together_cover_the_whole_file(toolbox: DevToolbox) -> None:
    """The property that actually matters: nothing is unreachable."""
    text = _numbered(600)
    (toolbox.workspace / "store.py").write_text(text)

    seen, offset = [], 1
    while True:
        result = _read(toolbox, "store.py", offset=offset)
        seen.append(result.stdout.split("\n---", 1)[0])
        if result.data["complete"]:
            break
        offset = result.data["last_line"] + 1

    assert "".join(seen) == text
    assert len(seen) == 3, "600 lines at 220 per window"


def test_limit_is_capped_not_trusted(toolbox: DevToolbox) -> None:
    (toolbox.workspace / "store.py").write_text(_numbered(5000))

    result = _read(toolbox, "store.py", limit=99999)

    assert result.data["last_line"] <= DevToolbox.READ_FILE_LINES


@pytest.mark.parametrize("junk", ["", "abc", "-4", None, 0])
def test_a_junk_offset_falls_back_instead_of_raising(toolbox: DevToolbox, junk) -> None:
    (toolbox.workspace / "store.py").write_text(_numbered(50))

    result = _read(toolbox, "store.py", offset=junk)

    assert result.ok and result.data["offset"] == 1


def test_an_offset_past_the_end_is_clamped(toolbox: DevToolbox) -> None:
    (toolbox.workspace / "store.py").write_text(_numbered(50))

    result = _read(toolbox, "store.py", offset=9000)

    assert result.ok, "reading past the end is not an error the model can act on"
    assert result.data["last_line"] == 50


# ----------------------------------------------------------------------
# the footer -- the model has to be TOLD there is more
# ----------------------------------------------------------------------


def test_an_incomplete_read_says_so_and_gives_the_next_call(toolbox: DevToolbox) -> None:
    """A silent truncation is what made the Developer invent the dump hack."""
    (toolbox.workspace / "store.py").write_text(_numbered(600))

    result = _read(toolbox, "store.py")

    assert "NOT THE WHOLE FILE" in result.stdout
    assert '"offset": 221' in result.stdout
    assert '"path": "store.py"' in result.stdout
    assert "of 600" in result.stdout


def test_the_footer_reports_the_real_totals(toolbox: DevToolbox) -> None:
    text = _numbered(600)
    (toolbox.workspace / "store.py").write_text(text)

    result = _read(toolbox, "store.py")

    assert f"({len(text)} chars total)" in result.stdout
    assert result.data["lines"] == 600


# ----------------------------------------------------------------------
# what survives into the scratchpad
# ----------------------------------------------------------------------


def test_the_observation_keeps_the_head_of_a_file(toolbox: DevToolbox) -> None:
    """THE regression. The old cap kept the tail, so the top was never visible."""
    (toolbox.workspace / "store.py").write_text(
        "import sqlite3\n" + _numbered(600))

    observation = _read(toolbox, "store.py").observation()

    assert "import sqlite3" in observation, "the file's first line must reach the model"


def test_a_traceback_still_keeps_its_tail() -> None:
    """Diagnostics truncate the other way: the exception line is at the bottom."""
    from nodes.dev_tools import ToolResult

    trace = "x\n" * 5000 + "AssertionError: the thing that actually failed\n"
    observation = ToolResult(False, "run_tests", stdout=trace).observation()

    assert "AssertionError: the thing that actually failed" in observation
    assert "[... truncated ...]" in observation


def test_a_file_read_gets_more_room_than_a_diagnostic(toolbox: DevToolbox) -> None:
    """220 lines of real source is ~9KB; the old budget showed 2400 characters."""
    from nodes.dev_tools import ToolResult

    (toolbox.workspace / "store.py").write_text(
        "".join(f"    self.field_{i} = compute_something({i})  # a realistic line\n"
                for i in range(1, 221)))
    observation = _read(toolbox, "store.py").observation()

    assert len(observation) > ToolResult.DEFAULT_OBSERVATION_CHARS * 2
    assert "field_1 " in observation and "field_220 " in observation


# ----------------------------------------------------------------------
# the schema is what the model actually reads
# ----------------------------------------------------------------------


def test_the_advertised_schema_offers_offset_and_limit() -> None:
    """A capability the tool has and the schema hides does not exist."""
    from nodes.dev_tools import TOOL_SCHEMAS

    schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "read_file")
    properties = schema["function"]["parameters"]["properties"]

    assert "offset" in properties and "limit" in properties
    assert "8000 characters" not in schema["function"]["description"]
