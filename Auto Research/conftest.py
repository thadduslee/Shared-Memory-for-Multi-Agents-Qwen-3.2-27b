"""Make the project root importable from anywhere.

Necessary because the project lives in a directory whose name contains a space,
so it cannot itself be a Python package and `pytest`'s rootdir insertion does
not give the modules a stable import path.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _pin_rollback_to_best(monkeypatch):
    """Insulate the suite from `.env`'s `ROLLBACK_TO_BEST`.

    `config` loads the project `.env` at import time, so an OPERATOR setting --
    `ROLLBACK_TO_BEST=false`, which selects strict linear N-1 lineage for real
    runs -- silently rewrote what nine tests were measuring. They assert the
    ROLLBACK path (`scoreboard.champion_iteration` returning the champion, a
    failed build never becoming a parent, the Architect prompt carrying the
    rollback line), and with the flag off they were asserting it against a
    configuration in which it is deliberately disabled.

    The tests are about the CODE, so they get the code's default rather than
    whatever the machine's `.env` happens to say. The two tests that exercise
    the other setting monkeypatch it in their own body, which runs after this
    fixture and therefore still wins.
    """
    import config

    monkeypatch.setattr(config, "ROLLBACK_TO_BEST", True)
