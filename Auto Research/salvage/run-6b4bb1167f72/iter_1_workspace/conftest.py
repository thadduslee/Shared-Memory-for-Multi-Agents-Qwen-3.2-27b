"""Root conftest: make the ``memory_system`` package importable from tests.

Pytest inserts only the ``tests/`` directory onto ``sys.path`` for a bare
``pytest tests/`` run, so sibling-package imports fail unless the workspace
root is on the path.  Placing this at the workspace root lets the Developer's
``run_tests`` gate pass regardless of how the orchestrator invokes pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))