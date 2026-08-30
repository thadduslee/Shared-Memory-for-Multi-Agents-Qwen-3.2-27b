"""Make the project root importable from anywhere.

Necessary because the project lives in a directory whose name contains a space,
so it cannot itself be a Python package and `pytest`'s rootdir insertion does
not give the modules a stable import path.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
