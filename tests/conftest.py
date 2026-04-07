"""Pytest configuration: ensure the repository root is on sys.path.

The package is not installed in editable mode in every dev environment,
and an existing ``active_memory.egg-info`` next to ``active_memory/``
prevents pytest's automatic rootdir detection from picking up the
source directory. Inserting the repo root here lets ``pytest -q`` work
without needing ``PYTHONPATH=.``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
