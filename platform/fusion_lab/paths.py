"""Resolve platform and student roots for Day 23."""

from __future__ import annotations

import os
from pathlib import Path

_PLATFORM_PKG = Path(__file__).resolve().parent
PLATFORM_ROOT = Path(os.environ.get("FUSION_LAB_PLATFORM", _PLATFORM_PKG.parent))
THIRD_PARTY = PLATFORM_ROOT / "third_party"


def student_root() -> Path:
    """Resolve the directory containing ``workspace/``.

    Returns:
        Resolved path from ``DAY23_STUDENT_ROOT`` or default ``platform/../student``.
    """
    env = os.environ.get("DAY23_STUDENT_ROOT")
    if env:
        return Path(env).resolve()
    default = PLATFORM_ROOT.parent / "student"
    return default.resolve()


def ensure_platform_on_path() -> None:
    """Insert platform ``third_party`` on ``sys.path`` if missing."""
    import sys

    tp = str(THIRD_PARTY)
    if tp not in sys.path:
        sys.path.insert(0, tp)
