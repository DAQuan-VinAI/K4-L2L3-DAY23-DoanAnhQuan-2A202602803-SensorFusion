"""Import student workspace modules from DAY23_STUDENT_ROOT."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

from fusion_lab.paths import PLATFORM_ROOT, student_root


def _load_module(name: str, path: Path) -> ModuleType:
    """Load a single workspace module from disk.

    Args:
        name: Module stem (filename without `.py`).
        path: Absolute path to the `.py` file.

    Returns:
        Executed module object registered in ``sys.modules``.

    Raises:
        ImportError: If the spec or loader cannot be created.
    """
    spec = importlib.util.spec_from_file_location(f"day23_workspace.{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_workspace() -> dict[str, ModuleType]:
    """Load all workspace/*.py modules as a dict keyed by stem name.

    Returns:
        Mapping from module stem (e.g. ``kalman``) to loaded module.

    Raises:
        FileNotFoundError: If ``workspace/`` is missing under ``DAY23_STUDENT_ROOT``.
    """
    os.environ.setdefault("FUSION_LAB_PLATFORM", str(PLATFORM_ROOT))
    root = student_root()
    ws = root / "workspace"
    if not ws.is_dir():
        raise FileNotFoundError(f"No workspace/ under {root}")
    ws_str = str(ws)
    if ws_str not in sys.path:
        sys.path.insert(0, ws_str)
    modules = {}
    for py in sorted(ws.glob("*.py")):
        if py.name.startswith("_"):
            continue
        modules[py.stem] = _load_module(py.stem, py)
    return modules
