"""Import student workspace modules from DAY23_STUDENT_ROOT."""

from __future__ import annotations

import importlib.util
import hashlib
import os
import sys
from pathlib import Path
from types import ModuleType

from fusion_lab.paths import PLATFORM_ROOT, student_root


def _package_name(workspace: Path) -> str:
    """Return the module namespace isolated by the workspace absolute path."""
    return "day23_workspace_" + hashlib.sha256(str(workspace).encode()).hexdigest()[:12]


def _bind_package(workspace: Path, reset: bool = False) -> str:
    """Register the namespace package of ``workspace`` and return its name.

    ``from . import kalman`` inside a workspace module and
    ``load_workspace_module("kalman")`` then resolve to the same module object.
    ``reset`` drops cached modules so edited files are executed again.

    Args:
        workspace: The ``workspace/`` directory of the active student root.
        reset: Start a fresh namespace even when one is registered.

    Returns:
        Name of the registered namespace package.
    """
    package = _package_name(workspace)
    if not reset and package in sys.modules:
        return package
    for cached in [key for key in sys.modules if key.startswith(package + ".")]:
        del sys.modules[cached]
    namespace = ModuleType(package)
    namespace.__path__ = [str(workspace)]
    namespace.__package__ = package
    sys.modules[package] = namespace
    return package


def _load_module(name: str, path: Path, package: str) -> ModuleType:
    """Load a single workspace module from disk, reusing one already loaded.

    Args:
        name: Module stem (filename without `.py`).
        path: Absolute path to the `.py` file.
        package: Namespace isolated by the workspace absolute path.

    Returns:
        Executed module object registered in ``sys.modules``.

    Raises:
        ImportError: If the spec or loader cannot be created.
    """
    qualified = f"{package}.{name}"
    cached = sys.modules.get(qualified)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(qualified, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = mod
    setattr(sys.modules[package], name, mod)
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(qualified, None)
        raise
    return mod


def load_workspace_module(name: str) -> ModuleType:
    """Load one module strictly from the currently selected workspace.

    Use it inside a workspace file, e.g. ``kalman = load_workspace_module("kalman")``,
    so a stale ``kalman`` on ``sys.path`` is never picked up.

    Args:
        name: Workspace module stem, such as ``kalman``.

    Returns:
        Module loaded from the active root, shared with ``load_workspace``.
    """
    workspace = student_root() / "workspace"
    return _load_module(name, workspace / f"{name}.py", _bind_package(workspace))


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
    package = _bind_package(ws, reset=True)
    modules = {}
    for py in sorted(ws.glob("*.py"), key=lambda path: (path.stem != "kalman", path.name)):
        if py.name.startswith("_"):
            continue
        modules[py.stem] = _load_module(py.stem, py, package)
    return modules
