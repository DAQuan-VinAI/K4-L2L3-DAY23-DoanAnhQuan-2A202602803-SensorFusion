"""Shared student workspace setup for the self-check tests."""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def workspace_modules():
    """Load the student workspace, respecting DAY23_STUDENT_ROOT overrides."""
    student_dir = Path(__file__).resolve().parents[1]
    student_root = Path(os.environ.get("DAY23_STUDENT_ROOT", student_dir)).resolve()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DAY23_STUDENT_ROOT", str(student_root))
        patch.setenv("FUSION_LAB_PLATFORM", str(student_dir.parent / "platform"))
        from fusion_lab.workspace_loader import load_workspace

        yield load_workspace()


def pytest_configure(config):
    """Register the narrowly scoped marker for unfinished E–H exercises."""
    config.addinivalue_line(
        "markers", "student_exercise: behavioural check of a Part E–H learner function"
    )


def pytest_collection_modifyitems(config, items):
    """Allow only NotImplementedError from the bundled unfinished student pack."""
    student_dir = Path(__file__).resolve().parents[1]
    selected_root = Path(os.environ.get("DAY23_STUDENT_ROOT", student_dir)).resolve()
    if selected_root != student_dir:
        return
    exercise_files = ("kalman", "association", "camera_fusion", "track_management")
    unfinished = any(
        'raise NotImplementedError(' in (student_dir / "workspace" / f"{name}.py").read_text()
        for name in exercise_files
    )
    if not unfinished:
        return
    marker = pytest.mark.xfail(
        raises=NotImplementedError,
        reason="Implement the bundled Part E–H student exercises to run this check.",
        strict=False,
    )
    for item in items:
        if item.get_closest_marker("student_exercise"):
            item.add_marker(marker)
