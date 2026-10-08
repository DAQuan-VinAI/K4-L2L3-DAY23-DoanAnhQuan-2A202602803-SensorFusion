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
