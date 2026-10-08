"""Optional CVAT export hooks (extension stub)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def export_tracks_json(track_results: Sequence[Any], output_path: Path) -> None:
    """Write minimal track JSON for external CVAT tooling.

    Args:
        track_results: Per-frame track outputs (JSON-serializable).
        output_path: Destination ``.json`` file.
    """
    payload = {"frames": track_results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))
