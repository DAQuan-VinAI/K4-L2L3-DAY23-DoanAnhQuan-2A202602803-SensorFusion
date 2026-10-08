"""Local imshow/matplotlib vs headless PNG export."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Used when an OpenCV build without GUI support forces a fallback to PNG export.
DEFAULT_FALLBACK_DIR = Path("artifacts/viz")


@dataclass
class VizConfig:
    mode: str = "local"
    pause_ms: int = 0
    save_dir: Path | None = None

    @classmethod
    def from_env(cls) -> "VizConfig":
        mode = os.environ.get("DAY23_VIZ_MODE", "local")
        if os.environ.get("DAY23_HEADLESS", "").lower() in ("1", "true", "yes"):
            mode = "headless"
        save = os.environ.get("DAY23_VIZ_SAVE_DIR")
        return cls(mode=mode, save_dir=Path(save) if save else None)


def show_or_save(
    name: str,
    image: np.ndarray,
    cfg: VizConfig,
    frame_id: int = 0,
) -> None:
    """Show with OpenCV locally or write PNG when headless.

    The pip wheel ``opencv-python-headless`` has no GUI backend on Linux and
    Windows, so ``cv2.imshow`` raises ``cv2.error`` there. In that case the
    config switches to headless mode once and later frames are saved as PNG.
    """
    if cfg.mode == "local":
        try:
            cv2.imshow(name, image)
            cv2.waitKey(cfg.pause_ms)
        except cv2.error as exc:
            cfg.mode = "headless"
            cfg.save_dir = cfg.save_dir or DEFAULT_FALLBACK_DIR
            warnings.warn(
                f"cv2.imshow is unavailable ({exc.err or exc}); "
                f"saving PNG frames to {cfg.save_dir} instead.",
                RuntimeWarning,
                stacklevel=2,
            )
    if cfg.save_dir:
        cfg.save_dir.mkdir(parents=True, exist_ok=True)
        out = cfg.save_dir / f"{name}_{frame_id:04d}.png"
        cv2.imwrite(str(out), image)


def show_figure_or_save(
    name: str,
    fig: Any,
    cfg: VizConfig,
    frame_id: int = 0,
) -> None:
    """Save a matplotlib figure to PNG or refresh the display in local mode."""
    if cfg.save_dir:
        cfg.save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.save_dir / f"{name}_{frame_id:04d}.png")
    if cfg.mode == "local":
        plt.pause(0.001)
    else:
        plt.close(fig)
