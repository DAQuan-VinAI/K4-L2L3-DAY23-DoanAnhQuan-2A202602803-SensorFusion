"""Regression test for the headless OpenCV fallback in the visualisation helper."""

import cv2
import numpy as np
import pytest

from fusion_lab.viz import display


def test_imshow_error_falls_back_to_png(monkeypatch, tmp_path):
    def fail_imshow(*_args, **_kwargs):
        raise cv2.error("The function is not implemented")

    monkeypatch.setattr(display.cv2, "imshow", fail_imshow)
    cfg = display.VizConfig(mode="local", save_dir=tmp_path)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.warns(RuntimeWarning, match="imshow is unavailable"):
        display.show_or_save("bev", image, cfg, frame_id=3)
    display.show_or_save("bev", image, cfg, frame_id=4)

    assert cfg.mode == "headless"
    assert (tmp_path / "bev_0003.png").exists()
    assert (tmp_path / "bev_0004.png").exists()
