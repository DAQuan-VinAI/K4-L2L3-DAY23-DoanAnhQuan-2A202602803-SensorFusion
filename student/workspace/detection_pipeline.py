"""FPN-ResNet BEV object detection pipeline.

Provided Part C — pretrained FPN inference; do not edit for fusion submission.
"""

# vi: Phần cung cấp sẵn — detector có weight sẵn; không cần sửa để nộp fusion.

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from easydict import EasyDict as edict

from fusion_lab.workspace_support import third_party_root

_THIRD_PARTY = third_party_root()
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

# objdet_models live under third_party; import after sys.path setup.
from objdet_models.resnet.models import fpn_resnet
from objdet_models.resnet.utils.evaluation_utils import decode, post_processing
from objdet_models.resnet.utils.torch_utils import _sigmoid


def _default_weights_path() -> Path:
    """Return default pretrained weights path under platform third_party."""
    return (
        _THIRD_PARTY
        / "objdet_models"
        / "resnet"
        / "pretrained"
        / "fpn_resnet_18_epoch_300.pth"
    )


def load_fpn_resnet_config(weights_path: Optional[str | Path] = None) -> Any:
    """Load FPN-ResNet and BEV detection configuration.

    Args:
        weights_path: Optional path to `.pth` weights; defaults under third_party.

    Returns:
        EasyDict with BEV limits, model paths, and inference settings.
    """
    configs = edict()
    configs.lim_x = [0, 50]
    configs.lim_y = [-25, 25]
    configs.lim_z = [-1, 3]
    configs.lim_r = [0, 1.0]
    configs.bev_width = 608
    configs.bev_height = 608
    configs.arch = "fpn_resnet"
    configs.model_path = str(_THIRD_PARTY / "objdet_models" / "resnet")
    if weights_path is not None:
        weights = str(weights_path)
    else:
        weights = str(_default_weights_path())
    configs.pretrained_filename = weights
    configs.k = 50
    configs.conf_thresh = 0.5
    configs.batch_size = 1
    configs.down_ratio = 4
    configs.imagenet_pretrained = False
    configs.head_conv = 64
    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2
    configs.heads = {
        "hm_cen": configs.num_classes,
        "cen_offset": configs.num_center_offset,
        "direction": configs.num_direction,
        "z_coor": configs.num_z,
        "dim": configs.num_dim,
    }
    configs.no_cuda = True
    configs.gpu_idx = 0
    configs.device = torch.device(
        "cpu" if configs.no_cuda else "cuda:{}".format(configs.gpu_idx)
    )
    configs.obj_colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]
    return configs


def create_fpn_model(configs: Any, weights_path: Optional[str | Path] = None) -> Any:
    """Instantiate FPN-ResNet-18 and load pretrained weights.

    Args:
        configs: Detection config from ``load_fpn_resnet_config``.
        weights_path: Optional override for weight file path.

    Returns:
        Eval-mode PyTorch model on ``configs.device``.

    Raises:
        FileNotFoundError: If the weights file is missing.
    """
    if weights_path is not None:
        configs.pretrained_filename = str(weights_path)
    weights_file = configs.pretrained_filename
    if not os.path.isfile(weights_file):
        raise FileNotFoundError("No file at {}".format(weights_file))
    model = fpn_resnet.get_pose_net(
        num_layers=18,
        heads=configs.heads,
        head_conv=configs.head_conv,
        imagenet_pretrained=configs.imagenet_pretrained,
    )
    model.load_state_dict(torch.load(weights_file, map_location="cpu"))
    model = model.to(device=configs.device)
    model.eval()
    return model


def detect_objects_from_bev(
    input_bev_maps: Any, model: Any, configs: Any
) -> list[list[float]]:
    """Run inference on BEV maps and return metric 3D box detections.

    Args:
        input_bev_maps: BEV tensor ``(1, 3, H, W)``.
        model: FPN-ResNet from ``create_fpn_model``.
        configs: Detection config.

    Returns:
        List of detections ``[class_id, x, y, z, h, w, l, yaw]`` in vehicle frame.
    """
    objects = []
    with torch.no_grad():
        outputs = model(input_bev_maps)
        outputs["hm_cen"] = _sigmoid(outputs["hm_cen"])
        outputs["cen_offset"] = _sigmoid(outputs["cen_offset"])
        detections = decode(
            outputs["hm_cen"],
            outputs["cen_offset"],
            outputs["direction"],
            outputs["z_coor"],
            outputs["dim"],
            K=40,
        )
        detections = detections.cpu().numpy().astype(np.float32)
        detections = post_processing(detections, configs)
        if not detections or 1 not in detections[0]:
            return objects
        car_detections = detections[0][1]
        for obj in car_detections:
            _id, bev_x, bev_y, z, h, bev_w, bev_l, yaw = obj
            x = bev_y / configs.bev_height * (configs.lim_x[1] - configs.lim_x[0])
            y = (
                bev_x / configs.bev_width * (configs.lim_y[1] - configs.lim_y[0])
                - (configs.lim_y[1] - configs.lim_y[0]) / 2.0
            )
            w = bev_w / configs.bev_width * (configs.lim_y[1] - configs.lim_y[0])
            l = bev_l / configs.bev_height * (configs.lim_x[1] - configs.lim_x[0])
            if (
                configs.lim_x[0] <= x <= configs.lim_x[1]
                and configs.lim_y[0] <= y <= configs.lim_y[1]
                and configs.lim_z[0] <= z <= configs.lim_z[1]
            ):
                objects.append([1, x, y, z, h, w, l, yaw])
    return objects
