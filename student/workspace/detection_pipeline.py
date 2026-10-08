"""FPN-ResNet BEV object detection pipeline.

Provided Part C — pretrained FPN inference; do not edit for fusion submission.
"""

# vi: Phần cung cấp sẵn — detector có weight sẵn; không cần sửa để nộp fusion.

from __future__ import annotations

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
    # Spatial window and network tensor sizes are independent configuration
    # concerns; head names are the MIT model's public decoding contract.
    configs = edict({
        "bev_height": 608, "bev_width": 608, "down_ratio": 4,
        "lim_z": [-1, 3], "lim_r": [0, 1.0],
        "lim_y": [-25, 25], "lim_x": [0, 50],
        "pretrained_filename": str(weights_path if weights_path is not None else _default_weights_path()),
        "model_path": str(_THIRD_PARTY / "objdet_models" / "resnet"),
        "arch": "fpn_resnet", "head_conv": 64, "imagenet_pretrained": False,
        "num_classes": 3, "num_center_offset": 2, "num_z": 1,
        "num_dim": 3, "num_direction": 2,
        "conf_thresh": 0.5, "k": 50, "batch_size": 1,
        "gpu_idx": 0, "no_cuda": True, "device": torch.device("cpu"),
        "obj_colors": [[0, 255, 255], [0, 0, 255], [255, 0, 0]],
    })
    head_dimensions = {
        "dim": "num_dim", "z_coor": "num_z", "direction": "num_direction",
        "cen_offset": "num_center_offset", "hm_cen": "num_classes",
    }
    configs.heads = {head: configs[field] for head, field in head_dimensions.items()}
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
    checkpoint = Path(configs.pretrained_filename)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Detector checkpoint does not exist: {checkpoint}")
    network = fpn_resnet.get_pose_net(
        heads=configs.heads, num_layers=18,
        imagenet_pretrained=configs.imagenet_pretrained, head_conv=configs.head_conv,
    )
    parameters = torch.load(checkpoint, map_location="cpu", weights_only=True)
    network.load_state_dict(parameters)
    return network.to(device=configs.device).eval()


def detect_objects_from_bev(
    input_bev_maps: Any, model: Any, configs: Any
) -> list[list[float]]:
    """Run inference on BEV maps and return metric 3D box detections.

    Args:
        input_bev_maps: BEV tensor ``(1, 3, H, W)``.
        model: FPN-ResNet from ``create_fpn_model``.
        configs: Detection config.

    Returns:
        Detections ``[class_id, x, y, z, h, w, l, yaw]`` in vehicle frame.
        XYZ denotes the box centre; length is along yaw, width perpendicular.
        SFA3D predicts bottom height relative to ``lim_z[0]`` and BEV yaw,
        which are converted here to centre height and vehicle-frame heading.
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
            K=configs.k,
        )
        detections = detections.cpu().numpy().astype(np.float32)
        detections = post_processing(detections, configs)
        if not detections or 1 not in detections[0]:
            return objects
        car_detections = detections[0][1]
        for obj in car_detections:
            _score, bev_x, bev_y, bottom_offset, h, bev_w, bev_l, bev_yaw = obj
            x = (
                bev_y / configs.bev_height * (configs.lim_x[1] - configs.lim_x[0])
                + configs.lim_x[0]
            )
            y = (
                bev_x / configs.bev_width * (configs.lim_y[1] - configs.lim_y[0])
                + configs.lim_y[0]
            )
            z = bottom_offset + configs.lim_z[0] + h / 2.0
            yaw = -bev_yaw
            w = bev_w / configs.bev_width * (configs.lim_y[1] - configs.lim_y[0])
            l = bev_l / configs.bev_height * (configs.lim_x[1] - configs.lim_x[0])
            if (
                configs.lim_x[0] <= x <= configs.lim_x[1]
                and configs.lim_y[0] <= y <= configs.lim_y[1]
                and configs.lim_z[0] <= z <= configs.lim_z[1]
            ):
                objects.append([1, x, y, z, h, w, l, yaw])
    return objects
