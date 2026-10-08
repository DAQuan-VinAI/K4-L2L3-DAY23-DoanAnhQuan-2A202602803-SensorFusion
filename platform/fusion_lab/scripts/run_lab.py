"""Waymo end-to-end lab runner (detection + tracking + optional fusion compare)."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from fusion_lab.paths import (
    PLATFORM_ROOT,
    THIRD_PARTY,
    ensure_platform_on_path,
    student_root,
)
from fusion_lab.tracking.filter import Filter
from fusion_lab.tracking.manager import TrackManager
from fusion_lab.tracking.sensors import Sensor
from fusion_lab.viz.display import VizConfig
from fusion_lab.workspace_loader import load_workspace


def _setup_import_paths() -> None:
    ensure_platform_on_path()
    waymo_pkg = THIRD_PARTY / "waymo_reader"
    if str(waymo_pkg) not in sys.path:
        sys.path.insert(0, str(waymo_pkg))
    os.environ.setdefault("FUSION_LAB_PLATFORM", str(PLATFORM_ROOT))


def _load_paths_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    for key in ("waymo_dir", "weights_dir"):
        if cfg.get(key) and not Path(cfg[key]).is_absolute():
            cfg[key] = str((student_root() / cfg[key]).resolve())
    return cfg


def _resolve_weights(cfg: dict[str, Any]) -> Path | None:
    weights_dir = cfg.get("weights_dir")
    if not weights_dir:
        return None
    base = Path(weights_dir)
    for name in (
        "fpn_resnet_18_epoch_300.pth",
        "pretrained_fpn-resnet/fpn_resnet_18_epoch_300.pth",
    ):
        candidate = base / name
        if candidate.is_file():
            return candidate
    zip_path = base / "pretrained_fpn-resnet.zip"
    if zip_path.is_file():
        import zipfile

        extract_to = base / "pretrained_fpn-resnet"
        extract_to.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_to)
        for p in extract_to.rglob("*.pth"):
            return p
    return None


def run(
    config_path: Path,
    fusion_mode: str = "fused",
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Run detection and tracking over a Waymo segment; write metrics and log."""
    _setup_import_paths()
    cfg = _load_paths_config(config_path)
    ws = load_workspace()
    kalman = ws["kalman"]
    assoc = ws["association"]
    cam = ws["camera_fusion"]
    bev = ws["bev_mapping"]
    det_pipe = ws["detection_pipeline"]
    det_metrics = ws["detection_metrics"]

    from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
    from simple_waymo_open_dataset_reader import utils as waymo_utils

    from fusion_lab.lidar_pcl import pcl_from_range_image

    tfrecord = Path(cfg["waymo_dir"]) / cfg.get(
        "segment",
        "training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord",
    )
    frame_start = int(cfg.get("frame_start", 0))
    frame_end = int(cfg.get("frame_end", 20))
    if max_frames is not None:
        frame_end = min(frame_end, frame_start + max_frames - 1)

    weights = _resolve_weights(cfg)
    det_cfg = det_pipe.load_fpn_resnet_config(str(weights) if weights else None)
    model = det_pipe.create_fpn_model(det_cfg, str(weights) if weights else None)

    artifacts = student_root() / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    log_path = artifacts / "grade_run.log"
    metrics_path = artifacts / "metrics.json"

    viz = VizConfig.from_env()
    if viz.save_dir is None:
        viz.save_dir = artifacts / "viz"

    reader = WaymoDataFileReader(str(tfrecord))
    data_iter = iter(reader)

    KF = Filter(kalman)
    manager = TrackManager(ws["track_management"])
    lidar_sensor = None
    camera_sensor = None

    det_tp = det_fp = 0
    det_fn = 0
    tracking_rmse_lidar = []
    tracking_rmse_fused = []

    cnt = 0
    with log_path.open("w") as log:
        log.write(f"segment={tfrecord.name} frames={frame_start}-{frame_end} fusion={fusion_mode}\n")

        while True:
            try:
                frame = next(data_iter)
            except StopIteration:
                break
            if cnt < frame_start:
                cnt += 1
                continue
            if cnt > frame_end:
                break

            lidar_name = dataset_pb2.LaserName.TOP
            camera_name = dataset_pb2.CameraName.FRONT
            lidar_calib = waymo_utils.get(frame.context.laser_calibrations, lidar_name)
            camera_calib = waymo_utils.get(frame.context.camera_calibrations, camera_name)

            lidar_pcl = pcl_from_range_image(frame, lidar_name)
            bev_maps = bev.bev_maps_from_pcl(lidar_pcl, det_cfg)
            tensor = torch.from_numpy(bev_maps).unsqueeze(0).float()
            detections = det_pipe.detect_objects_from_bev(tensor, model, det_cfg)

            labels = frame.laser_labels
            valid = [lbl.type == label_pb2.Label.Type.TYPE_VEHICLE for lbl in labels]
            for lbl, ok in zip(labels, valid):
                if not ok:
                    continue
                matches = det_metrics.match_label_to_detections(lbl, detections)
                if matches:
                    det_tp += 1
                else:
                    det_fn += 1
            det_fp += max(0, len(detections) - det_tp)

            if fusion_mode in ("fused", "lidar", "compare"):
                if lidar_sensor is None:
                    lidar_sensor = Sensor("lidar", lidar_calib, cam)
                if camera_sensor is None:
                    camera_sensor = Sensor("camera", camera_calib, cam)

                meas_lidar = []
                for det in detections:
                    if (
                        det_cfg.lim_x[0] < det[1] < det_cfg.lim_x[1]
                        and det_cfg.lim_y[0] < det[2] < det_cfg.lim_y[1]
                    ):
                        lidar_sensor.generate_measurement(cnt, det[1:], meas_lidar)

                for track in manager.track_list:
                    KF.predict(track)
                    track.set_t((cnt - 1) * 0.1)

                assoc.associate_and_update(manager, meas_lidar, KF, cam)

                if fusion_mode in ("fused", "compare"):
                    meas_cam = []
                    if frame.camera_labels:
                        for label in frame.camera_labels[0].labels:
                            if label.type != label_pb2.Label.Type.TYPE_VEHICLE:
                                continue
                            box = label.box
                            z = [
                                box.center_x + np.random.normal(0, 0.5),
                                box.center_y + np.random.normal(0, 0.5),
                            ]
                            camera_sensor.generate_measurement(cnt, z, meas_cam)
                    assoc.associate_and_update(manager, meas_cam, KF, cam)

                if labels and valid:
                    ref = labels[0]
                    if manager.track_list:
                        tr = manager.track_list[0]
                        err = np.sqrt(
                            (float(tr.x[0, 0]) - ref.box.center_x) ** 2
                            + (float(tr.x[1, 0]) - ref.box.center_y) ** 2
                        )
                        if fusion_mode == "lidar":
                            tracking_rmse_lidar.append(err)
                        else:
                            tracking_rmse_fused.append(err)

            log.write(
                f"frame={cnt} dets={len(detections)} "
                f"tracks={len(manager.track_list)}\n"
            )
            cnt += 1

    precision = det_tp / (det_tp + det_fp) if (det_tp + det_fp) else 0.0
    recall = det_tp / (det_tp + det_fn) if (det_tp + det_fn) else 0.0
    metrics = {
        "detection": {
            "tp": det_tp,
            "fp": det_fp,
            "fn": det_fn,
            "precision": precision,
            "recall": recall,
        },
        "tracking": {
            "rmse_lidar_only_mean": (
                float(np.mean(tracking_rmse_lidar)) if tracking_rmse_lidar else None
            ),
            "rmse_fused_mean": (
                float(np.mean(tracking_rmse_fused)) if tracking_rmse_fused else None
            ),
        },
        "fusion_mode": fusion_mode,
        "frames": [frame_start, frame_end],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Day 23 sensor fusion lab runner")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="paths.yaml or paths.example.yaml",
    )
    parser.add_argument(
        "--fusion",
        choices=["lidar", "fused", "compare"],
        default="fused",
        help="Tracking fusion mode",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)
    if args.fusion == "compare":
        run(args.config, "lidar", args.max_frames)
        run(args.config, "fused", args.max_frames)
    else:
        run(args.config, args.fusion, args.max_frames)


if __name__ == "__main__":
    main()
