"""Waymo end-to-end lab runner (detection + tracking + optional fusion compare)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from fusion_lab.evaluation import (
    aggregate_records, detection_counts, read_records, record_json,
    tracking_counts, valid_ground_truth, validate_metrics_records,
)
from fusion_lab import tracking_params
from fusion_lab.paths import (
    PLATFORM_ROOT,
    THIRD_PARTY,
    ensure_platform_on_path,
    student_root,
)
from fusion_lab.tracking.filter import Filter
from fusion_lab.tracking.manager import TrackManager
from fusion_lab.tracking.sensors import Sensor
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
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError(f"paths config must be a non-empty YAML mapping: {path}")
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

        extract_to = student_root() / "artifacts" / "weights-cache"
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_to)
        for p in extract_to.rglob("*.pth"):
            return p
    return None


def _lidar_observations(frame_index, detections, sensor, config):
    """Build vehicle-frame centre observations within the configured xy window."""
    observations = []
    for detection in detections:
        coordinates = detection[1:3]
        if all(low <= value <= high for value, (low, high) in
               zip(coordinates, (config.lim_x, config.lim_y))):
            sensor.generate_measurement(frame_index, detection[1:], observations)
    return observations


def _front_observations(frame, frame_index, sensor, rng, camera_name, vehicle_type):
    """Build noisy GT-box pixels; None means no FRONT data, [] means no vehicles."""
    group = next((group for group in frame.camera_labels if group.name == camera_name), None)
    if group is None:
        return None
    observations = []
    for label in group.labels:
        if label.type == vehicle_type:
            centre = np.array([label.box.center_x, label.box.center_y])
            sensor.generate_measurement(frame_index, centre + rng.normal(0, 0.5, 2), observations)
    return observations


def run(
    config_path: Path,
    fusion_mode: str = "fused",
    max_frames: int | None = None,
    artifact_suffix: str = "",
    seed: int = 0,
) -> dict[str, Any]:
    """Run detection and tracking over a Waymo segment; write metrics and log.

    Args:
        config_path: Path to ``paths.yaml``.
        fusion_mode: ``lidar``, ``fused`` or ``compare``. ``compare`` runs both
            single-sensor modes and writes one merged ``metrics.json``.
        max_frames: Optional cap on the number of processed frames.
        seed: Reproducible camera-noise seed (default zero).
        artifact_suffix: Suffix for ``metrics*.json`` and ``grade_run*.log`` so
            several runs can share one artifacts directory.

    Returns:
        The metrics dictionary written to ``artifacts/metrics<suffix>.json``.

    Raises:
        ValueError: If ``fusion_mode`` is not a supported mode.
    """
    if fusion_mode == "compare":
        return run_compare(config_path, max_frames, seed)
    if fusion_mode not in ("lidar", "fused"):
        raise ValueError(f"unsupported fusion_mode: {fusion_mode!r}")
    _setup_import_paths()
    cfg = _load_paths_config(config_path)
    rng = np.random.default_rng(seed)
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

    if frame_start < 0 or frame_end < frame_start:
        raise ValueError("frame range and max_frames must select at least one frame")

    weights = _resolve_weights(cfg)
    det_cfg = det_pipe.load_fpn_resnet_config(str(weights) if weights else None)
    model = det_pipe.create_fpn_model(det_cfg, str(weights) if weights else None)

    artifacts = student_root() / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    log_path = artifacts / f"grade_run{artifact_suffix}.log"
    metrics_path = artifacts / f"metrics{artifact_suffix}.json"

    reader = WaymoDataFileReader(str(tfrecord))
    data_iter = iter(reader)

    KF = Filter(kalman)
    manager = TrackManager(ws["track_management"])
    lidar_sensor = None
    camera_sensor = None

    records = []
    # A crashed run must not leave the previous run's metrics next to a fresh log.
    metrics_path.unlink(missing_ok=True)
    with log_path.open("w") as log:
        for cnt, frame in enumerate(data_iter):
            if cnt < frame_start:
                continue
            if cnt > frame_end:
                break
            # Calibrations are keyed identities; serialized order is irrelevant.
            if lidar_sensor is None:
                lidar_sensor = Sensor(
                    "lidar", waymo_utils.get(frame.context.laser_calibrations, dataset_pb2.LaserName.TOP), cam
                )
            if fusion_mode == "fused" and camera_sensor is None:
                camera_sensor = Sensor(
                    "camera", waymo_utils.get(frame.context.camera_calibrations, dataset_pb2.CameraName.FRONT), cam
                )
            points = pcl_from_range_image(frame, dataset_pb2.LaserName.TOP)
            tensor = torch.from_numpy(bev.bev_maps_from_pcl(points, det_cfg)).unsqueeze(0).float()
            detections = det_pipe.detect_objects_from_bev(tensor, model, det_cfg)
            labels = valid_ground_truth(
                frame.laser_labels, det_cfg, label_pb2.Label.Type.TYPE_VEHICLE
            )
            observations = _lidar_observations(cnt, detections, lidar_sensor, det_cfg)
            for track in manager.track_list:
                KF.predict(track)
                track.set_t(cnt * tracking_params.dt)
            assoc.associate_and_update(manager, observations, KF, lidar_sensor)
            if camera_sensor is not None:
                observations = _front_observations(
                    frame, cnt, camera_sensor, rng,
                    dataset_pb2.CameraName.FRONT, label_pb2.Label.Type.TYPE_VEHICLE,
                )
                if observations is not None:
                    assoc.associate_and_update(manager, observations, KF, camera_sensor)
            record = {
                "mode": fusion_mode, "frame": cnt,
                **detection_counts(labels, detections, det_metrics),
                "valid_gt": len(labels), **tracking_counts(manager.track_list, labels),
            }
            log.write(record_json(record) + "\n")
            records.append(record)

    if not records:
        raise ValueError("segment contains no frames in the requested range")
    metrics = aggregate_records(records, fusion_mode, [frame_start, records[-1]["frame"]],
                                seed, tfrecord.name)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


def merge_compare_metrics(
    lidar: dict[str, Any], fused: dict[str, Any]
) -> dict[str, Any]:
    """Merge compatible single-mode reports while preserving each tracking result."""
    for key in ("detection", "frames", "seed", "segment"):
        if lidar[key] != fused[key]:
            raise ValueError(f"compare run mismatch: {key}")
    if lidar["fusion_mode"] != "lidar" or fused["fusion_mode"] != "fused":
        raise ValueError("compare requires lidar and fused single-mode reports")
    return {**fused, "tracking": {"lidar": lidar["tracking"]["lidar"],
                                  "fused": fused["tracking"]["fused"]},
            "fusion_mode": "compare"}


def run_compare(config_path: Path, max_frames: int | None = None, seed: int = 0) -> dict[str, Any]:
    """Run lidar-only and fused tracking, then write one merged report.

    Per-mode outputs stay in ``metrics_{lidar,fused}.json`` and
    ``grade_run_{lidar,fused}.log``; ``metrics.json`` and ``grade_run.log`` hold
    the merged compare result read by the grader.

    Args:
        config_path: Path to ``paths.yaml``.
        max_frames: Optional cap on the number of processed frames.
        seed: Shared camera-noise seed.

    Returns:
        The merged compare-mode metrics dictionary.
    """
    artifacts = student_root() / "artifacts"
    # Remove the previous merged report first: if either mode fails, no stale pair survives.
    for name in ("metrics.json", "grade_run.log"):
        (artifacts / name).unlink(missing_ok=True)
    lidar = run(config_path, "lidar", max_frames, artifact_suffix="_lidar", seed=seed)
    fused = run(config_path, "fused", max_frames, artifact_suffix="_fused", seed=seed)
    merged = merge_compare_metrics(lidar, fused)

    logs = []
    for mode in ("lidar", "fused"):
        logs.append((artifacts / f"grade_run_{mode}.log").read_text())
    (artifacts / "grade_run.log").write_text("".join(logs))
    validate_metrics_records(merged, read_records(artifacts / "grade_run.log"))
    (artifacts / "metrics.json").write_text(json.dumps(merged, indent=2))
    print(json.dumps(merged, indent=2))
    return merged


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
    parser.add_argument("--seed", type=int, default=0, help="Camera-noise random seed")
    args = parser.parse_args(argv)
    run(args.config, args.fusion, args.max_frames, seed=args.seed)


if __name__ == "__main__":
    main()
