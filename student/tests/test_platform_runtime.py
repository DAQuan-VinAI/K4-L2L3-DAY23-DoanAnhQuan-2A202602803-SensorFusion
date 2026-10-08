"""Check the relocated configuration and Apache reader adapter without dataset files."""

import zlib
from pathlib import Path

import numpy as np
from simple_waymo_open_dataset_reader import dataset_pb2

from fusion_lab.lidar_pcl import pcl_from_range_image
from fusion_lab.scripts.run_lab import _load_paths_config


def test_example_paths_resolve_from_repo_root(workspace_modules, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    monkeypatch.setenv("DAY23_STUDENT_ROOT", str(root / "student"))
    cfg = _load_paths_config(root / "student/config/paths.example.yaml")
    assert Path(cfg["waymo_dir"]) == root / "data/Waymo"
    assert Path(cfg["weights_dir"]) == root / "data/weights"


def test_point_cloud_keeps_positive_ranges_and_intensities():
    frame = dataset_pb2.Frame()
    frame.pose.transform.extend(np.eye(4).ravel())
    laser = frame.lasers.add(name=dataset_pb2.LaserName.FRONT)
    calibration = frame.context.laser_calibrations.add(name=laser.name)
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    calibration.extrinsic.transform.extend(transform.ravel())
    calibration.beam_inclinations.extend([0.0, 0.0])
    ri = np.array([
        [[1, 0.2, 0, 0], [0, 0.3, 0, 0], [2, 0.4, 0, 0]],
        [[-1, 0.5, 0, 0], [3, 0.6, 0, 0], [0, 0.7, 0, 0]],
    ], dtype=np.float32)
    matrix = dataset_pb2.MatrixFloat()
    matrix.shape.dims.extend(ri.shape)
    matrix.data.extend(ri.ravel())
    laser.ri_return1.range_image_compressed = zlib.compress(matrix.SerializeToString())
    projection = dataset_pb2.MatrixInt32()
    projection.shape.dims.extend([2, 3, 6])
    projection.data.extend([0] * 36)
    laser.ri_return1.camera_projection_compressed = zlib.compress(
        projection.SerializeToString()
    )
    points = pcl_from_range_image(frame, laser.name)
    assert points.shape == (3, 4)
    np.testing.assert_allclose(
        points[:, :3], [[0, 2, 3], [-1, 2, 3], [4, 2, 3]], atol=1e-7
    )
    np.testing.assert_allclose(points[:, 3], ri[ri[:, :, 0] > 0][:, 1])


def test_empty_yaml_has_clear_error(tmp_path):
    import pytest

    config = tmp_path / "empty.yaml"
    config.write_text("")
    with pytest.raises(ValueError, match="non-empty YAML mapping"):
        _load_paths_config(config)


def test_zipped_weights_extract_to_writable_artifacts(tmp_path, monkeypatch):
    import zipfile

    from fusion_lab.scripts.run_lab import _resolve_weights

    weights = tmp_path / "readonly-data"
    weights.mkdir()
    with zipfile.ZipFile(weights / "pretrained_fpn-resnet.zip", "w") as archive:
        archive.writestr("model/fpn_resnet_18_epoch_300.pth", b"fixture")
    monkeypatch.setenv("DAY23_STUDENT_ROOT", str(tmp_path / "student"))
    weights.chmod(0o555)
    try:
        result = _resolve_weights({"weights_dir": str(weights)})
        assert result.is_relative_to(tmp_path / "student/artifacts/weights-cache")
        assert result.read_bytes() == b"fixture"
        assert list(weights.iterdir()) == [weights / "pretrained_fpn-resnet.zip"]
    finally:
        weights.chmod(0o755)


def test_front_group_selected_by_identity_and_seed():
    from types import SimpleNamespace as NS

    from fusion_lab.scripts.run_lab import _front_observations

    class Sensor:
        def generate_measurement(self, frame, z, observations):
            observations.append((frame, np.asarray(z)))

    front = NS(name=1, labels=[NS(type=1, box=NS(center_x=100, center_y=200))])
    other = NS(name=2, labels=[NS(type=1, box=NS(center_x=-100, center_y=-200))])
    frame = NS(camera_labels=[other, front])
    first = _front_observations(frame, 0, Sensor(), np.random.default_rng(7), 1, 1)
    second = _front_observations(frame, 0, Sensor(), np.random.default_rng(7), 1, 1)
    np.testing.assert_array_equal(first[0][1], second[0][1])
    np.testing.assert_allclose(first[0][1], [100, 200], atol=1)
    assert first[0][0] == 0
    assert _front_observations(NS(camera_labels=[other]), 0, Sensor(), np.random.default_rng(0), 1, 1) is None
