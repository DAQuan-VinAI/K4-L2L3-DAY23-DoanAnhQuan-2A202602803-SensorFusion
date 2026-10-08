"""Compare mode preserves mode artifacts and validates shared frame evidence."""

import json
from pathlib import Path

import pytest

from fusion_lab.evaluation import aggregate_records, record_json
from fusion_lab.scripts import run_lab


def _record(mode, rmse):
    return dict(mode=mode, frame=0, det_tp=3, det_fp=1, det_fn=0, valid_gt=3,
                confirmed=2, matches=2, sum_sq_err=2 * rmse**2, ghosts=0, misses=1)


def _fake_metrics(mode, rmse):
    return aggregate_records([_record(mode, rmse)], mode, [0, 0], 42, "test.tfrecord")


def test_merge_compare_metrics_keeps_both_rmse_values():
    lidar = _fake_metrics("lidar", 1.5)
    fused = _fake_metrics("fused", 1.2)
    merged = run_lab.merge_compare_metrics(lidar, fused)
    assert merged["tracking"]["lidar"]["rmse"] == 1.5
    assert merged["tracking"]["fused"]["rmse"] == 1.2
    assert merged["fusion_mode"] == "compare"
    assert fused["tracking"]["lidar"] is None


@pytest.mark.parametrize("key,value", [("seed", 1), ("segment", "other"),
    ("frames", [0, 4]), ("detection", {})])
def test_merge_rejects_incompatible_runs(key, value):
    fused = _fake_metrics("fused", 1.2)
    fused[key] = value
    with pytest.raises(ValueError, match="mismatch"):
        run_lab.merge_compare_metrics(_fake_metrics("lidar", 1.5), fused)


def test_compare_writes_merged_metrics_and_both_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("DAY23_STUDENT_ROOT", str(tmp_path))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    real_run = run_lab.run
    seeds = []

    def fake_run(config_path, fusion_mode="fused", max_frames=None, artifact_suffix="", seed=0):
        if fusion_mode == "compare":
            return real_run(config_path, fusion_mode, max_frames, artifact_suffix, seed)
        seeds.append(seed)
        rmse = 1.5 if fusion_mode == "lidar" else 1.2
        metrics = _fake_metrics(fusion_mode, rmse)
        (artifacts / f"metrics{artifact_suffix}.json").write_text(json.dumps(metrics))
        (artifacts / f"grade_run{artifact_suffix}.log").write_text(
            record_json(_record(fusion_mode, rmse)) + "\n")
        return metrics

    monkeypatch.setattr(run_lab, "run", fake_run)
    run_lab.main(["--config", str(Path("unused.yaml")), "--fusion", "compare", "--seed", "42"])
    assert seeds == [42, 42]
    merged = json.loads((artifacts / "metrics.json").read_text())
    assert merged["tracking"]["lidar"]["rmse"] == 1.5
    assert merged["tracking"]["fused"]["rmse"] == 1.2
    records = [json.loads(line) for line in (artifacts / "grade_run.log").read_text().splitlines()]
    assert [r["mode"] for r in records] == ["lidar", "fused"]
    for mode in ("lidar", "fused"):
        assert json.loads((artifacts / f"metrics_{mode}.json").read_text())["fusion_mode"] == mode
        assert json.loads((artifacts / f"grade_run_{mode}.log").read_text())["mode"] == mode


def test_failed_compare_leaves_no_stale_merged_report(tmp_path, monkeypatch):
    monkeypatch.setenv("DAY23_STUDENT_ROOT", str(tmp_path))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "metrics.json").write_text("{}")
    (artifacts / "grade_run.log").write_text("{}\n")

    def fake_run(config_path, fusion_mode="fused", max_frames=None, artifact_suffix="", seed=0):
        if fusion_mode == "fused":
            raise RuntimeError("fused run failed")
        return {}

    monkeypatch.setattr(run_lab, "run", fake_run)
    with pytest.raises(RuntimeError, match="fused run failed"):
        run_lab.run_compare(Path("unused.yaml"))
    assert not (artifacts / "metrics.json").exists()
    assert not (artifacts / "grade_run.log").exists()
