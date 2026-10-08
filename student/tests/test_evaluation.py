"""Deterministic matching and strict frame evidence regression tests."""

import copy
from types import SimpleNamespace as NS

import numpy as np
import pytest

from fusion_lab.evaluation import (
    RECORD_FIELDS,
    aggregate_records,
    detection_counts,
    record_json,
    tracking_counts,
    valid_ground_truth,
    validate_metrics_records,
)


def label(x=0, y=0, z=0, type_=1):
    return NS(type=type_, box=NS(center_x=x, center_y=y, center_z=z,
                               width=2, length=4, heading=0))


def track(x=0, y=0, z=0, state="confirmed"):
    return NS(x=np.array([x, y, z, 0, 0, 0]).reshape(6, 1), state=state)


@pytest.mark.parametrize("tracks,labels,matches,ghosts,misses", [
    ([track(), track(0, 1)], [label(), label(2, 0)], 2, 0, 0),
    ([track(10)], [label()], 0, 1, 1),
    ([track(), track(), track()], [label()], 1, 2, 0),
    ([track(0, 0.5)], [label(), label(0, 1)], 1, 0, 1),
    ([], [label()], 0, 0, 1), ([track()], [], 0, 1, 0), ([], [], 0, 0, 0),
    ([track(state="tentative")], [label()], 0, 0, 1),
])
def test_partial_matching_invariants(tracks, labels, matches, ghosts, misses):
    counts = tracking_counts(tracks, labels)
    assert counts["matches"] == matches
    assert counts["ghosts"] == ghosts
    assert counts["misses"] == misses
    assert counts["matches"] + counts["ghosts"] == counts["confirmed"]
    assert counts["matches"] + counts["misses"] == len(labels)


def test_matching_minimises_distance_and_uses_3d_error():
    counts = tracking_counts([track(0, 0, 3), track(1)], [label(0.1), label(1.1)])
    assert counts["sum_sq_err"] == pytest.approx(9.02)
    assert counts == tracking_counts([track(0, 0, 3), track(1)], [label(0.1), label(1.1)])


def test_valid_gt_window_type_and_optional_z():
    cfg = NS(lim_x=[0, 50], lim_y=[-25, 25], lim_z=[-1, 3])
    labels = [label(), label(-1), label(50, 25, 3), label(z=4), label(type_=2), label(x=np.nan)]
    assert valid_ground_truth(labels, cfg) == [labels[0], labels[2]]
    del cfg.lim_z
    assert valid_ground_truth([label(z=4)], cfg)


def test_detection_one_to_one(workspace_modules):
    helpers = workspace_modules["detection_metrics"]
    detection = [1, 0, 0, 0, 1, 2, 4, 0]
    assert detection_counts([label(), label()], [detection], helpers) == {
        "det_tp": 1, "det_fp": 0, "det_fn": 1}
    assert detection_counts([label()], [detection, detection], helpers) == {
        "det_tp": 1, "det_fp": 1, "det_fn": 0}


def record(mode="lidar", frame=0):
    return dict(mode=mode, frame=frame, det_tp=1, det_fp=1, det_fn=1, valid_gt=2,
                confirmed=2, matches=1, sum_sq_err=0.25, ghosts=1, misses=1)


def test_aggregates_recomputed_exactly():
    records = [record(m, f) for m in ("lidar", "fused") for f in (0, 1)]
    metrics = aggregate_records(records, "compare", [0, 1], 0, "segment")
    assert metrics["detection"] == {"tp": 2, "fp": 2, "fn": 2, "precision": .5, "recall": .5}
    assert metrics["tracking"]["fused"] == dict(rmse=.5, matches=2, sum_sq_err=.5,
        ghost_track_frames=2, missed_gt_frames=2, mean_confirmed_tracks=2)
    validate_metrics_records(metrics, records)
    metrics["tracking"]["fused"]["rmse"] = .1
    with pytest.raises(ValueError, match="inconsistent"):
        validate_metrics_records(metrics, records)


@pytest.mark.parametrize("records", [[], [record(), record()],
    [record("lidar"), {**record("fused"), "det_fp": 2}]])
def test_missing_duplicate_inconsistent_records(records):
    with pytest.raises(ValueError):
        aggregate_records(records, "compare", [0, 0], 0, "segment")


@pytest.mark.parametrize("key,value", [("confirmed", 0), ("valid_gt", 0),
    ("det_tp", 2), ("matches", -1), ("sum_sq_err", float("nan")), ("frame", True)])
def test_invalid_frame_records(key, value):
    r = copy.deepcopy(record())
    r[key] = value
    with pytest.raises(ValueError):
        record_json(r)


def test_exact_schema_and_no_match_rmse():
    r = record()
    assert set(r) == set(RECORD_FIELDS)
    r.update(matches=0, sum_sq_err=0, ghosts=2, misses=2)
    metrics = aggregate_records([r], "lidar", [0, 0], 0, "s")
    assert metrics["tracking"]["lidar"]["rmse"] is None
    assert metrics["tracking"]["fused"] is None
    with pytest.raises(ValueError):
        record_json({**r, "extra": 0})


@pytest.mark.parametrize("target,key", [("detection", "tp"), ("tracking", "matches")])
def test_metrics_boolean_counts_are_rejected(target, key):
    records = [record()]
    metrics = aggregate_records(records, "lidar", [0, 0], 0, "segment")
    stats = metrics["detection"] if target == "detection" else metrics["tracking"]["lidar"]
    stats[key] = True  # Numerically equals 1 but violates the count type contract.
    with pytest.raises(ValueError, match="count type"):
        validate_metrics_records(metrics, records)
