"""Pure frame evaluation and the shared runner/grader JSON Lines contract."""

from __future__ import annotations

import json
import math
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

TRACK_GATE_METERS = 2.0
RECORD_FIELDS = (
    "mode", "frame", "det_tp", "det_fp", "det_fn", "valid_gt", "confirmed",
    "matches", "sum_sq_err", "ghosts", "misses",
)


def valid_ground_truth(labels: Sequence[Any], config: Any, vehicle_type: int = 1):
    """Select vehicle labels whose finite centres lie inside the detector window."""
    selected = []
    for label in labels:
        if label.type != vehicle_type:
            continue
        centre = (label.box.center_x, label.box.center_y, label.box.center_z)
        if all(
            math.isfinite(value) and bounds[0] <= value <= bounds[1]
            for value, bounds in zip(
                centre, (config.lim_x, config.lim_y, getattr(config, "lim_z", (-math.inf, math.inf)))
            )
        ):
            selected.append(label)
    return selected


def _partial_assignment(costs: np.ndarray, eligible: np.ndarray):
    """Maximise gated pair count, then minimise cost using finite square padding."""
    n, m = costs.shape
    if not n or not m:
        return []
    # Every non-edge, including dummy-to-dummy, has the same finite cost.
    # Replacing it with a gated edge saves more than all gated costs combined.
    penalty = (n + m + 1) * (float(np.max(costs[eligible])) + 1) if eligible.any() else 1.0
    padded = np.full((n + m, n + m), penalty)
    padded[:n, :m] = np.where(eligible, costs, penalty)
    rows, cols = linear_sum_assignment(padded)
    return [(int(i), int(j)) for i, j in zip(rows, cols)
            if i < n and j < m and eligible[i, j]]


def detection_counts(labels: Sequence[Any], detections: Sequence[Any], helpers: Any):
    """Count one-to-one IoU matches using the provided detection helpers."""
    costs = np.zeros((len(labels), len(detections)))
    eligible = np.zeros_like(costs, dtype=bool)
    for i, label in enumerate(labels):
        for j, detection in enumerate(detections):
            matches = helpers.match_label_to_detections(label, [detection])
            if matches:
                eligible[i, j] = True
                costs[i, j] = 1.0 - matches[0][0]
    tp = len(_partial_assignment(costs, eligible))
    counts = helpers.precision_recall_counts([True] * len(labels), tp, len(detections))
    return {"det_tp": counts["true_positives"], "det_fp": counts["false_positives"],
            "det_fn": counts["false_negatives"]}


def tracking_counts(tracks: Sequence[Any], labels: Sequence[Any]):
    """Evaluate confirmed tracks at the documented 2 m xy gate; errors are 3-D."""
    confirmed = [track for track in tracks if track.state == "confirmed"]
    positions = np.array([np.asarray(t.x).reshape(-1)[:3] for t in confirmed]).reshape(-1, 3)
    centres = np.array([(l.box.center_x, l.box.center_y, l.box.center_z)
                        for l in labels]).reshape(-1, 3)
    delta = positions[:, None, :] - centres[None, :, :]
    distances = np.linalg.norm(delta[:, :, :2], axis=2)
    pairs = _partial_assignment(distances, np.isfinite(distances) & (distances <= TRACK_GATE_METERS))
    result = {"confirmed": len(confirmed), "matches": len(pairs),
              "sum_sq_err": float(sum(np.dot(delta[i, j], delta[i, j]) for i, j in pairs)),
              "ghosts": len(confirmed) - len(pairs), "misses": len(labels) - len(pairs)}
    assert result["matches"] + result["ghosts"] == result["confirmed"]
    assert result["matches"] + result["misses"] == len(labels)
    return result


def validate_record(record: dict[str, Any]) -> None:
    """Reject extra/missing fields, invalid counts and broken frame invariants."""
    if not isinstance(record, dict) or set(record) != set(RECORD_FIELDS):
        raise ValueError("frame record fields do not match RECORD_FIELDS")
    if record["mode"] not in ("lidar", "fused"):
        raise ValueError("invalid frame mode")
    for key in RECORD_FIELDS[1:]:
        value = record[key]
        if key == "sum_sq_err":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid squared error")
        elif type(value) is not int or value < 0:
            raise ValueError(f"invalid nonnegative integer: {key}")
    if record["matches"] + record["ghosts"] != record["confirmed"]:
        raise ValueError("confirmed invariant broken")
    if record["matches"] + record["misses"] != record["valid_gt"]:
        raise ValueError("ground truth invariant broken")
    if record["det_tp"] + record["det_fn"] != record["valid_gt"]:
        raise ValueError("detection ground truth invariant broken")
    if record["matches"] == 0 and record["sum_sq_err"] != 0:
        raise ValueError("squared error without matches")


def record_json(record: dict[str, Any]) -> str:
    """Serialize exactly one validated frame record (without its newline)."""
    validate_record(record)
    return json.dumps(record, allow_nan=False)


def read_records(path) -> list[dict[str, Any]]:
    """Read strict JSON Lines; blank lines and non-record headers are invalid."""
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        validate_record(record)
    return records


def aggregate_records(records: Iterable[dict[str, Any]], fusion_mode: str,
                      frames: Sequence[int], seed: int, segment: str) -> dict[str, Any]:
    """Recompute metrics; reject missing, duplicate or inconsistent mode frames."""
    if type(seed) is not int or seed < 0 or not isinstance(segment, str) or not segment:
        raise ValueError("seed must be nonnegative and segment must be nonempty")
    modes = ("lidar", "fused") if fusion_mode == "compare" else (fusion_mode,)
    if any(mode not in ("lidar", "fused") for mode in modes):
        raise ValueError("invalid fusion mode")
    if (len(frames) != 2 or any(type(value) is not int for value in frames)
            or frames[0] < 0 or frames[1] < frames[0]):
        raise ValueError("invalid inclusive frame range")
    by_key = {}
    for record in records:
        validate_record(record)
        key = (record["mode"], record["frame"])
        if key in by_key:
            raise ValueError("duplicate mode/frame record")
        by_key[key] = record
    expected = {(mode, frame) for mode in modes for frame in range(frames[0], frames[1] + 1)}
    if set(by_key) != expected:
        raise ValueError("missing or unexpected mode/frame records")
    if fusion_mode == "compare":
        for frame in range(frames[0], frames[1] + 1):
            if any(by_key[("lidar", frame)][k] != by_key[("fused", frame)][k]
                   for k in ("det_tp", "det_fp", "det_fn", "valid_gt")):
                raise ValueError("detection differs between modes")
    tracking = {"lidar": None, "fused": None}
    for mode in modes:
        group = [by_key[(mode, frame)] for frame in range(frames[0], frames[1] + 1)]
        total = {key: sum(r[key] for r in group) for key in RECORD_FIELDS[2:]}
        if not math.isfinite(total["sum_sq_err"]):
            raise ValueError("nonfinite aggregate squared error")
        tracking[mode] = {"rmse": math.sqrt(total["sum_sq_err"] / total["matches"]) if total["matches"] else None,
                          "matches": total["matches"], "sum_sq_err": total["sum_sq_err"],
                          "ghost_track_frames": total["ghosts"], "missed_gt_frames": total["misses"],
                          "mean_confirmed_tracks": total["confirmed"] / len(group)}
    # Detection is shared; count each frame once in compare mode.
    tp, fp, fn = (total[k] for k in ("det_tp", "det_fp", "det_fn"))
    return {"detection": {"tp": tp, "fp": fp, "fn": fn,
                          "precision": tp / (tp + fp) if tp + fp else 0.0,
                          "recall": tp / (tp + fn) if tp + fn else 0.0},
            "tracking": tracking, "fusion_mode": fusion_mode, "frames": list(frames),
            "seed": seed, "segment": segment}


def validate_metrics_records(metrics: dict[str, Any], records: Iterable[dict[str, Any]]) -> None:
    """Require the submitted metrics to equal their complete frame evidence."""
    if not isinstance(metrics, dict) or set(metrics) != {
        "detection", "tracking", "fusion_mode", "frames", "seed", "segment"
    }:
        raise ValueError("invalid metrics fields")
    detection = metrics["detection"]
    if not isinstance(detection, dict) or set(detection) != {"tp", "fp", "fn", "precision", "recall"}:
        raise ValueError("invalid detection fields")
    for key in ("tp", "fp", "fn"):
        if type(detection[key]) is not int or detection[key] < 0:
            raise ValueError("invalid detection count type")
    tracking = metrics["tracking"]
    if not isinstance(tracking, dict) or set(tracking) != {"lidar", "fused"}:
        raise ValueError("invalid tracking modes")
    for stats in tracking.values():
        if stats is None:
            continue
        if not isinstance(stats, dict) or set(stats) != {
            "rmse", "matches", "sum_sq_err", "ghost_track_frames",
            "missed_gt_frames", "mean_confirmed_tracks"
        }:
            raise ValueError("invalid tracking fields")
        for key in ("matches", "ghost_track_frames", "missed_gt_frames"):
            if type(stats[key]) is not int or stats[key] < 0:
                raise ValueError("invalid tracking count type")
        for key in ("rmse", "sum_sq_err", "mean_confirmed_tracks"):
            value = stats[key]
            if key == "rmse" and value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid tracking numeric type")
    for key in ("precision", "recall"):
        value = detection[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid detection ratio")
    expected = aggregate_records(records, metrics["fusion_mode"], metrics["frames"],
                                 metrics["seed"], metrics["segment"])
    if metrics != expected:
        raise ValueError("metrics are inconsistent with frame records")
