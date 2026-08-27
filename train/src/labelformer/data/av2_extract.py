"""Extract per-track BEV trajectories + cropped lidar points from an AV2 log.

Raw ArgoVerse 2 sensor logs store annotations and lidar sweeps in the *ego*
frame of each timestamp. LabelFormer needs, per object track, a temporally
consistent point cloud, so everything is lifted into a shared static frame:
the "log frame" = city frame minus the log's first ego position (kept in
float64) so that coordinates stay small enough for float32 storage.

Boxes follow the project convention ``(x, y, yaw, l, w)``; the BEV yaw of a box
is the yaw of its quaternion (roll/pitch are ignored -- AV2 boxes are nearly
upright), while *points* are moved with the full 3D ego->city rotation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pyarrow.feather import read_feather

from ..geometry import points_in_box_mask, wrap_angle

__all__ = ["extract_log_tracks", "quat_to_rotation", "quat_to_yaw"]


def quat_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """3x3 rotation matrix of a (possibly unnormalized) quaternion ``(w, x, y, z)``."""
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_to_yaw(qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
    """Yaw (rotation about +z) of quaternion arrays, vectorized."""
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def _crop_half_extents(size: np.ndarray, scale: float, margin: float) -> np.ndarray:
    """Per-side half extents: at least ``scale * size / 2`` and ``size / 2 + margin``."""
    return np.maximum(scale * size / 2.0, size / 2.0 + margin)


def _read_ego_poses(log_dir: Path) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Map timestamp -> (R 3x3, t 3) ego->city, plus the earliest ego position."""
    df = read_feather(log_dir / "city_SE3_egovehicle.feather").sort_values("timestamp_ns")
    ts = df["timestamp_ns"].to_numpy(dtype=np.int64)
    quat = df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=np.float64)
    trans = df[["tx_m", "ty_m", "tz_m"]].to_numpy(dtype=np.float64)
    poses = {int(t): (quat_to_rotation(*quat[i]), trans[i]) for i, t in enumerate(ts)}
    origin = trans[0].copy() if len(trans) else np.zeros(3)
    return poses, origin


def _read_sweep(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a lidar sweep: ego-frame xyz (N, 3) float64 and intensity (N,) float32."""
    df = read_feather(path, columns=["x", "y", "z", "intensity"])
    xyz = np.stack(
        [df[c].to_numpy().astype(np.float64) for c in ("x", "y", "z")], axis=1
    )
    return xyz, df["intensity"].to_numpy().astype(np.float32)


def extract_log_tracks(
    log_dir: Path,
    categories: set[str],
    min_frames: int = 5,
    crop_scale: float = 1.5,
    crop_margin: float = 0.5,
    min_total_points: int = 20,
) -> list[dict[str, Any]]:
    """Extract every qualifying object track of one AV2 log.

    Sweeps are visited in timestamp order so each lidar file is read exactly
    once and shared by all tracks annotated at that timestamp. The crop around
    the ground-truth box is deliberately generous (``crop_scale`` /
    ``crop_margin``): at train time points are re-cropped with the perturbed
    box, which may sit outside the GT box.

    Args:
        log_dir: directory holding ``annotations.feather``,
            ``city_SE3_egovehicle.feather`` and ``sensors/lidar/*.feather``.
        categories: AV2 category names to keep.
        min_frames: minimum number of usable annotated frames per track.
        crop_scale: multiplicative box enlargement for the point crop.
        crop_margin: minimum absolute per-side crop margin in metres.
        min_total_points: drop tracks with fewer points over the whole track.

    Returns:
        One dict per track (sorted by ``track_uuid``) with keys
        ``track_uuid, category, origin, timestamps_ns, boxes_bev, z_center,
        height, points, point_counts``.
    """
    log_dir = Path(log_dir)
    ann = read_feather(log_dir / "annotations.feather")
    ann = ann[ann["category"].isin(categories)]
    if ann.empty:
        return []

    poses, origin = _read_ego_poses(log_dir)
    sweep_paths = {
        int(p.stem): p for p in (log_dir / "sensors" / "lidar").glob("*.feather")
    }
    usable = ann["timestamp_ns"].isin(sweep_paths.keys()) & ann["timestamp_ns"].isin(
        poses.keys()
    )
    ann = ann[usable]
    counts = ann["track_uuid"].value_counts()
    keep = set(counts[counts >= min_frames].index)
    ann = ann[ann["track_uuid"].isin(keep)]
    if ann.empty:
        return []

    tracks: dict[str, dict[str, list]] = {
        uuid: {
            "category": None,
            "timestamps_ns": [],
            "boxes_bev": [],
            "z_center": [],
            "height": [],
            "points": [],
        }
        for uuid in sorted(keep)
    }

    for ts, rows in ann.groupby("timestamp_ns", sort=True):
        rot, trans = poses[int(ts)]
        xyz_ego, intensity = _read_sweep(sweep_paths[int(ts)])
        xyz = xyz_ego @ rot.T + (trans - origin)  # log frame
        xy = xyz[:, :2]
        ego_yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))

        centers = rows[["tx_m", "ty_m", "tz_m"]].to_numpy(dtype=np.float64) @ rot.T + (
            trans - origin
        )
        quat = rows[["qw", "qx", "qy", "qz"]].to_numpy(dtype=np.float64)
        yaws = wrap_angle(quat_to_yaw(*quat.T) + ego_yaw)
        sizes = rows[["length_m", "width_m", "height_m"]].to_numpy(dtype=np.float64)
        uuids = rows["track_uuid"].to_numpy()

        for i, uuid in enumerate(uuids):
            length, width, height = sizes[i]
            box = np.array([centers[i, 0], centers[i, 1], yaws[i], length, width])
            half = _crop_half_extents(np.array([length, width]), crop_scale, crop_margin)
            crop_box = np.array([box[0], box[1], box[2], 2 * half[0], 2 * half[1]])
            mask = points_in_box_mask(xy, crop_box)
            mask &= np.abs(xyz[:, 2] - centers[i, 2]) <= 0.6 * height + 0.3
            pts = np.concatenate(
                [xyz[mask], intensity[mask, None]], axis=1, dtype=np.float32
            )
            tr = tracks[str(uuid)]
            tr["category"] = str(rows["category"].to_numpy()[i])
            tr["timestamps_ns"].append(int(ts))
            tr["boxes_bev"].append(box)
            tr["z_center"].append(centers[i, 2])
            tr["height"].append(height)
            tr["points"].append(pts)

    out: list[dict[str, Any]] = []
    for uuid, tr in tracks.items():
        n_frames = len(tr["timestamps_ns"])
        point_counts = np.array([len(p) for p in tr["points"]], dtype=np.int32)
        if n_frames < min_frames or int(point_counts.sum()) < min_total_points:
            continue
        points = (
            np.concatenate(tr["points"], axis=0)
            if n_frames
            else np.zeros((0, 4), np.float32)
        )
        out.append(
            {
                "track_uuid": uuid,
                "category": tr["category"],
                "origin": origin.astype(np.float64),
                "timestamps_ns": np.array(tr["timestamps_ns"], dtype=np.int64),
                "boxes_bev": np.asarray(tr["boxes_bev"], dtype=np.float32),
                "z_center": np.asarray(tr["z_center"], dtype=np.float32),
                "height": np.asarray(tr["height"], dtype=np.float32),
                "points": points.astype(np.float32),
                "point_counts": point_counts,
            }
        )
    return out
