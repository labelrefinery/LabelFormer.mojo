"""Torch dataset over preprocessed LabelFormer tracks.

Each sample is one object trajectory: the noisy *initial* boxes (simulated by
perturbing the ground truth per frame, as in the paper's experiments), the
ground-truth boxes, and per-frame lidar points expressed in that frame's
initial-box frame. Boxes are reported in the *trajectory frame*, defined by the
initial box of the middle frame.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..geometry import (
    canonicalize_headings,
    points_in_box_mask,
    se2_from_xyt,
    se2_inv,
    transform_boxes_bev,
    wrap_angle,
)

__all__ = ["PerturbConfig", "TrajectoryDataset", "collate_tracks"]

_STR_FIELDS = ("track_uuid", "log_id")


@dataclass
class PerturbConfig:
    """Bounds of the synthetic noise applied to ground-truth boxes per frame."""

    max_translation: float = 0.25
    max_rotation_deg: float = 10.0
    size_scale_range: tuple[float, float] = (0.9, 1.1)


def _stable_seed(key: str) -> int:
    """Deterministic 64-bit seed from a string (``hash()`` is salted per run)."""
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")


class TrajectoryDataset(Dataset):
    """Dataset of preprocessed track NPZs under ``root/split/<log_id>/<uuid>.npz``."""

    def __init__(
        self,
        root: Path,
        split: str,
        max_frames: int = 64,
        max_points_per_frame: int = 1024,
        perturb: PerturbConfig | None = None,
        subsequence: bool = True,
        deterministic: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.max_frames = max_frames
        self.max_points_per_frame = max_points_per_frame
        self.perturb = perturb if perturb is not None else PerturbConfig()
        self.subsequence = subsequence
        self.deterministic = deterministic
        self.files: list[Path] = sorted((self.root / split).rglob("*.npz"))

    def __len__(self) -> int:
        return len(self.files)

    def _rng(self, path: Path) -> np.random.Generator:
        if self.deterministic:
            return np.random.default_rng(_stable_seed(f"{path.parent.name}/{path.stem}"))
        return np.random.default_rng()

    def _perturb_boxes(
        self, boxes_gt: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Per-frame independent jitter of position, heading and dimensions."""
        cfg = self.perturb
        n = len(boxes_gt)
        out = boxes_gt.copy()
        out[:, 0] += rng.uniform(-cfg.max_translation, cfg.max_translation, n)
        out[:, 1] += rng.uniform(-cfg.max_translation, cfg.max_translation, n)
        max_rot = np.deg2rad(cfg.max_rotation_deg)
        out[:, 2] = wrap_angle(out[:, 2] + rng.uniform(-max_rot, max_rot, n))
        lo, hi = cfg.size_scale_range
        out[:, 3] *= rng.uniform(lo, hi, n)
        out[:, 4] *= rng.uniform(lo, hi, n)
        return out

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        rng = self._rng(path)
        with np.load(path) as data:
            boxes_gt = data["boxes_bev"].astype(np.float64)
            z_center = data["z_center"].astype(np.float64)
            counts = data["point_counts"].astype(np.int64)
            all_points = data["points"].astype(np.float64)

        offsets = np.concatenate([[0], np.cumsum(counts)])
        n_frames = len(counts)
        if n_frames > self.max_frames:
            span = n_frames - self.max_frames
            if self.subsequence and not self.deterministic:
                start = int(rng.integers(0, span + 1))
            else:
                start = span // 2
            stop = start + self.max_frames
            boxes_gt = boxes_gt[start:stop]
            z_center = z_center[start:stop]
            counts = counts[start:stop]
            offsets = offsets[start : stop + 1]
            n_frames = self.max_frames

        boxes_init = self._perturb_boxes(boxes_gt, rng)

        # Trajectory frame: the middle frame's initial box.
        mid = n_frames // 2
        t_traj = se2_inv(se2_from_xyt(*boxes_init[mid, :3]))
        boxes_init_traj = transform_boxes_bev(t_traj, boxes_init)
        boxes_gt_traj = transform_boxes_bev(t_traj, boxes_gt)

        # Canonicalize the *initial* headings only; GT stays untouched because
        # the losses/metrics downstream are pi-symmetric.
        canonical, flipped = canonicalize_headings(boxes_init_traj[:, 2])
        boxes_init_traj[:, 2] = canonical
        yaw_log = wrap_angle(boxes_init[:, 2] + np.where(flipped, np.pi, 0.0))

        frame_points: list[np.ndarray] = []
        for t in range(n_frames):
            pts = all_points[offsets[t] : offsets[t + 1]]
            box = boxes_init[t]
            if len(pts):
                keep = points_in_box_mask(pts[:, :2], box, scale=1.1)
                pts = pts[keep]
            if len(pts) > self.max_points_per_frame:
                sel = rng.choice(len(pts), self.max_points_per_frame, replace=False)
                pts = pts[sel]
            local = np.zeros((len(pts), 4), dtype=np.float32)
            if len(pts):
                c, s = np.cos(yaw_log[t]), np.sin(yaw_log[t])
                dx = pts[:, 0] - box[0]
                dy = pts[:, 1] - box[1]
                local[:, 0] = c * dx + s * dy
                local[:, 1] = -s * dx + c * dy
                local[:, 2] = pts[:, 2] - z_center[t]
                local[:, 3] = pts[:, 3] / 255.0
            frame_points.append(local)

        n_max = max((len(p) for p in frame_points), default=0)
        points = np.zeros((n_frames, n_max, 4), dtype=np.float32)
        points_mask = np.zeros((n_frames, n_max), dtype=bool)
        for t, p in enumerate(frame_points):
            points[t, : len(p)] = p
            points_mask[t, : len(p)] = True

        return {
            "boxes_init": torch.from_numpy(boxes_init_traj.astype(np.float32)),
            "boxes_gt": torch.from_numpy(boxes_gt_traj.astype(np.float32)),
            "points": torch.from_numpy(points),
            "points_mask": torch.from_numpy(points_mask),
            "frame_mask": torch.ones(n_frames, dtype=torch.bool),
            "track_uuid": path.stem,
            "log_id": path.parent.name,
        }


def collate_tracks(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad ragged frame/point counts and stack samples into a batch."""
    batch = len(samples)
    max_t = max(s["frame_mask"].shape[0] for s in samples)
    max_n = max(s["points"].shape[1] for s in samples)

    boxes_init = torch.zeros(batch, max_t, 5)
    boxes_gt = torch.zeros(batch, max_t, 5)
    points = torch.zeros(batch, max_t, max_n, 4)
    points_mask = torch.zeros(batch, max_t, max_n, dtype=torch.bool)
    frame_mask = torch.zeros(batch, max_t, dtype=torch.bool)

    for i, s in enumerate(samples):
        t, n = s["frame_mask"].shape[0], s["points"].shape[1]
        boxes_init[i, :t] = s["boxes_init"]
        boxes_gt[i, :t] = s["boxes_gt"]
        points[i, :t, :n] = s["points"]
        points_mask[i, :t, :n] = s["points_mask"]
        frame_mask[i, :t] = s["frame_mask"]

    out: dict[str, Any] = {
        "boxes_init": boxes_init,
        "boxes_gt": boxes_gt,
        "points": points,
        "points_mask": points_mask,
        "frame_mask": frame_mask,
    }
    for key in _STR_FIELDS:
        out[key] = [s[key] for s in samples]
    return out
