"""PointPillars-style per-frame featurization: points -> pillar pseudo-image.

Mirrors ``PillarEncoder._scatter_pillars`` + ``point_net`` from LabelFormer.py:
per valid in-range point, features ``(x, y, z, intensity, dx, dy)`` (offsets to
the pillar center) go through ``Linear(6 -> C) + ReLU`` and are max-pooled per
pillar into a ``(C, H, W)`` grid; empty pillars stay exactly zero (valid because
the ReLU makes all pooled features non-negative).
"""

from std.math import floor

from .io import Config
from .tensor import Tensor


def pillar_grid(
    points: Tensor,
    points_mask: Tensor,
    frame: Int,
    w: Tensor,
    b: Tensor,
    cfg: Config,
) raises -> Tensor:
    """Build the pillar pseudo-image ``(C, H, W)`` for one frame.

    Args:
        points: ``(T, N, 4)`` object points ``(x, y, z, intensity)``.
        points_mask: ``(T, N)`` 0/1 validity mask.
        frame: frame index into the leading dimension.
        w: PointNet weight ``(C, 6)`` (torch Linear layout).
        b: PointNet bias ``(C,)``.
    """
    var c_out = w.dim(0)
    var h = cfg.grid_h()
    var wd = cfg.grid_w()
    var n = points.dim(1)
    var grid = Tensor([c_out, h, wd])

    for p in range(n):
        if points_mask.at2(frame, p) < 0.5:
            continue
        var px = points.at3(frame, p, 0)
        var py = points.at3(frame, p, 1)
        var col = Int(floor((px - cfg.x_min) / cfg.pillar_size))
        var row = Int(floor((py - cfg.y_min) / cfg.pillar_size))
        if col < 0 or col >= wd or row < 0 or row >= h:
            continue
        var cx = cfg.x_min + (Float32(col) + 0.5) * cfg.pillar_size
        var cy = cfg.y_min + (Float32(row) + 0.5) * cfg.pillar_size

        var feat = SIMD[DType.float32, 8](0.0)
        feat[0] = px
        feat[1] = py
        feat[2] = points.at3(frame, p, 2)
        feat[3] = points.at3(frame, p, 3)
        feat[4] = px - cx
        feat[5] = py - cy

        for c in range(c_out):
            var acc = b[c]
            for k in range(6):
                acc += w.at2(c, k) * feat[k]
            if acc < 0.0:
                acc = 0.0
            var idx = (c * h + row) * wd + col
            if acc > grid[idx]:
                grid[idx] = acc
    return grid^
