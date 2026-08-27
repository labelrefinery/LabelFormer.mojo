"""Stage-by-stage parity debug against data/debug.lft (see export_mojo.export_debug)."""

from labelformer.io import load_lft
from labelformer.layers import conv2d, upsample_nearest_to
from labelformer.model import LabelFormerModel
from labelformer.pillars import pillar_grid
from labelformer.tensor import Tensor, add_, matmul_bias, relu_
from main import max_abs_diff


def report(name: String, got: Tensor, ref_tensors: Dict[String, Tensor]) raises:
    var d = max_abs_diff(got, ref_tensors[name])
    print(name, "shape", got, "max|diff| =", d)


def main() raises:
    var model = LabelFormerModel("data/weights.lft")
    var sample = load_lft("data/sample_0.lft")
    var dbg = load_lft("data/debug.lft")
    var fi = Int(dbg["frame_index"][0])
    print("debug frame:", fi)

    var grid = pillar_grid(
        sample["points"], sample["points_mask"], fi,
        model.w("pointnet.w"), model.w("pointnet.b"), model.cfg,
    )
    report("dbg_grid", grid, dbg)

    var stem = conv2d(grid, model.w("stem.w"), model.w("stem.b"), 2, 1)
    relu_(stem)
    report("dbg_stem", stem, dbg)

    var f1 = model._block(stem, "s1b0", 2, True)
    f1 = model._block(f1, "s1b1", 1, False)
    report("dbg_f1", f1, dbg)

    var f2 = model._block(f1, "s2b0", 2, True)
    f2 = model._block(f2, "s2b1", 1, False)
    report("dbg_f2", f2, dbg)

    var l1 = conv2d(f1, model.w("lat1.w"), model.w("lat1.b"), 1, 0)
    report("dbg_l1", l1, dbg)
    var l2 = conv2d(f2, model.w("lat2.w"), model.w("lat2.b"), 1, 0)
    report("dbg_l2", l2, dbg)
    var up = upsample_nearest_to(l2, l1.dim(1), l1.dim(2))
    add_(l1, up)
    report("dbg_fpn", l1, dbg)

    # Full per-frame center features and the box embedding.
    var t = sample["boxes_init"].dim(0)
    var pfeat = Tensor([t, model.cfg.out_dim])
    for i in range(t):
        var g = pillar_grid(
            sample["points"], sample["points_mask"], i,
            model.w("pointnet.w"), model.w("pointnet.b"), model.cfg,
        )
        var center = model._backbone_center(g)
        for c in range(model.cfg.out_dim):
            pfeat.set2(i, c, center[c])
    report("pfeat", pfeat, dbg)

    ref ref_pfeat = dbg["pfeat"]
    for i in range(t):
        var m: Float32 = 0.0
        var npts: Float32 = 0.0
        for c in range(model.cfg.out_dim):
            var d = pfeat.at2(i, c) - ref_pfeat.at2(i, c)
            if d < 0.0:
                d = -d
            if d > m:
                m = d
        for p in range(sample["points_mask"].dim(1)):
            npts += sample["points_mask"].at2(i, p)
        if m > 1e-4:
            print("frame", i, "pts", npts, "max|diff|", m)
