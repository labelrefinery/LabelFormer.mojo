"""LabelFormer Mojo inferencer CLI.

Usage:
    pixi run mojo run src/main.mojo data/weights.lft data/sample_0.lft [more samples...]

Runs the exported model on each exported trajectory sample and, when the sample
carries expected tensors from PyTorch, reports the max-abs difference per stage.
"""

from std.sys import argv

from labelformer.model import LabelFormerModel
from labelformer.io import load_lft
from labelformer.tensor import Tensor


def max_abs_diff(a: Tensor, b: Tensor) raises -> Float32:
    if a.numel() != b.numel():
        raise Error("size mismatch: " + String(a.numel()) + " vs " + String(b.numel()))
    var m: Float32 = 0.0
    for i in range(a.numel()):
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m


def main() raises:
    var args = argv()
    if len(args) < 3:
        print("usage: mojo run src/main.mojo <weights.lft> <sample.lft>...")
        return

    var model = LabelFormerModel(String(args[1]))
    print(
        "loaded weights:", len(model.tensors), "tensors | d_model", model.cfg.d_model,
        "layers", model.cfg.num_layers, "grid", model.cfg.grid_w(), "x", model.cfg.grid_h(),
    )

    var stages: List[String] = ["tokens", "hidden", "pose_residual", "size_residual", "boxes_refined"]
    var all_pass = True
    for s in range(2, len(args)):
        var path = String(args[s])
        var sample = load_lft(path)
        var t = sample["boxes_init"].dim(0)
        var out = model.forward(sample)
        print(path, "| T =", t)

        for name in stages:
            if String(name) in sample:
                var diff = max_abs_diff(out[String(name)], sample[String(name)])
                var tol: Float32 = 1e-3
                var ok = diff <= tol
                if String(name) == "boxes_refined" and not ok:
                    all_pass = False
                print("  ", name, "max|diff| =", diff, "PASS" if ok else "FAIL")

        ref boxes = out["boxes_refined"]
        print("  refined[0]  =", boxes.at2(0, 0), boxes.at2(0, 1), boxes.at2(0, 2), boxes.at2(0, 3), boxes.at2(0, 4))
        var last = t - 1
        print("  refined[-1] =", boxes.at2(last, 0), boxes.at2(last, 1), boxes.at2(last, 2), boxes.at2(last, 3), boxes.at2(last, 4))

    print("PARITY:", "PASS" if all_pass else "FAIL")
