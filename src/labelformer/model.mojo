"""Full LabelFormer forward pass (inference only) from exported LFT1 weights.

Faithfully mirrors LabelFormer.py's eval-mode forward: box MLP + pillar CNN
fused into per-frame tokens, ALiBi transformer over the trajectory, pose and
size heads, residual box assembly. BatchNorms were folded into the convs at
export time, so every conv here is plain conv + bias.
"""

from std.math import cos, floor, pi, sin

from .attention import mha
from .io import Config, decode_config, load_lft
from .layers import conv2d, layernorm, upsample_nearest_to
from .pillars import pillar_grid
from .tensor import Tensor, add_, matmul_bias, relu_


def wrap_angle(a: Float32) -> Float32:
    """Wrap to ``(-pi, pi]`` using Python-style floored modulo (matches torch)."""
    var x = -a + Float32(pi)
    var m = Float32(2.0 * pi)
    var r = x - m * floor(x / m)
    return -(r - Float32(pi))


struct LabelFormerModel(Movable):
    """Weight store + forward implementation."""

    var tensors: Dict[String, Tensor]
    var cfg: Config

    def __init__(out self, weights_path: String) raises:
        self.tensors = load_lft(weights_path)
        self.cfg = decode_config(self.tensors)

    def w(self, name: String) raises -> Tensor:
        return self.tensors[name].copy()

    def _block(
        self, x: Tensor, prefix: String, stride: Int, has_down: Bool
    ) raises -> Tensor:
        """ResNet BasicBlock with export-folded BNs: relu(conv2(relu(conv1(x))) + shortcut)."""
        var out = conv2d(x, self.w(prefix + ".conv1.w"), self.w(prefix + ".conv1.b"), stride, 1)
        relu_(out)
        out = conv2d(out, self.w(prefix + ".conv2.w"), self.w(prefix + ".conv2.b"), 1, 1)
        var identity: Tensor
        if has_down:
            identity = conv2d(x, self.w(prefix + ".down.w"), self.w(prefix + ".down.b"), stride, 0)
        else:
            identity = x.copy()
        add_(out, identity)
        relu_(out)
        return out^

    def _backbone_center(self, grid: Tensor) raises -> Tensor:
        """Pillar pseudo-image ``(C, H, W)`` -> center feature ``(out_dim,)``."""
        var x = conv2d(grid, self.w("stem.w"), self.w("stem.b"), 2, 1)
        relu_(x)
        var f1 = self._block(x, "s1b0", 2, True)
        f1 = self._block(f1, "s1b1", 1, False)
        var f2 = self._block(f1, "s2b0", 2, True)
        f2 = self._block(f2, "s2b1", 1, False)

        var l1 = conv2d(f1, self.w("lat1.w"), self.w("lat1.b"), 1, 0)
        var l2 = conv2d(f2, self.w("lat2.w"), self.w("lat2.b"), 1, 0)
        var up = upsample_nearest_to(l2, l1.dim(1), l1.dim(2))
        add_(l1, up)

        var fh = l1.dim(1)
        var fw = l1.dim(2)
        # Exported with PyTorch's exact float64 semantics; see export_mojo.py.
        var col0 = self.cfg.center_col
        var row0 = self.cfg.center_row
        if col0 > fw - 1:
            col0 = fw - 1
        if row0 > fh - 1:
            row0 = fh - 1

        var out = Tensor([l1.dim(0)])
        for c in range(l1.dim(0)):
            out[c] = l1.at3(c, row0, col0)
        return out^

    def encode_tokens(
        self, boxes_init: Tensor, points: Tensor, points_mask: Tensor
    ) raises -> Tensor:
        """Fused per-frame tokens ``(T, D)`` = BoxEncoder(boxes) + fusion(pillar feats)."""
        var t = boxes_init.dim(0)

        var box_feat = Tensor([t, 6])
        for i in range(t):
            var yaw2 = 2.0 * boxes_init.at2(i, 2)
            box_feat.set2(i, 0, boxes_init.at2(i, 0))
            box_feat.set2(i, 1, boxes_init.at2(i, 1))
            box_feat.set2(i, 2, sin(yaw2))
            box_feat.set2(i, 3, cos(yaw2))
            box_feat.set2(i, 4, boxes_init.at2(i, 3))
            box_feat.set2(i, 5, boxes_init.at2(i, 4))
        var box_emb = matmul_bias(box_feat, self.w("box.0.w"), self.w("box.0.b"))
        relu_(box_emb)
        box_emb = matmul_bias(box_emb, self.w("box.2.w"), self.w("box.2.b"))

        var pn_w = self.w("pointnet.w")
        var pn_b = self.w("pointnet.b")
        var pfeat = Tensor([t, self.cfg.out_dim])
        for i in range(t):
            var grid = pillar_grid(points, points_mask, i, pn_w, pn_b, self.cfg)
            var center = self._backbone_center(grid)
            for c in range(self.cfg.out_dim):
                pfeat.set2(i, c, center[c])

        var fused = matmul_bias(pfeat, self.w("fusion.w"), self.w("fusion.b"))
        add_(fused, box_emb)
        return fused^

    def transform(self, tokens: Tensor, frame_mask: Tensor) raises -> Tensor:
        """ALiBi transformer stack over ``(T, D)`` tokens; padded rows zeroed."""
        var t = tokens.dim(0)
        var d = tokens.dim(1)
        var x = tokens.copy()
        for i in range(t):
            if frame_mask[i] < 0.5:
                for j in range(d):
                    x.set2(i, j, 0.0)

        for li in range(self.cfg.num_layers):
            var p = "layer" + String(li)
            var normed = layernorm(x, self.w(p + ".norm1.w"), self.w(p + ".norm1.b"))
            var att = mha(
                normed,
                self.w(p + ".qkv.w"),
                self.w(p + ".qkv.b"),
                self.w(p + ".out.w"),
                self.w(p + ".out.b"),
                self.cfg.nhead,
                frame_mask,
            )
            add_(x, att)
            var normed2 = layernorm(x, self.w(p + ".norm2.w"), self.w(p + ".norm2.b"))
            var ff = matmul_bias(normed2, self.w(p + ".ffn1.w"), self.w(p + ".ffn1.b"))
            relu_(ff)
            ff = matmul_bias(ff, self.w(p + ".ffn2.w"), self.w(p + ".ffn2.b"))
            add_(x, ff)

        x = layernorm(x, self.w("final_norm.w"), self.w("final_norm.b"))
        for i in range(t):
            if frame_mask[i] < 0.5:
                for j in range(d):
                    x.set2(i, j, 0.0)
        return x^

    def forward(self, sample: Dict[String, Tensor]) raises -> Dict[String, Tensor]:
        """Refine one trajectory; returns outputs plus intermediates for parity checks."""
        ref boxes_init = sample["boxes_init"]
        ref frame_mask = sample["frame_mask"]
        var t = boxes_init.dim(0)

        var tokens = self.encode_tokens(boxes_init, sample["points"], sample["points_mask"])
        var hidden = self.transform(tokens, frame_mask)

        var pose = matmul_bias(hidden, self.w("pose.0.w"), self.w("pose.0.b"))
        relu_(pose)
        pose = matmul_bias(pose, self.w("pose.2.w"), self.w("pose.2.b"))

        var pooled = _masked_mean(hidden, frame_mask)
        var size = matmul_bias(pooled, self.w("size.0.w"), self.w("size.0.b"))
        relu_(size)
        size = matmul_bias(size, self.w("size.2.w"), self.w("size.2.b"))

        var mean_lw = _masked_mean(boxes_init, frame_mask)
        var out_l = mean_lw.at2(0, 3) + size.at2(0, 0)
        var out_w = mean_lw.at2(0, 4) + size.at2(0, 1)

        var boxes = Tensor([t, 5])
        for i in range(t):
            boxes.set2(i, 0, boxes_init.at2(i, 0) + pose.at2(i, 0))
            boxes.set2(i, 1, boxes_init.at2(i, 1) + pose.at2(i, 1))
            boxes.set2(i, 2, wrap_angle(boxes_init.at2(i, 2) + pose.at2(i, 2)))
            boxes.set2(i, 3, out_l)
            boxes.set2(i, 4, out_w)

        var size_flat = Tensor([2])
        size_flat[0] = size.at2(0, 0)
        size_flat[1] = size.at2(0, 1)

        var out = Dict[String, Tensor]()
        out["tokens"] = tokens^
        out["hidden"] = hidden^
        out["pose_residual"] = pose^
        out["size_residual"] = size_flat^
        out["boxes_refined"] = boxes^
        return out^


def _masked_mean(x: Tensor, mask: Tensor) raises -> Tensor:
    """Mean of ``(T, D)`` rows selected by 0/1 ``mask``; returns ``(1, D)``."""
    var t = x.dim(0)
    var d = x.dim(1)
    var out = Tensor([1, d])
    var denom: Float32 = 0.0
    for i in range(t):
        denom += mask[i]
    if denom < 1.0:
        denom = 1.0
    for j in range(d):
        var acc: Float32 = 0.0
        for i in range(t):
            acc += x.at2(i, j) * mask[i]
        out.set2(0, j, acc / denom)
    return out^
