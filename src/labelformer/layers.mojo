"""Elementary neural-net layers used by LabelFormer: layernorm, conv2d, upsample."""

from std.math import sqrt

from labelformer.tensor import Tensor


def layernorm(x: Tensor, gamma: Tensor, beta: Tensor) raises -> Tensor:
    """Row-wise layer normalization matching ``torch.nn.LayerNorm`` in eval mode.

    Args:
        x: Input of shape ``(T, D)``.
        gamma: Per-feature scale of shape ``(D,)``.
        beta: Per-feature shift of shape ``(D,)``.

    Returns:
        Tensor of shape ``(T, D)``: ``(x - mean) / sqrt(var + 1e-5) * gamma + beta``
        where ``var`` is the biased (divide-by-D) variance of each row.
    """
    if x.rank() != 2:
        raise Error("layernorm: x must be rank 2")
    var t = x.dim(0)
    var d = x.dim(1)
    if gamma.numel() != d or beta.numel() != d:
        raise Error("layernorm: gamma/beta size mismatch")

    var eps: Float32 = 1e-5
    var inv_d = 1.0 / Float32(d)
    var out = Tensor([t, d])
    for i in range(t):
        var mean: Float32 = 0.0
        for j in range(d):
            mean += x.at2(i, j)
        mean *= inv_d

        var var_acc: Float32 = 0.0
        for j in range(d):
            var diff = x.at2(i, j) - mean
            var_acc += diff * diff
        var_acc *= inv_d

        var inv_std = 1.0 / sqrt(var_acc + eps)
        for j in range(d):
            var norm = (x.at2(i, j) - mean) * inv_std
            out.set2(i, j, norm * gamma[j] + beta[j])
    return out^


def conv2d(x: Tensor, w: Tensor, b: Tensor, stride: Int, pad: Int) raises -> Tensor:
    """2-D convolution (cross-correlation) matching ``torch.nn.Conv2d``.

    Args:
        x: Input of shape ``(C_in, H, W)``.
        w: Weights of shape ``(C_out, C_in, KH, KW)``.
        b: Bias of shape ``(C_out,)``.
        stride: Stride applied on both H and W.
        pad: Zero padding applied on both sides of H and W.

    Returns:
        Tensor of shape ``(C_out, (H + 2*pad - KH)//stride + 1,
        (W + 2*pad - KW)//stride + 1)``.
    """
    if x.rank() != 3:
        raise Error("conv2d: x must be rank 3 (C_in, H, W)")
    if w.rank() != 4:
        raise Error("conv2d: w must be rank 4 (C_out, C_in, KH, KW)")
    if stride < 1:
        raise Error("conv2d: stride must be >= 1")

    var cin = x.dim(0)
    var h = x.dim(1)
    var wd = x.dim(2)
    var cout = w.dim(0)
    var kh = w.dim(2)
    var kw = w.dim(3)
    if w.dim(1) != cin:
        raise Error("conv2d: weight input channels mismatch")
    if b.numel() != cout:
        raise Error("conv2d: bias size mismatch")

    var oh = (h + 2 * pad - kh) // stride + 1
    var ow = (wd + 2 * pad - kw) // stride + 1
    if oh < 1 or ow < 1:
        raise Error("conv2d: empty output")

    var out = Tensor([cout, oh, ow])
    for oc in range(cout):
        for oy in range(oh):
            for ox in range(ow):
                var acc = b[oc]
                for ic in range(cin):
                    for ky in range(kh):
                        var iy = oy * stride + ky - pad
                        if iy < 0 or iy >= h:
                            continue
                        for kx in range(kw):
                            var ix = ox * stride + kx - pad
                            if ix < 0 or ix >= wd:
                                continue
                            var wi = ((oc * cin + ic) * kh + ky) * kw + kx
                            acc += x.at3(ic, iy, ix) * w[wi]
                out.set3(oc, oy, ox, acc)
    return out^


def upsample_nearest_to(x: Tensor, oh: Int, ow: Int) raises -> Tensor:
    """Nearest-neighbour resize matching ``F.interpolate(mode="nearest")``.

    Args:
        x: Input of shape ``(C, H, W)``.
        oh: Output height.
        ow: Output width.

    Returns:
        Tensor of shape ``(C, oh, ow)``; source index is
        ``min(floor(i * H / oh), H - 1)`` with the float scale ``H / oh``.
    """
    if x.rank() != 3:
        raise Error("upsample_nearest_to: x must be rank 3 (C, H, W)")
    if oh < 1 or ow < 1:
        raise Error("upsample_nearest_to: output dims must be >= 1")

    var c = x.dim(0)
    var h = x.dim(1)
    var wd = x.dim(2)
    var scale_y = Float32(h) / Float32(oh)
    var scale_x = Float32(wd) / Float32(ow)

    var out = Tensor([c, oh, ow])
    for oy in range(oh):
        var sy = Int(Float32(oy) * scale_y)
        if sy > h - 1:
            sy = h - 1
        for ox in range(ow):
            var sx = Int(Float32(ox) * scale_x)
            if sx > wd - 1:
                sx = wd - 1
            for ch in range(c):
                out.set3(ch, oy, ox, x.at3(ch, sy, sx))
    return out^
