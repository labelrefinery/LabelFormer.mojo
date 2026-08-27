"""Multi-head self-attention with ALiBi positional bias and a frame mask."""

from std.math import exp, exp2, sqrt

from labelformer.tensor import Tensor, matmul_bias

comptime NEG_MASK: Float32 = -1.7014118e38
"""``torch.finfo(float32).min / 2`` — the additive mask value for invalid keys."""


def alibi_slope(h: Int, nhead: Int) -> Float32:
    """ALiBi slope for head ``h`` of ``nhead``: ``2 ** (-8 * (h + 1) / nhead)``.

    Args:
        h: Zero-based head index.
        nhead: Total number of heads.

    Returns:
        The slope as float32.
    """
    var expo = Float32(-8.0) * Float32(h + 1) / Float32(nhead)
    return exp2(expo)


def softmax_row_(mut logits: Tensor) raises:
    """In-place row-wise softmax of a rank-2 tensor, with row-max subtraction.

    Args:
        logits: Rank-2 tensor ``(R, C)``; each row is replaced by its softmax.
    """
    if logits.rank() != 2:
        raise Error("softmax_row_: expected rank 2")
    var rows = logits.dim(0)
    var cols = logits.dim(1)
    for i in range(rows):
        var m = logits.at2(i, 0)
        for j in range(1, cols):
            if logits.at2(i, j) > m:
                m = logits.at2(i, j)
        var total: Float32 = 0.0
        for j in range(cols):
            var e = exp(logits.at2(i, j) - m)
            logits.set2(i, j, e)
            total += e
        var inv = 1.0 / total
        for j in range(cols):
            logits.set2(i, j, logits.at2(i, j) * inv)


def mha(
    x: Tensor,
    qkv_w: Tensor,
    qkv_b: Tensor,
    out_w: Tensor,
    out_b: Tensor,
    nhead: Int,
    frame_mask: Tensor,
) raises -> Tensor:
    """Multi-head self-attention with ALiBi bias and key masking (eval, no dropout).

    Args:
        x: Input of shape ``(T, D)``.
        qkv_w: Fused QKV weight of shape ``(3D, D)`` (torch Linear layout).
        qkv_b: Fused QKV bias of shape ``(3D,)``.
        out_w: Output projection weight of shape ``(D, D)``.
        out_b: Output projection bias of shape ``(D,)``.
        nhead: Number of attention heads; must divide ``D``.
        frame_mask: Shape ``(T,)`` of 0.0/1.0; keys with value <= 0.5 are masked out.

    Returns:
        Tensor of shape ``(T, D)``.
    """
    if x.rank() != 2:
        raise Error("mha: x must be rank 2 (T, D)")
    var t = x.dim(0)
    var d = x.dim(1)
    if nhead < 1 or d % nhead != 0:
        raise Error("mha: nhead must divide D")
    if qkv_w.dim(0) != 3 * d or qkv_w.dim(1) != d:
        raise Error("mha: qkv_w must be (3D, D)")
    if frame_mask.numel() != t:
        raise Error("mha: frame_mask must have T entries")
    var hd = d // nhead
    var inv_scale = 1.0 / sqrt(Float32(hd))

    var qkv = matmul_bias(x, qkv_w, qkv_b)  # (T, 3D)

    var concat = Tensor([t, d])
    var logits = Tensor([t, t])
    for h in range(nhead):
        var slope = alibi_slope(h, nhead)
        var qo = h * hd
        var ko = d + h * hd
        var vo = 2 * d + h * hd

        for i in range(t):
            for j in range(t):
                var dot: Float32 = 0.0
                for c in range(hd):
                    dot += qkv.at2(i, qo + c) * qkv.at2(j, ko + c)
                var dist = i - j
                if dist < 0:
                    dist = -dist
                var bias = -slope * Float32(dist)
                if frame_mask[j] <= 0.5:
                    bias += NEG_MASK
                logits.set2(i, j, dot * inv_scale + bias)

        softmax_row_(logits)

        for i in range(t):
            for c in range(hd):
                var acc: Float32 = 0.0
                for j in range(t):
                    acc += logits.at2(i, j) * qkv.at2(j, vo + c)
                concat.set2(i, h * hd + c, acc)

    return matmul_bias(concat, out_w, out_b)
