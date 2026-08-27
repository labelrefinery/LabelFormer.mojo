"""Unit tests for the LabelFormer tensor ops, layers and attention."""

from std.math import exp
from std.testing import assert_almost_equal, assert_equal, assert_true, TestSuite

from labelformer.attention import alibi_slope, mha, softmax_row_
from labelformer.layers import conv2d, layernorm, upsample_nearest_to
from labelformer.tensor import Tensor, matmul_bias

comptime TOL: Float64 = 1e-5


def make(shape: List[Int], vals: List[Float32]) raises -> Tensor:
    """Build a tensor of `shape` filled row-major from `vals`."""
    var t = Tensor(shape)
    if t.numel() != len(vals):
        raise Error("make: value count mismatch")
    for i in range(len(vals)):
        t[i] = vals[i]
    return t^


def close(got: Float32, expected: Float32) raises:
    """Assert `got` is within TOL of `expected`."""
    assert_almost_equal(got, expected, atol=TOL, rtol=TOL)


# ----------------------------------------------------------------- matmul_bias


def test_matmul_bias() raises:
    var x = make([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var w = make([2, 3], [1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    var b = make([2], [1.0, -1.0])
    var y = matmul_bias(x, w, b)
    assert_equal(y.dim(0), 2)
    assert_equal(y.dim(1), 2)
    close(y.at2(0, 0), 2.0)
    close(y.at2(0, 1), 4.0)
    close(y.at2(1, 0), 5.0)
    close(y.at2(1, 1), 10.0)


# ------------------------------------------------------------------- layernorm


def test_layernorm_basic() raises:
    var x = make([1, 4], [1.0, 2.0, 3.0, 4.0])
    var gamma = make([4], [1.0, 1.0, 1.0, 1.0])
    var beta = make([4], [0.0, 0.0, 0.0, 0.0])
    var y = layernorm(x, gamma, beta)
    assert_equal(y.dim(0), 1)
    assert_equal(y.dim(1), 4)
    close(y.at2(0, 0), -1.3416355)
    close(y.at2(0, 1), -0.4472118)
    close(y.at2(0, 2), 0.4472118)
    close(y.at2(0, 3), 1.3416355)


def test_layernorm_affine() raises:
    # Row 0 is the reference row; row 1 is constant (normalizes to zeros).
    var x = make([2, 4], [1.0, 2.0, 3.0, 4.0, 7.0, 7.0, 7.0, 7.0])
    var gamma = make([4], [2.0, 2.0, 0.5, 0.5])
    var beta = make([4], [1.0, -1.0, 0.0, 3.0])
    var y = layernorm(x, gamma, beta)
    close(y.at2(0, 0), -1.3416355 * 2.0 + 1.0)
    close(y.at2(0, 1), -0.4472118 * 2.0 - 1.0)
    close(y.at2(0, 2), 0.4472118 * 0.5)
    close(y.at2(0, 3), 1.3416355 * 0.5 + 3.0)
    # Constant row: mean == value, var == 0 -> output is just beta.
    close(y.at2(1, 0), 1.0)
    close(y.at2(1, 1), -1.0)
    close(y.at2(1, 2), 0.0)
    close(y.at2(1, 3), 3.0)


# ---------------------------------------------------------------------- conv2d


def test_conv2d_ones_pad1() raises:
    var x = make([1, 3, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    var w = make([1, 1, 3, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    var b = make([1], [0.0])
    var y = conv2d(x, w, b, 1, 1)
    assert_equal(y.dim(0), 1)
    assert_equal(y.dim(1), 3)
    assert_equal(y.dim(2), 3)
    var expected: List[Float32] = [
        12.0, 21.0, 16.0,
        27.0, 45.0, 33.0,
        24.0, 39.0, 28.0,
    ]
    for i in range(9):
        close(y[i], expected[i])


def test_conv2d_stride2() raises:
    var x = make([1, 3, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    var w = make([1, 1, 3, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    var b = make([1], [0.0])
    var y = conv2d(x, w, b, 2, 1)
    assert_equal(y.dim(1), 2)
    assert_equal(y.dim(2), 2)
    close(y.at3(0, 0, 0), 12.0)
    close(y.at3(0, 0, 1), 16.0)
    close(y.at3(0, 1, 0), 24.0)
    close(y.at3(0, 1, 1), 28.0)


def test_conv2d_1x1_channels() raises:
    var x = make([2, 1, 1], [3.0, 5.0])
    # w[oc, ic] with oc=0 -> [1, 0], oc=1 -> [0, 2]
    var w = make([2, 2, 1, 1], [1.0, 0.0, 0.0, 2.0])
    var b = make([2], [0.5, -1.0])
    var y = conv2d(x, w, b, 1, 0)
    assert_equal(y.dim(0), 2)
    assert_equal(y.dim(1), 1)
    assert_equal(y.dim(2), 1)
    close(y.at3(0, 0, 0), 3.5)
    close(y.at3(1, 0, 0), 9.0)


# ---------------------------------------------------------------- upsample 2x


def test_upsample_exact_2x() raises:
    var x = make([1, 2, 2], [1.0, 2.0, 3.0, 4.0])
    var y = upsample_nearest_to(x, 4, 4)
    assert_equal(y.dim(0), 1)
    assert_equal(y.dim(1), 4)
    assert_equal(y.dim(2), 4)
    for oy in range(4):
        for ox in range(4):
            close(y.at3(0, oy, ox), x.at3(0, oy // 2, ox // 2))


def test_upsample_non_integer() raises:
    var x = make([1, 3, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    var y = upsample_nearest_to(x, 4, 4)
    # src index for i in 0..3 with scale 3/4: floor(i * 3 / 4) = 0, 0, 1, 2
    var src: List[Int] = [0, 0, 1, 2]
    for oy in range(4):
        for ox in range(4):
            close(y.at3(0, oy, ox), x.at3(0, src[oy], src[ox]))


# ---------------------------------------------------------------- alibi slopes


def test_alibi_slope_nhead4() raises:
    close(alibi_slope(0, 4), 0.25)
    close(alibi_slope(1, 4), 0.0625)
    close(alibi_slope(2, 4), 0.015625)
    close(alibi_slope(3, 4), 0.00390625)


# --------------------------------------------------------------------- softmax


def test_softmax_uniform() raises:
    var logits = make([1, 2], [0.0, 0.0])
    softmax_row_(logits)
    close(logits.at2(0, 0), 0.5)
    close(logits.at2(0, 1), 0.5)


def test_softmax_stability() raises:
    var logits = make([1, 2], [1000.0, 0.0])
    softmax_row_(logits)
    assert_true(logits.at2(0, 0) == logits.at2(0, 0), "softmax produced NaN")
    assert_true(logits.at2(0, 1) == logits.at2(0, 1), "softmax produced NaN")
    close(logits.at2(0, 0), 1.0)
    assert_equal(logits.at2(0, 1), Float32(0.0))


def test_softmax_masked_logit_is_zero() raises:
    var neg: Float32 = -1.7014118e38
    var logits = make([2, 2], [0.0, neg, neg, 0.0])
    softmax_row_(logits)
    close(logits.at2(0, 0), 1.0)
    assert_equal(logits.at2(0, 1), Float32(0.0))
    assert_equal(logits.at2(1, 0), Float32(0.0))
    close(logits.at2(1, 1), 1.0)


# ------------------------------------------------------------------------- mha


def test_mha_identity_values() raises:
    var t = 3
    var d = 4
    var nhead = 2
    var hd = d // nhead

    var x = Tensor([t, d])
    for i in range(t * d):
        x[i] = Float32(i + 1)

    # qkv_w: (3D, D). Rows [0, 2D) zero -> q = k = 0. Rows [2D, 3D) identity -> v = x.
    var qkv_w = Tensor([3 * d, d])
    for i in range(d):
        qkv_w.set2(2 * d + i, i, 1.0)
    var qkv_b = Tensor([3 * d])

    var out_w = Tensor([d, d])
    for i in range(d):
        out_w.set2(i, i, 1.0)
    var out_b = Tensor([d])

    var frame_mask = Tensor([t])
    for i in range(t):
        frame_mask[i] = 1.0

    var y = mha(x, qkv_w, qkv_b, out_w, out_b, nhead, frame_mask)
    assert_equal(y.dim(0), t)
    assert_equal(y.dim(1), d)

    # With q = k = 0 every logit is pure ALiBi bias: row 0 is [0, -m, -2m].
    for h in range(nhead):
        var m = alibi_slope(h, nhead)
        var e0 = exp(Float32(0.0))
        var e1 = exp(-m)
        var e2 = exp(-2.0 * m)
        var total = e0 + e1 + e2
        var w0 = e0 / total
        var w1 = e1 / total
        var w2 = e2 / total
        for c in range(hd):
            var lane = h * hd + c
            var expected = w0 * x.at2(0, lane) + w1 * x.at2(1, lane) + w2 * x.at2(
                2, lane
            )
            close(y.at2(0, lane), expected)


def test_mha_frame_mask_drops_key() raises:
    var t = 3
    var d = 4
    var nhead = 2
    var hd = d // nhead

    var x = Tensor([t, d])
    for i in range(t * d):
        x[i] = Float32(i + 1)

    var qkv_w = Tensor([3 * d, d])
    for i in range(d):
        qkv_w.set2(2 * d + i, i, 1.0)
    var qkv_b = Tensor([3 * d])
    var out_w = Tensor([d, d])
    for i in range(d):
        out_w.set2(i, i, 1.0)
    var out_b = Tensor([d])

    # Mask out key j = 1.
    var frame_mask = make([3], [1.0, 0.0, 1.0])

    var y = mha(x, qkv_w, qkv_b, out_w, out_b, nhead, frame_mask)
    for h in range(nhead):
        var m = alibi_slope(h, nhead)
        var e0 = exp(Float32(0.0))
        var e2 = exp(-2.0 * m)
        var total = e0 + e2
        var w0 = e0 / total
        var w2 = e2 / total
        for c in range(hd):
            var lane = h * hd + c
            var expected = w0 * x.at2(0, lane) + w2 * x.at2(2, lane)
            close(y.at2(0, lane), expected)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
