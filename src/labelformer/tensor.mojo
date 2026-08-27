"""Minimal owned float32 tensor (row-major) plus the few ops LabelFormer needs."""


struct Tensor(Copyable, Movable, Writable):
    """Row-major float32 tensor with up to 4 dims."""

    var shape: List[Int]
    var data: List[Float32]

    def __init__(out self, shape: List[Int]):
        var n = 1
        for d in shape:
            n *= d
        self.shape = shape.copy()
        self.data = List[Float32](length=n, fill=0.0)

    def __init__(out self, *, copy: Self):
        self.shape = copy.shape.copy()
        self.data = copy.data.copy()

    def __init__(out self, *, deinit move: Self):
        self.shape = move.shape^
        self.data = move.data^

    def numel(self) -> Int:
        return len(self.data)

    def dim(self, i: Int) -> Int:
        return self.shape[i]

    def rank(self) -> Int:
        return len(self.shape)

    def __getitem__(self, i: Int) -> Float32:
        return self.data[i]

    def __setitem__(mut self, i: Int, v: Float32):
        self.data[i] = v

    def at2(self, i: Int, j: Int) -> Float32:
        return self.data[i * self.shape[1] + j]

    def at3(self, i: Int, j: Int, k: Int) -> Float32:
        return self.data[(i * self.shape[1] + j) * self.shape[2] + k]

    def set2(mut self, i: Int, j: Int, v: Float32):
        self.data[i * self.shape[1] + j] = v

    def set3(mut self, i: Int, j: Int, k: Int, v: Float32):
        self.data[(i * self.shape[1] + j) * self.shape[2] + k] = v

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Tensor(shape=[")
        for i in range(len(self.shape)):
            if i > 0:
                writer.write(", ")
            writer.write(self.shape[i])
        writer.write("], numel=", len(self.data), ")")


def matmul_bias(x: Tensor, w: Tensor, b: Tensor) raises -> Tensor:
    """``(m, k) @ (n, k)^T + (n,)`` with torch-layout weight ``w``: rows are outputs."""
    var m = x.dim(0)
    var k = x.dim(1)
    var n = w.dim(0)
    if w.dim(1) != k or b.dim(0) != n:
        raise Error("matmul_bias: shape mismatch")
    var out = Tensor([m, n])
    for i in range(m):
        for j in range(n):
            var acc: Float32 = 0.0
            for t in range(k):
                acc += x.at2(i, t) * w.at2(j, t)
            out.set2(i, j, acc + b[j])
    return out^


def relu_(mut x: Tensor):
    for i in range(x.numel()):
        if x[i] < 0.0:
            x[i] = 0.0


def add_(mut x: Tensor, y: Tensor) raises:
    if x.numel() != y.numel():
        raise Error("add_: shape mismatch")
    for i in range(x.numel()):
        x[i] = x[i] + y[i]
