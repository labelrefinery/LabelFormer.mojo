"""Reader for the LFT1 tensor container written by LabelFormer.py's export_mojo.py.

Layout (little-endian):
    b"LFT1" | u32 n_tensors | per tensor:
        u32 name_len | name utf8 | u32 ndim | u32 shape[ndim] | f32 data (C order)
"""

from std.memory import bitcast

from .tensor import Tensor


@fieldwise_init
struct Config(Copyable, Movable, Writable):
    """Model hyper-parameters decoded from the ``__config__`` tensor."""

    var d_model: Int
    var nhead: Int
    var num_layers: Int
    var dim_feedforward: Int
    var x_min: Float32
    var x_max: Float32
    var y_min: Float32
    var y_max: Float32
    var pillar_size: Float32
    var point_feat_dim: Int
    var out_dim: Int
    var center_row: Int
    var center_col: Int

    def grid_w(self) -> Int:
        return Int(((self.x_max - self.x_min) / self.pillar_size).__round__())

    def grid_h(self) -> Int:
        return Int(((self.y_max - self.y_min) / self.pillar_size).__round__())


def _u32(bytes: List[UInt8], off: Int) -> UInt32:
    var v: UInt32 = 0
    v |= UInt32(bytes[off])
    v |= UInt32(bytes[off + 1]) << 8
    v |= UInt32(bytes[off + 2]) << 16
    v |= UInt32(bytes[off + 3]) << 24
    return v


def _f32(bytes: List[UInt8], off: Int) -> Float32:
    return bitcast[DType.float32, 1](SIMD[DType.uint32, 1](_u32(bytes, off)))[0]


def load_lft(path: String) raises -> Dict[String, Tensor]:
    """Parse an LFT1 file into a name -> Tensor dictionary."""
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()
    if len(bytes) < 8 or bytes[0] != 76 or bytes[1] != 70 or bytes[2] != 84 or bytes[3] != 49:
        raise Error("not an LFT1 file: " + path)

    var out = Dict[String, Tensor]()
    var n_tensors = Int(_u32(bytes, 4))
    var off = 8
    for _ in range(n_tensors):
        var name_len = Int(_u32(bytes, off))
        off += 4
        var name = String("")
        for i in range(name_len):
            name += chr(Int(bytes[off + i]))
        off += name_len

        var ndim = Int(_u32(bytes, off))
        off += 4
        var shape = List[Int]()
        var numel = 1
        for _ in range(ndim):
            var d = Int(_u32(bytes, off))
            off += 4
            shape.append(d)
            numel *= d
        if ndim == 0:
            shape.append(1)

        var t = Tensor(shape)
        for i in range(numel):
            t[i] = _f32(bytes, off)
            off += 4
        out[name] = t^
    if off != len(bytes):
        raise Error("trailing bytes in " + path)
    return out^


def decode_config(tensors: Dict[String, Tensor]) raises -> Config:
    """Build a Config from the ``__config__`` tensor of a weights file."""
    ref c = tensors["__config__"]
    if c.numel() != 13:
        raise Error("bad __config__ tensor")
    return Config(
        d_model=Int(c[0]),
        nhead=Int(c[1]),
        num_layers=Int(c[2]),
        dim_feedforward=Int(c[3]),
        x_min=c[4],
        x_max=c[5],
        y_min=c[6],
        y_max=c[7],
        pillar_size=c[8],
        point_feat_dim=Int(c[9]),
        out_dim=Int(c[10]),
        center_row=Int(c[11]),
        center_col=Int(c[12]),
    )
