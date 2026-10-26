#!/usr/bin/env python3
"""IQ4_XS block format: the byte-golden layout (llama.cpp ggml-common.h) and the
reference dequant, in one place.

An IQ4_XS super-block is 256 weights in 136 bytes (ggml-common.h block_iq4_xs):

    off  0  f16  d          super-block scale
    off  2  u16  scales_h   2-bit high nibbles of the 8 sub-block scales
    off  4  u8   scales_l[QK_K/64]   4-bit low nibbles, two sub-blocks a byte
    off  8  u8   qs[QK_K/2]          4-bit codebook indices, two per byte

Dequant (ggml-quants.c dequantize_row_iq4_xs), per super-block, eight 32-weight
sub-blocks (ib = 0..7):

    ls = ((scales_l[ib/2] >> 4*(ib%2)) & 0xf)
       | (((scales_h >> 2*ib) & 3) << 4)      # 0..63
    dl = d * (ls - 32)
    for j in 0..15:
        y[ib*32 + j]      = dl * kvalues_iq4nl[qs[ib*16 + j] & 0xf]
        y[ib*32 + 16 + j] = dl * kvalues_iq4nl[qs[ib*16 + j] >>  4]

kvalues_iq4nl (ggml-common.h:1120, int8) is the non-linear 16-entry codebook
[-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113].

Every product is exact: d is a f16 significand (<=11 bits), (ls-32) is in
[-32, 31] (5 bits) and kvalues_iq4nl is <= 7 bits, so d*(ls-32) is <=16 bits and
dl*code <= 23 bits, both inside f32's 24-bit significand. The whole dequant is
therefore exact in f32 -- there is no rounding to reproduce, and the CUDA, CPU
and numpy paths agree bit for bit on the finite d the pack contract allows.

Rows are kept in the GGUF block layout verbatim (no row-split): 136 is a
multiple of 8 (but not 16), so a row of `cols` columns (cols % 256 == 0) is
row_bytes = cols / 256 * 136. The qs region starts at offset 8, so the 4-byte
loads the group-of-four GEMV uses are always 4-aligned.
"""

import ctypes as C

import numpy as np

QK_K = 256
BLOCK_BYTES = 2 + 2 + QK_K // 64 + QK_K // 2  # 136
GGML_IQ4_XS = 23

# kvalues_iq4nl, ggml-common.h:1120
KV = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    np.int8,
)

f32, f16 = np.float32, np.float16


def sub_scales(raw):
    """raw u8 [..., 136] -> ls u8 [nb, 8], ggml's per-sub-block scale (0..63)."""
    B = raw.reshape(-1, BLOCK_BYTES)
    nb = B.shape[0]
    sh = B[:, 2:4].copy().view(np.uint16).astype(np.uint32).reshape(nb)
    sl = B[:, 4:8].astype(np.uint32)
    ls = np.empty((nb, 8), np.uint32)
    for ib in range(8):
        ls[:, ib] = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (
            ((sh >> (2 * ib)) & 3) << 4
        )
    return (ls - 32).astype(np.int32).reshape(nb, 8)


def codes(raw):
    """raw u8 [..., 136] -> code u8 [nb, 256]: the codebook index per column."""
    B = raw.reshape(-1, BLOCK_BYTES)
    nb = B.shape[0]
    qs = B[:, 8:136].astype(np.uint16).reshape(nb, 8, 16)
    out = np.empty((nb, QK_K), np.uint8)
    for ib in range(8):
        out[:, ib * 32 : ib * 32 + 16] = (qs[:, ib] & 0xF).astype(np.uint8)
        out[:, ib * 32 + 16 : ib * 32 + 32] = (qs[:, ib] >> 4).astype(np.uint8)
    return out


def dequant_blocks_arrays(d, raw):
    """d f16-bits [nb]; raw u8 [nb,136] -> f32 [nb,256]. Exact (see module doc)."""
    nb = d.shape[0]
    df = d.copy().view(np.float16).astype(f32).reshape(nb)
    ls = sub_scales(raw).astype(f32)  # ls-32, exact
    KVf = KV.astype(f32)[codes(raw)]  # [nb,256]
    with np.errstate(invalid="ignore", over="ignore"):
        dl = df[:, None] * ls  # [nb,8], exact
        return (dl.repeat(32, axis=1).astype(f32) * KVf).astype(f32)


def dequant_blocks(raw):
    """raw u8 [nb, 136] (GGUF block layout) -> f32 [nb, 256]."""
    raw = np.ascontiguousarray(raw, np.uint8).reshape(-1, BLOCK_BYTES)
    return dequant_blocks_arrays(raw[:, 0:2], raw)


def dequant_rows(raw, rows, cols):
    """raw u8 [rows * cols/256 * 136] -> f32 [rows, cols]."""
    return dequant_blocks(raw).reshape(rows, cols)


def row_bytes(cols):
    assert cols % QK_K == 0, cols
    return cols // QK_K * BLOCK_BYTES


# ------------------------------------------------------------------ ggml oracle
_ggml = None


def ggml():
    global _ggml
    if _ggml is None:
        g = C.CDLL(__import__("os").environ.get("GGML_BASE_LIB", "libggml-base.so"))
        g.dequantize_row_iq4_xs.argtypes = [C.c_void_p, C.c_void_p, C.c_int64]
        g.quantize_row_iq4_xs_ref.argtypes = [C.c_void_p, C.c_void_p, C.c_int64]
        _ggml = g
    return _ggml


def ggml_dequant(blocks):
    """blocks u8 [..., 136] -> f32 [nb * 256] through llama.cpp."""
    b = np.ascontiguousarray(blocks, np.uint8).reshape(-1, BLOCK_BYTES)
    y = np.empty(b.shape[0] * QK_K, f32)
    ggml().dequantize_row_iq4_xs(b.ctypes.data, y.ctypes.data, y.size)
    return y


def ggml_quantize(x):
    """x f32 [k], k % 256 == 0 -> u8 [k/256, 136] through llama.cpp."""
    x = np.ascontiguousarray(x, f32)
    assert x.size % QK_K == 0
    b = np.empty((x.size // QK_K, BLOCK_BYTES), np.uint8)
    ggml().quantize_row_iq4_xs_ref(x.ctypes.data, b.ctypes.data, x.size)
    return b


if __name__ == "__main__":
    rng = np.random.default_rng(12)
    x = rng.standard_normal(4096).astype(f32)
    b = ggml_quantize(x)
    got = dequant_blocks(b)
    want = ggml_dequant(b).reshape(-1, QK_K)
    bad = int((got.view(np.uint32) != want.view(np.uint32)).sum())
    print(f"smoke: {b.shape[0]} blocks, {bad} of {got.size} words differ from ggml")
