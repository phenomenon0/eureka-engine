#!/usr/bin/env python3
"""Q4_K block format: the byte-golden layout (llama.cpp ggml-common.h) and the
reference dequant, in one place.

A Q4_K super-block is 256 weights in 144 bytes (ggml-common.h block_q4_K):

    off  0  f16  d       super-block scale for the quantized scales
    off  2  f16  dmin    super-block scale for the quantized mins
    off  4  u8   scales[12]   8 sub-blocks of 32, scale and min 6-bit packed
    off 16  u8   qs[128]      4-bit quants, two per byte

Dequant (ggml-quants.c dequantize_row_q4_K), per super-block, in four 64-weight
steps (is = 0,2,4,6):

    (sc, m) = get_scale_min_k4(is + 0)
    d1, m1  = d * sc, dmin * m
    y[0..31]  = d1 * (qs[l] & 0xF) - m1        l = 0..31
    (sc, m) = get_scale_min_k4(is + 1)
    d2, m2  = d * sc, dmin * m
    y[32..63] = d2 * (qs[l] >> 4) - m2         qs advances 32 bytes

get_scale_min_k4 (ggml-quants.c:880):
    j < 4 : d = q[j] & 63,        m = q[j+4] & 63
    j >= 4: d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
            m = (q[j+4] >> 4)  | ((q[j]   >> 6) << 4)

Rows are kept in the GGUF block layout verbatim (no row-split): 144 is a
multiple of 16, so a row of `cols` columns (cols % 256 == 0) is
row_bytes = cols / 256 * 144 and every 16-byte load inside it is aligned
provided the tensor start is.
"""

import ctypes as C

import numpy as np

QK_K = 256
K_SCALE_SIZE = 12
BLOCK_BYTES = 2 + 2 + K_SCALE_SIZE + QK_K // 2  # 144
GGML_Q4_K = 12

f32, f16 = np.float32, np.float16


def decode_scales(sc):
    """sc u8 [..., 12] -> (dsc, msc) u8 [..., 8], ggml get_scale_min_k4."""
    nb = sc.shape[0]
    d = np.empty((nb, 8), np.uint8)
    m = np.empty((nb, 8), np.uint8)
    for j in range(4):
        d[:, j] = sc[:, j] & 63
        m[:, j] = sc[:, j + 4] & 63
    for j in range(4, 8):
        d[:, j] = (sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4)
        m[:, j] = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
    return d, m


# column w (0..255) -> sub-block (0..7), and which nibble of qs
SB = np.empty(QK_K, np.int64)
HI = np.empty(QK_K, np.int64)
for _b in range(4):
    SB[_b * 64 : _b * 64 + 32] = 2 * _b
    SB[_b * 64 + 32 : _b * 64 + 64] = 2 * _b + 1
    HI[_b * 64 : _b * 64 + 32] = 0
    HI[_b * 64 + 32 : _b * 64 + 64] = 1


def dequant_blocks_arrays(d, dm, sc, qs):
    """d,dm f16-bits [nb]; sc u8 [nb,12]; qs u8 [nb,128] -> f32 [nb,256].

    The products d*sc and dmin*m and the final d1*nib are exact (f16 significand
    * 6-bit int, then * 4-bit int); the only rounding is  d1*nib - m1, done in
    f32 exactly as ggml's C floats do.
    """
    nb = d.shape[0]
    df = d.copy().view(np.float16).astype(f32).reshape(nb)
    dmf = dm.copy().view(np.float16).astype(f32).reshape(nb)
    dsc, msc = decode_scales(sc)
    q = qs.astype(np.uint16)
    lo = (q & 0xF).astype(f32)
    hi = (q >> 4).astype(f32)
    nib = np.empty((nb, QK_K), f32)
    for b in range(4):
        nib[:, b * 64 : b * 64 + 32] = lo[:, b * 32 : b * 32 + 32]
        nib[:, b * 64 + 32 : b * 64 + 64] = hi[:, b * 32 : b * 32 + 32]
    with np.errstate(invalid="ignore", over="ignore"):
        d1 = (df[:, None] * dsc[:, SB].astype(f32)).astype(f32)
        m1 = (dmf[:, None] * msc[:, SB].astype(f32)).astype(f32)
        return (d1 * nib).astype(f32) - m1


def dequant_blocks(raw):
    """raw u8 [nb, 144] (GGUF block layout) -> f32 [nb, 256]."""
    raw = np.ascontiguousarray(raw, np.uint8).reshape(-1, BLOCK_BYTES)
    return dequant_blocks_arrays(
        raw[:, 0:2], raw[:, 2:4], raw[:, 4:16], raw[:, 16:144]
    )


def dequant_rows(raw, rows, cols):
    """raw u8 [rows * cols/256 * 144] -> f32 [rows, cols]."""
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
        g.dequantize_row_q4_K.argtypes = [C.c_void_p, C.c_void_p, C.c_int64]
        g.quantize_row_q4_K_ref.argtypes = [C.c_void_p, C.c_void_p, C.c_int64]
        _ggml = g
    return _ggml


def ggml_dequant(blocks):
    """blocks u8 [..., 144] -> f32 [nb * 256] through llama.cpp."""
    b = np.ascontiguousarray(blocks, np.uint8).reshape(-1, BLOCK_BYTES)
    y = np.empty(b.shape[0] * QK_K, f32)
    ggml().dequantize_row_q4_K(b.ctypes.data, y.ctypes.data, y.size)
    return y


def ggml_quantize(x):
    """x f32 [k], k % 256 == 0 -> u8 [k/256, 144] through llama.cpp."""
    x = np.ascontiguousarray(x, f32)
    assert x.size % QK_K == 0
    b = np.empty((x.size // QK_K, BLOCK_BYTES), np.uint8)
    ggml().quantize_row_q4_K_ref(x.ctypes.data, b.ctypes.data, x.size)
    return b


if __name__ == "__main__":
    rng = np.random.default_rng(12)
    x = rng.standard_normal(4096).astype(f32)
    b = ggml_quantize(x)
    got = dequant_blocks(b)
    want = ggml_dequant(b).reshape(-1, QK_K)
    bad = int((got.view(np.uint32) != want.view(np.uint32)).sum())
    print(f"smoke: {b.shape[0]} blocks, {bad} of {got.size} words differ from ggml")
