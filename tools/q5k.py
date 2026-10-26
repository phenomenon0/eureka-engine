#!/usr/bin/env python3
"""Q5_K block format: the byte-golden layout (llama.cpp ggml-common.h) and the
reference dequant, in one place.

A Q5_K super-block is 256 weights in 176 bytes (ggml-common.h block_q5_K):

    off  0  f16  d       super-block scale for the quantized scales
    off  2  f16  dmin    super-block scale for the quantized mins
    off  4  u8   scales[12]   8 sub-blocks of 32, scale/min 6-bit packed
    off 16  u8   qh[32]       5th (high) bit, one per weight
    off 48  u8   qs[128]      low 4-bit quants, two per byte

Dequant (ggml-quants.c dequantize_row_q5_K), per super-block, four 64-weight
groups (is = 0,2,4,6; u1 = 1,4,16,64; u2 = 2,8,32,128):

    (sc, m) = get_scale_min_k4(is + 0); d1, m1 = d*sc, dmin*m
    (sc, m) = get_scale_min_k4(is + 1); d2, m2 = d*sc, dmin*m
    y[0..31]  = d1 * ((qs[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1   l = 0..31
    y[32..63] = d2 * ((qs[l] >> 4)  + (qh[l] & u2 ? 16 : 0)) - m2   same qs bytes

`get_scale_min_k4` is Q4_K's (q4k.decode_scales); d*sc and dmin*m and d1*nib
are exact in f32, so the only rounding is `d1*nib - m1`, ggml's own floats.

Rows are kept in the GGUF block layout verbatim: 176 is a multiple of 16, so a
row of `cols` columns is row_bytes = cols / 256 * 176 and 16-byte loads are
aligned when the tensor start is.
"""

import ctypes as C

import numpy as np

QK_K = 256
K_SCALE_SIZE = 12
BLOCK_BYTES = 2 + 2 + K_SCALE_SIZE + QK_K // 8 + QK_K // 2  # 176
GGML_Q5_K = 13

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


# column c (0..255) -> (qs byte, high-nibble?, qh byte, qh bit, scale index)
QB = np.empty(QK_K, np.int64)
HI = np.empty(QK_K, np.int64)
QHB = np.empty(QK_K, np.int64)
QHS = np.empty(QK_K, np.int64)
SB = np.empty(QK_K, np.int64)
for _c in range(QK_K):
    _j = _c >> 6            # 64-wide group (is pair)
    _l = _c & 63
    _lo = _l < 32
    QB[_c] = _j * 32 + (_l & 31)
    HI[_c] = 0 if _lo else 1
    QHB[_c] = _l & 31
    QHS[_c] = 2 * _j + (0 if _lo else 1)
    SB[_c] = 2 * _j + (0 if _lo else 1)


def dequant_blocks_arrays(d, dm, sc, qh, qs):
    """d,dm f16-bits [nb]; sc u8 [nb,12]; qh u8 [nb,32]; qs u8 [nb,128]
    -> f32 [nb,256]. Both multiplies follow ggml: (d*sc)*nib - dmin*m."""
    nb = d.shape[0]
    df = d.copy().view(np.float16).astype(f32).reshape(nb)
    dmf = dm.copy().view(np.float16).astype(f32).reshape(nb)
    dsc, msc = decode_scales(sc)
    qb = qs.astype(np.uint16)[:, QB]  # [nb,256] selected qs byte per column
    nib = np.where(
        HI[None, :].astype(bool),
        (qb >> 4),
        (qb & 0xF),
    ).astype(np.uint16)
    nib = nib + (((qh[:, QHB] >> QHS) & 1).astype(np.uint16) * 16)
    nib = nib.astype(f32)
    with np.errstate(invalid="ignore", over="ignore"):
        d1 = (df[:, None] * dsc[:, SB].astype(f32)).astype(f32)
        m1 = (dmf[:, None] * msc[:, SB].astype(f32)).astype(f32)
        return (d1 * nib).astype(f32) - m1


def dequant_blocks(raw):
    raw = np.ascontiguousarray(raw, np.uint8).reshape(-1, BLOCK_BYTES)
    return dequant_blocks_arrays(
        raw[:, 0:2], raw[:, 2:4], raw[:, 4:16], raw[:, 16:48], raw[:, 48:176]
    )


def dequant_rows(raw, rows, cols):
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
        g.dequantize_row_q5_K.argtypes = [C.c_void_p, C.c_void_p, C.c_int64]
        g.quantize_row_q5_K_ref.argtypes = [C.c_void_p, C.c_void_p, C.c_int64]
        _ggml = g
    return _ggml


def ggml_dequant(blocks):
    b = np.ascontiguousarray(blocks, np.uint8).reshape(-1, BLOCK_BYTES)
    y = np.empty(b.shape[0] * QK_K, f32)
    ggml().dequantize_row_q5_K(b.ctypes.data, y.ctypes.data, y.size)
    return y


def ggml_quantize(x):
    x = np.ascontiguousarray(x, f32)
    assert x.size % QK_K == 0
    b = np.empty((x.size // QK_K, BLOCK_BYTES), np.uint8)
    ggml().quantize_row_q5_K_ref(x.ctypes.data, b.ctypes.data, x.size)
    return b


if __name__ == "__main__":
    rng = np.random.default_rng(15)
    x = rng.standard_normal(4096).astype(f32)
    b = ggml_quantize(x)
    got = dequant_blocks(b)
    want = ggml_dequant(b).reshape(-1, QK_K)
    bad = int((got.view(np.uint32) != want.view(np.uint32)).sum())
    print(f"smoke: {b.shape[0]} blocks, {bad} of {got.size} words differ from ggml")
