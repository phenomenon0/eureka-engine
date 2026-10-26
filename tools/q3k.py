#!/usr/bin/env python3
"""Q3_K block format: the byte-golden layout (llama.cpp ggml-common.h) and the
reference dequant, in one place.

A Q3_K super-block is 256 weights in 110 bytes (ggml-common.h block_q3_K):

    off  0  u8  hmask[32]    a high bit per weight (one of four 2-bit groups)
    off 32  u8  qs[64]       low 2 bits of the quants, four per byte
    off 96  u8  scales[12]   16 6-bit scales, four packed into three words
    off108  f16 d            super-block scale

Dequant (ggml-quants.c dequantize_row_q3_K): the 12 scale bytes are expanded
into 16 int8 scales (scales[is]-32), then for each 128-weight half (n = 0,1),
for j = 0..3 (shift = 2j, high-bit mask m = 1 << (j + 4n)):

    dl = d * (scales[is++] - 32)
    y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4))
    y[l+16] = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4))

for l = 0..15, q advancing 32 per half. d*(scales-32) and dl*code are exact, so
the only f32 multiply pair is ggml's own, in ggml's order.

Rows are kept in the GGUF block layout verbatim: 110 is not 4-aligned, so the
device and CPU kernels use byte loads.
"""

import ctypes as C

import numpy as np

QK_K = 256
BLOCK_BYTES = QK_K // 8 + QK_K // 4 + 12 + 2  # 110
GGML_Q3_K = 11

f32, f16 = np.float32, np.float16


def decode_scales(sc):
    """sc u8 [nb,12] -> int8 [nb,16], ggml's four-word unpack."""
    nb = sc.shape[0]
    a = np.zeros((nb, 4), np.uint32)
    for w in range(3):
        for byte in range(4):
            a[:, w] |= sc[:, w * 4 + byte].astype(np.uint32) << (8 * byte)
    k1, k2 = np.uint32(0x03030303), np.uint32(0x0F0F0F0F)
    tmp = a[:, 2].copy()
    a2 = ((a[:, 0] >> 4) & k2) | (((tmp >> 4) & k1) << 4)
    a3 = ((a[:, 1] >> 4) & k2) | (((tmp >> 6) & k1) << 4)
    a0 = (a[:, 0] & k2) | (((tmp >> 0) & k1) << 4)
    a1 = (a[:, 1] & k2) | (((tmp >> 2) & k1) << 4)
    out = np.empty((nb, 16), np.uint8)
    for wi, w in enumerate((a0, a1, a2, a3)):
        for byte in range(4):
            out[:, wi * 4 + byte] = ((w >> (8 * byte)) & 0xFF).astype(np.uint8)
    return out.view(np.int8)


QLB = np.empty(QK_K, np.int64)   # qs byte index
HM = np.empty(QK_K, np.int64)    # hmask byte index
SH = np.empty(QK_K, np.int64)    # qs shift
BIT = np.empty(QK_K, np.int64)   # hmask bit
SCI = np.empty(QK_K, np.int64)   # scale index
for _c in range(QK_K):
    _n = _c >> 7
    _rr = _c & 127
    _j = _rr >> 5
    _l = _rr & 31
    _w16 = _l >> 4
    _lb = _l & 15
    QLB[_c] = _n * 32 + _lb + 16 * _w16
    HM[_c] = _lb + 16 * _w16
    SH[_c] = 2 * _j
    BIT[_c] = _j + 4 * _n
    SCI[_c] = _n * 8 + _j * 2 + _w16


def dequant_blocks_arrays(d, hm, qs, sc):
    """d f16-bits [nb]; hm u8 [nb,32]; qs u8 [nb,64]; sc u8 [nb,12]
    -> f32 [nb,256]."""
    nb = d.shape[0]
    df = d.copy().view(np.float16).astype(f32).reshape(nb)
    scf = decode_scales(sc).astype(f32)
    code = ((qs[:, QLB] >> SH) & 3).astype(np.int32) - np.where(
        ((hm[:, HM] >> BIT) & 1).astype(bool), 0, 4
    )
    with np.errstate(invalid="ignore", over="ignore"):
        dl = df[:, None] * (scf[:, SCI] - 32.0)
        return (dl * code).astype(f32)


def dequant_blocks(raw):
    """raw u8 [nb, 110] (GGUF block layout) -> f32 [nb, 256]."""
    raw = np.ascontiguousarray(raw, np.uint8).reshape(-1, BLOCK_BYTES)
    return dequant_blocks_arrays(
        raw[:, 108:110], raw[:, 0:32], raw[:, 32:96], raw[:, 96:108]
    )


def dequant_rows(raw, rows, cols):
    """raw u8 [rows * cols/256 * 110] -> f32 [rows, cols]."""
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
        g.dequantize_row_q3_K.argtypes = [C.c_void_p, C.c_void_p, C.c_int64]
        g.quantize_row_q3_K_ref.argtypes = [C.c_void_p, C.c_void_p, C.c_int64]
        _ggml = g
    return _ggml


def ggml_dequant(blocks):
    b = np.ascontiguousarray(blocks, np.uint8).reshape(-1, BLOCK_BYTES)
    y = np.empty(b.shape[0] * QK_K, f32)
    ggml().dequantize_row_q3_K(b.ctypes.data, y.ctypes.data, y.size)
    return y


def ggml_quantize(x):
    x = np.ascontiguousarray(x, f32)
    assert x.size % QK_K == 0
    b = np.empty((x.size // QK_K, BLOCK_BYTES), np.uint8)
    ggml().quantize_row_q3_K_ref(x.ctypes.data, b.ctypes.data, x.size)
    return b


if __name__ == "__main__":
    rng = np.random.default_rng(13)
    x = rng.standard_normal(4096).astype(f32)
    b = ggml_quantize(x)
    got = dequant_blocks(b)
    want = ggml_dequant(b).reshape(-1, QK_K)
    bad = int((got.view(np.uint32) != want.view(np.uint32)).sum())
    print(f"smoke: {b.shape[0]} blocks, {bad} of {got.size} words differ from ggml")
