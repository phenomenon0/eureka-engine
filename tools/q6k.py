#!/usr/bin/env python3
"""Q6_K block format: the byte-golden layout (llama.cpp ggml-common.h) and the
reference dequant, in one place.

A Q6_K super-block is 256 weights in 210 bytes (ggml-common.h block_q6_K):

    off   0  u8   ql[128]     low 4 bits of the quants, two per byte
    off 128  u8   qh[64]      upper 2 bits, four per byte (2 bits each)
    off 192  i8   scales[16]  16 sub-blocks of 16, signed 8-bit
    off 208  f16  d           super-block scale

Dequant (ggml-quants.c dequantize_row_q6_K), per super-block, in two 128-weight
halves (n = 0,1); for l = 0..31, is = l/16:

    q1 = (ql[l+ 0]&0xF) | ((qh[l]>>0 &3)<<4)) - 32   y[l+ 0] = d*sc[is+0]*q1
    q2 = (ql[l+32]&0xF) | ((qh[l]>>2 &3)<<4)) - 32   y[l+32] = d*sc[is+2]*q2
    q3 = (ql[l+ 0]>>4 ) | ((qh[l]>>4 &3)<<4)) - 32   y[l+64] = d*sc[is+4]*q3
    q4 = (ql[l+32]>>4 ) | ((qh[l]>>6 &3)<<4)) - 32   y[l+96] = d*sc[is+6]*q4

then y,ql,qh advance 128,64,32 and sc 8. `d*sc` is exact (f16 significand,
8-bit int) and `(d*sc)*q` with |q| <= 32 is exact too, so the only ops are the
two f32 multiplies ggml performs, in ggml's order (d*sc first).

Rows are kept in the GGUF block layout verbatim (no row-split): 210 is not a
multiple of 16, so a row's super-blocks are 210 bytes each and a 16-byte load
inside a super-block is not guaranteed aligned; the device kernel uses byte
loads (see tools/q6k_probe.cu).
"""

import ctypes as C

import numpy as np

QK_K = 256
BLOCK_BYTES = QK_K // 2 + QK_K // 4 + QK_K // 16 + 2  # 210
GGML_Q6_K = 14

f32, f16 = np.float32, np.float16


# column c (0..255) -> (ql byte, ql nibble high?, qh byte, qh shift, scale index)
QLB = np.empty(QK_K, np.int64)
NIB = np.empty(QK_K, np.int64)
QHB = np.empty(QK_K, np.int64)
QHS = np.empty(QK_K, np.int64)
SCI = np.empty(QK_K, np.int64)
for _c in range(QK_K):
    _n = _c >> 7
    _rr = _c & 127
    _quad = _rr >> 5
    _l = _rr & 31
    QLB[_c] = _n * 64 + (_quad & 1) * 32 + _l
    NIB[_c] = 4 if _quad >= 2 else 0
    QHB[_c] = _n * 32 + _l
    QHS[_c] = 2 * _quad
    SCI[_c] = _n * 8 + _quad * 2 + (_l >> 4)


def dequant_blocks_arrays(d, ql, qh, sc):
    """d f16-bits [nb]; ql u8 [nb,128]; qh u8 [nb,64]; sc i8-bits [nb,16]
    -> f32 [nb,256]. Both multiplies follow ggml: (d*sc)*q in f32."""
    nb = d.shape[0]
    df = d.copy().view(np.float16).astype(f32).reshape(nb)
    scf = sc.copy().view(np.int8).astype(f32).reshape(nb, QK_K // 16)
    q = (
        ((ql[:, QLB] >> NIB) & 0xF)
        | (((qh[:, QHB] >> QHS) & 3) << 4)
    ).astype(f32)
    code = q - 32.0
    with np.errstate(invalid="ignore", over="ignore"):
        return (df[:, None] * scf[:, SCI]).astype(f32) * code


def dequant_blocks(raw):
    """raw u8 [nb, 210] (GGUF block layout) -> f32 [nb, 256]."""
    raw = np.ascontiguousarray(raw, np.uint8).reshape(-1, BLOCK_BYTES)
    return dequant_blocks_arrays(
        raw[:, 208:210], raw[:, 0:128], raw[:, 128:192], raw[:, 192:208]
    )


def dequant_rows(raw, rows, cols):
    """raw u8 [rows * cols/256 * 210] -> f32 [rows, cols]."""
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
        g.dequantize_row_q6_K.argtypes = [C.c_void_p, C.c_void_p, C.c_int64]
        g.quantize_row_q6_K_ref.argtypes = [C.c_void_p, C.c_void_p, C.c_int64]
        _ggml = g
    return _ggml


def ggml_dequant(blocks):
    """blocks u8 [..., 210] -> f32 [nb * 256] through llama.cpp."""
    b = np.ascontiguousarray(blocks, np.uint8).reshape(-1, BLOCK_BYTES)
    y = np.empty(b.shape[0] * QK_K, f32)
    ggml().dequantize_row_q6_K(b.ctypes.data, y.ctypes.data, y.size)
    return y


def ggml_quantize(x):
    """x f32 [k], k % 256 == 0 -> u8 [k/256, 210] through llama.cpp."""
    x = np.ascontiguousarray(x, f32)
    assert x.size % QK_K == 0
    b = np.empty((x.size // QK_K, BLOCK_BYTES), np.uint8)
    ggml().quantize_row_q6_K_ref(x.ctypes.data, b.ctypes.data, x.size)
    return b


if __name__ == "__main__":
    rng = np.random.default_rng(12)
    x = rng.standard_normal(4096).astype(f32)
    b = ggml_quantize(x)
    got = dequant_blocks(b)
    want = ggml_dequant(b).reshape(-1, QK_K)
    bad = int((got.view(np.uint32) != want.view(np.uint32)).sum())
    print(f"smoke: {b.shape[0]} blocks, {bad} of {got.size} words differ from ggml")
