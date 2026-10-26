#!/usr/bin/env python3
"""Q2_0 (BitNet b1.58, group 64) and PQ2_0 (Bonsai, group 128) block formats.

Both are the same 2-bit ternary codec, one f16 scale per group:

    block_q2_0  (QK2_0  = 64) : f16 d + qs[16] = 34 bytes
    block_pq2_0 (QK_PQ2_0=128): f16 d + qs[32] = 34 bytes

Quant q in 0..3 maps to (q-1) in {-1, 0, +1, +2}; the value is (q-1)*d.
Weight j sits in qs[j/4] at bit (j%4)*2. d is f16, (q-1) is a small int, so
(q-1)*d is exact. Q2_0 is llama.cpp's line (GGML_TYPE_Q2_0 = 42); PQ2_0 is the
PrismML fork's private group-128 variant (GGML_TYPE_PQ2_0 = 142,
ggml-common.h). Both dequantizers are exported by the pinned libraries.

Rows are kept in the GGUF block layout verbatim: 34 is not 4-aligned, but qs
is a byte array and the kernels read bytes, so alignment is moot.
"""

import ctypes as C
import os

import numpy as np

BLOCK_BYTES = {64: 2 + 64 // 4, 128: 2 + 128 // 4}  # 18 / 34
GGML_Q2_0 = 42
GGML_PQ2_0 = 142
MAINLINE = os.environ.get("GGML_BASE_LIB", "libggml-base.so")
FORK = os.environ.get("GGML_BASE_LIB_FORK", "libggml-base.so")

f32, f16 = np.float32, np.float16


def _idx(qk):
    j = np.arange(qk, dtype=np.int64)
    return j // 4, (j % 4) * 2


BYTE64, BIT64 = _idx(64)
BYTE128, BIT128 = _idx(128)


def dequant_blocks(raw, qk):
    """raw u8 [nb, 34] -> f32 [nb, qk]."""
    raw = np.ascontiguousarray(raw, np.uint8).reshape(-1, BLOCK_BYTES[qk])
    df = raw[:, 0:2].copy().view(np.float16).astype(f32).reshape(-1)
    qs = raw[:, 2:]
    byte, bit = (BYTE64, BIT64) if qk == 64 else (BYTE128, BIT128)
    q = ((qs[:, byte] >> bit) & 3).astype(f32)
    return (q - 1.0) * df[:, None]


def dequant_rows(raw, rows, cols, qk):
    return dequant_blocks(raw, qk).reshape(rows, cols)


def row_bytes(cols, qk):
    assert cols % qk == 0, cols
    return cols // qk * BLOCK_BYTES[qk]


# ------------------------------------------------------------------ ggml oracle
_libs = {}


def _lib(path):
    if path not in _libs:
        g = C.CDLL(path)
        for fn in ("dequantize_row_q2_0", "dequantize_row_pq2_0",
                   "quantize_row_q2_0_ref", "quantize_row_pq2_0_ref"):
            try:
                getattr(g, fn).argtypes = [C.c_void_p, C.c_void_p, C.c_int64]
            except AttributeError:
                pass
        _libs[path] = g
    return _libs[path]


def ggml_dequant(blocks, qk):
    b = np.ascontiguousarray(blocks, np.uint8).reshape(-1, BLOCK_BYTES[qk])
    y = np.empty(b.shape[0] * qk, f32)
    lib = _lib(FORK if qk == 128 else MAINLINE)
    fn = "dequantize_row_pq2_0" if qk == 128 else "dequantize_row_q2_0"
    getattr(lib, fn)(b.ctypes.data, y.ctypes.data, y.size)
    return y


def ggml_quantize(x, qk):
    x = np.ascontiguousarray(x, f32)
    assert x.size % qk == 0
    b = np.empty((x.size // qk, BLOCK_BYTES[qk]), np.uint8)
    lib = _lib(FORK if qk == 128 else MAINLINE)
    fn = "quantize_row_pq2_0_ref" if qk == 128 else "quantize_row_q2_0_ref"
    getattr(lib, fn)(x.ctypes.data, b.ctypes.data, x.size)
    return b


if __name__ == "__main__":
    rng = np.random.default_rng(14)
    for qk, name in ((64, "Q2_0"), (128, "PQ2_0")):
        x = rng.standard_normal(qk * 16).astype(f32)
        b = ggml_quantize(x, qk)
        got = dequant_blocks(b, qk)
        want = ggml_dequant(b, qk).reshape(-1, qk)
        bad = int((got.view(np.uint32) != want.view(np.uint32)).sum())
        print(f"smoke {name}: {b.shape[0]} blocks, {bad} of {got.size} words differ from ggml")
