#!/usr/bin/env python3
"""Verify the Bend kernels against numpy, on both lanes.

usage: tools/verify_kernels.py [--quick] [--order v2]
  --quick     skips the logits gemv
  --order v2  gemv cases only (the only dot-order kernel), built by tools/build.sh v2,
              interpreted from build/src-v2, oracle gemv_v2 -> llb-blocked-kernels-v2.verify.md

Per case:
  lanes   interp (`bun bend2/main.ts k.bend`, Bend's JS evaluator) and C (`-o` binary);
          their stdout must be byte-identical.
  oracle  numpy f32 in the kernel's fixed op order (tree sums, libm-f64-rounded
          transcendentals) -> must be bit-exact.
  native  numpy's own idiom (pairwise np.sum/mean, BLAS matmul, f32 SIMD exp/cos)
          -> max ULP recorded, reason named. Informational, not a gate.
Cases 26/27 (tools/quant-fix.patch): the Q8_0 and F16 GEMV known-answer tests
of tools/quant_audit.py (t3's oracle, the kernel's association) through
kernels/probe_rb.bend's F.llb.gemv_warp! with tagged row words, built by the
patched compiler (tools/build_gpu.sh, BEND_BUILD, default build/bend-qfix).
Exit 1 on any lane or oracle mismatch. Needs data/kernels/ from tools/mk_inputs.py.
"""

import json, os, subprocess, sys, time
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
D = f"{ROOT}/data/kernels"
BEND = os.path.join(os.environ.get("LLB_BEND", os.path.expanduser("~/bend")), "bend2", "main.ts")
f32, f64 = np.float32, np.float64
EPS = f32(9.999999974752427e-07)
BEND_BUILD = os.environ.setdefault("BEND_BUILD", f"{ROOT}/build/bend-qfix")  # build_gpu.sh reads it
PATCHED = {"probe_rb"}  # kernels on the patched compiler (the F.llb bangs)


def ld(n):
    return np.fromfile(f"{D}/{n}.bin", "<f4")


def tsum(P):  # the kernels' reduction: adjacent pairs, then pairs of pairs (last axis)
    while P.shape[-1] > 1:
        P = P[..., 0::2] + P[..., 1::2]
    return P[..., 0]


# oracle-v2 (reports/llb-blocked.md): lane j sums P[..., j::B] left to right,
# starting from +0.0, one f32 add per step; then the B lane sums merge by tsum.
# Each step is one elementwise f32 add, so this is the order itself, not a model of it.
def bsum(P, B):
    s = np.zeros(P.shape[:-1] + (B,), f32)
    for k in range(0, P.shape[-1], B):
        s = s + P[..., k : k + B]
    return tsum(s)


V2_LANES = 16  # the frozen v2 block size (measured choice, see the report)


def lib(fn, x):  # Base's F32 transcendental: (f32) libm_double((double) x)
    return fn(np.asarray(x, f32).astype(f64)).astype(f32)


def pad(x, n, v=0.0):
    return np.concatenate([x, np.full(n - len(x), v, f32)])


# oracles (same op order as kernels/ops.bend) and numpy-native forms
def rms_oracle(x, w):
    s = f32(1) / np.sqrt(tsum(x * x) / f32(2048) + EPS)
    return (x * s) * w


def rms_native(x, w):
    return x / np.sqrt(np.mean(x * x) + EPS) * w


def qknorm_oracle(v, g, rows=None, d=128):
    if rows is None:
        rows = len(v) // d
    v = np.asarray(v, f32).reshape(rows, d)
    g = np.asarray(g, f32)
    out = np.empty_like(v)
    for r in range(rows):
        row = v[r]
        s = f32(1) / np.sqrt(tsum(row * row) / f32(d) + EPS)
        out[r] = (row * s) * g
    return out.reshape(-1)


def qknorm_native(v, g, rows=None, d=128):
    if rows is None:
        rows = len(v) // d
    v = np.asarray(v, f32).reshape(rows, d)
    g = np.asarray(g, f32)
    s = f32(1) / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + EPS)
    return (v * s * g).reshape(-1)


def output_gate_oracle(x, g):
    sig = f32(1.0) / (f32(1.0) + lib(np.exp, -g))
    return x * sig


def output_gate_native(x, g):
    sig = f32(1.0) / (f32(1.0) + np.exp(-g))
    return x * sig


def rope_oracle(x, pos):
    i = np.arange(64)
    freq = lib(lambda e: np.power(f64(5e6), e), -(2 * i).astype(f32) / f32(128))
    th = np.tile(f32(pos) * freq, len(x) // 128)
    c, s = lib(np.cos, th), lib(np.sin, th)
    a, b = x[0::2], x[1::2]
    y = np.empty_like(x)
    y[0::2], y[1::2] = a * c - b * s, a * s + b * c
    return y


def rope_native(x, pos):
    th = np.tile(
        f32(pos) * np.power(f32(5e6), -(2 * np.arange(64)).astype(f32) / f32(128)),
        len(x) // 128,
    )
    c, s = np.cos(th), np.sin(th)
    a, b = x[0::2], x[1::2]
    y = np.empty_like(x)
    y[0::2], y[1::2] = a * c - b * s, a * s + b * c
    return y


def soft_oracle(s, pos):
    s = np.where(np.arange(len(s)) <= pos, s, f32(-np.inf)).astype(f32)
    e = lib(np.exp, s - s.max())
    return e / tsum(e)


def soft_native(s, pos):
    s = np.where(np.arange(len(s)) <= pos, s, f32(-np.inf)).astype(f32)
    e = np.exp(s - s.max())
    return e / e.sum()


def silu_oracle(x):
    return x / (f32(1) + lib(np.exp, -x))


def silu_native(x):
    return x / (f32(1) + np.exp(-x))


def gemv_oracle(W, x):
    return np.concatenate(
        [tsum(np.asarray(W[i : i + 8192]) * x) for i in range(0, len(W), 8192)]
    )


def gemv_v2(W, x, B=V2_LANES):  # over W's real columns: no padding leaves in v2
    x = x[: W.shape[1]]
    return np.concatenate(
        [bsum(np.asarray(W[i : i + 8192]) * x, B) for i in range(0, len(W), 8192)]
    )


def gemv_native(W, x):
    return np.concatenate(
        [np.asarray(W[i : i + 8192]) @ x for i in range(0, len(W), 8192)]
    )


def ulp(a, b):
    o = lambda v: (lambda s: np.where(s < 0, -(s & 0x7FFFFFFF), s))(
        v.view(np.int32).astype(np.int64)
    )
    return int(np.abs(o(a) - o(b)).max()) if len(a) else 0


def run(cmd):
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True)
    dt = time.perf_counter() - t
    if p.returncode:
        sys.exit(f"FAIL {' '.join(cmd)}: rc={p.returncode}\n{p.stderr.decode()}")
    return p.stdout, dt


def cases(quick, v2=False):
    inp = json.load(open(f"{D}/inputs.json"))
    pack = inp["pack"]
    mat = lambda n: np.memmap(
        pack, "<f4", "r", inp["mats"][n]["offset"], (inp["mats"][n]["rows"], 2048)
    )
    h = {k: ld(k) for k in ("h_attn", "h_ffn", "h_out")}
    for j in range(0 if v2 else 3):
        x, w = ld(f"x{j}"), ld("attn_norm")
        yield (
            f"rms_norm x{j}(tok {inp['x_rows'][j]})·attn_norm",
            "rms_norm",
            [f"{D}/x{j}.bin", f"{D}/attn_norm.bin"],
            rms_oracle(x, w),
            rms_native(x, w),
            "pairwise np.mean vs our tree",
        )
    for v, dv, rows in () if v2 else (("q", 11, 16), ("k", 8, 2)):
        x = ld(v)
        kd = f"{ROOT}/build/kat"
        os.makedirs(kd, exist_ok=True)
        g_head = ld("attn_norm")[:128]
        g_head.tofile(f"{kd}/g_head.bin")
        yield (
            f"qknorm {v}[{rows},128]·g_head",
            "qknorm",
            [f"{D}/{v}.bin", f"{kd}/g_head.bin", str(dv), "7"],
            qknorm_oracle(x, g_head, rows=rows, d=128),
            qknorm_native(x, g_head, rows=rows, d=128),
            "pairwise np.mean vs our tree",
        )
    x0 = ld("x0")
    g0 = ld("attn_norm")[:2048]
    yield (
        "output_gate x0[2048]·attn_norm",
        "output_gate",
        [f"{D}/x0.bin", f"{D}/attn_norm.bin", "11"],
        output_gate_oracle(x0, g0),
        output_gate_native(x0, g0),
        "numpy f32 SIMD exp vs libm-f64-rounded",
    )
    for v, d in () if v2 else (("q", 11), ("k", 8)):
        x = ld(v)
        for pos in (0, 1, 37, 4095, 131071):
            yield (
                f"rope {v}[{len(x) // 128},128] pos={pos}",
                "rope",
                [f"{D}/{v}.bin", str(d), str(pos)],
                rope_oracle(x, pos),
                rope_native(x, pos),
                "numpy f32 SIMD pow/cos/sin vs libm-f64-rounded",
            )
    s = ld("scores")
    for pos in () if v2 else (40, 63):
        yield (
            f"softmax T=64 pos={pos}",
            "softmax",
            [f"{D}/scores.bin", "6", str(pos)],
            soft_oracle(s, pos),
            soft_native(s, pos),
            "numpy f32 SIMD exp + pairwise sum",
        )
    g = ld("g")
    yield from [] if v2 else [(
        "silu g[6144] (Wg·h real)",
        "silu",
        [f"{D}/g.bin", "13", "6144"],
        silu_oracle(g),
        silu_native(g),
        "numpy f32 SIMD exp vs libm-f64-rounded",
    )]
    for n in () if v2 else ("z", "am_ties", "am_zeros", "am_nan", "am_neginf", "am_last"):
        z = ld(n)
        i = np.array([np.argmax(z)], np.uint32)
        yield (
            f"argmax {n}[130560]",
            "argmax",
            [f"{D}/{n}.bin", "17"],
            i,
            i,
            "numpy's rule is the spec",
        )
    for n, hv in (
        ("blk.0.attn_q.weight", "h_attn"),
        ("blk.0.ffn_gate.weight", "h_ffn"),
        ("output.weight", "h_out"),
    ):
        if quick and n == "output.weight":
            continue
        W, m = mat(n), inp["mats"][n]
        yield (
            f"gemv {n} [{m['rows']},2048]",
            "gemv",
            [pack, str(m["offset"]), str(m["rows"]), f"{D}/{hv}.bin"],
            (gemv_v2 if v2 else gemv_oracle)(W, h[hv]),
            gemv_native(W, h[hv]),
            "BLAS sgemv blocking/FMA vs our tree",
        )
    yield from [] if v2 else kat_cases()


def kat_cases():
    """26/27: Q8_0 / F16 GEMV KATs (quant_audit.py t3) on the CPU body of the
    warp bang, F.mv, with the engine's tagged row words"""
    sys.path.insert(0, os.path.dirname(__file__))
    from quant_audit import finite16, gemv_oracle, ref_h16, ref_q8

    rng = np.random.default_rng(27)
    n, c, d = 64, 2048, 9
    kd = f"{ROOT}/build/kat"
    os.makedirs(kd, exist_ok=True)
    x = rng.standard_normal(c).astype(f32)
    np.concatenate([x, np.zeros(8192 - c, f32)]).tofile(f"{kd}/x.bin")
    q = rng.integers(-128, 128, (n, c), dtype=np.int8)
    dh = (np.abs(rng.standard_normal((n, c // 32))) * 2e-3).astype(np.float16).view(np.uint16)
    h = finite16(rng, n * c).reshape(n, c)
    for tag, fm, b, W in (
        ("Q8_0", 1, np.concatenate([q.view(np.uint8), dh.view(np.uint8)], axis=1),
         ref_q8(q, np.repeat(dh, 32, axis=1))),
        ("F16", 2, h.view(np.uint8), ref_h16(h)),
    ):
        b.tofile(f"{kd}/{tag}.bin")
        rb = b.shape[1]
        yield (
            f"{tag} gemv KAT [{n},{c}] rw=0x{1 << 30 | fm << 28 | rb:08x}",
            "probe_rb",
            [f"{kd}/{tag}.bin", f"{kd}/x.bin", str(b.nbytes), str(n), str(d), str(1 << 30 | fm << 28 | rb)],
            gemv_oracle(W, x, d),
            W @ x,
            "BLAS sgemv blocking/FMA vs our tree",
        )
    # 28: Q4_K GEMV KAT (tools/q4k.patch) on the same CPU body, blocks from
    # llama.cpp's quantizer, weights dequantized by q4k.dequant_rows
    from q4k import dequant_rows, ggml_quantize, row_bytes

    qW = rng.standard_normal((n, c)).astype(f32)
    qb = ggml_quantize(qW.ravel()).reshape(n, row_bytes(c))
    qb.tofile(f"{kd}/Q4_K.bin")
    W = dequant_rows(qb.ravel(), n, c)
    yield (
        f"Q4_K gemv KAT [{n},{c}] rw=0x{1 << 30 | 3 << 28 | qb.shape[1]:08x}",
        "probe_rb",
        [f"{kd}/Q4_K.bin", f"{kd}/x.bin", str(qb.nbytes), str(n), str(d),
         str(1 << 30 | 3 << 28 | qb.shape[1])],
        gemv_oracle(W, x, d),
        W @ x,
        "Q4_K blocks from ggml's quantizer, tree association",
    )
    # 29: IQ4_XS GEMV KAT (tools/iq4xs.patch) on the same CPU body, blocks from
    # llama.cpp's quantizer, weights dequantized by iq4xs.dequant_rows
    from iq4xs import dequant_rows as xdeq, ggml_quantize as xquant, row_bytes as xrb

    xb = xquant(rng.standard_normal(n * c).astype(f32)).reshape(n, xrb(c))
    xb.tofile(f"{kd}/IQ4_XS.bin")
    XW = xdeq(xb.ravel(), n, c)
    yield (
        f"IQ4_XS gemv KAT [{n},{c}] rw=0x{1 << 30 | 1 << 27 | xb.shape[1]:08x}",
        "probe_rb",
        [f"{kd}/IQ4_XS.bin", f"{kd}/x.bin", str(xb.nbytes), str(n), str(d),
         str(1 << 30 | 1 << 27 | xb.shape[1])],
        gemv_oracle(XW, x, d),
        XW @ x,
        "IQ4_XS blocks from ggml's quantizer, tree association",
    )


def main():
    quick = "--quick" in sys.argv
    v2 = "--order" in sys.argv and sys.argv[sys.argv.index("--order") + 1] == "v2"
    os.makedirs(f"{ROOT}/build", exist_ok=True)
    src, exe = (f"{ROOT}/build/src-v2", f"{ROOT}/build/v2-") if v2 else (ROOT, f"{ROOT}/build/")
    if v2:
        run([f"{ROOT}/tools/build.sh", "v2", "kernels/gemv.bend", f"{exe}gemv"])
    else:
        for k in ("rms_norm", "qknorm", "output_gate", "rope", "softmax", "silu", "argmax", "gemv"):
            run(["bun", BEND, f"{ROOT}/kernels/{k}.bend", "-o", f"{ROOT}/build/{k}"])
        run([f"{ROOT}/tools/build_gpu.sh", f"{ROOT}/kernels/probe_rb.bend", "-o", f"{ROOT}/build/probe_rb"])
    rows, bad = [], 0
    for name, k, args, want, native, why in cases(quick, v2):
        cc = f"{BEND_BUILD}/bend2/main.ts" if k in PATCHED else BEND
        oi, ti = run(["bun", cc, f"{src}/kernels/{k}.bend", *args])
        oc, tc = run([f"{exe}{k}", *(["--gpu", "off", "--"] if k in PATCHED else []), *args])
        got = np.array(oc.split(), np.uint32)
        if k == "argmax":
            ok = got.tolist() == want.tolist()
            res, nat = (
                (f"exact (idx {got[0]})" if ok else f"MISMATCH {got} vs {want}"),
                "same rule",
            )
        else:
            wb = want.astype(f32).view(np.uint32)
            ok = got.shape == wb.shape and bool((got == wb).all())
            res = (
                "bit-exact"
                if ok
                else (
                    f"MISMATCH {int((got != wb).sum())}/{len(wb)}, max {ulp(got.view(f32), want)} ULP"
                    if got.shape == wb.shape
                    else f"MISMATCH shape {got.shape} vs {wb.shape}"
                )
            )
            nb = native.astype(f32)
            nd = int((got != nb.view(np.uint32)).sum())
            rel = float(np.abs(got.view(f32).astype(f64) - nb).max() / np.abs(nb).max())
            nat = (
                f"{nd}/{len(nb)} differ, max {ulp(got.view(f32), nb)} ULP, max|d|/max|y| {rel:.1e} ({why})"
                if nd
                else "bit-exact"
            )
        lane = "identical" if oi == oc else "DIFFER"
        bad += (not ok) + (oi != oc)
        rows.append((name, res, nat, lane, f"{ti:.2f}", f"{tc:.3f}", len(oc)))
        print(
            f"{name:42} | {res:22} | {lane:9} | interp {ti:7.2f}s | C {tc:7.3f}s | native: {nat}",
            flush=True,
        )
    out = f"{ROOT}/receipts/{'llb-blocked-kernels-v2' if v2 else 'llb-kernels'}.verify.md"
    with open(out, "w") as f:
        f.write(
            f"# verify_kernels {time.strftime('%Y-%m-%dT%H:%M:%S%z')}{' (--quick)' if quick else ''}{' order v2 (oracle gemv_v2)' if v2 else ''}\n\n"
        )
        f.write(
            "| case | vs numpy oracle | interp vs C stdout | interp s | C s | stdout bytes | vs numpy-native |\n|---|---|---|---|---|---|---|\n"
        )
        for r in rows:
            f.write(
                f"| {r[0]} | {r[1]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[2]} |\n"
            )
        f.write(f"\n{len(rows)} cases, {bad} failures\n")
    print(f"{len(rows)} cases, {bad} failures -> {os.path.relpath(out, ROOT)}")
    sys.exit(bool(bad))


if __name__ == "__main__":
    main()
