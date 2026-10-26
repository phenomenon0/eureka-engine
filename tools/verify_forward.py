#!/usr/bin/env python3
"""Check a complete engine log against the fixed-order numpy F32 forward oracle.

usage: verify_forward.py PACK LOG TOK... [--steps N] [--layers L] [--run BINARY]

--run executes an already-built C binary (one thread by default), exclusively
creates LOG, and records its command and binary/source snapshot hashes. Without
--run, LOG must already exist; current sources are not evidence of its origin.
Every forward, including prefill, requires `step POS MS` and
`logits POS ID1 F32BITS1 ID2 F32BITS2` (decimal U32 bits). LOG must end in wall_ms.
Old token-only receipts are deliberately rejected. Tables show both margins;
comparison stops at the first different top1, before prefixes can diverge.
--exact additionally requires both reported top-two IDs and F32 bits to match;
it does not inspect unreported logits. The numerical oracle is unchanged.
--order v2 swaps every gemv row dot for oracle-v2 (verify_kernels.bsum, reports/
llb-blocked.md); the log's `order` line must agree (a log without one is v1).

Numerics retain kernels/ops.bend's tree reductions and verify_kernels' libm-f64-
rounded transcendental convention. Weights come from pack.bin via read_pack.load,
or, for a Q8 pack (llama-bend-pack-q8/v0), q8_pack.load_q8: ggml's Q8_0 dequant
w = f32(q) * f32(d) in numpy, then the same F32 forward; an F16 pack's f16le
weights are upconverted to F32 at use (exact).
"""

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from read_pack import load
from q8_pack import load_q8
from verify_kernels import (
    tsum,
    pad,
    rms_oracle,
    qknorm_oracle,
    output_gate_oracle,
    rope_oracle,
    soft_oracle,
    silu_oracle,
    gemv_v2,
)
from gated_delta_oracle import (
    recurrent_gated_delta_step,
    recurrent_gated_delta_seq,
    chunk_gated_delta_seq,
    rmsnorm_gated,
    causal_conv1d_step,
    causal_conv1d_seq,
    attn_output_gate,
)

f32 = np.float32
KQ = f32(1) / np.sqrt(f32(128))
ROOT = Path(__file__).resolve().parent.parent
ORDER = "v1"  # set by --order; v1 = the strict tree, THE ground truth


def gemv(W, x):  # y_i = tsum(W[i] * x); rows narrower than x are zero-padded
    if ORDER == "v2":
        return gemv_v2(W, x)
    n = len(x)
    return np.concatenate(
        [
            tsum(np.pad(np.asarray(W[i : i + 8192], f32), ((0, 0), (0, n - W.shape[1]))) * x)
            for i in range(0, len(W), 8192)
        ]
    )


def attn(q, K, V, pos):  # q [2048], K/V [T, 256]; head h reads kv head h >> 3
    o = []
    for h in range(16):
        g = h >> 3
        s = tsum(q[h * 128 : (h + 1) * 128] * K[:, g * 128 : (g + 1) * 128]) * KQ
        p = soft_oracle(s, pos)
        o.append(tsum((V[:, g * 128 : (g + 1) * 128] * p[:, None]).T))
    return np.concatenate(o)


def forward(get, L, caches, tok, pos):
    x = np.array(get("token_embd.weight")[tok], f32)
    for i in range(L):
        w = lambda n: get(f"blk.{i}.{n}.weight")
        K, V = caches[i]
        h = rms_oracle(x, np.array(w("attn_norm")))
        q = rope_oracle(gemv(w("attn_q"), h), pos)
        K[pos] = rope_oracle(gemv(w("attn_k"), h), pos)
        V[pos] = gemv(w("attn_v"), h)
        x = x + gemv(w("attn_output"), attn(q, K, V, pos))
        h = rms_oracle(x, np.array(w("ffn_norm")))
        a = pad(silu_oracle(gemv(w("ffn_gate"), h)), 8192) * pad(
            gemv(w("ffn_up"), h), 8192
        )
        x = x + gemv(w("ffn_down"), a)
    return gemv(
        get("output.weight"), rms_oracle(x, np.array(get("output_norm.weight")))
    )


def validate_manifest(m, pack_size=None):
    """Validate the one supported v0 layout, without reading/hashing tensor data."""
    if (
        not isinstance(m, dict)
        or not isinstance(m.get("config"), dict)
        or not isinstance(m.get("tensors"), dict)
    ):
        raise ValueError("manifest must contain config and tensors objects")
    expected = {
        "architecture": "llama",
        "n_layers": 42,
        "n_heads": 16,
        "n_kv_heads": 2,
        "d_model": 2048,
        "head_dim": 128,
        "d_ff": 6144,
        "vocab_size": 130560,
        "context_length": 131072,
        "rope_freq_base": 5000000.0,
        "rope_dim": 128,
        "rope_type": "norm",
        "rope_scaling": None,
        "rms_norm_eps": 9.999999974752427e-07,
        "tied_embeddings": False,
    }
    for k, v in expected.items():
        if (
            k not in m["config"]
            or type(m["config"][k]) is not type(v)
            or m["config"][k] != v
        ):
            raise ValueError(
                f"unsupported config {k}={m.get('config', {}).get(k)!r}; expected {v!r}"
            )
    q8 = m.get("format") == "llama-bend-pack-q8/v0"
    # f16 pack (the quant-f16 lane): every matrix f16le (the source bits), every vector f32le
    f16 = m.get("format") == "llama-bend-pack-f16/v0"
    for k, v in {
        "format": "llama-bend-pack-q8/v0" if q8 else "llama-bend-pack-f16/v0" if f16 else "llama-bend-pack-f32/v0",
        "dtype": "q8_0-rowsplit" if q8 else "f16le" if f16 else "f32le",
        "alignment": 64,
    }.items():
        if type(m.get(k)) is not type(v) or m.get(k) != v:
            raise ValueError(f"unsupported manifest {k}={m.get(k)!r}; expected {v!r}")
    digest = m.get("pack_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid pack_sha256")
    size = m.get("pack_bytes")
    if (
        type(size) is not int
        or size <= 0
        or (pack_size is not None and size != pack_size)
    ):
        raise ValueError(f"pack size mismatch: manifest {size}, file {pack_size}")
    shapes = {
        "token_embd.weight": [130560, 2048],
        "output.weight": [130560, 2048],
        "output_norm.weight": [2048],
    }
    layer_shapes = {
        "attn_norm": [2048],
        "ffn_norm": [2048],
        "attn_q": [2048, 2048],
        "attn_k": [256, 2048],
        "attn_v": [256, 2048],
        "attn_output": [2048, 2048],
        "ffn_gate": [6144, 2048],
        "ffn_up": [6144, 2048],
        "ffn_down": [2048, 6144],
    }
    shapes.update(
        {f"blk.{i}.{n}.weight": s for i in range(42) for n, s in layer_shapes.items()}
    )
    if set(m["tensors"]) != set(shapes):
        raise ValueError("tensor names do not match the 381-tensor v0 layout")
    spans = []
    for n, s in shapes.items():
        t = m["tensors"][n]
        if not isinstance(t, dict):
            raise ValueError(f"invalid tensor metadata: {n}")
        o, b = t.get("offset"), t.get("bytes")
        if (
            t.get("shape") != s
            or any(type(d) is not int for d in t["shape"])
            or not isinstance(t.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", t["sha256"])
            or type(o) is not int
            or o < 0
            or o % 64
            or type(b) is not int
            or b != nbytes(q8, t, s, f16)
            or o + b > size
        ):
            raise ValueError(f"invalid tensor metadata: {n}")
        spans.append((o, o + b, n))
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("overlapping tensor byte ranges")


def nbytes(q8, t, s, f16=False):
    """F32 bytes, a Q8_0 matrix's row-split bytes (tools/q8_pack.py), or an F16 matrix's."""
    if f16:
        e = 2 if len(s) == 2 else 4
        return e * math.prod(s) if t.get("dtype", "f32le") == ("f16le" if e == 2 else "f32le") else -1
    if not q8 or t.get("src_type") == "F32":
        return 4 * math.prod(s)
    if t.get("src_type") != "Q8_0" or len(s) != 2 or s[1] % 32 or t.get("row_bytes") != s[1] * 17 // 16:
        return -1
    return s[0] * t["row_bytes"]


def parse_log(text, prompt_count, steps, vocab=130560):
    """Reject partial, mixed, or token-only logs before spending time on the oracle."""
    ids, step_ms, logits, wall = [], [], [], None
    for number, line in enumerate(text.splitlines(), 1):
        if not line:
            continue
        if wall is not None:
            raise ValueError(f"line {number}: data after final wall_ms")
        if re.fullmatch(r"\d+", line):
            tok = int(line)
            if tok >= vocab:
                raise ValueError(f"line {number}: token outside vocabulary")
            ids.append(tok)
        elif m := re.fullmatch(r"step (\d+) (\d+)", line):
            pos, ms = map(int, m.groups())
            if pos != len(step_ms):
                raise ValueError(
                    f"line {number}: expected step {len(step_ms)}, got {pos}"
                )
            step_ms.append(ms)
        elif m := re.fullmatch(r"logits (\d+) (\d+) (\d+) (\d+) (\d+)", line):
            pos, a, ab, b, bb = map(int, m.groups())
            if pos != len(logits) or not (a < vocab and b < vocab and a != b):
                raise ValueError(f"line {number}: invalid logits position or ids")
            if max(ab, bb) >= 2**32:
                raise ValueError(f"line {number}: logits bits outside U32")
            za, zb = np.array([ab, bb], dtype="<u4").view("<f4")
            if not np.isfinite([za, zb]).all() or za < zb or (za == zb and a > b):
                raise ValueError(f"line {number}: nonfinite or misordered top2 logits")
            logits.append((a, za, b, zb))
        elif m := re.fullmatch(r"wall_ms (\d+)", line):
            wall = int(m[1])
        elif re.fullmatch(r"order v[12]", line):
            pass
        elif not re.fullmatch(r"op [a-zA-Z_][a-zA-Z_0-9]* \d+", line):
            raise ValueError(f"line {number}: unrecognized engine output: {line!r}")
    count = prompt_count + steps - 1
    if (
        len(ids) != steps
        or len(step_ms) != count
        or len(logits) != count
        or wall is None
    ):
        raise ValueError(
            f"incomplete log: ids {len(ids)}/{steps}, steps {len(step_ms)}/{count}, "
            f"logits {len(logits)}/{count}, final wall_ms {wall}"
        )
    if wall < sum(step_ms):
        raise ValueError("wall_ms is smaller than the sum of step times")
    for k, tok in enumerate(ids):
        if tok != logits[prompt_count - 1 + k][0]:
            raise ValueError(f"generated token {k + 1} disagrees with its logits top1")
    return ids, step_ms, logits, wall


def verify(get, layers, toks, steps, run, exact=False):
    ids, step_ms, logits, wall = run
    count = len(toks) + steps - 1
    dT = (count - 1).bit_length()
    caches = [
        (np.zeros((2**dT, 256), f32), np.zeros((2**dT, 256), f32))
        for _ in range(layers)
    ]
    print(
        f"prompt {toks}, steps {steps}, layers {layers}, T = 2^{dT}, engine wall_ms {wall}\n"
    )
    print(
        "| gen | pos | oracle | engine | oracle top1-top2 | engine top1-top2 | engine step ms | oracle ms |"
    )
    print("|---|---|---|---|---|---|---|---|")
    seq = list(toks)
    for pos in range(count):
        t0 = time.perf_counter()
        z = forward(get, layers, caches, seq[pos], pos)
        elapsed = (time.perf_counter() - t0) * 1000
        if not np.isfinite(z).all():
            raise ValueError(f"nonfinite oracle logits at pos {pos}")
        top = np.argsort(-z.astype(np.float64), kind="stable")[:2]
        mine, second = map(int, top)
        e, ze, e2, ze2 = logits[pos]
        margin, emargin = z[mine] - z[second], ze - ze2
        gen = pos - len(toks) + 2
        label = str(gen) if gen > 0 else "prefill"
        print(
            f"| {label} | {pos} | {mine} | {e} | {margin:.9g} | {emargin:.9g} | "
            f"{step_ms[pos]} | {elapsed:.3f} |",
            flush=True,
        )
        if e != mine:
            print(
                f"\n**DIVERGENCE at {('generated token ' + label) if gen > 0 else 'prefill'} "
                f"(pos {pos})**: oracle top2 {mine},{second} "
                f"z={z[mine]:.9g},{z[second]:.9g} margin={margin:.9g}; "
                f"engine top2 {e},{e2} z={ze:.9g},{ze2:.9g} margin={emargin:.9g}; "
                f"oracle z[engine]={z[e]:.9g}. Stopped before different-prefix comparisons."
            )
            return 1
        if exact and (
            e2 != second
            or ze.view(np.uint32) != z[mine].view(np.uint32)
            or ze2.view(np.uint32) != z[second].view(np.uint32)
        ):
            print(
                f"\nBIT DIVERGENCE at pos {pos}: oracle top2 {mine},{second} "
                f"bits={z[mine].view(np.uint32)},{z[second].view(np.uint32)}; "
                f"engine top2 {e},{e2} bits={ze.view(np.uint32)},{ze2.view(np.uint32)}"
            )
            return 1
        if gen > 0:
            seq.append(mine)
    print(
        f"\nPASS: {len(ids)}/{steps} generated tokens match; {count}/{count} forward top1 ids match"
    )
    if exact:
        print(f"PASS: {count}/{count} reported top-two IDs and F32 bits match")
    return 0


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main(argv):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("pack", type=Path)
    p.add_argument("log", type=Path)
    p.add_argument("toks", type=int, nargs="+")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--layers", type=int, default=42)
    p.add_argument(
        "--exact",
        action="store_true",
        help="also compare reported top-two IDs and F32 bits",
    )
    p.add_argument(
        "--order",
        choices=("v1", "v2"),
        default="v1",
        help="dot order the engine was built with (default v1)",
    )
    p.add_argument(
        "--run",
        type=Path,
        metavar="BINARY",
        help="execute an already-built C binary into a new LOG",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=1,
        help="C runtime threads with --run (default 1)",
    )
    args = p.parse_args(argv)
    if not (
        1 <= args.steps <= 131072
        and 1 <= args.layers <= 42
        and 1 <= args.threads <= 128
    ):
        p.error("steps must be 1..131072, layers 1..42, threads 1..128")
    if any(t < 0 or t >= 130560 for t in args.toks):
        p.error("token ids must be 0..130559")
    if len(args.toks) + args.steps - 1 > 131072:
        p.error("prompt + steps - 1 exceeds the 131072-position context")
    try:
        pack, log = args.pack.resolve(), args.log.resolve()
        m, raw, get = load(pack)
        if m.get("format") == "llama-bend-pack-q8/v0":
            m, raw, get = load_q8(pack)
        validate_manifest(m, len(raw))
        print(f"verify_utc {datetime.now(timezone.utc).isoformat()}")
        print(f"manifest_sha256 {sha256(pack / 'manifest.json')}")
        print(
            f"pack_sha256_manifest {m.get('pack_sha256')} (declared, not rehashed here)"
        )
        for path in (
            Path(__file__),
            ROOT / "tools/verify_kernels.py",
            ROOT / "tools/read_pack.py",
            ROOT / "tools/q8_pack.py",
        ):
            print(f"oracle_source_sha256 {path.resolve()} {sha256(path)}")
        if args.run:
            binary = args.run.resolve(strict=True)
            cmd = [
                str(binary),
                "--threads",
                str(args.threads),
                "--",
                str(pack),
                *map(str, args.toks),
                "--steps",
                str(args.steps),
                "--layers",
                str(args.layers),
            ]
            print(f"command {shlex.join(cmd)}")
            print(f"binary_sha256 {binary} {sha256(binary)}")
            for path in (
                ROOT / "engine.bend",
                ROOT / "kernels/ops.bend",
                ROOT / "kernels/io.bend",
                ROOT / "kernels/order.bend",
            ):
                print(f"source_snapshot_sha256 {path} {sha256(path)}")
            print("source snapshot is not a binary build attestation", flush=True)
            with log.open("x") as out:
                result = subprocess.run(
                    cmd, stdout=out, stderr=subprocess.STDOUT, check=False
                )
            print(f"engine_exit {result.returncode}", flush=True)
            if result.returncode:
                raise ValueError(f"engine exited {result.returncode}; see {log}")
        else:
            print(
                "existing log: engine command/binary/source origin not established by this verification"
            )
        text = log.read_text()
        run = parse_log(text, len(args.toks), args.steps)
        declared = re.findall(r"^order (v[12])$", text, re.M) or ["v1"]
        if declared != [args.order]:
            raise ValueError(f"log declares order {declared}, verifying {args.order}")
        global ORDER
        ORDER = args.order
        print(f"order {ORDER}", flush=True)
        print(
            f"engine_log_sha256 {log} {sha256(log)}",
            flush=True,
        )
        return verify(get, args.layers, args.toks, args.steps, run, exact=args.exact)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"FAIL: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
