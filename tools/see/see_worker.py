#!/usr/bin/env python3
"""see_worker — the detector child for llb-serve's POST /v1/see (D-FINE-N, ONNX).

One process, one ORT session, one JSON line in / one JSON line out (the same
request-loop shape as llb-serve's other worker children). llb-serve forks it on
the first /v1/see request (tools/sys1dual/see_route.c).

  in : {"id":N,"image_b64":"...","threshold":0.4,"max_detections":100}
  out: {"model":"dfine-n","task":"detect","image_wh":[W,H],
        "detections":[{"box":[x0,y0,x1,y1],"label":"person","class_id":0,"score":0.9397}],
        "usage":{"images":1},
        "llb":{"images":1,"encode_ms":41.2,"preprocess_ms":9.1,"infer_ms":31.4,
               "ep":"cpu","runtime":"onnxruntime","model_sha256":"4daf..."}}
  err: {"error":{"message":"...","type":"invalid_request_error","param":null,"code":400}}

Boxes are xyxy in ORIGINAL image pixels (the graph bakes in sigmoid -> top-300 ->
rescale by orig_target_sizes). No NMS: D-FINE is a DETR, one query per object.
Preprocessing is the measured contract: RGB, plain stretch to 640x640 bilinear,
/255, no mean/std.

Usage:
  python3 see_worker.py --model dfine_n_coco.onnx [--labels coco80.json] [--ep auto|cuda|cpu]
  --labels defaults to coco80.json beside this script. --ep cuda refuses to start
  without CUDAExecutionProvider (no silent CPU fallback under a GPU label).
"""
import argparse
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import time

MAX_IMAGE_BYTES = 4 * 1024 * 1024
IN = 640


def log(*a):
    print("see_worker:", *a, file=sys.stderr, flush=True)


def err_obj(code, msg, typ=None):
    if typ is None:
        typ = "invalid_request_error" if code < 500 else "unavailable_error"
    return {"error": {"message": msg, "type": typ, "param": None, "code": code}}


def load_labels(path):
    with open(path, "r") as f:
        v = json.load(f)
    if not isinstance(v, list) or len(v) != 80:
        raise SystemExit("labels must be a JSON array of 80 COCO names")
    return v


def pick_providers(ep):
    import onnxruntime as ort
    have = ort.get_available_providers()
    if ep == "cpu":
        return ["CPUExecutionProvider"]
    if ep == "cuda":
        # hard fail: a GPU-labelled run must never serve CPU numbers
        if "CUDAExecutionProvider" not in have:
            raise SystemExit("see_worker: LLB_SEE_EP=cuda but this runtime has no CUDAExecutionProvider (have=%s)" % have)
        return ["CUDAExecutionProvider"]
    # auto: prefer CUDA, silently fall back (a CPU-only wheel must still serve)
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in have \
        else ["CPUExecutionProvider"]


def make_session(model, ep, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    t0 = time.perf_counter()
    sess = ort.InferenceSession(model, so, providers=pick_providers(ep))
    return sess, (time.perf_counter() - t0) * 1e3


def decode_rgb(raw):
    """bytes -> (HxW RGB uint8 array, (W,H)). PIL if present, else ffmpeg."""
    try:
        from PIL import Image
    except ImportError:
        Image = None
    if Image is not None:
        try:
            im = Image.open(io.BytesIO(raw))
            im.load()
            w, h = im.size
            bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
            return im.convert("RGB").resize((IN, IN), bilinear), (w, h)
        except Exception as e:
            raise ValueError("image could not be decoded (unsupported or truncated format)")
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-vf", "scale=%d:%d:flags=bilinear" % (IN, IN),
         "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
        input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or len(p.stdout) != IN * IN * 3:
        raise ValueError((p.stderr or b"ffmpeg could not decode the image").decode("utf-8", "replace")[:300])
    # original size via ffprobe would cost another spawn; the C side only needs a
    # truthful size, so probe once (cheap relative to a failed request)
    q = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0:s=x", "pipe:0"],
        input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        w, h = [int(x) for x in q.stdout.decode().strip().split("x")[:2]]
    except Exception:
        raise ValueError("could not determine image size")
    return p.stdout, (w, h)


def prep(raw):
    """bytes -> (1,3,640,640) float32 NCHW, (W,H)."""
    import numpy as np
    img, (w, h) = decode_rgb(raw)
    if isinstance(img, (bytes, bytearray)):
        a = np.frombuffer(img, dtype=np.uint8).reshape(IN, IN, 3)
    else:
        a = np.asarray(img, dtype=np.uint8)
    x = a.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x), (w, h)


def run_one(sess, labels, raw, threshold, max_det):
    import numpy as np
    t0 = time.perf_counter()
    x, (w, h) = prep(raw)
    t1 = time.perf_counter()
    sz = np.array([[w, h]], dtype=np.int64)
    outs = sess.run(None, {"images": x, "orig_target_sizes": sz})
    t2 = time.perf_counter()
    # graph output order: labels, boxes, scores (boxes already in original px)
    by_name = {o.name: v for o, v in zip(sess.get_outputs(), outs)}
    lab = by_name.get("labels", outs[0] if len(outs) == 3 else None)
    box = by_name.get("boxes", outs[1] if len(outs) == 3 else None)
    sco = by_name.get("scores", outs[2] if len(outs) == 3 else None)
    if lab is None or box is None or sco is None:
        raise ValueError("model did not return labels/boxes/scores")
    dets = []
    for l, b, s in zip(lab[0], box[0], sco[0]):
        s = float(s)
        if s <= threshold:
            continue
        ci = int(l)
        dets.append({
            "box": [round(float(v), 1) for v in b],
            "label": labels[ci] if 0 <= ci < len(labels) else str(ci),
            "class_id": ci,
            "score": round(s, 4),
        })
    dets.sort(key=lambda d: -d["score"])
    if len(dets) > max_det:
        dets = dets[:max_det]
    return dets, (w, h), (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t2 - t0) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "coco80.json"))
    ap.add_argument("--ep", default=os.environ.get("LLB_SEE_EP", "auto"), choices=["auto", "cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=int(os.environ.get("LLB_SEE_THREADS", "4")))
    ap.add_argument("--max-image-bytes", type=int, default=int(os.environ.get("LLB_SEE_MAX_IMAGE_BYTES", str(MAX_IMAGE_BYTES))))
    args = ap.parse_args()

    labels = load_labels(args.labels)
    model_sha = hashlib.sha256()
    with open(args.model, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            model_sha.update(chunk)
    model_sha = model_sha.hexdigest()
    sess, load_ms = make_session(args.model, args.ep, args.threads)
    ep = sess.get_providers()[0]
    if args.ep == "cuda" and ep != "CUDAExecutionProvider":
        raise SystemExit("see_worker: LLB_SEE_EP=cuda but session bound %s — refusing to serve" % ep)
    log("session ready %.0f ms ep=%s model=%s sha=%s" % (load_ms, ep, os.path.basename(args.model), model_sha[:16]))

    out = sys.stdout
    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            if req.get("task", "detect") != "detect":
                raise ValueError("this worker serves task=detect only")
            s = req.get("image_b64")
            if not isinstance(s, str) or not s:
                raise ValueError("image must be a base64 string or data URL")
            if s.startswith("data:"):
                s = s.split(",", 1)[1] if "," in s else ""
            try:
                raw = base64.b64decode(s, validate=True)
            except Exception:
                raise ValueError("image is not valid base64")
            if not raw:
                raise ValueError("image is empty")
            if len(raw) > args.max_image_bytes:
                raise OverflowError(len(raw))
            thr = req.get("threshold", 0.4)
            if not isinstance(thr, (int, float)) or not (0.0 <= float(thr) <= 1.0):
                raise ValueError("threshold must be a number in [0,1]")
            md = req.get("max_detections", 100)
            if not isinstance(md, int) or not (1 <= md <= 300):
                raise ValueError("max_detections must be an integer in [1,300]")
            dets, (w, h), pre_ms, inf_ms, tot_ms = run_one(sess, labels, raw, float(thr), md)
            body = {
                "model": "dfine-n",
                "task": "detect",
                "image_wh": [w, h],
                "detections": dets,
                "usage": {"images": 1},
                "llb": {"images": 1, "encode_ms": round(tot_ms, 2),
                        "preprocess_ms": round(pre_ms, 2), "infer_ms": round(inf_ms, 2),
                        "ep": ep, "runtime": "onnxruntime", "model_sha256": model_sha},
            }
            resp = body
        except OverflowError as e:
            resp = err_obj(413, "image exceeds %d bytes" % args.max_image_bytes)
        except ValueError as e:
            resp = err_obj(400, str(e))
        except Exception as e:  # never die on one bad frame
            log("request failed:", repr(e))
            resp = err_obj(500, "detector failed: %s" % str(e)[:200], "server_error")
        # protocol: "<id> <json>\n" — the id stays out of the public response body
        out.write("%s %s\n" % (rid if rid is not None else 0, json.dumps(resp, separators=(",", ":"))))
        out.flush()


if __name__ == "__main__":
    main()
