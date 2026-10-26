#!/usr/bin/env python3
"""detect_cli.py — D-FINE-n (COCO-80) on one image file, one JSON line on stdout.

The same pipeline the app ships (public/senses/detect-core.js / detect.html):
  * RGB, stretched to 640x640 (PIL bilinear — the reference that produced
    public/senses/assets/dfine_n_ref.json), /255, NCHW float32
  * the shipped ONNX graph (bend-models dfine-n/dfine_n.onnx)
    on onnxruntime CPU fp32
  * sigmoid over logits, score >= 0.5, best-first, top 100 (no NMS, as shipped)
  * cx,cy,w,h (normalised) -> pixel xyxy in the SOURCE image

Contract (see SCHEMAS.md / M3-SEE): {"detections":[{"label","score","box"}],
"inference_ms", "width", "height", "model"}.
"""
import argparse
import json
import sys
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

SZ = 640
THR = 0.5
NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    a = ap.parse_args()

    try:
        im = Image.open(a.image).convert("RGB")
    except Exception as e:  # noqa: BLE001 - the CLI's whole error surface
        print(f"detect_cli: cannot read {a.image}: {e}", file=sys.stderr)
        return 2
    w, h = im.size
    x = np.ascontiguousarray(
        np.asarray(im.resize((SZ, SZ), Image.BILINEAR), np.float32).transpose(2, 0, 1)[None] / 255.0
    )

    try:
        sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
        t0 = time.perf_counter()
        logits, boxes = sess.run(None, {sess.get_inputs()[0].name: x})
        inference_ms = (time.perf_counter() - t0) * 1e3
    except Exception as e:  # noqa: BLE001
        print(f"detect_cli: inference failed: {e}", file=sys.stderr)
        return 3

    lg = logits.reshape(-1)
    nc = logits.shape[2]
    bx = boxes.reshape(-1, 4)
    hit = [(i, float(1.0 / (1.0 + np.exp(-float(v))))) for i, v in enumerate(lg)]
    hit = sorted((p for p in hit if p[1] >= THR), key=lambda p: -p[1])[:100]
    dets = []
    for i, score in hit:
        q, c = i // nc, i % nc
        b = bx[q]
        dets.append({
            "label": NAMES[c] if c < len(NAMES) else str(c),
            "score": round(score, 4),
            "box": [round(float((b[0] - b[2] / 2) * w), 1), round(float((b[1] - b[3] / 2) * h), 1),
                    round(float((b[0] + b[2] / 2) * w), 1), round(float((b[1] + b[3] / 2) * h), 1)],
        })
    print(json.dumps({"model": "dfine-n", "detections": dets, "inference_ms": round(inference_ms, 1),
                      "width": w, "height": h}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
