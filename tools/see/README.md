# see: object detection for `POST /v1/see`

This directory holds the detector that `llb-serve` runs for `POST /v1/see`:
D-FINE-N (COCO-80 classes) in ONNX Runtime. The server forks
`see_worker.py` on the first `/v1/see` request and keeps one session open for
its lifetime. For the request and response shapes, see
[HTTP API: Senses](../../docs/reference/http-api.md#senses).

| File | Role |
|---|---|
| `see_worker.py` | The detector child. One ONNX Runtime session; one JSON line in, one line out. |
| `coco80.json` | The COCO-80 label table, indexed by the model's `class_id`. |
| `export_dfine.py` | Exports the official D-FINE-N checkpoint to `dfine_n_coco.onnx`. |

## Before you begin

- Python 3 with `numpy`, `onnxruntime` (CPU) or `onnxruntime-gpu`
  (CUDA 12), and `Pillow`. Without Pillow, the worker falls back to `ffmpeg`
  and `ffprobe` on `PATH`.
- The model file `dfine_n_coco.onnx` (not in the repo, see below).

## Get the model

The model is D-FINE-N exported with its postprocessor baked in (sigmoid, top-300,
box rescale). No NMS step is needed. The graded file is:

| File | Size | sha256 |
|---|---|---|
| `dfine_n_coco.onnx` | 15,534,295 bytes | `4daf426f09e5e1496dd879f2818a65e8706137bd1d9d9f785173df86cee38604` |

To produce it:

1. Clone [D-FINE](https://github.com/Peterande/D-FINE) (Apache-2.0) and check out
   commit `956d1709314c2c6a4df6f34de232054578a7449f`.
2. Download the checkpoint
   [`dfine_n_coco.pth`](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth)
   (sha256 `41973938d2784d38a9836990d805b8392855ebf611aba55f0f7add90e110744c`).
3. Export, with `torch` and `onnx` installed:

   ```sh
   python tools/see/export_dfine.py D_FINE_DIR \
     D_FINE_DIR/configs/dfine/dfine_hgnetv2_n_coco.yml dfine_n_coco.pth MODEL_DIR/dfine_n_coco.onnx
   sha256sum MODEL_DIR/dfine_n_coco.onnx
   ```

   Replace `D_FINE_DIR` with the D-FINE checkout and `MODEL_DIR` with the output
   directory. The script also writes `coco80.json` and a `.deploy.pt` reference
   next to the model.

The graded file was exported with torch 2.14.1 (CPU) and onnx 1.23.2, opset 17.
Other versions can produce different bytes. If the sha256 differs, check the
detections against the example below instead.

## Run it with llb-serve

`build-serve.sh` copies `see_worker.py` and `coco80.json` next to `llb-serve`.
Set these variables before you start the server:

```sh
export LLB_SEE_SCRIPT=$OUT/see_worker.py
export LLB_SEE_MODEL=MODEL_DIR/dfine_n_coco.onnx
export LLB_SEE_PYTHON=PYTHON      # an interpreter with onnxruntime
export LLB_SEE_EP=auto            # auto | cuda | cpu
```

| Variable | Default | Description |
|---|---|---|
| `LLB_SEE_SCRIPT` | Unset (route answers 503) | Path to `see_worker.py`. Turns the route on. |
| `LLB_SEE_MODEL` | Unset | Path to `dfine_n_coco.onnx`. Required. |
| `LLB_SEE_LABELS` | `coco80.json` beside the script | Label table. |
| `LLB_SEE_PYTHON` | `python3` | Interpreter for the worker. |
| `LLB_SEE_EP` | `auto` | `auto` uses CUDA when the wheel has it, otherwise CPU. `cuda` refuses to start without `CUDAExecutionProvider`, so a GPU label never carries CPU numbers. `cpu` forces CPU. |
| `LLB_SEE_THREADS` | `4` | ONNX Runtime intra-op threads. |
| `LLB_SEE_TIMEOUT_MS` | `30000` | Per-request budget. A timed-out worker is restarted on the next request. |
| `LLB_SEE_MAX_IMAGE_BYTES` | `4194304` | Decoded image cap. Larger images get 413. |

The first request pays the session load (about 0.2 s on CPU). The worker's
stderr goes to the server's stderr.

## Example

```sh
printf '{"image":"data:image/jpeg;base64,%s"}' "$(base64 -w0 bus.jpg)" > req.json
curl -s localhost:8080/v1/see -H 'Content-Type: application/json' --data-binary @req.json
```

```text
{"model":"dfine-n","task":"detect","image_wh":[810,1080],
 "detections":[{"box":[48.5,396.0,242.9,906.1],"label":"person","class_id":0,"score":0.9397},...],
 "usage":{"images":1},
 "llb":{"images":1,"encode_ms":39.0,"preprocess_ms":9.27,"infer_ms":29.73,
        "ep":"CPUExecutionProvider","runtime":"onnxruntime","model_sha256":"4daf426f..."}}
```

## Preprocessing contract

RGB, plain stretch to 640 x 640 (bilinear), scaled to [0, 1], no mean or
standard-deviation normalization, no letterbox. The graph takes the original
`[W, H]` and returns boxes in original-image pixels.
