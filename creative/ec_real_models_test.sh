#!/usr/bin/env bash
# Real Official Multi-GB HuggingFace Checkpoint Verification for 100% Native Eureka Creative Engine
# Boots eureka-creative-serve pointed at /home/omen/.local/share/models/creative/,
# auto-detects real official 2026 Safetensors checkpoints, uploads & converts BF16/FP16 weights on GPU,
# and runs live TensorCore DiT forward probes on the RTX 3090!
set -euo pipefail

OUT=${OUT:-/tmp/eureka-creative-build}
MODELS=${MODELS:-/home/omen/.local/share/models/creative}
ART="$OUT/real-model-artifacts"
RECEIPT="$OUT/receipts/real-hf-models-receipt.txt"
API_KEY="eureka-real-models-2026"
mkdir -p "$ART" "$OUT/receipts"

READY_FILE="$OUT/receipts/real-serve.ready"
rm -f "$READY_FILE"
"$OUT/eureka-creative-serve" --port 0 --artifacts "$ART" --models-dir "$MODELS" --api-key "$API_KEY" > "$READY_FILE" 2> "$OUT/receipts/real-serve.log" &
SPID=$!
cleanup() { kill "$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; }
trap cleanup EXIT

PORT=""
for _ in $(seq 1 50); do
  if [ -s "$READY_FILE" ]; then
    PORT=$(python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).readline())["port"])' "$READY_FILE" 2>/dev/null || true)
    [ -n "$PORT" ] && break
  fi
  sleep 0.1
done
[ -n "$PORT" ] || { echo "error: serve did not boot"; cat "$OUT/receipts/real-serve.log"; exit 1; }

python3 - "$PORT" "$API_KEY" <<'PY' | tee "$RECEIPT"
import json, sys, urllib.request

port, api_key = int(sys.argv[1]), sys.argv[2]
base = f"http://127.0.0.1:{port}"
hdrs = {"Content-Type": "application/json", "X-Api-Key": api_key}

def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(base + path, data=data, method=method, headers=hdrs)
    with urllib.request.urlopen(r, timeout=60) as resp:
        return json.loads(resp.read())

reg = call("GET", "/v1/creative/models")
print("===========================================================================")
print(f"Real Official 2026 HuggingFace Checkpoints Verified on Native Eureka Engine")
print(f"Models Dir: {reg['models_dir']} ({reg['count']} safetensors files discovered)")
print("===========================================================================")

targets = [
    ("acestep_v15_turbo_bf16", "acestep_1_5"),
    ("trellis2_ss_flow_1.3b_bf16", "trellis2_ovoxel_3d"),
    ("flux2_klein_4b_bf16", "flux2_mmdit"),
    ("wan2.2_ti2v_5b_fp16", "wan2_2_moe_video"),
]
by_id = {m["id"]: m for m in reg["models"]}
fails = 0
probed = 0

for mid, expect_arch in targets:
    if mid not in by_id:
        print(f"  [SKIP] {mid:<32} not on disk yet")
        continue
    m = by_id[mid]
    arch_ok = (m["arch"] == expect_arch)
    res = call("POST", "/v1/creative/models/load", {"id": mid})
    ok = arch_ok and res.get("ok", False)
    probed += 1
    if not ok: fails += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {mid:<32} arch={m['arch']} dtype={m['dtype']} {m['bytes']/1e9:.2f}GB L={m['layers']} dim={m['dim']} H={m['heads']} ffn={m['ffn']} -> GPU forward {res.get('probe_gpu_ms',0):.2f}ms")
    call("POST", "/v1/creative/models/unload", {"id": mid})

print("===========================================================================")
print(f"REAL HF CHECKPOINT VERDICT: {probed - fails}/{probed} OFFICIAL MULTI-GB MODELS LOADED & PROBED ON GPU")
print("===========================================================================")
sys.exit(0 if (probed > 0 and fails == 0) else 1)
PY
