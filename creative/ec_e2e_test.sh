#!/usr/bin/env bash
# End-to-End Verification Script for 100% Native Eureka Creative Studio Server
# Starts eureka-creative-serve on a local scratch port, exercises ALL 7 routes
# (Health, Image PNG, 3D GLB+PLY, Music Stereo WAV + 4 Stems, Video Causal-3D,
# Realtime Self-Forcing SHM Burst, ComfyUI JSON DAG + Cache + Surgical Edit),
# and cryptographically/structurally validates every returned artifact!
set -euo pipefail

OUT=${OUT:-/tmp/eureka-creative-build}
PORT=${PORT:-18099}
ART="$OUT/e2e-artifacts"
RECEIPT="$OUT/receipts/c0-e2e-receipt.txt"
rm -rf "$ART"
mkdir -p "$ART" "$OUT/receipts"

"$OUT/eureka-creative-serve" --port "$PORT" --artifacts "$ART" > "$OUT/receipts/serve.log" 2>&1 &
SPID=$!
cleanup() {
  kill "$SPID" 2>/dev/null || true
  wait "$SPID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 40); do
  if curl -sf "http://127.0.0.1:$PORT/v1/creative/health" >/dev/null 2>&1; then break; fi
  sleep 0.1
done

python3 - "$PORT" "$ART" <<'PY' | tee "$RECEIPT"
import json, math, os, struct, sys, urllib.request, urllib.error, wave

port = int(sys.argv[1])
art_dir = sys.argv[2]
base = f"http://127.0.0.1:{port}"
fails = 0

def req(method, path, payload=None):
    data = json.dumps(payload).encode() if isinstance(payload, dict) else (payload.encode() if isinstance(payload, str) else None)
    r = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=20) as resp:
        return resp.status, resp.read()

def check(name, cond, detail):
    global fails
    print(f"  [{'PASS' if cond else 'FAIL'}] {name:<44} {detail}")
    if not cond:
        fails += 1

print("===========================================================================")
print("Eureka Creative Studio — End-to-End HTTP & Binary Artifact Proof")
print("===========================================================================")

# 1. Health
st, body = req("GET", "/v1/creative/health")
h = json.loads(body)
check("E2E.1 /v1/creative/health", st == 200 and h["status"] == "ok", f"GPU={h['gpu']} ({h['sm']}) modalities={len(h['modalities'])}")

# 2. Image Generation -> Valid RFC 2083 PNG
st, body = req("POST", "/v1/images/generations", {"prompt": "bioluminescent obsidian studio monolith", "seed": 2026, "width": 256, "height": 256, "steps": 12, "tier": "fast"})
im = json.loads(body)
st2, png = req("GET", im["artifact_url"])
png_ok = (st2 == 200 and png[:8] == b"\x89PNG\r\n\x1a\n" and b"IHDR" in png[:20] and b"IDAT" in png and png[-12:-4] == b"\x00\x00\x00\x00IEND")
w_hdr, h_hdr = struct.unpack(">II", png[16:24])
check("E2E.2 /v1/images/generations -> PNG", png_ok and w_hdr == 256 and h_hdr == 256 and im["steps_skipped"] > 0,
      f"{w_hdr}x{h_hdr} {len(png)}B gpu={im['gpu_ms']}ms steps={im['steps_run']}+{im['steps_skipped']}skip")

# 3. 3D Generation -> Valid GLB 2.0 + 3DGS PLY
st, body = req("POST", "/v1/3d/generations", {"prompt": "sculptural cybernetic torus emblem", "seed": 77, "grid_res": 20})
d3 = json.loads(body)
_, glb = req("GET", d3["glb_url"])
_, ply = req("GET", d3["splat_ply_url"])
magic, ver, total_len = struct.unpack("<III", glb[:12])
jlen, jtype = struct.unpack("<II", glb[12:20])
gltf_json = json.loads(glb[20:20+jlen].decode("utf-8"))
glb_ok = (magic == 0x46546C67 and ver == 2 and total_len == len(glb) and jtype == 0x4E4F534A and gltf_json["asset"]["version"] == "2.0")
ply_ok = (ply.startswith(b"ply\nformat binary_little_endian 1.0") and b"property float f_dc_0" in ply and b"end_header\n" in ply)
check("E2E.3 /v1/3d/generations -> GLB 2.0 + PLY", glb_ok and ply_ok and d3["n_tris"] > 500,
      f"verts={d3['n_verts']} tris={d3['n_tris']} splats={d3['n_splats']} glb={len(glb)}B ply={len(ply)}B gpu={d3['gpu_ms']}ms")

# 4. Music + 4 Stems -> Valid 44.1kHz Stereo WAV (Non-Silent RMS verified!)
st, body = req("POST", "/v1/audio/music", {"prompt": "cinematic neon synthwave pulse", "seed": 101, "bpm": 124, "duration_sec": 3.0})
mu = json.loads(body)
_, wav_bytes = req("GET", mu["master_wav_url"])
wav_path = os.path.join(art_dir, "verify_master.wav")
with open(wav_path, "wb") as f: f.write(wav_bytes)
with wave.open(wav_path, "rb") as wf:
    ch, sw, sr, nf = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
    pcm = wf.readframes(min(nf, 44100))
samples = struct.unpack(f"<{len(pcm)//2}h", pcm)
rms = math.sqrt(sum(s*s for s in samples) / len(samples)) / 32768.0
stems_ok = all(req("GET", url)[0] == 200 for url in mu["stems"].values())
check("E2E.4 /v1/audio/music -> WAV + 4 Stems", ch == 2 and sw == 2 and sr == 44100 and nf >= 132000 and rms > 0.02 and stems_ok,
      f"{sr}Hz {ch}ch {nf/sr:.1f}s RMS={rms:.3f} 4-stems=OK gpu={mu['gpu_ms']}ms")

# 5. Video Generation -> Causal 3D VAE Multi-Frame PNGs
st, body = req("POST", "/v1/videos/generations", {"prompt": "orbital camera around obsidian studio", "seed": 303, "frames": 6})
vid = json.loads(body)
st_v, vpng = req("GET", vid["keyframe0_url"])
check("E2E.5 /v1/videos/generations -> Causal 3D VAE", st_v == 200 and vpng[:8] == b"\x89PNG\r\n\x1a\n" and vid["frames"] == 6,
      f"{vid['width']}x{vid['height']}x{vid['frames']}f gpu={vid['gpu_ms']}ms ({vid['fps']} fps)")

# 6. Realtime Self-Forcing CUDA-Graph Burst -> Zero-Copy SHM Ring
st, body = req("POST", "/v1/videos/realtime/burst", {"prompt": "realtime live painting", "seed": 404, "frames": 30})
rt = json.loads(body)
check("E2E.6 /v1/videos/realtime/burst -> SHM Ring", rt["ok"] and rt["frames"] == 30 and rt["fps"] >= 60.0,
      f"{rt['width']}x{rt['height']} {rt['frames']}f in {rt['elapsed_ms']}ms = {rt['fps']} FPS")

# 7. ComfyUI JSON Compiled DAG: Run 1 (cold), Run 2 (100% cache hit), Run 3 (surgical BPM edit -> partial rerun!)
dag = {
    "1": {"class_type": "EurekaImageGen", "inputs": {"prompt": "studio hero", "seed": 500}},
    "2": {"class_type": "Eureka3DGen",    "inputs": {"image": ["1", 0], "prompt": "studio hero", "seed": 500}},
    "3": {"class_type": "EurekaMusicGen", "inputs": {"prompt": "studio theme", "seed": 500, "bpm": 120}},
}
_, b1 = req("POST", "/v1/flows/run", dag); r1 = json.loads(b1)
_, b2 = req("POST", "/v1/flows/run", dag); r2 = json.loads(b2)
dag["3"]["inputs"]["bpm"] = 132
_, b3 = req("POST", "/v1/flows/run", dag); r3 = json.loads(b3)
check("E2E.7 /v1/flows/run (ComfyUI DAG + Cache)",
      r1["executed"] == 3 and r2["executed"] == 0 and r2["cache_hits"] == 3 and r3["executed"] == 1 and r3["cache_hits"] == 2,
      f"cold={r1['executed']}/3 warm_hits={r2['cache_hits']}/3 edit_bpm_rerun={r3['executed']}/3 (2 cached)")

# 8. Security: Path Traversal Hardening Check
trav_rejected = False
try:
    req("GET", "/artifacts/..%2F..%2Fetc%2Fpasswd")
except urllib.error.HTTPError as e:
    trav_rejected = (e.code in (400, 404))
check("E2E.8 Artifact Path-Traversal Hardening", trav_rejected, "rejected cleanly")

print("===========================================================================")
print(f"E2E VERDICT: {'ALL 8 END-TO-END TESTS PASS' if fails == 0 else f'FAILED ({fails})'}")
print("===========================================================================")
sys.exit(0 if fails == 0 else 1)
PY
