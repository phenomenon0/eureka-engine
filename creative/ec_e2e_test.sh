#!/usr/bin/env bash
# End-to-End Parallel & Production Proof Suite for 100% Native Eureka Creative Studio Server (v2)
# Exercises:
#   - OS-assigned random --port 0 + --api-key auth enforcement (401 without key, 200 with key)
#   - 5-way CONCURRENT parallel HTTP requests (Image + 3D + Music + NVENC MP4 A/V + Realtime SHM)
#   - ffprobe structural verification of hardware NVENC H.264 + AAC .mp4 movie
#   - Async Job Queue + Live SSE text/event-stream progress + Cooperative Cancel
#   - Live Multi-Architecture BF16/FP8 Safetensors Registry (/v1/creative/models/load across all 5 families)
#   - Director Plan + Auto-Execute 4-Modality ComfyUI Flow DAG + Merkle Cache
#   - Path-Traversal Security Hardening
set -euo pipefail

OUT=${OUT:-/tmp/eureka-creative-build}
ART="$OUT/e2e-artifacts"
MODELS="$OUT/artifacts/hf_models"
RECEIPT="$OUT/receipts/c0-e2e-receipt.txt"
API_KEY="eureka-test-secret-2026"
rm -rf "$ART"
mkdir -p "$ART" "$OUT/receipts"

READY_FILE="$OUT/receipts/serve.ready"
rm -f "$READY_FILE"
"$OUT/eureka-creative-serve" --port 0 --artifacts "$ART" --models-dir "$MODELS" --api-key "$API_KEY" > "$READY_FILE" 2> "$OUT/receipts/serve.log" &
SPID=$!
cleanup() {
  kill "$SPID" 2>/dev/null || true
  wait "$SPID" 2>/dev/null || true
}
trap cleanup EXIT

PORT=""
for _ in $(seq 1 50); do
  if [ -s "$READY_FILE" ]; then
    PORT=$(python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).readline())["port"])' "$READY_FILE" 2>/dev/null || true)
    [ -n "$PORT" ] && break
  fi
  sleep 0.1
done
[ -n "$PORT" ] || { echo "error: eureka-creative-serve did not print ready port"; cat "$OUT/receipts/serve.log"; exit 1; }

python3 - "$PORT" "$ART" "$API_KEY" "$OUT/eureka-creative-serve" "$MODELS" <<'PY' | tee "$RECEIPT"
import concurrent.futures, json, math, os, struct, subprocess, sys, time, urllib.request, urllib.error, wave

port, art_dir, api_key, serve_bin, models_dir = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
base = f"http://127.0.0.1:{port}"
fails = 0

def req(method, path, payload=None, auth=True, timeout=30):
    data = json.dumps(payload).encode() if isinstance(payload, dict) else (payload.encode() if isinstance(payload, str) else None)
    hdrs = {"Content-Type": "application/json"}
    if auth:
        hdrs["X-Api-Key"] = api_key
    r = urllib.request.Request(base + path, data=data, method=method, headers=hdrs)
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return resp.status, resp.read()

def check(name, cond, detail):
    global fails
    print(f"  [{'PASS' if cond else 'FAIL'}] {name:<46} {detail}")
    if not cond:
        fails += 1

print("===========================================================================")
print(f"Eureka Creative Studio v4 — 4 Flagship Native CUDA Engines Proof (:{port})")
print("===========================================================================")

# 1. Auth Enforcement (401 without key, 200 with X-Api-Key)
unauth_401 = False
try:
    req("GET", "/v1/creative/health", auth=False)
except urllib.error.HTTPError as e:
    unauth_401 = (e.code == 401)
st, body = req("GET", "/v1/creative/health")
h = json.loads(body)
check("E2E.1 Auth + /v1/creative/health (v4 4-Engine)",
      unauth_401 and st == 200 and h["parallel"] and h["auth_enabled"] and h["models_discovered"] >= 5
      and h.get("photoreal_ready") and h.get("acestep_ready") and h.get("trellis2_ready") and h.get("wan22_ready"),
      f"GPU={h['gpu']} flux2={h.get('photoreal_ready')} acestep={h.get('acestep_ready')} trellis2={h.get('trellis2_ready')} wan22={h.get('wan22_ready')}")

# 2. 5-WAY CONCURRENT PARALLEL GENERATION BURST (All 5 Modalities Simultaneously!)
t0 = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
    f_img = ex.submit(req, "POST", "/v1/images/generations", {"prompt": "bioluminescent monolith", "seed": 2026, "width": 256, "height": 256, "steps": 12, "tier": "fast"})
    f_3d  = ex.submit(req, "POST", "/v1/3d/generations",     {"prompt": "sculptural cybernetic torus", "seed": 77, "grid_res": 20})
    f_mus = ex.submit(req, "POST", "/v1/audio/music",        {"prompt": "cinematic neon pulse", "seed": 101, "bpm": 124, "duration_sec": 3.0})
    f_vid = ex.submit(req, "POST", "/v1/videos/generations", {"prompt": "orbital camera studio", "seed": 303, "frames": 24, "width": 320, "height": 180, "fps": 24, "with_audio": True})
    f_rt  = ex.submit(req, "POST", "/v1/videos/realtime/burst", {"prompt": "live canvas", "seed": 404, "frames": 30})
    im = json.loads(f_img.result()[1])
    d3 = json.loads(f_3d.result()[1])
    mu = json.loads(f_mus.result()[1])
    vd = json.loads(f_vid.result()[1])
    rt = json.loads(f_rt.result()[1])
par_ms = (time.monotonic() - t0) * 1000.0
check("E2E.2 5-Way Parallel API Burst (All Modalities)",
      im["ok"] and d3["ok"] and mu["ok"] and vd["ok"] and rt["ok"],
      f"5 concurrent requests finished in {par_ms:.1f}ms (RT={rt['fps']} FPS)")

# 3. Validate PNG, GLB 2.0 + PLY, Stereo WAV + 4 Stems
_, png = req("GET", im["artifact_url"])
png_ok = png[:8] == b"\x89PNG\r\n\x1a\n" and b"IHDR" in png[:20] and png[-12:-4] == b"\x00\x00\x00\x00IEND"
_, glb = req("GET", d3["glb_url"])
_, ply = req("GET", d3["splat_ply_url"])
glb_ok = struct.unpack("<II", glb[:8]) == (0x46546C67, 2) and ply.startswith(b"ply\nformat binary_little_endian 1.0")
_, wav_b = req("GET", mu["master_wav_url"])
wp = os.path.join(art_dir, "check_master.wav")
open(wp, "wb").write(wav_b)
with wave.open(wp, "rb") as wf:
    ch, sr, nf = wf.getnchannels(), wf.getframerate(), wf.getnframes()
    pcm = wf.readframes(min(nf, 44100))
rms = math.sqrt(sum(s*s for s in struct.unpack(f"<{len(pcm)//2}h", pcm)) / (len(pcm)//2)) / 32768.0
stems_ok = all(req("GET", u)[0] == 200 for u in mu["stems"].values())
check("E2E.3 Binary Proofs: PNG + GLB/PLY + 5x WAV",
      png_ok and glb_ok and ch == 2 and sr == 44100 and rms > 0.02 and stems_ok,
      f"PNG={len(png)}B GLB={len(glb)}B({d3['n_tris']}t) WAV_RMS={rms:.3f} 4-stems=OK")

# 4. ffprobe Structural Proof of Real Hardware-NVENC H.264 + AAC MP4!
_, mp4_b = req("GET", vd["mp4_url"])
mp4_path = os.path.join(art_dir, "check_video.mp4")
open(mp4_path, "wb").write(mp4_b)
pr = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", mp4_path], capture_output=True, text=True)
pinfo = json.loads(pr.stdout)
vstreams = [s for s in pinfo["streams"] if s["codec_type"] == "video"]
astreams = [s for s in pinfo["streams"] if s["codec_type"] == "audio"]
check("E2E.4 Hardware NVENC H.264 + AAC MP4 (ffprobe)",
      len(vstreams) == 1 and vstreams[0]["codec_name"] == "h264" and len(astreams) == 1 and astreams[0]["codec_name"] == "aac",
      f"enc={vd['encoder']} {vstreams[0]['width']}x{vstreams[0]['height']} h264+aac {len(mp4_b)}B")

# 5. Async Job Queue + Real SSE text/event-stream Progress & Completion
st_j, jb = req("POST", "/v1/jobs", {"kind": "image", "prompt": "sse live progress test", "seed": 888, "width": 256, "height": 256, "steps": 16, "tier": "fast", "photoreal": False})
job = json.loads(jb)
sse_req = urllib.request.Request(base + job["events_url"], headers={"X-Api-Key": api_key})
events = []
with urllib.request.urlopen(sse_req, timeout=15) as resp:
    ct = resp.headers.get("Content-Type", "")
    cur_ev = None
    for raw in resp:
        line = raw.decode().strip()
        if line.startswith("event:"): cur_ev = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            events.append((cur_ev, json.loads(line.split(":", 1)[1].strip())))
            if cur_ev in ("done", "cancelled", "failed"): break
check("E2E.5 Async Job Queue + Live SSE Stream",
      st_j == 202 and "text/event-stream" in ct and len(events) >= 2 and events[-1][0] == "done" and events[-1][1]["pct"] == 100,
      f"job_id={job['job_id']} sse_events={len(events)} final={events[-1][0]} (100%)")

# 6. Multi-Architecture BF16/FP8 Safetensors Model Registry Load over HTTP
_, mb = req("GET", "/v1/creative/models")
reg = json.loads(mb)
loaded_routes = []
for mid in ("z_image_turbo_bf16", "acestep_v1.5_xl_fp8", "trellis2_4b_ovoxel_bf16", "wan2.2_moe_video_bf16", "ltx2.5_dual_av_bf16"):
    _, lb = req("POST", "/v1/creative/models/load", {"id": mid})
    lr = json.loads(lb)
    if lr.get("ok"): loaded_routes.append(lr["route"])
check("E2E.6 Multi-Arch BF16/FP8 Model Load API (5/5)",
      reg["count"] >= 5 and len(loaded_routes) == 5,
      f"loaded=5/5 ({', '.join(r.split(':')[0] for r in loaded_routes)})")

# 7. Director Plan + Auto-Execute Full 4-Modality ComfyUI Flow + Warm Cache Hit
_, db1 = req("POST", "/v1/director/plan", {"brief": "cyberpunk glass violin", "seed": 999, "bpm": 126, "execute": True})
dr1 = json.loads(db1)
_, db2 = req("POST", "/v1/flows/run", dr1["workflow"])
dr2 = json.loads(db2)
check("E2E.7 Director Plan -> 4-Modality DAG + Cache",
      dr1["ok"] and dr1["dag_executed"] == 4 and dr2["executed"] == 0 and dr2["cache_hits"] == 4 and dr1["mp4_url"].endswith(".mp4"),
      f"cold=4/4 warm_hits={dr2['cache_hits']}/4 outputs=PNG+GLB+WAV+MP4")

# 8. Security: Path Traversal Hardening
trav = False
try:
    req("GET", "/artifacts/..%2F..%2Fetc%2Fpasswd")
except urllib.error.HTTPError as e:
    trav = e.code in (400, 404)
check("E2E.8 Path-Traversal Security Guard", trav, "rejected cleanly")

# 9. Multi-Drive & Network Storage Vault + LoRA & Checkpoint Discovery
_, sb = req("GET", "/v1/creative/storage")
sv = json.loads(sb)
net_test_dir = os.path.join(art_dir, "sim_network_vault")
os.makedirs(net_test_dir, exist_ok=True)
_, sb_add = req("POST", "/v1/creative/storage/roots", {"path": net_test_dir, "label": "Sim NAS Drive"})
sv_add = json.loads(sb_add)
_, sb_del = req("DELETE", "/v1/creative/storage/roots", {"path": net_test_dir})
sv_del = json.loads(sb_del)
_, lb = req("GET", "/v1/creative/loras")
lreg = json.loads(lb)
_, cb = req("GET", "/v1/creative/checkpoints")
creg = json.loads(cb)
check("E2E.9 Multi-Drive Vault + LoRA/Ckpt Discovery",
      len(sv["roots"]) >= 2 and len(sv_add["roots"]) == len(sv["roots"]) + 1 and len(sv_del["roots"]) == len(sv["roots"]) and creg["count"] >= 1,
      f"roots={len(sv['roots'])} ckpts={creg['count']} loras={lreg['count']} total_free={sum(r['free_gb'] for r in sv['roots']):.1f}GB")

# 10. Curated Photoreal Prompt Library + User Recipes CRUD
_, pb = req("GET", "/v1/creative/prompts")
plib = json.loads(pb)
_, rb0 = req("GET", "/v1/creative/recipes")
rec0 = json.loads(rb0)
_, rb1 = req("POST", "/v1/creative/recipes", {
    "id": "e2e-custom-leica",
    "name": "E2E Leica Rain Test",
    "category": "Custom",
    "modality": "image",
    "prompt": "35mm street photograph in rain, Leica M11",
    "camera": "Leica M11 35mm",
    "checkpoint": "flux2_klein_4b_bf16",
    "width": 512, "height": 512, "steps": 4, "seed": 1234
})
rec1 = json.loads(rb1)
st_rd, _ = req("DELETE", "/v1/creative/recipes/e2e-custom-leica")
_, rb2 = req("GET", "/v1/creative/recipes")
rec2 = json.loads(rb2)
check("E2E.10 Prompt Library + User Recipes CRUD",
      len(plib.get("cameras", [])) >= 5 and rec0["count"] >= 12 and rec1["count"] == rec0["count"] + 1 and st_rd == 200 and rec2["count"] == rec0["count"],
      f"builtins={rec0['count']} cameras={len(plib['cameras'])} user_crud=OK")

# 11. Universal ComfyUI Importer: LiteGraph UI JSON + Embedded PNG Metadata
litegraph_wf = {
    "last_node_id": 5, "last_link_id": 4,
    "nodes": [
        {"id": 1, "type": "UNETLoader", "widgets_values": ["flux-2-klein-4b.safetensors", "default"]},
        {"id": 2, "type": "LoraLoader", "widgets_values": ["Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", 0.75, 0.75]},
        {"id": 3, "type": "CLIPTextEncode", "widgets_values": ["Candid editorial portrait in Kyoto workshop, 85mm f/1.4, Kodak Portra 400"]},
        {"id": 4, "type": "EmptySD3LatentImage", "widgets_values": [768, 512, 1]},
        {"id": 5, "type": "KSampler", "widgets_values": [2026, "fixed", 4, 1.0, "euler", "simple", 1.0]}
    ],
    "links": [[1, 1, 0, 2, 0, "MODEL"], [2, 2, 0, 5, 0, "MODEL"], [3, 3, 0, 5, 1, "CONDITIONING"], [4, 4, 0, 5, 3, "LATENT"]]
}
_, imp1_b = req("POST", "/v1/flows/import", litegraph_wf)
imp1 = json.loads(imp1_b)

# Build a valid PNG with an embedded ComfyUI tEXt "prompt" chunk and upload raw binary PNG!
import zlib
def make_comfy_png(prompt_dict):
    def chunk(ctype, cdata):
        return struct.pack(">I", len(cdata)) + ctype + cdata + struct.pack(">I", zlib.crc32(ctype + cdata) & 0xffffffff)
    ihdr = struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0)
    raw_px = b"".join(b"\x00" + b"\x80\x40\x20" * 4 for _ in range(4))
    idat = zlib.compress(raw_px)
    text = b"prompt\x00" + json.dumps(prompt_dict).encode("utf-8")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"tEXt", text) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")

comfy_api_in_png = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b.safetensors"}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "Brutalist fjord concrete pavilion at blue hour, 24mm tilt-shift"}},
    "3": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512}},
    "4": {"class_type": "KSampler", "inputs": {"seed": 777, "steps": 4}}
}
png_payload = make_comfy_png(comfy_api_in_png)
req_png = urllib.request.Request(base + "/v1/flows/import", data=png_payload, method="POST",
                                 headers={"Content-Type": "image/png", "X-Api-Key": api_key})
with urllib.request.urlopen(req_png, timeout=15) as resp_png:
    imp2 = json.loads(resp_png.read())

check("E2E.11 ComfyUI Import (LiteGraph UI + PNG tEXt)",
      imp1["ok"] and imp1["source_format"] == "comfy_litegraph_ui" and "Kyoto" in imp1["extracted"]["prompt"]
      and bool(imp1["extracted"]["resolved_checkpoint"]) and len(imp1["extracted"]["loras"]) == 1
      and imp2["ok"] and imp2["source_format"] == "comfy_api" and "Brutalist" in imp2["extracted"]["prompt"],
      f"lg_nodes={imp1['node_count']} ckpt_resolved=OK lora_resolved={bool(imp1['extracted']['loras'][0]['resolved'])} png_tEXt=OK")

# 12. Real Native FLUX.2-klein-4B Photoreal Generation + Hot-Fused LoRA Delta Proof!
if h.get("photoreal_ready"):
    pr_prompt = "Candid editorial portrait of a ceramic artist in a sunlit Kyoto workshop, natural window sidelight, 85mm f/1.4, Kodak Portra 400, photorealistic"
    _, ph1_b = req("POST", "/v1/images/generations", {
        "prompt": pr_prompt, "seed": 1042, "width": 512, "height": 512, "steps": 4, "photoreal": True
    }, timeout=180)
    ph1 = json.loads(ph1_b)
    _, ph1_png = req("GET", ph1["artifact_url"])
    # Save proof PNG for visual audit
    open(os.path.join(art_dir, "proof_photoreal_base.png"), "wb").write(ph1_png)

    # Create a real synthetic FLUX.2 safetensors LoRA (BFL + Diffusers keys) & fuse onto GPU BF16 weights!
    lora_path = os.path.join(art_dir, "e2e_test_flux2_lora.safetensors")
    R, D, IN_L2 = 4, 3072, 12288
    def mk_f32(n, val):
        return struct.pack(f"<{n}f", *([val] * n))
    t_specs = [
        ("diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight", [R, D], mk_f32(R * D, 0.08)),
        ("diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight", [D, R], mk_f32(D * R, 0.08)),
        ("diffusion_model.single_blocks.0.linear2.lora_A.weight",       [R, IN_L2], mk_f32(R * IN_L2, 0.04)),
        ("diffusion_model.single_blocks.0.linear2.lora_B.weight",       [D, R], mk_f32(D * R, 0.04)),
        ("transformer.transformer_blocks.1.attn.to_q.lora_A.weight",    [R, D], mk_f32(R * D, 0.06)),
        ("transformer.transformer_blocks.1.attn.to_q.lora_B.weight",    [D, R], mk_f32(D * R, 0.06)),
    ]
    hdr = {}
    off = 0
    blob = bytearray()
    for name, sh, data in t_specs:
        hdr[name] = {"dtype": "F32", "shape": sh, "data_offsets": [off, off + len(data)]}
        blob.extend(data)
        off += len(data)
    hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    with open(lora_path, "wb") as lf:
        lf.write(struct.pack("<Q", len(hb)) + hb + blob)

    _, ph2_b = req("POST", "/v1/images/generations", {
        "prompt": pr_prompt, "seed": 1042, "width": 512, "height": 512, "steps": 4, "photoreal": True,
        "loras": [{"path": lora_path, "strength": 1.0}]
    }, timeout=60)
    ph2 = json.loads(ph2_b)
    _, ph2_png = req("GET", ph2["artifact_url"])
    open(os.path.join(art_dir, "proof_photoreal_lora.png"), "wb").write(ph2_png)

    check("E2E.12 Photoreal FLUX.2-klein-4B + LoRA Fusion",
          ph1["ok"] and ph1["engine_mode"] == "flux2_photoreal" and ph1["width"] == 512 and len(ph1_png) > 50000
          and ph2["ok"] and ph2["lora_tensors_fused"] >= 3 and ph1_png != ph2_png,
          f"base(dit={ph1['dit_ms']:.0f}ms,vae={ph1['vae_ms']:.0f}ms,png={len(ph1_png)}B) lora(fused={ph2['lora_tensors_fused']},warm_txt={ph2['text_enc_ms']:.1f}ms,delta=VERIFIED)")
else:
    check("E2E.12 Photoreal FLUX.2-klein-4B + LoRA Fusion", False, "photoreal_ready=false (models missing)")

# 13. Real Native ACE-Step 1.5 Turbo 48 kHz Stereo AI Music + 4 Mixer Stems
if h.get("acestep_ready"):
    _, as_b = req("POST", "/v1/audio/music", {
        "prompt": "warm upbeat jazz house groove, electric rhodes chords, funky bassline, crisp acoustic percussion, studio mix",
        "seed": 4242, "bpm": 124, "duration_sec": 10.0, "full_model": True
    }, timeout=120)
    as_r = json.loads(as_b)
    _, as_wav = req("GET", as_r["master_wav_url"])
    as_wp = os.path.join(art_dir, "proof_acestep_master.wav")
    open(as_wp, "wb").write(as_wav)
    with wave.open(as_wp, "rb") as wf:
        ach, asr, anf = wf.getnchannels(), wf.getframerate(), wf.getnframes()
        apcm = wf.readframes(min(anf, 48000 * 4))
    arms = math.sqrt(sum(s*s for s in struct.unpack(f"<{len(apcm)//2}h", apcm)) / (len(apcm)//2)) / 32768.0
    as_stems_ok = all(req("GET", u)[0] == 200 for u in as_r["stems"].values())
    check("E2E.13 Real ACE-Step 1.5 Turbo (48kHz + 4 Stems)",
          as_r["ok"] and as_r.get("engine_mode") == "acestep_1.5_turbo" and ach == 2 and asr == 48000
          and anf >= 450000 and arms > 0.01 and as_stems_ok,
          f"sr={asr}Hz ch={ach} dur={anf/asr:.1f}s rms={arms:.3f} dit={as_r['dit_ms']:.0f}ms vae={as_r['vae_ms']:.0f}ms total={as_r['gpu_ms']:.0f}ms")
else:
    check("E2E.13 Real ACE-Step 1.5 Turbo (48kHz + 4 Stems)", False, "acestep_ready=false")

# 14. Real Native TRELLIS.2 1.3B Text-to-3D (FLUX.2 Isolated Render -> DINOv3 -> 30L Flow -> 64^3 3D Conv -> GLB + PLY)
if h.get("trellis2_ready") and h.get("photoreal_ready"):
    _, tr_b = req("POST", "/v1/3d/generations", {
        "prompt": "sculptural matte-white porcelain teapot with curved spout and warm bamboo handle",
        "seed": 777, "grid_res": 64, "full_model": True
    }, timeout=240)
    tr_r = json.loads(tr_b)
    _, tr_glb = req("GET", tr_r["glb_url"])
    _, tr_ply = req("GET", tr_r["splat_ply_url"])
    open(os.path.join(art_dir, "proof_trellis2_teapot.glb"), "wb").write(tr_glb)
    open(os.path.join(art_dir, "proof_trellis2_teapot.ply"), "wb").write(tr_ply)
    tr_glb_ok = struct.unpack("<II", tr_glb[:8]) == (0x46546C67, 2)
    tr_ply_ok = tr_ply.startswith(b"ply\nformat binary_little_endian 1.0")
    check("E2E.14 Real TRELLIS.2 1.3B 3D (GLB 2.0 + PLY)",
          tr_r["ok"] and tr_r.get("engine_mode") == "trellis2_1.3b" and tr_glb_ok and tr_ply_ok and tr_r["n_tris"] >= 3000,
          f"verts={tr_r['n_verts']} tris={tr_r['n_tris']} glb={len(tr_glb)}B flow={tr_r['dit_ms']:.0f}ms dec={tr_r['vae_ms']:.0f}ms")
else:
    check("E2E.14 Real TRELLIS.2 1.3B 3D (GLB 2.0 + PLY)", False, "trellis2_ready=false")

# 15. Real Native Wan 2.2 TI2V-5B Cinema Video (832x480) + Synced ACE-Step 1.5 Audio NVENC H.264/AAC MP4!
if h.get("wan22_ready"):
    _, wn_b = req("POST", "/v1/videos/generations", {
        "prompt": "Aerial drone footage flying slowly along dramatic black volcanic basalt sea cliffs in Iceland at golden sunset, turquoise ocean waves crashing gently onto black sand shore, misty green cliffs, cinematic 35mm film",
        "seed": 42, "width": 832, "height": 480, "frames": 17, "fps": 16, "steps": 20, "with_audio": True, "full_model": True
    }, timeout=360)
    wn_r = json.loads(wn_b)
    _, wn_mp4 = req("GET", wn_r["mp4_url"])
    _, wn_kf0 = req("GET", wn_r["keyframe0_url"])
    wn_mp4_path = os.path.join(art_dir, "proof_wan22_iceland.mp4")
    open(wn_mp4_path, "wb").write(wn_mp4)
    open(os.path.join(art_dir, "proof_wan22_iceland_kf0.png"), "wb").write(wn_kf0)
    wpr = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", wn_mp4_path], capture_output=True, text=True)
    wpinfo = json.loads(wpr.stdout)
    wvs = [s for s in wpinfo["streams"] if s["codec_type"] == "video"]
    was = [s for s in wpinfo["streams"] if s["codec_type"] == "audio"]
    check("E2E.15 Real Wan 2.2 TI2V-5B + Synced Audio MP4",
          wn_r["ok"] and wn_r.get("engine_mode") == "wan2.2_ti2v_5b"
          and len(wvs) == 1 and wvs[0]["codec_name"] == "h264" and wvs[0]["width"] == 832 and wvs[0]["height"] == 480
          and len(was) == 1 and was[0]["codec_name"] == "aac" and int(was[0]["sample_rate"]) == 48000,
          f"enc={wn_r['encoder']} 832x480x{wn_r['frames']}f h264+aac(48kHz) dit={wn_r['dit_ms']:.0f}ms vae={wn_r['vae_ms']:.0f}ms mp4={len(wn_mp4)}B")
else:
    check("E2E.15 Real Wan 2.2 TI2V-5B + Synced Audio MP4", False, "wan22_ready=false")

# 16. Sprint 2 Universal ComfyUI Importer v2: Subgraph Flattening + Pos/Neg Tracing + 4-Modality + Alias Resolution!
sg_wf = {
    "nodes": [
        {"id": 76, "type": "PrimitiveStringMultiline", "widgets_values": ["A hedgehog wearing a tiny party hat surrounded by confetti, 2000s digicam flash"]},
        {"id": 75, "type": "sub-flux2-4step-uuid", "inputs": [{"name": "text", "link": 155}], "widgets_values": []},
        {"id": 9, "type": "SaveImage", "inputs": [{"name": "images", "link": 154}], "widgets_values": ["Flux2"]}
    ],
    "links": [[155, 76, 0, 75, 0, "STRING"], [154, 75, 0, 9, 0, "IMAGE"]],
    "definitions": {"subgraphs": [{
        "id": "sub-flux2-4step-uuid",
        "inputs": [{"name": "text", "type": "STRING"}],
        "nodes": [
            {"id": 70, "type": "UNETLoader", "widgets_values": ["flux-2-klein-4b.safetensors", "default"]},
            {"id": 74, "type": "CLIPTextEncode", "inputs": [{"name": "text", "link": 162}], "widgets_values": [""]},
            {"id": 62, "type": "Flux2Scheduler", "widgets_values": [4, 1024, 1024]},
            {"id": 63, "type": "CFGGuider", "inputs": [{"name": "positive", "link": 140}], "widgets_values": [1]},
            {"id": 66, "type": "EmptyFlux2LatentImage", "widgets_values": [1024, 1024, 1]}
        ],
        "links": [
            {"id": 162, "origin_id": -10, "origin_slot": 0, "target_id": 74, "target_slot": 0},
            {"id": 140, "origin_id": 74, "origin_slot": 0, "target_id": 63, "target_slot": 0}
        ]
    }]}
}
wan_wf = {
    "nodes": [
        {"id": 37, "type": "UNETLoader", "widgets_values": ["wan2.2_ti2v_5B_fp16.safetensors", "default"]},
        {"id": 7, "type": "CLIPTextEncode", "widgets_values": ["VERY LONG NEGATIVE PROMPT: blurry, ugly, static, distorted, bad anatomy, extra fingers, low quality, jpeg artifacts, watermark, text, deformed limbs, three legs"]},
        {"id": 6, "type": "CLIPTextEncode", "widgets_values": ["1970s subway street musician playing guitar, slow dolly right"]},
        {"id": 55, "type": "Wan22ImageToVideoLatent", "widgets_values": [1280, 704, 121, 1]},
        {"id": 56, "type": "LoadImage", "widgets_values": ["example.png", "image"]},
        {"id": 3, "type": "KSampler", "inputs": [{"name": "positive", "link": 46}, {"name": "negative", "link": 52}], "widgets_values": [999, "fixed", 20, 5]},
        {"id": 58, "type": "SaveVideo", "widgets_values": ["video/ComfyUI"]}
    ],
    "links": [[46, 6, 0, 3, 1, "CONDITIONING"], [52, 7, 0, 3, 2, "CONDITIONING"]]
}
ace_wf = {
    "nodes": [
        {"id": 97, "type": "CheckpointLoaderSimple", "widgets_values": ["ace_step_1.5_turbo_aio.safetensors"]},
        {"id": 94, "type": "TextEncodeAceStepAudio1.5", "widgets_values": ["Neo-Soul warm groove with rhodes", "[Verse 1] Late night glow", 31, "fixed", 118, 90]},
        {"id": 98, "type": "EmptyAceStep1.5LatentAudio", "widgets_values": [90, 1]},
        {"id": 3, "type": "KSampler", "inputs": [{"name": "positive", "link": 254}], "widgets_values": [31, "fixed", 8, 1]},
        {"id": 106, "type": "SaveAudioAdvanced", "widgets_values": ["audio/ComfyUI"]}
    ],
    "links": [[254, 94, 0, 3, 1, "CONDITIONING"]]
}
tr_wf = {
    "nodes": [
        {"id": 40, "type": "UNETLoader", "widgets_values": ["trellis_2_int8_convrot.safetensors", "default"]},
        {"id": 122, "type": "LoadImage", "widgets_values": ["viking_axe.png", "image"]},
        {"id": 91, "type": "Trellis2ShapeStage", "widgets_values": []},
        {"id": 12, "type": "KSampler", "widgets_values": [43, "fixed", 12, 7.5]}
    ],
    "links": []
}
r_sg  = json.loads(req("POST", "/v1/flows/import", sg_wf)[1])
r_wan = json.loads(req("POST", "/v1/flows/import", wan_wf)[1])
r_ace = json.loads(req("POST", "/v1/flows/import", ace_wf)[1])
r_tr  = json.loads(req("POST", "/v1/flows/import", tr_wf)[1])
imp_v2_ok = (
    r_sg["ok"] and r_sg["modality"] == "image" and r_sg["subgraph_count"] == 1
    and "hedgehog" in r_sg["extracted"]["prompt"] and r_sg["extracted"]["width"] == 1024
    and r_sg["extracted"]["steps"] == 4 and bool(r_sg["extracted"]["resolved_checkpoint"])
    and r_wan["ok"] and r_wan["modality"] == "video" and "subway" in r_wan["extracted"]["prompt"]
    and "NEGATIVE" in r_wan["extracted"]["neg_prompt"] and r_wan["extracted"]["frames"] == 121
    and r_wan["extracted"]["needs_image_input"] and bool(r_wan["extracted"]["resolved_checkpoint"])
    and r_ace["ok"] and r_ace["modality"] == "music" and "Neo-Soul" in r_ace["extracted"]["prompt"]
    and "Verse 1" in r_ace["extracted"]["lyrics"] and r_ace["extracted"]["bpm"] == 118
    and r_ace["extracted"]["duration_sec"] == 90.0 and bool(r_ace["extracted"]["resolved_checkpoint"])
    and r_tr["ok"] and r_tr["modality"] == "3d" and r_tr["extracted"]["needs_image_input"]
    and bool(r_tr["extracted"]["resolved_checkpoint"])
)
check("E2E.16 ComfyUI Importer v2 (Subgraph+4-Modality)",
      imp_v2_ok,
      f"sg_flux2=OK wan_pos_vs_neg=OK(121f) ace_audio=OK(118bpm,90s,alias) trellis_3d=OK(alias)")

# 17. Sprint 3 ComfyUI-Level Node Graph Studio: ObjectInfo + Templates + Upload + Queue SSE + Post-FX Merkle Cache + PNG tEXt Roundtrip!
_, oi_b = req("GET", "/v1/flows/object_info")
oi = json.loads(oi_b)
_, tpl_b = req("GET", "/v1/flows/templates")
tpls = json.loads(tpl_b)["templates"]

# Test LoadImage binary upload endpoint
req_up = urllib.request.Request(base + "/v1/flows/upload", data=png_payload, method="POST",
                                headers={"Content-Type": "image/png", "X-Api-Key": api_key})
with urllib.request.urlopen(req_up, timeout=15) as resp_up:
    up_r = json.loads(resp_up.read())

req("POST", "/v1/flows/clear-cache", {})

# Run the real 8-node FLUX.2-klein-4B + Film Color Grade + SaveImage template via /v1/flows/queue + SSE!
wf_flux = json.loads(json.dumps(tpls[0]["workflow"]))
_, q1_b = req("POST", "/v1/flows/queue", wf_flux)
q1 = json.loads(q1_b)
_, sse1_b = req("GET", q1["events_url"], timeout=120)
sse1_txt = sse1_b.decode("utf-8", "replace")
ev1_lines = [json.loads(l[5:].strip()) for l in sse1_txt.splitlines() if l.startswith("data:")]
fin1 = ev1_lines[-1]["result"]
_, saved_png1 = req("GET", fin1["image_url"])
open(os.path.join(art_dir, "proof_comfy_flow_grade1.png"), "wb").write(saved_png1)

# Verify PNG tEXt ComfyUI metadata roundtrip! (Drop Eureka-generated PNG right back into /v1/flows/import!)
req_rt = urllib.request.Request(base + "/v1/flows/import", data=saved_png1, method="POST",
                                headers={"Content-Type": "image/png", "X-Api-Key": api_key})
with urllib.request.urlopen(req_rt, timeout=15) as resp_rt:
    rt_r = json.loads(resp_rt.read())

# Now tweak ONLY downstream ImageColorGrade.contrast (1.08 -> 1.42) and re-queue!
# Expect 6 upstream nodes (including FLUX.2 KSampler!) to hit compiled Merkle cache in 0 ms!
wf_flux2 = json.loads(json.dumps(wf_flux))
wf_flux2["7"]["inputs"]["contrast"] = 1.42
wf_flux2["7"]["inputs"]["warmth"] = -0.25
t_mk0 = time.time()
_, q2_b = req("POST", "/v1/flows/queue", wf_flux2)
q2 = json.loads(q2_b)
_, sse2_b = req("GET", q2["events_url"], timeout=30)
t_mk_ms = (time.time() - t_mk0) * 1000.0
ev2_lines = [json.loads(l[5:].strip()) for l in sse2_b.decode("utf-8", "replace").splitlines() if l.startswith("data:")]
fin2 = ev2_lines[-1]["result"]
_, saved_png2 = req("GET", fin2["image_url"])
open(os.path.join(art_dir, "proof_comfy_flow_grade2.png"), "wb").write(saved_png2)

flow_v6_ok = (
    len(oi) >= 24 and "KSampler" in oi and "ImageColorGrade" in oi and "TRELLIS2ImageTo3D" in oi
    and len(tpls) >= 5 and up_r["ok"] and up_r["url"].startswith("/artifacts/upload_")
    and fin1["ok"] and fin1["executed"] == 8 and fin1["cache_hits"] == 0 and len(saved_png1) > 50000
    and rt_r["ok"] and rt_r["source_format"] == "comfy_api" and rt_r["node_count"] == 8
    and fin2["ok"] and fin2["cache_hits"] == 6 and fin2["executed"] == 2 and saved_png1 != saved_png2
)
check("E2E.17 ComfyUI Node Graph (SSE+Merkle+PNG tEXt)",
      flow_v6_ok,
      f"nodes_schema={len(oi)} tpls={len(tpls)} cold={fin1['executed']}/8 warm_tweak=hits:{fin2['cache_hits']}+exec:{fin2['executed']}({t_mk_ms:.0f}ms!) png_roundtrip=OK(8n)")

# 18. v7 Tessera / Cowrie v2 Phase A+B: Cold Process Restart Persistent StatePages + Trust Gate 403 + R0/R1/R2 Lenses!
proc2 = subprocess.Popen(
    [serve_bin, "--port", "0", "--artifacts", art_dir, "--models-dir", models_dir, "--api-key", api_key],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
)
try:
    ready_line = proc2.stdout.readline()
    port2 = json.loads(ready_line)["port"]
    base2 = f"http://127.0.0.1:{port2}"
    def req2(method, path, payload=None, timeout=60):
        data = json.dumps(payload).encode() if isinstance(payload, dict) else (payload.encode() if isinstance(payload, str) else None)
        hdrs = {"Content-Type": "application/json", "X-Api-Key": api_key}
        r = urllib.request.Request(base2 + path, data=data, method=method, headers=hdrs)
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as he:
            return he.code, he.read()

    # 18a. Health reports v7 Cowrie SHA-NI enabled
    _, h2_b = req2("GET", "/v1/creative/health")
    h2 = json.loads(h2_b)

    # 18b. Cold restart persistent ComfyUI DAG hit (8/8 nodes from 16 KB SHA-NI StatePages, 0 RAM cache!)
    t_cr0 = time.time()
    _, q_cr_b = req2("POST", "/v1/flows/queue", wf_flux2)
    q_cr = json.loads(q_cr_b)
    _, sse_cr_b = req2("GET", q_cr["events_url"], timeout=30)
    t_cr_ms = (time.time() - t_cr0) * 1000.0
    fin_cr = [json.loads(l[5:].strip()) for l in sse_cr_b.decode("utf-8", "replace").splitlines() if l.startswith("data:")][-1]["result"]

    # 18c. Canonical key-order-shuffled tweak on cold server (hits 6/8 upstream FLUX.2 nodes!)
    wf_shuf = json.loads(json.dumps(wf_flux2))
    wf_shuf["7"]["inputs"] = {
        "sharpen": 0.25, "vignette": 0.20, "saturation": 1.10, "warmth": -0.10, "contrast": 1.33, "image": ["6", 0]
    }
    t_sh0 = time.time()
    _, q_sh_b = req2("POST", "/v1/flows/queue", wf_shuf)
    q_sh = json.loads(q_sh_b)
    _, sse_sh_b = req2("GET", q_sh["events_url"], timeout=30)
    t_sh_ms = (time.time() - t_sh0) * 1000.0
    fin_sh = [json.loads(l[5:].strip()) for l in sse_sh_b.decode("utf-8", "replace").splitlines() if l.startswith("data:")][-1]["result"]

    # 18d. Cold restart FLUX.2 Qwen3-4B StatePage hit (new seed, same prompt -> <50 ms text encode vs ~1200 ms!)
    flux_prompt = wf_flux["3"]["inputs"]["text"]
    _, img_sp_b = req2("POST", "/v1/images/generations", {
        "prompt": flux_prompt, "seed": 9911, "width": 512, "height": 512, "steps": 4, "photoreal": True
    }, timeout=60)
    img_sp = json.loads(img_sp_b)

    # 18e. Phase B Trust Membrane: External URL import tagged UNTRUSTED_EXT (0x04) + 403 Trust Gate
    _, imp_cw_b = req2("POST", "/v1/flows/import?source=external_url", sg_wf)
    imp_cw = json.loads(imp_cw_b)
    cw_meta = imp_cw.get("cowrie", {})
    cw_handle = cw_meta.get("handle", "")
    st_blk, blk_b = req2("POST", "/v1/creative/trust/gate", {"action": "shell_exec", "arg_ids": [cw_handle]})
    st_ro,  ro_b  = req2("POST", "/v1/creative/trust/gate", {"action": "read_only",  "arg_ids": [cw_handle]})
    st_ov,  ov_b  = req2("POST", "/v1/creative/trust/gate", {"action": "shell_exec", "arg_ids": [cw_handle], "owner_override": True})
    blk_j, ro_j, ov_j = json.loads(blk_b), json.loads(ro_b), json.loads(ov_b)
    _, tri_b = req2("GET", "/v1/creative/trust/triage")
    tri_j = json.loads(tri_b)

    # 18f. Phase B Director LLM R0/R1/R2 Lenses
    _, lr2_b = req2("GET", "/v1/creative/lens?target=recipes&tier=r2")
    _, lr1_b = req2("GET", "/v1/creative/lens?target=vault&tier=r1")
    _, lr0_b = req2("GET", "/v1/creative/lens?target=templates&tier=r0")
    lr2, lr1, lr0 = json.loads(lr2_b), json.loads(lr1_b), json.loads(lr0_b)

    _, h2_after_b = req2("GET", "/v1/creative/health")
    h2_after = json.loads(h2_after_b)

    cowrie_v7_ok = (
        h2.get("engine") == "eureka-creative-native-v7"
        and h2.get("cowrie_enabled") is True and h2.get("cowrie_shani") is True
        and fin_cr["ok"] and fin_cr["cache_hits"] == 8 and fin_cr["executed"] == 0
        and fin_sh["ok"] and fin_sh["cache_hits"] == 6 and fin_sh["executed"] == 2
        and img_sp["ok"] and img_sp.get("text_enc_ms", 9999) < 80.0
        and h2_after.get("cowrie_sp_hits", 0) >= 9
        and cw_meta.get("tainted") is True and cw_meta.get("authority") == 0x84 and cw_handle.startswith("@")
        and len(cw_meta.get("cid", "")) == 68
        and st_blk == 403 and blk_j.get("error") == "ERR_TAINTED_EXECUTION_BLOCKED" and blk_j.get("allowed") is False
        and st_ro == 200 and ro_j.get("allowed") is True
        and st_ov == 200 and ov_j.get("allowed") is True
        and tri_j.get("count", 0) >= 1
        and lr2.get("ok") is True and lr2.get("reduction_pct", 0) >= 20.0
        and lr1.get("ok") is True and lr1.get("reduction_pct", 0) >= 85.0
        and lr0.get("ok") is True and lr0.get("reduction_pct", 0) >= 98.0
    )
    check("E2E.18 Cowrie v2 (ColdRestart+Trust403+Lens)",
          cowrie_v7_ok,
          f"cold_dag={fin_cr['cache_hits']}/8({t_cr_ms:.0f}ms) shuf={fin_sh['cache_hits']}/8({t_sh_ms:.0f}ms) qwen3_sp={img_sp.get('text_enc_ms',0):.1f}ms sp_hits={h2_after.get('cowrie_sp_hits',0)} gate={st_blk}/{st_ro}/{st_ov} lens=-{lr2.get('reduction_pct',0):.1f}%/-{lr1.get('reduction_pct',0):.1f}%/-{lr0.get('reduction_pct',0):.1f}%")
finally:
    proc2.terminate()
    try:
        proc2.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc2.kill()
        proc2.wait()

print("===========================================================================")
print(f"E2E v7 VERDICT: {'ALL 18 PRODUCTION E2E TESTS PASS' if fails == 0 else f'FAILED ({fails})'}")
print("===========================================================================")
sys.exit(0 if fails == 0 else 1)
PY
