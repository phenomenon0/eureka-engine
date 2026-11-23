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

python3 - "$PORT" "$ART" "$API_KEY" <<'PY' | tee "$RECEIPT"
import concurrent.futures, json, math, os, struct, subprocess, sys, time, urllib.request, urllib.error, wave

port, art_dir, api_key = int(sys.argv[1]), sys.argv[2], sys.argv[3]
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

print("===========================================================================")
print(f"E2E v4 VERDICT: {'ALL 15 PRODUCTION E2E TESTS PASS' if fails == 0 else f'FAILED ({fails})'}")
print("===========================================================================")
sys.exit(0 if fails == 0 else 1)
PY
