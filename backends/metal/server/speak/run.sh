#!/bin/bash
# run.sh <ep> <model> <outdir> — start serve.py, run bench.py, stop, count orphans.
set -uo pipefail
cd "$(dirname "$0")"
EP=$1 MODEL=$2 OUT=$3
mkdir -p "$OUT"
{ date -Iseconds; uptime; sysctl -n machdep.cpu.brand_string; sw_vers -productVersion; } > "$OUT/env.txt"
pgrep -f 'tts-mac/serve.py' > "$OUT/pre.pids" || true
TTS_EP=$EP TTS_MODEL=$MODEL nohup .venv/bin/python -I "$PWD/serve.py" > "$OUT/server.log" 2>&1 &
SPID=$!
for _ in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:19022/health > "$OUT/health-ready.json" && break; sleep 0.5; done
.venv/bin/python -I "$PWD/bench.py" --out "$OUT" ${BENCH_TEXTS:+--texts "$BENCH_TEXTS"} > "$OUT/bench.log" 2>&1; RC=$?
curl -s -m 5 -o "$OUT/health-after.json" -w '%{http_code}\n' http://127.0.0.1:19022/health > "$OUT/health-after.http"
kill "$SPID"; wait "$SPID" 2>/dev/null
sleep 1
pgrep -f 'tts-mac/serve.py' > "$OUT/post.pids" || true
lsof -nP -iTCP:19022 -sTCP:LISTEN > "$OUT/port-after.txt" 2>&1 || true
echo "orphans=$(comm -13 <(sort "$OUT/pre.pids") <(sort "$OUT/post.pids") | wc -l | tr -d ' ') bench_rc=$RC" | tee "$OUT/orphans.txt"
uptime >> "$OUT/env.txt"
