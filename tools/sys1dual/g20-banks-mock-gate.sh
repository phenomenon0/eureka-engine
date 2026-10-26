#!/usr/bin/env bash
# G20 CPU gate on the G6 mock engine (serve plumbing only; the mock ignores "bank"):
#   W1 single-slot server = serial reference
#   m1 G20 --parallel 2           (S0 regression: identity + real overlap)
#   m2 G20 --banks 2              (S1: one engine, 2 banks: identity, expect NO overlap)
#   m3 G20 --parallel 2 --banks 2 (2 engines x 2 banks: identity + overlap)
#   m4 G20 default vs W1          (N=1 regression)
#   fifo on --banks 2 (peak 1) and --parallel 2 --banks 2 (peak 2); /slots
set -uo pipefail
O=${1:?out dir}; mkdir -p "$O"/{m1,m2,m3,m4,f-b2,f-np2b2}
T=${G20_T:-$(cd "$(dirname "$0")" && pwd)/g6-slots}; G=${G20_SERVE:-/tmp/g20/llb-serve}; W1=${G20_W1:-/tmp/g6/w1art/llb-serve}  # merge-train: env-overridable
export LLB_GGUF=/mock/model.gguf TMPDIR=${TMPDIR:-/tmp/g20/scratch}
start() { local port=$1 bin=$2; shift 2
  "$bin" --port "$port" --alias mock "$@" --engine $T/mock-engine.py -- 2> "$O/server-$port.stderr" &
  echo $! > "$O/server-$port.pid"
  for i in $(seq 1 100); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health)" = 200 ] && return 0; sleep 0.1; done; return 1; }
start 18701 $W1 || echo "w1 start fail"
start 18702 $G --parallel 2 || echo "np2 start fail"
start 18703 $G --banks 2 || echo "b2 start fail"
start 18704 $G --parallel 2 --banks 2 || echo "np2b2 start fail"
start 18705 $G || echo "np1 start fail"
U=http://127.0.0.1
curl -s $U:18703/slots > "$O/slots-b2-idle.json"; echo >> "$O/slots-b2-idle.json"
python3 -I $T/conc.py pairs --serial $U:18701 --par $U:18702 --seeds 1-5 --out "$O/m1" > "$O/m1.log" 2>&1; echo "m1 np2 pairs rc=$?" >> "$O/gate.log"
python3 -I $T/conc.py pairs --serial $U:18701 --par $U:18703 --seeds 1-5 --expect-serial --out "$O/m2" > "$O/m2.log" 2>&1; echo "m2 banks2 pairs(expect-serial) rc=$?" >> "$O/gate.log"
python3 -I $T/conc.py pairs --serial $U:18701 --par $U:18704 --seeds 1-5 --out "$O/m3" > "$O/m3.log" 2>&1; echo "m3 np2xb2 pairs rc=$?" >> "$O/gate.log"
python3 -I $T/conc.py pairs --serial $U:18701 --par $U:18705 --seeds 6-7 --expect-serial --out "$O/m4" > "$O/m4.log" 2>&1; echo "m4 np1-vs-w1 rc=$?" >> "$O/gate.log"
python3 -I $T/conc.py fifo --url $U:18703 --k 6 --slots 1 --out "$O/f-b2" > "$O/fifo-b2.log" 2>&1; echo "fifo banks2 (peak 1) rc=$?" >> "$O/gate.log"
python3 -I $T/conc.py fifo --url $U:18704 --k 7 --slots 2 --out "$O/f-np2b2" > "$O/fifo-np2b2.log" 2>&1; echo "fifo np2xb2 (peak 2) rc=$?" >> "$O/gate.log"
curl -s $U:18703/slots > "$O/slots-b2-after.json"; echo >> "$O/slots-b2-after.json"
curl -s $U:18704/slots > "$O/slots-np2b2-after.json"; echo >> "$O/slots-np2b2-after.json"
curl -s -o /dev/null -w '%{http_code}\n' $U:18705/slots > "$O/slots-np1.code"
for p in 18701 18702 18703 18704 18705; do kill -TERM "$(cat $O/server-$p.pid)"; done; wait
grep -h "shutdown complete" "$O"/server-*.stderr >> "$O/gate.log"
