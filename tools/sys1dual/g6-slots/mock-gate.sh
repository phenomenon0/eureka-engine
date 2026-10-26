#!/usr/bin/env bash
# G6 CPU gate on the mock engine: W1 single-slot server (serial ref) vs G6 --parallel 2,
# plus FIFO fairness at N=1 and N=2, /slots, N=1 regression vs the W1 binary.
set -uo pipefail
O=${1:?out dir}; mkdir -p "$O"
T=/tmp/g6/tools; B=/tmp/g6/bin
export LLB_GGUF=/mock/model.gguf TMPDIR=/tmp/g6/scratch
W1=/tmp/w1-server/new/llb-serve
start() { # port binary extra... ; engine = mock
  local port=$1 bin=$2; shift 2
  "$bin" --port "$port" --alias mock "$@" --engine $T/mock-engine.py -- 2> "$O/server-$port.stderr" &
  echo $! > "$O/server-$port.pid"
  for i in $(seq 1 100); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health)" = 200 ] && return 0; sleep 0.1; done
  return 1
}
start 18601 $W1 || echo "w1 start fail"
start 18602 $B/llb-serve --parallel 2 || echo "g6 np2 start fail"
start 18603 $B/llb-serve || echo "g6 np1 start fail"
start 18604 $B/llb-serve --parallel 3 || echo "g6 np3 start fail"
curl -s http://127.0.0.1:18602/slots > "$O/slots-idle.json"; echo >> "$O/slots-idle.json"
python3 -I $T/conc.py pairs --serial http://127.0.0.1:18601 --par http://127.0.0.1:18602 --seeds 1-5 --out "$O" > "$O/pairs.log" 2>&1; echo "pairs rc=$?" >> "$O/gate.log"
# N=1 regression: G6 --parallel 1 vs W1 binary, same requests, serially
python3 -I $T/conc.py pairs --serial http://127.0.0.1:18601 --par http://127.0.0.1:18603 --seeds 6-7 --expect-serial --out "$O/np1" > "$O/np1.log" 2>&1; echo "np1-vs-w1 rc=$? " >> "$O/gate.log"
python3 -I $T/conc.py fifo --url http://127.0.0.1:18604 --k 7 --slots 3 --out "$O" > "$O/fifo-3.log" 2>&1; echo "fifo N=3 rc=$?" >> "$O/gate.log"
python3 -I $T/conc.py fifo --url http://127.0.0.1:18602 --k 6 --slots 2 --out "$O" > "$O/fifo-2.log" 2>&1; echo "fifo N=2 rc=$?" >> "$O/gate.log"
curl -s http://127.0.0.1:18602/slots > "$O/slots-after.json"; echo >> "$O/slots-after.json"
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:18603/slots > "$O/slots-np1.code"
for p in 18601 18602 18603 18604; do kill -TERM "$(cat $O/server-$p.pid)"; done; wait
grep -h "shutdown complete" "$O"/server-*.stderr >> "$O/gate.log"
