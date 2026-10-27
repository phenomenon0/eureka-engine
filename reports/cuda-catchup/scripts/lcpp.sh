#!/usr/bin/env bash
# lcpp.sh GGUF OUT [extra llama-bench args] : llama-bench (build 9d57ce45) tg128 -r 5 and pp512 -r 3, under locks + cap
LB=$HOME/llama.cpp/build/bin/llama-bench; g=$1 o=$2; shift 2
cap() { systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=0 flock -w 3600 /tmp/llb-gpu-bench.lock flock -w 3600 /tmp/bend-bench.lock timeout 1800 "$@"; }
[ -n "$NODEC" ] || cap $LB -m $g -ngl 99 "$@" -p 0 -n 128 -r 5 -o json > $o-dec.json 2> $o-dec.err
[ -n "$NOPF" ] || cap $LB -m $g -ngl 99 "$@" -p 512 -n 0 -r 3 -o json > $o-pf.json 2> $o-pf.err
python3 -c "
import json,sys
for f in sys.argv[1:]:
  try: d=json.load(open(f)); print(f.split('/')[-1], ' '.join('%.1f'%r['avg_ts'] for r in d))
  except Exception as e: pass" $o-dec.json $o-pf.json
