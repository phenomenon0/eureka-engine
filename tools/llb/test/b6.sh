#!/usr/bin/env bash
# b6.sh TOK DIR OUT FILE...: `llb-tok enc DIR < FILE` vs llama.cpp /tokenize (parse_special:false, add_special:false;
# server on $PORT, default 8795) on each file's whole content. PASS = every file identical; diffs -> OUT.diff.
set -u
tok=$1 dir=$2 out=$3 port=${PORT:-8795}; shift 3; : > $out.diff; n=0 bad=0
for f; do
  n=$((n + 1))
  a=$($tok enc $dir < "$f")
  b=$(jq -n --rawfile c "$f" '{content: $c, add_special: false, parse_special: false}' | curl -s localhost:$port/tokenize -d @- | jq -r '.tokens | map(tostring) | join(" ")')
  if [ "$a" != "$b" ] || [ -z "$b" ]; then bad=$((bad + 1)); { echo "$f"; echo "  ours  $a"; echo "  llama $b"; } >> $out.diff; fi
done
echo "B6 $((n - bad))/$n identical, $bad differ" | tee $out
