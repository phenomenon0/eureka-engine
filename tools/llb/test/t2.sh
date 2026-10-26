#!/usr/bin/env bash
# t2.sh TOK DIR CORPUS FUZZ OUT: llb-tok enc ids vs llama.cpp /tokenize (server on $PORT, default 8795) per line.
# Lines are unescaped (\n, \t, \r) before both sides. PASS = every line identical; diffs -> OUT.diff with the bytes.
set -u
tok=$1 dir=$2 port=${PORT:-8795}; out=$5; : > $out.diff; n=0 bad=0
for f in "$3" "$4"; do
  while IFS= read -r line; do
    s=$(printf '%b' "$line"x); s=${s%x}
    a=$($tok enc $dir "$s")
    b=$(jq -n --arg c "$s" '{content: $c, add_special: false, parse_special: true}' | curl -s localhost:$port/tokenize -d @- | jq -r '.tokens | map(tostring) | join(" ")')
    n=$((n + 1))
    if [ "$a" != "$b" ]; then bad=$((bad + 1)); { echo "line $n ($f): $(printf '%s' "$s" | od -An -c | tr -s ' ' | tr '\n' ' ')"; echo "  ours  $a"; echo "  llama $b"; } >> $out.diff; fi
  done < "$f"
done
echo "T2 $((n - bad))/$n identical, $bad differ" | tee $out
