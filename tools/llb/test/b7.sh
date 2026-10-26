#!/usr/bin/env bash
# b7.sh TOK DIR CONVS OUT: specials injection. Conversations whose content holds special-token text (JSON arrays of
# {role, content}; no reasoning). llama.cpp /apply-template renders each one (must equal `llb-tok chat --text`); the
# reference ids = template spans tokenized with parse_special:true + content spans with parse_special:false (llama.cpp
# /tokenize, server on $PORT, default 8795). PASS = ids identical AND the 248045/248046 count == the template's own.
# A content span is found as the first "CONTENT<|im_end|>" after the previous one (so contents must not occur in the
# template text in between). THINK as in t3.sh (default 2, llb chat's default).
set -u
tok=$1 dir=$2 port=${PORT:-8795} out=$4 th=${THINK:-2}; : > $out.diff; : > $out.log; n=0 bad=0; h=$(mktemp -p "${TMPDIR:-/tmp}")
tk() { [ -z "$1" ] && return; jq -n --arg c "$1" --argjson s $2 '{content: $c, add_special: false, parse_special: $s}' | curl -s localhost:$port/tokenize -d @- | jq -r '.tokens | map(tostring) | join(" ")'; }
while IFS= read -r conv; do
  n=$((n + 1)); why=
  echo "$conv" | jq -j '.[] | .role + "\u0000" + .content + "\u0000"' > $h
  a=$($tok chat $dir $h --think $th)
  t=$($tok chat $dir $h --think $th --text; echo x); t=${t%x}
  p=$(echo "$conv" | jq --arg th "$th" '{messages: ., chat_template_kwargs: {enable_thinking: ($th == "1")}}' | curl -s localhost:$port/apply-template -d @- | jq -j .prompt; echo x); p=${p%x}
  [ "$t" = "$p" ] || why="template text differs;"
  rest=$p ref= tpl=0
  while IFS= read -r -d '' c; do
    pre=${rest%%"$c<|im_end|>"*}
    [ "$pre" = "$rest" ] && { why="$why content not found;"; break; }
    ref="$ref $(tk "$pre" true) $(tk "$c" false)"; tpl=$((tpl + $(printf '%s' "$pre" | grep -o '<|im_\(start\|end\)|>' | wc -l)))
    rest=${rest#"$pre$c"}
  done < <(echo "$conv" | jq -j '.[] | (.content | sub("^\\s+"; "") | sub("\\s+$"; "")) + "\u0000"')
  ref="$ref $(tk "$rest" true)"; tpl=$((tpl + $(printf '%s' "$rest" | grep -o '<|im_\(start\|end\)|>' | wc -l)))
  ref=$(echo $ref)
  [ "$a" = "$ref" ] || why="$why ids differ;"
  k=$(echo "$a" | tr ' ' '\n' | grep -cx '24804[56]')
  [ "$k" = "$tpl" ] || why="$why $k im ids vs $tpl in the template;"
  echo "conv $n: im ids $k (template $tpl) ${why:-OK}" >> $out.log
  if [ -n "$why" ]; then bad=$((bad + 1)); { echo "conv $n: $why $conv"; echo "  ours  $a"; echo "  ref   $ref"; } >> $out.diff; fi
done < "$3"
rm -f $h
echo "B7 $((n - bad))/$n pass, $bad fail" | tee $out
