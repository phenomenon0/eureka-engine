#!/usr/bin/env bash
# t3.sh TOK DIR CONVS OUT: llb-tok chat ids vs llama.cpp /apply-template -> /tokenize (parse_special) per conversation.
# A conversation is a JSON array of {role, content, reasoning?}; the server gets reasoning as reasoning_content,
# llb-tok gets it the way a model emits it: "reasoning\n</think>\n\ncontent".
# THINK=1|2: both sides get it explicitly (server: chat_template_kwargs.enable_thinking true|false; ours: --think N).
# INLINE=1: the server gets the same inline form (for templates that split </think> themselves, e.g. Qwen3.5).
set -u
tok=$1 dir=$2 port=${PORT:-8795} out=$4; : > $out.diff; n=0 bad=0; h=$(mktemp -p "${TMPDIR:-/tmp}")
while IFS= read -r conv; do
  n=$((n + 1))
  echo "$conv" | jq -j '.[] | .role + "\u0000" + (if .reasoning then .reasoning + "\n</think>\n\n" else "" end) + .content + "\u0000"' > $h
  a=$($tok chat $dir $h ${THINK:+--think $THINK})
  p=$(echo "$conv" | jq --arg inl "${INLINE-}" --arg th "${THINK-}" '{messages: map(if .reasoning and $inl != "" then {role, content: (.reasoning + "\n</think>\n\n" + .content)} else {role, content} + (if .reasoning then {reasoning_content: .reasoning} else {} end) end)} + (if $th == "" then {} else {chat_template_kwargs: {enable_thinking: ($th == "1")}} end)' | curl -s localhost:$port/apply-template -d @- | jq -j .prompt; echo x); p=${p%x}
  b=$(jq -n --arg c "$p" '{content: $c, add_special: false, parse_special: true}' | curl -s localhost:$port/tokenize -d @- | jq -r '.tokens | map(tostring) | join(" ")')
  if [ "$a" != "$b" ] || [ -z "$b" ]; then bad=$((bad + 1)); { echo "conv $n: $conv"; echo "  ours  $a"; echo "  llama $b"; echo "  llama text: $(printf '%s' "$p" | od -An -c | tr -s ' ' | tr '\n' ' ')"; echo "  ours  text: $($tok chat $dir $h --text ${THINK:+--think $THINK} | od -An -c | tr -s ' ' | tr '\n' ' ')"; } >> $out.diff; fi
done < "$3"
rm -f $h
echo "T3 $((n - bad))/$n identical, $bad differ" | tee $out
