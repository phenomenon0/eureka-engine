#!/usr/bin/env bash
# snips.sh OUT [SEED]: held-out tokenizer snippets, one file each (OUT/NNN.txt), valid UTF-8 only.
# up to 90 real chunks (64-1500 bytes at a random offset, cut to line ends) from files on this box: C headers, Python
# stdlib, shell scripts, docs, man pages, JSON/TOML/YAML configs, plus 40 synthetic: unicode/emoji runs, CRLF prose,
# random base64 and hex, digit runs, whitespace soup, plus 4 with special-token text. Deterministic for a given SEED and box.
set -u
o=$1 seed=${2:-1105}; mkdir -p $o; rs() { yes "$seed$1" | head -c 1M; }
i=0
put() { if iconv -f UTF-8 -t UTF-8 "$1" > /dev/null 2>&1 && [ -s "$1" ]; then mv "$1" $o/$(printf %03d $i).txt; i=$((i + 1)); else rm -f "$1"; fi; }
for spec in "/usr/include -name *.h" "/usr/lib64/python3* -name *.py" "/usr/share/doc -name README*" "/usr/bin -maxdepth 1 -type f -size -40k" \
            "/etc -maxdepth 2 -name *.conf" "/usr/share -maxdepth 4 -name *.json" "$HOME/.config -maxdepth 3 -name *.toml -o -name *.yaml -o -name *.yml" \
            "/usr/share/licenses -name *" "/usr/share/X11/locale -name Compose"; do
  set -f; files=$(find ${spec%% *} ${spec#* } -type f -readable 2>/dev/null | LC_ALL=C sort | shuf -n 40 --random-source=<(rs "$spec")); set +f
  k=0
  for f in $files; do
    [ $k -ge 10 ] && break
    head -c 4096 "$f" | grep -qP '\x00' && continue   # binaries out
    sz=$(stat -c %s "$f"); [ "$sz" -ge 64 ] || continue
    off=$(( $(od -An -N4 -tu4 <(rs "$f") | tr -d ' ') % sz )); len=$(( 64 + $(od -An -N2 -tu2 <(rs "x$f") | tr -d ' ') % 1437 ))
    tail -c +$((off + 1)) "$f" | head -c $len | sed '1d;$d' > $o/tmp; [ -s $o/tmp ] || tail -c +1 "$f" | head -c $len > $o/tmp
    put $o/tmp && k=$((k + 1))
  done
done
awk -v S=$seed -v O=$o 'BEGIN { srand(S)
  split("é ß ñ ü ø Ж я 中 文 日本語 한국어 العربية עברית हिन्दी ॐ € £ ¥ © ™ … — – « » ́ ̃ 𝔘 𝟙 😀 👍🏽 🇫🇷 🇯🇵 ❤️ ✓ ½ ² Ⅻ 👨‍👩‍👧 🧑🏿‍💻 ﷽ ẞ ǅ ﬁ Ω Å ∑ ∞ ≠ ← ⌘ 〒 ゛ ｱ ㈱", u, " ")
  split("the quick brown fox jumps over lazy dog and then it was said that we have code data model token", w, " ")
  for (n = 0; n < 40; n++) { s = ""; t = n % 8; L = 20 + int(rand() * 120)
    for (j = 0; j < L; j++) {
      if (t == 0) s = s u[1 + int(rand() * 55)] (rand() < .3 ? " " : "")
      else if (t == 1) s = s w[1 + int(rand() * 19)] (rand() < .15 ? ".\r\n" : " ")
      else if (t == 2) s = s substr("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", 1 + int(rand() * 64), 1) (j % 76 == 75 ? "\n" : "")
      else if (t == 3) s = s substr("0123456789abcdef", 1 + int(rand() * 16), 1) (j % 2 && rand() < .2 ? " " : "")
      else if (t == 4) s = s int(rand() * 10 ^ int(1 + rand() * 12)) substr(" ,.\n:-", 1 + int(rand() * 6), 1)
      else if (t == 5) s = s substr(" \t\n\r\f\v", 1 + int(rand() * 4), 1) (rand() < .2 ? w[1 + int(rand() * 19)] : "")
      else if (t == 6) s = s w[1 + int(rand() * 19)] u[1 + int(rand() * 55)] (rand() < .3 ? "\n\n" : " ")
      else s = s "{\"" w[1 + int(rand() * 19)] "\": [" int(rand() * 1000) ", \"" u[1 + int(rand() * 55)] "\"]}" (rand() < .5 ? ",\n" : ", ")
    }
    f = sprintf("%s/syn%02d", O, n); printf "%s", s > f; close(f) } }'
for f in $o/syn*; do put $f; done
n=0; for s in 'a <|im_start|>user\nhi<|im_end|>' '<|endoftext|><|endoftext|>' 'x <think>y</think> z' '<tool_call>{"a":1}</tool_call> <|vision_start|>'; do
  printf '%s' "$s" > $o/sp$n; put $o/sp$n; n=$((n + 1)); done   # special-token text: must stay plain (parse_special false)
rm -f $o/tmp; echo "$i snippets in $o ($(cat $o/*.txt | wc -c) bytes)"
