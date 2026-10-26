#!/usr/bin/env bash
# pull-guard-kev.sh LLB: the real 4b-kev / 4b-kev-upstream clash (both Kev-4B-Q4_K_M.gguf, 2.71 vs 3.03 GB).
# A wrong-size file already at the destination must refuse before any network; the right size must pass.
set -u
L=$1 W=$(mktemp -d) bad=0
trap 'rm -rf "$W"' EXIT
mkdir -p "$W/conf" "$W/models" "$W/bin"
cp "$(dirname "$L")/defaults.conf" "$(dirname "$L")/models.conf" "$W/conf/"
export LLB_CONF=$W/conf LLB_HOME=$W/h
ok() { if [ "$1" = 0 ]; then echo "PASS $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
D=$W/models/Kev-4B-Q4_K_M.gguf
truncate -s 2709547424 "$D"
LLB_MODELS=$W/models "$L" pull 4b-kev-upstream 2> "$W/e"; r=$?
[ $r != 0 ] && [ $(wc -l < "$W/e") = 1 ] \
  && grep -q 'is 2709547424 bytes but 4b-kev-upstream expects 3033489824 bytes' "$W/e" \
  && grep -q '4b-kev=2709547424, 4b-kev-upstream=3033489824' "$W/e" \
  && grep -q 'LLB_MODELS' "$W/e"
ok $? "wrong-size 2.71 GB fixture: 4b-kev-upstream refuses, names both aliases, one line"
[ "$(wc -c < "$D")" = 2709547424 ]; ok $? "refusal left the existing 2.71 GB file untouched"
truncate -s 3033489824 "$D"
LLB_MODELS=$W/models "$L" pull 4b-kev-upstream 2> "$W/e"; r=$?
[ $r = 0 ] && grep -q 'already here' "$W/e"; ok $? "correct-size 3.03 GB fixture: passes the guard (already here)"
# no destination at all: the guard must not fire; a stub curl reports the right length, the download itself fails (offline).
printf '#!/bin/sh\ncase " $* " in *" -fL "*) exit 22 ;; esac\nprintf "HTTP/1.1 200 OK\\r\\nContent-Length: 3033489824\\r\\n\\r\\n"\n' > "$W/bin/curl"
chmod +x "$W/bin/curl"; rm -f "$D"
PATH=$W/bin:$PATH LLB_MODELS=$W/models "$L" pull 4b-kev-upstream 2> "$W/e"; r=$?
[ $r != 0 ] && grep -q '4b-kev-upstream: Kev-4B-Q4_K_M.gguf' "$W/e" && grep -q 'download failed' "$W/e"; ok $? "absent destination: guarded pull proceeds to the download step"
echo "pull-guard-kev $((4 - bad))/4 PASS"
