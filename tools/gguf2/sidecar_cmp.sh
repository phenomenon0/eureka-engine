#!/bin/sh
# S2: gguf2pack --sidecar-only vs an existing pack. table.bin byte-identical;
# manifest.json identical after blanking source.file and the sha256 fields
# (a header-only sidecar has none); bytes read by gguf2pack (strace) << weights.
#   sidecar_cmp.sh GGUF PACK_DIR OUT_DIR   (OUT_DIR must not hold a manifest)
set -eu
g=$1 pack=$2 out=$3
bin=$(dirname "$0")/../../build/g2/gguf2pack
s=$(date +%s.%N)
strace -f -e trace=read,pread64 -o "$out.strace" "$bin" "$g" --sidecar-only "$out" >/dev/null
e=$(date +%s.%N)
rd=$(awk -F'= ' '/(read|pread64)\(/ && $NF+0 > 0 {s += $NF} END {print s+0}' "$out.strace")
norm() { sed -e 's/^\(  "file": \).*/\1X/' -e 's/"\(pack_\)\{0,1\}sha256": "[0-9a-f]*"/"sha256": ""/' "$1"; }
cmp -s "$out/table.bin" "$pack/table.bin" && tb=identical || tb=DIFFERS
norm "$out/manifest.json" > "$out.m1"; norm "$pack/manifest.json" > "$out.m2"
cmp -s "$out.m1" "$out.m2" && mf=identical || mf=DIFFERS
lines=$(diff "$out/manifest.json" "$pack/manifest.json" | grep -c '^<' || true)
printf '%s: %.2f s, read %s bytes of %s; table.bin %s; manifest.json %s after normalising (%s raw lines differ: source.file + sha256s)\n' \
  "$(basename "$g")" "$(echo "$e - $s" | bc)" "$rd" "$(stat -L -c %s "$g")" "$tb" "$mf" "$lines"
mv "$out.m1" "$out.norm-sidecar"; mv "$out.m2" "$out.norm-pack"; mv "$out.strace" "$out.strace.txt"
