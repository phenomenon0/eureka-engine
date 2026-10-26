#!/bin/bash
# (ENGINE-4B copy (via engine-emcee) of lane m41's tools/fresh41.sh: only the fresh36.sh path differs)
# tools/fresh38.sh <dir>: lane m38's Mac tree v5 = M36's recipe (tools/fresh36.sh: M36 release/fresh.sh, same patches)
# + the m38 pair, each `patch -s --fuzz=0 -p1`: patches-m38/metal-m38.patch into cc/, engine-m38.patch into engine/.
set -eu
M=${M38:?lane m38 tree (legacy provenance only; not public)}; S=$1
(cd $M/patches-m38 && shasum -a 256 -c --quiet SHA256SUMS) || { echo "m38 patch hashes differ"; exit 3; }
bash "$(cd "$(dirname "$0")" && pwd)/fresh36.sh" "$S" >/dev/null
(cd "$S/cc" && patch -s --fuzz=0 -p1 < $M/patches-m38/metal-m38.patch) || { echo "FAILED metal-m38"; exit 4; }
(cd "$S/engine" && patch -s --fuzz=0 -p1 < $M/patches-m38/engine-m38.patch) || { echo "FAILED engine-m38"; exit 4; }
find "$S" -name '*.orig' -delete
shasum -a 256 "$S/cc/bend2/comp.ts" "$S/engine/engine_qwen35_gpu.bend" "$S/engine/host/llb_gguf.c" | sed "s|$S/||"
N=${M41:?lane m41 tree}
(cd "$N" && shasum -a 256 -c --quiet patches-m41/SHA256SUMS)
(cd "$S/cc" && patch -s --fuzz=0 -p1 < "$N/patches-m41/metal-m41.patch")
(cd "$S/engine" && patch -s --fuzz=0 -p1 < "$N/patches-m41/engine-m41.patch")
