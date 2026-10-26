#!/bin/bash
# (ENGINE-PHONE copy (via engine-4b) (via engine-emcee) of lane m41's tools/fresh36.sh: only the refuse path is re-rooted)
# tools/fresh36.sh <dir> (M38 copy of M36 release/fresh.sh, same patches, re-rooted to lane m38): lane m36 (from m34)'s ONE recipe for the Mac release tree. Fresh copies of the emitted-unlock/integrate
# compiler (build/m9d/cc) and engine (engine/, minus receipts/ reports/), then every patch in patches/cc and
# patches/engine in file-name order, each `patch -s --fuzz=0 -p1` (an empty patch is skipped). Hashes: patches/SHA256SUMS.
set -eu
# (Eureka import) legacy provenance only: the lane trees are not public; set M36, INTEGRATE and FRESH_ROOT to run it
H=${M36:?lane m36 tree}/release; S=$1; I=${INTEGRATE:?emitted-unlock/integrate tree}
case "$S" in "${FRESH_ROOT:?scratch root}"/*) ;; *) echo "refuse $S (must be under \$FRESH_ROOT)"; exit 2;; esac  # (PHONECAM) re-rooted
(cd "$H/patches" && shasum -a 256 -c --quiet SHA256SUMS) || { echo "patch hashes differ"; exit 3; }
rm -rf "$S"; mkdir -p "$S"
cp -R "$I/build/m9d/cc" "$S/cc"
rsync -a --exclude receipts --exclude reports "$I/engine/" "$S/engine/"
for d in cc engine; do for p in "$H"/patches/$d/*.patch; do [ -s "$p" ] || continue
  (cd "$S/$d" && patch -s --fuzz=0 -p1 < "$p") || { echo "FAILED $p"; exit 4; }; done; done
find "$S" -name '*.orig' -delete
shasum -a 256 "$S/cc/bend2/comp.ts" "$S/engine/engine_qwen35_gpu.bend" | sed "s|$S/||"
