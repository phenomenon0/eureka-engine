#!/usr/bin/env bash
# backends/metal/build.sh: the Mac Metal engine (sys1mac) from THIS tree. Opt-in: build.sh runs it only
# under LLB_BACKEND=metal; the CUDA build never reads backends/.
#   1. export the pinned Bend compiler (same commit as build.sh) and apply the SHARED head of the CUDA
#      patch stack: tools/build_gpu.sh's list up to and including $FORK (the Mac fork point; the 36 files
#      are byte-identical to the Mac integrate stack's, see README.md)
#   2. cc/metal-cc.patch: the Mac compiler delta from that point (integrate metal-port/9b/batch + M36 release
#      cc 01-14 + M38 + M41 + sys1mac + the 4B stack), --fuzz=0
#   3. a scratch copy of the engine sources + engine/metal-engine.patch (the Mac engine at engine-vision3's
#      default build); VISION=1 adds the three vendored vision patches, as mac/build.sh does there
#   4. emit C (platform-independent); on Darwin also link sys1mac exactly as engine-vision3/mac/build.sh
#   5. (default) cc/metal-anyshape-cc.patch + engine/metal-anyshape-engine.patch: any Qwen3.5 dense shape and any
#      context (docs/limits.md); LLB_METAL_BASE=1 leaves them out and reproduces the product hashes above
# Env: LLB_ROOT, LLB_BEND (Bend checkout), LLB_OUT (default $LLB_ROOT/build/metal), VISION, LLB_METAL_EMIT_ONLY,
#      LLB_METAL_BASE.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LLB_ROOT=${LLB_ROOT:-$(cd -- "$here/../.." && pwd)}
LLB_BEND=${LLB_BEND:-$LLB_ROOT/../bend}
LLB_OUT=${LLB_OUT:-$LLB_ROOT/build/metal}
BEND_COMMIT=$(sed -n 's/^BEND_COMMIT=//p' "$LLB_ROOT/build.sh")
FORK=tools/gemv-q8-rw.patch
# the product hashes this recipe must reproduce (engine-vision3 REPORT.md: default / VISION=1)
EXPECT_C=c13b9e28d36ca618f2e041e0648e53ad744482db8537c1586666599fa58b61ee
EXPECT_C_VISION=fbdefec75fe9f02a1f85e5bc8b7a3f1f461a8501c47db51dd8e0f454cd25f10a
# the same with the any-shape patches (the default build)
EXPECT_C_ANY=009794484c3860bd111eb131a9fc1d73f995a031ee81114847ecfc221b93fb55
EXPECT_C_ANY_VISION=85c8d3c489dda683d5bbd12b8928e2645a6268efdec6871d3e755fd1d6c56447
die() { echo "backends/metal/build.sh: $*" >&2; exit 1; }
BUN=$(command -v bun || echo "$HOME/.bun/bin/bun"); [ -x "$BUN" ] || die "missing tool: bun"
(cd "$here" && sha256sum -c --quiet SHA256SUMS 2>/dev/null || shasum -a 256 -c --quiet SHA256SUMS) \
  || die "vendored patch hashes differ (SHA256SUMS)"

rm -rf "$LLB_OUT/cc" "$LLB_OUT/cc.src" "$LLB_OUT/tree"; mkdir -p "$LLB_OUT/cc/bend2" "$LLB_OUT/tree"
if [ -f "$LLB_BEND/.commit" ] && [ "$(cat "$LLB_BEND/.commit")" = "$BEND_COMMIT" ]; then
  src=$LLB_BEND   # an already-exported bend-pinned dir (build.sh's $LLB_OUT/bend-pinned)
elif git -C "$LLB_BEND" cat-file -e "$BEND_COMMIT^{commit}" 2>/dev/null; then
  src=$LLB_OUT/cc.src; mkdir -p "$src"; git -C "$LLB_BEND" archive "$BEND_COMMIT" bend2 | tar -xm -C "$src"
elif BEND_TREE=$(sed -n 's/^BEND_TREE=//p' "$LLB_ROOT/build.sh") && [ -n "$BEND_TREE" ] &&
     [ "$(git -C "$LLB_ROOT" rev-parse -q --verify pins/bend-240ecef4:bend2 2>/dev/null)" = "$BEND_TREE" ]; then
  src=$LLB_OUT/cc.src; mkdir -p "$src"; git -C "$LLB_ROOT" archive pins/bend-240ecef4 bend2 | tar -xm -C "$src" # the content record (build.sh)
else die "LLB_BEND=$LLB_BEND is neither a Bend checkout nor a bend-pinned export of $BEND_COMMIT"; fi
cp -r "$src/bend2/"{bend.ts,comp.ts,main.ts,hand.ts,export.ts,base.bend,effs} "$LLB_OUT/cc/bend2/"
n=0
for p in $(grep -oE 'tools/[a-z0-9/_-]+\.patch' "$LLB_ROOT/tools/build_gpu.sh"); do
  patch --batch --silent --fuzz=0 --forward -d "$LLB_OUT/cc" -p1 < "$LLB_ROOT/$p" || die "shared $p"
  n=$((n + 1)); [ "$p" = "$FORK" ] && break
done
[ "$n" = 36 ] || die "fork point $FORK is entry $n of build_gpu.sh, expected 36"
patch --batch --silent --fuzz=0 -d "$LLB_OUT/cc" -p1 < "$here/cc/metal-cc.patch" || die "metal-cc.patch"

for x in engine_qwen35_gpu.bend engine_qwen35.bend engine_gpu.bend gguf.bend kernels host; do
  cp -r "$LLB_ROOT/$x" "$LLB_OUT/tree/"; done
mkdir -p "$LLB_OUT/tree/tools/sys1dual"; cp "$LLB_ROOT"/tools/sys1dual/sys1.{bend,c,js} "$LLB_OUT/tree/tools/sys1dual/"
# the engine patch is a diff against 543caec4 (the G7 base); files the CUDA line has moved since are pinned in base/
(cd "$here/base" && find . -type f ! -name SHA256SUMS -exec cp {} "$LLB_OUT/tree/{}" \;)
patch --batch --silent --fuzz=0 -E -d "$LLB_OUT/tree" -p1 < "$here/engine/metal-engine.patch" || die "metal-engine.patch"
expect=$EXPECT_C
if [ -n "${VISION:-}" ]; then
  patch --batch --silent --fuzz=0 -d "$LLB_OUT/tree" -p1 < "$here/vision/engine-mmproj.patch" || die "engine-mmproj"
  patch --batch --silent --fuzz=0 -d "$LLB_OUT/cc" -p1 < "$here/vision/metal-imrope.patch" || die "metal-imrope"
  patch --batch --silent --fuzz=0 -d "$LLB_OUT/cc" -p1 < "$here/vision/metal-vision.patch" || die "metal-vision"
  expect=$EXPECT_C_VISION
fi
if [ -z "${LLB_METAL_BASE:-}" ]; then
  patch --batch --silent --fuzz=0 -d "$LLB_OUT/cc" -p1 < "$here/cc/metal-anyshape-cc.patch" || die "metal-anyshape-cc"
  patch --batch --silent --fuzz=0 -d "$LLB_OUT/tree" -p1 < "$here/engine/metal-anyshape-engine.patch" \
    || die "metal-anyshape-engine"
  expect=$([ -n "${VISION:-}" ] && echo "$EXPECT_C_ANY_VISION" || echo "$EXPECT_C_ANY")
fi
find "$LLB_OUT/cc" "$LLB_OUT/tree" -name '*.orig' -delete
(cd "$LLB_OUT/tree" && BEND_NO_TELEMETRY=1 "$BUN" ../cc/bend2/main.ts engine_qwen35_gpu.bend -o "$LLB_OUT/sys1mac.c" \
  2>&1 | grep -v '^- \|bend update' | tail -3)
got=$( (sha256sum "$LLB_OUT/sys1mac.c" 2>/dev/null || shasum -a 256 "$LLB_OUT/sys1mac.c") | cut -d' ' -f1)
echo "sys1mac.c $got"
[ "$got" = "$expect" ] || [ -n "${LLB_METAL_NOHASH:-}" ] || die "emitted C $got != product $expect"
lab=$([ -n "${LLB_METAL_BASE:-}" ] || echo "any-shape ")
echo "emitted C == the Mac product build (${VISION:+VISION=1 }$lab${expect:0:8})"

[ "$(uname -s)" = Darwin ] && [ -z "${LLB_METAL_EMIT_ONLY:-}" ] || exit 0
# Darwin: the engine-vision3 mac/build.sh link (default build; VISION=1's tower objects live in that tree)
[ -z "${VISION:-}" ] || die "VISION=1 native link is not carried yet (tower sources are in ~/engine-vision3/vision)"
O=$LLB_OUT; mkdir -p "$O/obj" "$O/inc"; ln -sfn /opt/homebrew/opt/openblas/include "$O/inc/openblas"
# dual.c: the S1D pin (llama-bend-sys1dual @ 7e61bda, link/dual.c); the tree's own moved on (twoheads S3)
# (METAL-ANYSHAPE) + link/dual-anyshape.patch (a generate prompt up to the KV capacity) unless LLB_METAL_BASE
dual=$here/link/dual.c
if [ -z "${LLB_METAL_BASE:-}" ]; then
  mkdir -p "$O/link"; cp "$here/link/dual.c" "$O/link/dual.c"; dual=$O/link/dual.c
  patch --batch --silent --fuzz=0 -d "$O/link" -p1 < "$here/link/dual-anyshape.patch" || die "dual-anyshape"
fi
for f in "$dual" "$LLB_ROOT"/tools/clef3/{head,weights,spans,json}.c "$LLB_ROOT/tools/qwen35_tokenizer.c"; do
  clang -D_POSIX_C_SOURCE=200809L -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Wno-unused-function -I"$O/inc" \
    -I"$LLB_ROOT/tools/sys1dual" -c "$f" -o "$O/obj/$(basename "$f" .c).o"
done
(cd "$O" && clang -DBEND_METAL=1 -x objective-c -fobjc-arc -fmodules -fmodules-cache-path="$O/modules" \
  -std=c11 -O3 -ffp-contract=off sys1mac.c -x none obj/*.o -L/opt/homebrew/opt/openblas/lib -lopenblas -lpthread -lm \
  -o sys1mac)
shasum -a 256 "$O/sys1mac.c" "$O/sys1mac" "$O"/obj/*.o
