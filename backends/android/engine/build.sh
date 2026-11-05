#!/usr/bin/env bash
# android/engine/build.sh [android|host] OUT: the app's engine = feddecide's vendored qwen35 CPU engine
# + cpspeed4 patches 01-12 (g12d8 flags: the best phone build) + 20-kev-decide (feddecide's --decide
# JSON-lines loop, rebased onto the stack) + 21-app-eos (KEV_EOS: stop at <|im_end|>, stream pieces)
# + 30-vision (IMRoPE + image-row splice + generate's "image" field; the tower itself is vision/).
# The vendored source is never edited; patch --fuzz=0 refuses drift.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
which=${1:-android}
OUT=${2:-$here/../build/engine}
NDK=${NDK:-${ANDROID_HOME:-}/ndk/27.2.12479018}
CF="-O2 -ffp-contract=off -fno-fast-math"
FLAGS="-fopenmp -DLLB_SCHED=runtime -DLLB_PAR_CORE -DLLB_DOT8 -DLLB_DOT8_GEMM"
rm -rf "$OUT/src" && mkdir -p "$OUT/src" && cp -r "$here/vendor/." "$OUT/src/"
for p in "$here"/patches/*.patch; do patch --batch --silent --fuzz=0 -d "$OUT/src" -p1 < "$p"; done
s=$OUT/src/tools
if [ "$which" = android ]; then
  CC="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang"
  CF="$CF -march=armv8.2-a+dotprod"; LINK="-static -static-openmp -Wl,-z,max-page-size=16384"; LIBS="-lm -ldl"
else
  CC=gcc; CF="$CF -march=native"; LINK=""; LIBS="-lm"
fi
$CC -D_POSIX_C_SOURCE=200809L -std=c11 $CF -c "$s/kevserve/kev.c" -o "$OUT/kev.o"
$CC -D_POSIX_C_SOURCE=200809L -std=c11 $CF -c "$s/clef3/json.c" -o "$OUT/json.o"
for v in vision vpre; do $CC -std=c11 -D_POSIX_C_SOURCE=200809L $CF -fopenmp -I "$here/vision" -c "$here/vision/$v.c" -o "$OUT/$v.o"; done
$CC $CF $FLAGS -I "$s" -I "$here/vision" $LINK -o "$OUT/kev-engine-$which" "$s/qwen35_engine.c" "$OUT/kev.o" "$OUT/json.o" "$OUT/vision.o" "$OUT/vpre.o" $LIBS
{
  echo "Android engine build provenance v1"
  printf 'source_revision: %s\n' "${ENGINE_SOURCE_REVISION:-$(git -C "$here" rev-parse HEAD 2>/dev/null || echo unknown)}"
  printf 'target: %s\ncompiler: %s\n' "$which" "$CC"
  "$CC" --version | head -n 1
  printf 'cflags: %s %s\nlink: %s %s\n' "$CF" "$FLAGS" "$LINK" "$LIBS"
  echo 'source_sha256:'
  (cd "$here" && sha256sum build.sh)
  (cd "$here" && find vendor patches vision -type f ! -name .DS_Store -print | LC_ALL=C sort | while IFS= read -r f; do sha256sum "$f"; done)
  echo 'artifact_sha256:'
  (cd "$OUT" && sha256sum "kev-engine-$which")
} > "$OUT/BUILD-PROVENANCE.txt"
sha256sum "$OUT/kev-engine-$which"
