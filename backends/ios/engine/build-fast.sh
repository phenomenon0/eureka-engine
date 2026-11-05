#!/bin/bash
# (IMGFAST, from IMGPERF build-img.sh): the camera engine library (phonecam's build-cam.sh, unchanged recipe) with IMGPERF's wrapper:
# engine/llb_engine.{m,h} (+ llb_generate_rgb, llb_vision_has/forget) and vision/host/llbv.{c,h} (+ llbv_peek/hold).
# Everything else is read from phonecam's engine tree, READ ONLY ($ENG, default this directory; originally phonecam's engine tree @ 53aaa61):
# the emitted C b0ffd9af (hash-checked: the device program is byte-identical), head/, S1D/tools, the MMPROJ-patched
# llb_gguf.c, vision/host's vpre/vmm/vitm. Output: build/cam-{macos,ios,sim}/libllbengine.a, build/cam/LLBEngine.xcframework,
# build/cam/include/llb_engine.h, build/cam-macos/cam-host; SHA256 lines at the end.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENG=${ENG:-$here}  # (ISILDUR) this repo: phonecam@53aaa61 is an ancestor commit, its tree is ours
# (Eureka import) build outputs go under ISILDUR_OUT (default: this directory, as upstream; set it to a scratch dir);
# the S1D subset is read from ISILDUR_S1D (default $ENG/S1D)
OUTR=${ISILDUR_OUT:-$ENG}; S1DT=${ISILDUR_S1D:-$ENG/S1D}/tools
W=$OUTR/work/mac
SRC=${SRC:-$W/b-cam/sys1mac.c}
SRC_SHA=${SRC_SHA:-$(cat "$ENG/patches/CAM_SRC_SHA")}
IOS_MIN=27.0
B=$OUTR/build
export CLANG_MODULE_CACHE_PATH=$B/modules
got=$(shasum -a 256 "$SRC" | cut -d' ' -f1)
[ "$got" = "$SRC_SHA" ] || { echo "emitted C hash $got != $SRC_SHA ($SRC)"; exit 3; }
(cd "$ENG/head" && shasum -a 256 -c --quiet SHA256SUMS) || { echo "head/ differs from jev-head 8736c62"; exit 3; }
G=$B/cam-gen; rm -rf "$G"; mkdir -p "$G/src" "$G/host" "$B/cam/include"
cp "$SRC" "$G/src/sys1mac.c"
# (IMGFAST) the host-only prefix snapshots (no kernel / device-program text changes); PREFIX=0 builds b0ffd9af as is
if [ "${PREFIX:-1}" = 1 ]; then
  patch -s --no-backup-if-mismatch "$G/src/sys1mac.c" < "$here/patches/host-prefix.patch"
fi
cp "$W/tree/engine/host/llb_gguf.c" "$G/host/llb_gguf.c"   # the MMPROJ-patched loader vmm.c includes
grep -q gg_mm_ "$G/host/llb_gguf.c" || { echo "work/mac/tree is not the VISION tree (no gg_mm_ in llb_gguf.c)"; exit 3; }
(cd "$ENG/vision/host" && xxd -i vitm.metal) > "$G/vitm_msl.h"
mkdir -p "$G/wrap"
cp "$here/engine/llb_engine.m" "$here/engine/llb_engine.h" "$G/wrap/"
python3 "$here/patches/g3_emit.py" "$G/src/sys1mac.c"
python3 "$here/patches/g3_wrap.py" "$G/wrap/llb_engine.m" "$G/wrap/llb_engine.h"
TAG=$(shasum -a 256 "$G/src/sys1mac.c" | cut -c1-16)
cp "$G/wrap/llb_engine.h" "$B/cam/include/"

CFLAGS=(-DBEND_METAL=1 -DLLB_VISION=1 -x objective-c -fobjc-arc -fmodules -std=c11 -O3 -ffp-contract=off
  -Wno-unused-function -DLLB_SRC_TAG=\"$TAG\" -I"$G/src" -I"$G/wrap" -I"$ENG/engine" -I"$ENG/head" -I"$S1DT" -I"$here/vision/host"
  -I"$ENG/vision/host")
HFLAGS=(-O2 -std=c11 -Wall -Wextra -Wno-unused-parameter -ffp-contract=off -fno-fast-math
  -D_POSIX_C_SOURCE=200809L -D_DARWIN_C_SOURCE)
HSRC=("$ENG/head/jevsession.c" "$ENG/head/jevhead.c" "$ENG/head/json.c" "$ENG/head/gguf.c"
  "$S1DT/qwen35_tokenizer.c")
VFLAGS=(-std=c11 -D_POSIX_C_SOURCE=200809L -D_DARWIN_C_SOURCE -O3 -ffp-contract=off -fblocks -w
  -I"$here/vision/host" -I"$ENG/vision/host" -I"$G/host" -I"$G")

lib() {  # lib <name> <sdk> <target>
  local d=$B/cam-$1; rm -rf "$d"; mkdir -p "$d/obj"
  for f in "${HSRC[@]}"; do
    xcrun -sdk "$2" clang -target "$3" "${HFLAGS[@]}" -c "$f" -o "$d/obj/$(basename "$f" .c).o"
  done
  xcrun -sdk "$2" clang -target "$3" "${CFLAGS[@]}" -c "$G/wrap/llb_engine.m" -o "$d/obj/llb_engine.o"
  xcrun -sdk "$2" clang -target "$3" "${VFLAGS[@]}" -DLLBV_NO_CPU -c "$here/vision/host/llbv.c" -o "$d/obj/llbv.o"
  for f in vpre vmm; do
    xcrun -sdk "$2" clang -target "$3" "${VFLAGS[@]}" -c "$ENG/vision/host/$f.c" -o "$d/obj/$f.o"
  done
  xcrun -sdk "$2" clang -target "$3" "${VFLAGS[@]}" -fobjc-arc -x objective-c -c "$ENG/vision/host/vitm.m" -o "$d/obj/vitm.o"
  local dup; dup=$(for o in "$d"/obj/*.o; do nm -g -U "$o" | awk '{print $3}'; done | sort | uniq -d)
  [ -z "$dup" ] || { echo "duplicate symbols ($1): $dup"; exit 4; }
  xcrun -sdk "$2" libtool -static -o "$d/libllbengine.a" "$d"/obj/*.o
  local blas; blas=$(nm -u "$d/libllbengine.a" 2>/dev/null | grep -E '_(cblas|openblas)_' | sort -u || true)
  [ -z "$blas" ] || { echo "OpenBLAS symbols in $1: $blas"; exit 6; }
  printf '%s\n' '#include "llb_engine.h"' 'int main(void) { llb_prefix_forget(); return llb_classify_rgb(0, 0, 0, 0, 0, 0, 0, 0, 0, 0) + llb_generate_rgb(0, 0, 0, 0, 0, 0, 0, 0, 0, 0) + llb_vision_has(0, 0, 0); }' > "$d/probe.c"
  xcrun -sdk "$2" clang -target "$3" -O2 -I"$G/wrap" "$d/probe.c" -Wl,-all_load "$d/libllbengine.a" \
    -framework Metal -framework Foundation -o "$d/probe"
  echo "== cam-$1 ($3): probe linked (-all_load); $(shasum -a 256 "$d/libllbengine.a" | cut -c1-16) $(stat -f %z "$d/libllbengine.a") B"
}
lib macos macosx arm64-apple-macos27.0 & p1=$!
lib ios iphoneos arm64-apple-ios$IOS_MIN & p2=$!
lib sim iphonesimulator arm64-apple-ios$IOS_MIN-simulator & p3=$!
wait $p1 && wait $p2 && wait $p3 || { echo "a library slice failed (above)"; exit 5; }
xcrun -sdk macosx clang -target arm64-apple-macos27.0 -O2 -I"$here/engine" -I"$S1DT" "$ENG/tools/cam_host.c" \
  "$B/cam-macos/libllbengine.a" -framework Metal -framework Foundation -o "$B/cam-macos/cam-host"
rm -rf "$B/cam/LLBEngine.xcframework"
xcodebuild -create-xcframework -library "$B/cam-ios/libllbengine.a" -headers "$B/cam/include" \
  -library "$B/cam-sim/libllbengine.a" -headers "$B/cam/include" -output "$B/cam/LLBEngine.xcframework" >/dev/null
(cd "$B" && shasum -a 256 cam-gen/src/sys1mac.c cam-macos/libllbengine.a cam-ios/libllbengine.a cam-sim/libllbengine.a cam/include/llb_engine.h cam-macos/cam-host)
