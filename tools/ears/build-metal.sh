#!/usr/bin/env bash
# tools/ears/build-metal.sh: the Metal AuT tower (aut_metal.mm) and the host mel (aud_mel.cpp) as objects for the
# three Apple slices, with the exact flags EARS.xcframework uses (backends/ios/Vendor/ears-src/build-ears-v3.sh):
#   aud_mel.cpp  clang++ -std=c++17 -O3 -ffp-contract=off
#   aut_metal.mm clang++ -std=c++17 -O2 -fobjc-arc  (MSL is compiled at load, fast math off)
# Objects only: EARS.xcframework prelinks them with the slim-chunker ears.cpp and llama.cpp 2ca15f5's static libs
# (ld -r, only ears_* exported); see tools/ears/README.md "Metal tower".
# AUT_DIAG=1 also builds the diagnostic tower (teacher forcing, MSL variant prefix). Never ship that object.
# Env: OUT (output dir, outside the source tree), LC (llama.cpp 2ca15f5 source: gguf.h, ggml.h).
# Needs macOS with Xcode (xcrun). Writes $OUT/{macos,ios,sim}/{aud_mel,aut_metal}.o.
set -euo pipefail
e=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${OUT:?set OUT}" "${LC:?llama.cpp 2ca15f5 source}"
[ "$(git -C "$LC" rev-parse --short=7 HEAD)" = 2ca15f5 ] || { echo "build-metal: $LC is not llama.cpp 2ca15f5" >&2; exit 2; }
INC=(-I"$LC/ggml/include" -I"$LC/include" -I"$e")
one() {  # one <slice> <sdk> <target>
  local d=$OUT/$1; mkdir -p "$d"
  local cc=(xcrun -sdk "$2" clang++ -target "$3")
  "${cc[@]}" -std=c++17 -O3 -ffp-contract=off -c "$e/aud_mel.cpp" -o "$d/aud_mel.o"
  "${cc[@]}" -std=c++17 -O2 -fobjc-arc "${INC[@]}" -c "$e/aut_metal.mm" -o "$d/aut_metal.o"
  if [ "${AUT_DIAG:-0}" = 1 ]; then
    "${cc[@]}" -std=c++17 -O2 -fobjc-arc -DAUT_DIAG "${INC[@]}" -c "$e/aut_metal.mm" -o "$d/aut_metal-diag.o"
  fi
}
one macos macosx arm64-apple-macos14.0
one ios iphoneos arm64-apple-ios18.0
one sim iphonesimulator arm64-apple-ios18.0-simulator
shasum -a 256 "$e/aut_metal.mm" "$e/aud_mel.cpp" "$OUT"/*/*.o
