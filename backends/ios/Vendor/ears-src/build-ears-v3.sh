#!/usr/bin/env bash
# G38: EARS.xcframework v3 (ios + sim slices) + Vendor/macos/libears.a (the Mac harness), for the app.
# v3 = ios-ears-carry's v2 (slim-chunker ears.cpp + aud_mel.cpp + aut_metal.mm + llama.cpp 2ca15f5, prelinked with ld -r
# so only ears_* is exported) + ears_text.cpp (ears_detokenize: vocab-only). The v2 sources are used in place, read only;
# ears.h gains the one declaration. Scratch in $SCRATCH (default /tmp/g38/ears-build). Usage: build-ears-v3.sh <ios/Vendor>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENDOR="$(cd "${1:?ios/Vendor}" && pwd)"
CARRY=${EARS_CARRY:?set EARS_CARRY to the ios-ears-carry work tree}     # v2 sources + llama.cpp lc-{sim,ios} builds (read only)
EARS1=${EARS_LLAMA:?set EARS_LLAMA to the ios-ears tree (llama.cpp 2ca15f5)}                # llama.cpp 2ca15f5 tree + lc-mac build (read only)
S=${SCRATCH:-/tmp/g38/ears-build}; rm -rf "$S"; mkdir -p "$S/src"
cp "$CARRY/src/ears.h" "$CARRY/src/ears.cpp" "$CARRY/src/aud_mel.cpp" "$CARRY/src/aud_mel.h" "$CARRY/src/aut.h" \
   "$CARRY/src/aut_metal.mm" "$CARRY/src/mel_tables_glibc.h" "$HERE/ears_text.cpp" "$S/src/"
python3 "$HERE/ears_h_v3.patch.py" "$S/src/ears.h"
printf '_ears_transcribe\n_ears_mel_check\n_ears_version\n_ears_detokenize\n' > "$S/exports.txt"
LC=$EARS1/llama.cpp; INC="-I$LC/ggml/include -I$LC/include -I$S/src"
one() {  # slice sdk target lc-build-dir extra-libs...
  local s=$1 sdk=$2 tgt=$3 B=$4; shift 4
  local d=$S/ears-$s; mkdir -p $d/include
  local CC="xcrun -sdk $sdk clang++ -target $tgt"
  $CC -std=c++17 -O3 -ffp-contract=off -c $S/src/aud_mel.cpp -o $d/aud_mel.o
  $CC -std=c++17 -O2 -fobjc-arc $INC -c $S/src/aut_metal.mm -o $d/aut_metal.o
  $CC -std=c++17 -O2 $INC -c $S/src/ears.cpp -o $d/ears.o
  $CC -std=c++17 -O2 $INC -c $S/src/ears_text.cpp -o $d/ears_text.o
  local extra=(); for l in "$@"; do extra+=(-Wl,-force_load,$B/$l); done
  $CC -r -nostdlib -o $d/ears_all.o $d/ears.o $d/ears_text.o $d/aud_mel.o $d/aut_metal.o -Wl,-exported_symbols_list,$S/exports.txt \
    -Wl,-force_load,$B/src/libllama.a -Wl,-force_load,$B/ggml/src/libggml.a -Wl,-force_load,$B/ggml/src/libggml-cpu.a \
    -Wl,-force_load,$B/ggml/src/ggml-metal/libggml-metal.a ${extra[@]+"${extra[@]}"} -Wl,-force_load,$B/ggml/src/libggml-base.a
  xcrun -sdk $sdk libtool -static -o $d/libears.a $d/ears_all.o
  cp $S/src/ears.h $d/include/
  echo "== ears-$s exported: $(nm -g -U $d/libears.a | awk 'NF==3{print $3}' | grep -v '^ltmp' | tr '\n' ' ')"
}
one sim iphonesimulator arm64-apple-ios18.0-simulator $CARRY/build/lc-sim
one ios iphoneos arm64-apple-ios18.0 $CARRY/build/lc-ios
one mac macosx arm64-apple-macos14.0 $EARS1/build/lc-mac ggml/src/ggml-blas/libggml-blas.a
rm -rf "$VENDOR/EARS.xcframework"
xcodebuild -create-xcframework -library $S/ears-ios/libears.a -headers $S/ears-ios/include \
  -library $S/ears-sim/libears.a -headers $S/ears-sim/include -output "$VENDOR/EARS.xcframework" >/dev/null
mkdir -p "$VENDOR/macos/ears-include"
cp $S/ears-mac/libears.a "$VENDOR/macos/libears.a"; cp $S/src/ears.h "$VENDOR/macos/ears-include/ears.h"
echo "== $VENDOR/EARS.xcframework + macos/libears.a"
