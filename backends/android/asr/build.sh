#!/usr/bin/env bash
# ANDROIDVOICE: rebuilds the app's on-device ASR: whisper.cpp v1.9.2 (ggml, CPU) + whisper_jni.c -> jniLibs/libbendwhisper.so,
# and the bundled model -> assets/asr/ (ggml-base.en-q8_0.bin, pinned sha256 below). Needs network (clone, model), cmake, ninja,
# the NDK. Scratch dir: $1 (default /tmp/whisper-build).
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd); W=${1:-/tmp/whisper-build}
NDK=${NDK:-${ANDROID_HOME:?set ANDROID_HOME to your Android SDK}/ndk/27.2.12479018}
MODEL=ggml-base.en-q8_0.bin; SHA=$(grep " $MODEL\$" "$here/asr/SHA256SUMS" | cut -d' ' -f1)
mkdir -p "$W" && cd "$W"
[ -d whisper.cpp ] || git clone --depth 1 -b v1.9.2 https://github.com/ggml-org/whisper.cpp   # 306c88f
cmake -S "$here/asr" -B and -G Ninja -DWHISPER_SRC="$W/whisper.cpp" -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release
cmake --build and --target bendwhisper
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip -o "$here/app/src/main/jniLibs/arm64-v8a/libbendwhisper.so" and/libbendwhisper.so
D=$here/app/src/main/assets/asr; mkdir -p "$D"
[ -f "$D/$MODEL" ] || curl -fsSL -o "$D/$MODEL" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$MODEL"
echo "$SHA  $D/$MODEL" | sha256sum -c -
