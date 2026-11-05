#!/usr/bin/env bash
# Build and run against the already-built Mac slice; args: model directory, scratch cache directory.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=${ISILDUR_OUT:-$here}
: "${1:?usage: cancel-test.sh model-dir scratch-cache-dir}"
: "${2:?usage: cancel-test.sh model-dir scratch-cache-dir}"
lib="$out/build/cam-macos/libllbengine.a"
[ -f "$lib" ] || { echo "Build the Mac slice first: $lib" >&2; exit 1; }
mkdir -p "$2"
xcrun -sdk macosx clang -target arm64-apple-macos27.0 -O2 -I"$out/build/cam/include" \
  "$here/engine/llb_cancel_test.c" "$lib" -framework Metal -framework Foundation -o "$2/llb-cancel-test"
"$2/llb-cancel-test" "$1" "$2"
