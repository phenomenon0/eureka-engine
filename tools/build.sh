#!/bin/sh
# build.sh v1|v2 ENTRY OUT: compile ENTRY (engine.bend, kernels/gemv.bend, ...)
# in the dot order named. The tree is v1; v2 compiles a copy of engine.bend +
# kernels/ (build/src-v2) with kernels/order.bend flipped. Nothing else differs.
set -eu
order=$1 entry=$2 out=$3
root=$(cd "$(dirname "$0")/.." && pwd)
case $order in
  v1) src=$root ;;
  v2)
    src=$root/build/src-v2
    rm -rf "$src" && mkdir -p "$src"
    cp -r "$root/engine.bend" "$root/kernels" "$src/"
    sed -i 's/^  False{}$/  True{}/' "$src/kernels/order.bend"
    grep -qx '  True{}' "$src/kernels/order.bend" || { echo "build.sh: order flip failed" >&2; exit 1; } ;;
  *) echo "usage: build.sh v1|v2 ENTRY OUT" >&2; exit 2 ;;
esac
bun "$root/../bend/bend2/main.ts" "$src/$entry" -o "$out"
test -x "$out"
echo "built $out order $order from $src/$entry sha256 $(sha256sum "$out" | cut -d' ' -f1)"
