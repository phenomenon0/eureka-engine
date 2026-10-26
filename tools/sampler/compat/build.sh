#!/usr/bin/env bash
# build ref_chain (oracle) + gate drivers into $OUT. CPU only. usage: build.sh [-DMUT_X ...]
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); OUT=${OUT:-/tmp/llb-sampler}
LLAMA=${LLAMA_DIR:-$HOME/llama.cpp}; mkdir -p "$OUT"
CF="-O2 -ffp-contract=off -fno-fast-math -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra"
[ -x "$OUT/ref_chain" ] || nice -n 19 g++ -O2 -std=c++17 -I"$LLAMA/include" -I"$LLAMA/ggml/include" "$here/ref_chain.cpp" \
  -L"$LLAMA/build/bin" -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$LLAMA/build/bin" -o "$OUT/ref_chain"
for g in gate1 gate2 gate3 bench dump_misc; do
  [ -f "$here/$g.c" ] && nice -n 19 gcc $CF "$@" "$here/$g.c" "$here/q35compat.c" -o "$OUT/$g${MUTSUF:-}" -lm
done
true
