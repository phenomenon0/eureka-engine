#!/usr/bin/env bash
# tools/sys1dual/build-structured.sh: G16 CPU artifacts for llb-serve --structured and its tests
# (no GPU, no engine): llb-serve, fake_engine (dual.c's stream protocol + the real tokenizer +
# G1's real mask, no model), st_test (request-side unit tests). SAN=1: ASan+UBSan builds.
# Env: OUT (output dir), CC (default clang-21, any C11 compiler works here).
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${OUT:?set OUT}"
CC=${CC:-clang}
mkdir -p "$OUT"
f=(-D_GNU_SOURCE -std=c11 -O2 -Wall -Wextra -Werror -Wno-unused-function -pthread)
[ "${SAN:-0}" = 1 ] && f+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
d=$root/tools/sys1dual
st=("$d/serve_structured.c" "$root/host/llb_structured.c" "$root/tools/clef3/json.c")
sv=("$d/mm.c") # G5: serve.c also links mm.c (merge-train S8)
"$CC" "${f[@]}" "$d/serve.c" "${sv[@]}" "${st[@]}" -lm -o "$OUT/llb-serve"
"$CC" "${f[@]}" "$d/st_test.c" "${st[@]}" -lm -o "$OUT/st_test"
"$CC" "${f[@]}" "$d/fake_engine.c" "$root/host/llb_structured.c" "$root/tools/clef3/json.c" "$root/tools/qwen35_tokenizer.c" -lm -o "$OUT/fake_engine"
sha256sum "$OUT/llb-serve" "$OUT/st_test" "$OUT/fake_engine"
