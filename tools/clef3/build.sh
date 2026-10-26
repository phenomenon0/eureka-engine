#!/usr/bin/env bash
# build.sh GGUF: builds clef-decide, its span helper, the engine and the GGUF's sidecar into $CLEF_OUT.
#   CLEF_OUT   output directory (default: build/clef3)
#   LLB_BEND   pinned Bend compiler checkout (required for the engine build; see docs/guides/run-on-cuda.md)
#   CC         C compiler (default: clang)
# Run from the repository root. Needs clang, OpenBLAS, and for the engine the CUDA toolkit.
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: $0 CLEF_GGUF" >&2; exit 2; }
gguf=$1
out=${CLEF_OUT:-$PWD/build/clef3}
cc=${CC:-clang}
: "${LLB_BEND:?set LLB_BEND to the pinned Bend compiler checkout}"
mkdir -p "$out"

# span helper and tokenizer
"$cc" -D_POSIX_C_SOURCE=200809L -std=c11 -O2 -Wall -Wextra -DSPANS_MAIN \
  tools/clef3/spans.c tools/clef3/json.c tools/qwen35_tokenizer.c -lm -o "$out/clef-spans"

# decision CLI (CPU head)
"$cc" -D_POSIX_C_SOURCE=200809L -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Wno-unused-function \
  tools/clef3/decide.c tools/clef3/head.c tools/clef3/weights.c tools/clef3/spans.c tools/clef3/json.c \
  tools/qwen35_tokenizer.c -lopenblas -lpthread -lm -o "$out/clef-decide"

# engine
LLB_ROOT=$PWD LLB_OUT=$out CC=$cc bash build.sh

# sidecar for the Clef GGUF
"$out/gguf2pack" "$gguf" --sidecar-only "$out/sidecar"
echo "built: $out/clef-decide, $out/llb-engine, $out/sidecar"
