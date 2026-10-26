#!/usr/bin/env bash
# tools/build_qwen35_cli.sh — build the Qwen3.5 text CLI (qwen35-cli) and the
# tokenizer gate tools on the omen. CPU is built with -ffp-contract=off; the
# CUDA build keeps -fmad=false so it stays the byte-contract twin.
#   build/qwen35-cli       CPU
#   build/qwen35-cli-gpu   CUDA (--gpu), if nvcc is present
#   build/qwen35_toktest   C tokenizer driver
#   build/qwen35_ref_tokenize  libllama oracle (if ~/llama.cpp is built)
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$root/build"

gcc -O2 -ffp-contract=off -fno-fast-math "$root/tools/qwen35_toktest.c" -o "$root/build/qwen35_toktest" -lm
gcc -O2 -ffp-contract=off -fno-fast-math "$root/tools/qwen35_engine.c" -o "$root/build/qwen35-cli" -lm

LLAMA=${LLAMA_DIR:-"$HOME/llama.cpp"}
if [ -f "$LLAMA/build/bin/libllama.so" ]; then
  g++ -O2 -std=c++17 -I "$LLAMA/include" -I "$LLAMA/ggml/include" \
      "$root/tools/qwen35_ref_tokenize.cpp" \
      -L "$LLAMA/build/bin" -lllama -lggml -lggml-base -lggml-cpu \
      -Wl,-rpath,"$LLAMA/build/bin" -o "$root/build/qwen35_ref_tokenize"
  echo "built qwen35_ref_tokenize (libllama at $LLAMA)"
else
  echo "skip qwen35_ref_tokenize (no libllama at $LLAMA)"
fi

if command -v nvcc >/dev/null 2>&1; then
  TMPDIR=${TMPDIR:-/tmp} nvcc -O3 -fmad=false -std=c++14 -arch=sm_86 \
      -o "$root/build/qwen35-cli-gpu" "$root/tools/qwen35_engine.cu"
  echo "built qwen35-cli-gpu (sm_86)"
fi

shasum=$(command -v sha256sum || command -v shasum)
ls -la "$root/build" | grep -E "qwen35" || true
