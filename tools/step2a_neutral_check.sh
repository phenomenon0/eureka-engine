#!/usr/bin/env bash
# host/llb_host.c must compile with no CUDA header: a stub backend + stub runtime decls only.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
printf '#define BEND_CUDA 1\n#define LLB_WARP_FID 1\n#include "%s/tools/step2a_stub.h"\n#include "%s"\n' "$root" "${1:-$root/host/llb_host.c}" \
  | clang -fsyntax-only -Wall -Wno-unused-function -x c - && echo "neutral: ok"
