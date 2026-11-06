#!/usr/bin/env bash
# Linux host compile of the CUDA line's plain-C parts (no GPU, no CUDA headers needed):
# build.sh step 4 request objects + step 5 gguf2pack, and build-serve.sh's llb-serve line.
set -uo pipefail
cd /src; O=/out; rm -rf "$O"/*; mkdir -p "$O/obj"; fail=0
clang --version | head -1
# Ubuntu ships cblas.h under openblas-pthread/; head.c includes <openblas/cblas.h> (backends/metal/build.sh links the same way)
mkdir -p /tmp/inc && ln -sfn "$(dirname "$(find /usr/include -name cblas.h -path "*openblas*" | head -1)")" /tmp/inc/openblas; export CPATH=/tmp/inc
for f in sys1dual/dual.c metrics/metrics.c metrics/tokstream.c glue/glue_out.c clef3/head.c clef3/weights.c clef3/spans.c clef3/json.c qwen35_tokenizer.c; do
  clang -D_POSIX_C_SOURCE=200809L -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Wno-unused-function \
    -c "tools/$f" -o "$O/obj/$(basename "$f" .c).o"; rc=$?; echo "obj $f rc=$rc"; [ $rc = 0 ] || fail=1
done
clang -std=c11 -O2 -Wall -o "$O/gguf2pack" tools/gguf2pack.c; rc=$?; echo "gguf2pack rc=$rc"; [ $rc = 0 ] || fail=1
clang -D_GNU_SOURCE -std=c11 -O2 -Wall -Wextra -pthread -DLLB_GIT='"port"' tools/sys1dual/serve.c tools/sys1dual/serve_structured.c \
  host/llb_structured.c tools/sys1dual/mm.c tools/clef3/json.c -lm -o "$O/llb-serve"; rc=$?; echo "llb-serve rc=$rc"; [ $rc = 0 ] || fail=1
# the request objects link: a stub main supplies nothing; check the objects resolve against each other + openblas
echo "fail=$fail"; exit $fail
