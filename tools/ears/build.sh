#!/usr/bin/env bash
# tools/ears/build.sh: EARS-CANON. llb-ear, the served ear (ear_worker.cpp): host mel (aud_mel.cpp, the goldens' flags:
# g++ -O3 -ffp-contract=off, no -march) + the CUDA AuT tower (aut.cu, -fmad=false, sm_86) + the slim-chunker ears.cpp on
# llama.cpp 2ca15f5 (the goldens' commit; libllama/libggml from its CUDA build) + mm.c/json.c (sha256, sniff, JSON).
# No libmtmd: the mmproj is opened once, by the tower. The Metal tower (aut_metal.mm) stays the Mac/iOS path.
# Env: OUT (output dir), LC (llama.cpp 2ca15f5 source), LCB (its CUDA build, bin/ has libllama.so + libggml*.so).
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${OUT:?set OUT}" "${LC:?llama.cpp 2ca15f5 source}" "${LCB:?its CUDA build dir}"
[ "$(git -C "$LC" rev-parse --short=7 HEAD)" = 2ca15f5 ] || { echo "build: $LC is not llama.cpp 2ca15f5" >&2; exit 2; }
export PATH=/usr/local/cuda/bin:$PATH
e=$root/tools/ears o=$OUT/ear-obj
mkdir -p "$o"
g++ -std=c++17 -O3 -ffp-contract=off -c "$e/aud_mel.cpp" -o "$o/aud_mel.o"
nvcc -std=c++17 -O3 -arch=sm_86 -fmad=false -I"$LC/ggml/include" -c "$e/aut.cu" -o "$o/aut.o"
g++ -std=c++17 -O2 -I"$LC/include" -I"$LC/ggml/include" -c "$e/ears.cpp" -o "$o/ears.o"
g++ -std=c++17 -O2 -c "$e/ear_worker.cpp" -o "$o/ear_worker.o"
gcc -D_GNU_SOURCE -std=c11 -O2 -c "$root/tools/sys1dual/mm.c" -o "$o/mm.o"
gcc -D_GNU_SOURCE -std=c11 -O2 -c "$root/tools/clef3/json.c" -o "$o/json.o"
g++ "$o"/ear_worker.o "$o"/ears.o "$o"/aud_mel.o "$o"/aut.o "$o"/mm.o "$o"/json.o -o "$OUT/llb-ear" \
  -L/usr/local/cuda/lib64 -lcudart -lcublas -L"$LCB/bin" -lllama -lggml -lggml-base -lm \
  -Wl,-rpath,"$LCB/bin" -Wl,-rpath,/usr/local/cuda/lib64
sha256sum "$OUT/llb-ear" "$e"/aut.cu "$e"/aud_mel.cpp "$e"/ears.cpp "$e"/ear_worker.cpp
