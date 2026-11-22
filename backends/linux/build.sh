#!/usr/bin/env bash
# Linux desktop engine build: eureka-engine -> EUREKA_LINUX_ENGINE contract dir (scratch builds only).
#   SRC  engine clone (default: repo root of this script)
#   APP  bend-app tree (serve.c with Linux CUDA tower + SIGUSR1 cancel), read-only; copied to OUT
#   OUT  build scratch (default /tmp/lp-out)
#   DEST contract dir (default /tmp/eureka-linux-engine)
set -euo pipefail
SRC=${SRC:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
APP=${APP:-/home/omen/worktrees/eureka-linux-parity}
OUT=${OUT:-/tmp/lp-out}
DEST=${DEST:-/tmp/eureka-linux-engine}

# Auto-discover clang >= 19 (needed for #embed in Bend-emitted C)
if [ -z "${CC:-}" ]; then
  for c in /usr/lib64/rocm/llvm/bin/clang-19 clang-21 clang-20 clang-19 clang; do
    if command -v "$c" >/dev/null 2>&1; then
      ver=$("$c" --version 2>/dev/null | head -n1 | grep -oE '[0-9]+\.[0-9]+' | head -n1 | cut -d. -f1 || true)
      if [ -n "$ver" ] && [ "$ver" -ge 19 ] 2>/dev/null; then CC="$c"; break; fi
    fi
  done
  : "${CC:?clang >= 19 is required (set CC=/path/to/clang-19)}"
fi

CUDA=${CUDA_HOME:-/usr/local/cuda}
export PATH=$HOME/.bun/bin:$CUDA/bin:$PATH CC CUDA_HOME=$CUDA
mkdir -p "$OUT/libshim"

# Auto-discover libopenblas.so.0 across Fedora (/usr/lib64) and Debian/Ubuntu (/usr/lib/x86_64-linux-gnu, aarch64)
OPENBLAS_SO=""
for p in /usr/lib64/libopenblas.so.0 /usr/lib/x86_64-linux-gnu/libopenblas.so.0 /usr/lib/aarch64-linux-gnu/libopenblas.so.0 /usr/lib/libopenblas.so.0; do
  if [ -e "$p" ]; then OPENBLAS_SO="$p"; break; fi
done
[ -n "$OPENBLAS_SO" ] || { echo "error: libopenblas.so.0 not found (install openblas / libopenblas0)" >&2; exit 1; }
ln -sf "$OPENBLAS_SO" "$OUT/libshim/libopenblas.so"
export LIBRARY_PATH=$OUT/libshim${LIBRARY_PATH:+:$LIBRARY_PATH}

step() { echo; echo "=== $* ($(date +%T))"; }

step "1. engine_gpu: build.sh (pinned bend 240ecef4, patch stack, emit, linuxcuda.py, remap_cc)"
if [ "${SKIP_GPU:-0}" != 1 ]; then
  LLB_BEND=$SRC LLB_OUT=$OUT/gpu bash "$SRC/build.sh"
fi
python3 "$SRC/backends/linux/linuxcuda.py" "$OUT/gpu/llb-engine.c" "$OUT/gpu/llb-engine.linux.c"
(cd "$OUT/gpu" && LLB_CC_EXTRA="$(echo obj/*.o) -lopenblas" bash "$SRC/tools/remap_cc.sh" llb-engine.linux.c engine_gpu)

step "2. engine_cpu (backends/windows/engine_cpu.c, native POSIX)"
o=$OUT/cpu/obj; mkdir -p "$o"; objs=()
req=(-std=c11 -O2 -ffp-contract=off -fno-fast-math -D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE -Wall -Wextra -Wno-unused-function -Wno-unused-parameter)
for u in tools/sys1dual/dual.c tools/metrics/metrics.c tools/metrics/tokstream.c tools/glue/glue_out.c tools/clef3/head.c \
         tools/clef3/weights.c tools/clef3/spans.c tools/clef3/json.c host/llb_sampler.c \
         backends/windows/engine_cpu_gguf.c backends/windows/engine_cpu.c; do
  "$CC" "${req[@]}" -c "$SRC/$u" -o "$o/$(basename "$u" .c).o"; objs+=("$o/$(basename "$u" .c).o")
done
"$CC" -O2 -o "$OUT/cpu/engine_cpu" "${objs[@]}" -lopenblas -lpthread -lm -Wl,--disable-new-dtags,-rpath,'$ORIGIN'

step "3. gguf2pack + llb-tok (this engine tree: one pack contract)"
"$CC" -std=c11 -O2 -Wall -D_GNU_SOURCE -o "$OUT/gguf2pack" "$SRC/tools/gguf2pack.c"
"$CC" -O2 -o "$OUT/llb-tok" "$SRC/tools/llb/llb-tok.c"

step "4. cudavis-cli (VITC_NO_CUBLAS, cudart static; fatbin sm_80..sm_120 + PTX)"
v=$OUT/vision; mkdir -p "$v"
CF=(-std=c11 -D_GNU_SOURCE -O3 -ffp-contract=off -I"$SRC/vision" -I"$SRC/host")
gcc "${CF[@]}" -w -c "$SRC/vision/vpre.c" -o "$v/vpre.o"
gcc "${CF[@]}" -w -c "$SRC/vision/vmm.c" -o "$v/vmm.o"
gcc "${CF[@]}" -w -c "$SRC/vision/cudavis_cli.c" -o "$v/cli.o"
GENC=(-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86
      -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=[sm_120,compute_120])
nvcc -O3 -fmad=false "${GENC[@]}" -std=c++17 -DVITC_NO_CUBLAS -Xcompiler -ffp-contract=off -I"$SRC/vision" -I"$SRC/host" \
  -c "$SRC/vision/vitc.cu" -o "$v/vitc.o"
nvcc "${GENC[@]}" -cudart static "$v/cli.o" "$v/vitc.o" "$v/vpre.o" "$v/vmm.o" -lm -lpthread -o "$v/cudavis-cli"

step "5. llb-serve (bend-app desktop/engine/metal/src/serve.c, POSIX build, + vpre.o from app vision host)"
s=$OUT/serve; rm -rf "$s"; mkdir -p "$s"; cp -a "$APP/desktop/engine/metal/src/." "$s/src/"
VH=$s/vision; mkdir -p "$VH"; cp -R "$APP/engine-src/vision/host" "$APP/engine-src/vision/third_party" "$VH/"
(cd "$VH" && patch -s --fuzz=0 -p1 < "$APP/desktop/engine/metal/patches/vision-mintok.patch")
"$CC" -D_GNU_SOURCE -std=c11 -O2 -w -I"$VH/host" -I"$VH/third_party" -c "$VH/host/vpre.c" -o "$s/vpre.o"
"$CC" -D_GNU_SOURCE -std=c11 -O2 -Wall -Wextra -pthread -I"$VH/host" "$s/src/serve.c" "$s/src/json.c" "$s/src/speak_route.c" \
  "$s/src/vendor/tools/sys1dual/serve_structured.c" "$s/src/vendor/host/llb_structured.c" "$s/vpre.o" -lm -o "$s/llb-serve"

step "5b. engine_inkling_gpu + engine_inkling_cpu (native Inkling 2 Mini CUDA/CPU shape engine)"
INK_SRC="$SRC/inkling2/engine/src/inkling2"
if [ -f "$INK_SRC/CMakeLists.txt" ] && command -v cmake >/dev/null 2>&1; then
  rm -rf "$OUT/inkling-cuda" "$OUT/inkling-cpu"
  cmake -S "$INK_SRC" -B "$OUT/inkling-cuda" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
    -DSTRATA_INK_CUDA_ARCHS="80;86;89;120" -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc"
  cmake --build "$OUT/inkling-cuda" --target strata-inkling -j"$(nproc)"
  cmake -S "$INK_SRC" -B "$OUT/inkling-cpu" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF
  cmake --build "$OUT/inkling-cpu" --target strata-inkling -j"$(nproc)"
fi

step "6. stage $DEST/{cuda,cpu}"
rm -rf "$DEST"; mkdir -p "$DEST/cuda" "$DEST/cpu"
cp "$s/llb-serve" "$OUT/gguf2pack" "$OUT/llb-tok" "$DEST/cuda/"
cp "$OUT/gpu/engine_gpu" "$DEST/cuda/engine_gpu"
[ -x "$OUT/inkling-cuda/strata-inkling" ] && install -m 0755 "$OUT/inkling-cuda/strata-inkling" "$DEST/cuda/engine_inkling_gpu"
# Auto-discover NVRTC sonames for the installed CUDA major version
NVRTC_MAIN=$(ls "$CUDA"/lib64/libnvrtc.so.[0-9]* 2>/dev/null | grep -E 'libnvrtc\.so\.[0-9]+$' | head -n1 || true)
NVRTC_BUILTINS=$(ls "$CUDA"/lib64/libnvrtc-builtins.so.[0-9]*.[0-9]* 2>/dev/null | grep -E 'libnvrtc-builtins\.so\.[0-9]+\.[0-9]+$' | head -n1 || true)
[ -n "$NVRTC_MAIN" ] && [ -n "$NVRTC_BUILTINS" ] || { echo "error: NVRTC libs not found in $CUDA/lib64" >&2; exit 1; }
cp -L "$NVRTC_MAIN" "$NVRTC_BUILTINS" "$DEST/cuda/"
cp "$v/cudavis-cli" "$DEST/cuda/"
cp "$s/llb-serve" "$OUT/gguf2pack" "$OUT/llb-tok" "$OUT/cpu/engine_cpu" "$DEST/cpu/"
[ -x "$OUT/inkling-cpu/strata-inkling" ] && install -m 0755 "$OUT/inkling-cpu/strata-inkling" "$DEST/cpu/engine_inkling_cpu"
ls -la "$DEST"/*
