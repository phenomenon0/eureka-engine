#!/usr/bin/env bash
# lane D: eureka-engine (Linux) -> the bend-app EUREKA_LINUX_ENGINE contract dir. Scratch only.
#   SRC  engine clone (default /tmp/lp-engine-build)   APP  bend-app tree for llb-serve's serve.c (read-only)
#   OUT  build scratch (default /tmp/lp-out)            DEST contract dir (default /tmp/eureka-linux-engine)
set -euo pipefail
SRC=${SRC:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}; APP=${APP:-/home/omen/worktrees/bend-app-android}
OUT=${OUT:-/tmp/lp-out}; DEST=${DEST:-/tmp/eureka-linux-engine}
CC=${CC:-/usr/lib64/rocm/llvm/bin/clang-19}   # clang >= 19 (bend cc_find, #embed); Fedora 43 ships clang-18 only
CUDA=${CUDA_HOME:-/usr/local/cuda}
export PATH=$HOME/.bun/bin:$CUDA/bin:$PATH CC CUDA_HOME=$CUDA
mkdir -p "$OUT/libshim"
# Fedora has libopenblas.so.0 but no -devel symlink: a link-time shim (runtime needs the openblas package)
ln -sf /usr/lib64/libopenblas.so.0 "$OUT/libshim/libopenblas.so"
export LIBRARY_PATH=$OUT/libshim${LIBRARY_PATH:+:$LIBRARY_PATH}
step() { echo; echo "=== $* ($(date +%T))"; }

step "1. engine_gpu: build.sh (pinned bend 240ecef4 tree aada4cbf, patch stack, emit, remap_cc)"
if [ "${SKIP_GPU:-0}" != 1 ]; then
  LLB_BEND=$SRC LLB_OUT=$OUT/gpu bash "$SRC/build.sh"
fi
# the served edits every OS needs (wincuda.ps1 edits 2 + 3: LLB_GEN_EOG, the decode run under SYS1), then the same link
python3 "$SRC/backends/linux/linuxcuda.py" "$OUT/gpu/llb-engine.c" "$OUT/gpu/llb-engine.linux.c"
(cd "$OUT/gpu" && LLB_CC_EXTRA="$(echo obj/*.o) -lopenblas" bash "$SRC/tools/remap_cc.sh" llb-engine.linux.c engine_gpu)

step "2. engine_cpu (backends/windows/engine_cpu.c, native POSIX: no shim)"
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

step "4. cudavis-cli (vision/: as build-vision.ps1: VITC_NO_CUBLAS, cudart static; fatbin sm_80..sm_120 (+PTX); the tower needs mma.m16n8k16 = sm_80+ + PTX)"
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

step "5. llb-serve (bend-app desktop/engine/metal/src/serve.c, POSIX build, + vpre.o from the app's vision host)"
s=$OUT/serve; rm -rf "$s"; mkdir -p "$s"; cp -a "$APP/desktop/engine/metal/src/." "$s/src/"
# the app tree's vision host (build-engine.sh: engine-src vision/host + patches/vision-mintok.patch -> obj-vis/vpre.o)
VH=$s/vision; mkdir -p "$VH"; cp -R "$APP/engine-src/vision/host" "$APP/engine-src/vision/third_party" "$VH/"
(cd "$VH" && patch -s --fuzz=0 -p1 < "$APP/desktop/engine/metal/patches/vision-mintok.patch")
"$CC" -D_GNU_SOURCE -std=c11 -O2 -w -I"$VH/host" -I"$VH/third_party" -c "$VH/host/vpre.c" -o "$s/vpre.o"
if [ -n "${SERVE_PATCH:-}" ]; then (cd "$s/src" && patch -p1 --fuzz=0 < "$SERVE_PATCH"); fi
"$CC" -D_GNU_SOURCE -std=c11 -O2 -Wall -Wextra -pthread -I"$VH/host" "$s/src/serve.c" "$s/src/json.c" "$s/src/speak_route.c" \
  "$s/src/vendor/tools/sys1dual/serve_structured.c" "$s/src/vendor/host/llb_structured.c" "$s/vpre.o" -lm -o "$s/llb-serve"

step "6. stage $DEST/{cuda,cpu}"
rm -rf "$DEST"; mkdir -p "$DEST/cuda" "$DEST/cpu"
cp "$s/llb-serve" "$OUT/gguf2pack" "$OUT/llb-tok" "$DEST/cuda/"; cp "$OUT/gpu/engine_gpu" "$DEST/cuda/engine_gpu"
# NVRTC: engine_gpu compiles its kernels at first run; DT_RPATH $ORIGIN finds these beside it (a driver-only box has none)
cp -L "$CUDA"/lib64/libnvrtc.so.13 "$CUDA"/lib64/libnvrtc-builtins.so.13.[0-9] "$DEST/cuda/"
cp "$v/cudavis-cli" "$DEST/cuda/"
cp "$s/llb-serve" "$OUT/gguf2pack" "$OUT/llb-tok" "$OUT/cpu/engine_cpu" "$DEST/cpu/"
ls -la "$DEST"/*
