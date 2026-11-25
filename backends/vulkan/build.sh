#!/usr/bin/env bash
# backends/vulkan/build.sh — Build Native Cross-Vendor Vulkan 1.2+ LLM Engine (`engine_vulkan`)
# Accelerates Qwen3.5 (0.8B..27B) on AMD Radeon (RADV), Intel Arc (ANV), and NVIDIA Vulkan
# using DEVICE_LOCAL VRAM weight slabs + Naga-compiled SPIR-V 1.3 GEMV shaders (llb_vulkan_spv.h).
set -euo pipefail
SRC=${SRC:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
OUT=${OUT:-/tmp/eureka-vulkan-llm}
CC=${CC:-cc}
mkdir -p "$OUT/obj" "$OUT/libshim"

OPENBLAS_SO=""
for p in /usr/lib64/libopenblas.so.0 /usr/lib/x86_64-linux-gnu/libopenblas.so.0 /usr/lib/aarch64-linux-gnu/libopenblas.so.0 /usr/lib/libopenblas.so.0; do
  if [ -e "$p" ]; then OPENBLAS_SO="$p"; break; fi
done
[ -n "$OPENBLAS_SO" ] || { echo "error: libopenblas.so.0 not found" >&2; exit 1; }
ln -sf "$OPENBLAS_SO" "$OUT/libshim/libopenblas.so"
export LIBRARY_PATH=$OUT/libshim${LIBRARY_PATH:+:$LIBRARY_PATH}

req=(-std=c11 -O2 -ffp-contract=off -fno-fast-math -D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE -Wall -Wextra -Wno-unused-function -Wno-unused-parameter)
objs=()
for u in tools/sys1dual/dual.c tools/metrics/metrics.c tools/metrics/tokstream.c tools/glue/glue_out.c tools/clef3/head.c \
         tools/clef3/weights.c tools/clef3/spans.c tools/clef3/json.c host/llb_sampler.c \
         backends/windows/engine_cpu_gguf.c backends/vulkan/llb_vulkan_gemv.c; do
  "$CC" "${req[@]}" -c "$SRC/$u" -o "$OUT/obj/$(basename "$u" .c).o"
  objs+=("$OUT/obj/$(basename "$u" .c).o")
done
"$CC" "${req[@]}" -DLLB_VULKAN=1 -c "$SRC/backends/windows/engine_cpu.c" -o "$OUT/obj/engine_vulkan.o"
objs+=("$OUT/obj/engine_vulkan.o")

"$CC" -O2 -o "$OUT/engine_vulkan" "${objs[@]}" -lvulkan -lopenblas -lpthread -lm -ldl -Wl,--disable-new-dtags,-rpath,'$ORIGIN'
echo "=== Built native Vulkan LLM engine: $OUT/engine_vulkan ($(stat -c%s "$OUT/engine_vulkan") bytes) ==="
