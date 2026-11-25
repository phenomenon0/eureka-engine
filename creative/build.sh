#!/usr/bin/env bash
# Scratch-only Unified Multi-Backend Build for Eureka Creative Engine v16
# Supports:
#   EC_BACKEND=cuda   (default on Linux/Windows w/ nvcc — 100% Native PTX TensorCore)
#   EC_BACKEND=metal  (default on macOS Darwin — Native Apple Silicon Metal 3.1 simdgroup_matrix + Unified Memory)
#   EC_BACKEND=vulkan (Cross-Vendor Linux/Windows AMD Radeon, Intel Arc, NVIDIA, Lavapipe SPIR-V Compute)
set -euo pipefail

SRC=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ -z "${EC_BACKEND:-}" ]; then
  if [ "$(uname -s)" = "Darwin" ]; then
    EC_BACKEND=metal
  else
    EC_BACKEND=cuda
  fi
fi

OUT=${1:-${OUT:-/tmp/eureka-creative-build-${EC_BACKEND}}}
mkdir -p "$OUT/obj" "$OUT/artifacts" "$OUT/receipts"

CC=${CC:-cc}
CXX=${CXX:-c++}

echo "=== Building Eureka Creative Engine v16 [EC_BACKEND=$EC_BACKEND] in $OUT ==="

# 1. Shared Pure C11 Runtime, Server, ComfyUI Graph, Tokenizers, Safetensors & Media
C_FLAGS=(-std=c11 -O3 -Wall -Wextra -Wno-unused-parameter -I"$SRC")
for cf in ec_safetensors ec_cowrie ec_flow ec_media ec_nvenc ec_storage ec_bpe ec_unigram ec_recipes ec_serve; do
  "$CC" "${C_FLAGS[@]}" -c "$SRC/${cf}.c" -o "$OUT/obj/${cf}.o"
done

C_OBJS=(
  "$OUT/obj/ec_safetensors.o"
  "$OUT/obj/ec_cowrie.o"
  "$OUT/obj/ec_flow.o"
  "$OUT/obj/ec_media.o"
  "$OUT/obj/ec_nvenc.o"
  "$OUT/obj/ec_storage.o"
  "$OUT/obj/ec_bpe.o"
  "$OUT/obj/ec_unigram.o"
  "$OUT/obj/ec_recipes.o"
)

MODEL_CUS=(ec_dit ec_vram_pool ec_models ec_lora ec_flux2 ec_acestep ec_trellis2 ec_wan22)

if [ "$EC_BACKEND" = "cuda" ]; then
  NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
  GENC=(-gencode arch=compute_80,code=sm_80
        -gencode arch=compute_86,code=sm_86
        -gencode arch=compute_89,code=sm_89
        -gencode arch=compute_90,code=sm_90
        -gencode arch=compute_120,code=[sm_120,compute_120])

  for kf in ec_kernels ec_kernels_v2 ec_ops "${MODEL_CUS[@]}" ec_studio ec_bench; do
    "$NVCC" -O3 -std=c++17 -DEC_BACKEND_CUDA=1 "${GENC[@]}" -I"$SRC" -c "$SRC/${kf}.cu" -o "$OUT/obj/${kf}.o"
  done

  GPU_OBJS=(
    "$OUT/obj/ec_kernels.o" "$OUT/obj/ec_kernels_v2.o" "$OUT/obj/ec_ops.o"
    "$OUT/obj/ec_dit.o" "$OUT/obj/ec_vram_pool.o" "$OUT/obj/ec_models.o" "$OUT/obj/ec_lora.o"
    "$OUT/obj/ec_flux2.o" "$OUT/obj/ec_acestep.o" "$OUT/obj/ec_trellis2.o" "$OUT/obj/ec_wan22.o"
  )

  "$NVCC" -O3 "${GENC[@]}" -cudart static "${C_OBJS[@]}" "${GPU_OBJS[@]}" "$OUT/obj/ec_bench.o" \
    -lm -lrt -lpthread -ldl -o "$OUT/ec-bench"
  "$NVCC" -O3 "${GENC[@]}" -cudart static "${C_OBJS[@]}" "${GPU_OBJS[@]}" "$OUT/obj/ec_studio.o" "$OUT/obj/ec_serve.o" \
    -lm -lrt -lpthread -ldl -o "$OUT/eureka-creative-serve"

elif [ "$EC_BACKEND" = "vulkan" ]; then
  "$CC" -std=c11 -O3 -fopenmp -DEC_BACKEND_VULKAN=1 -I"$SRC" \
    -c "$SRC/backends/ec_backend_vulkan.c" -o "$OUT/obj/ec_backend_vulkan.o"

  for mf in "${MODEL_CUS[@]}" ec_studio ec_bench; do
    "$CXX" -O3 -std=c++17 -fopenmp -DEC_BACKEND_VULKAN=1 -x c++ -I"$SRC" \
      -c "$SRC/${mf}.cu" -o "$OUT/obj/${mf}.o"
  done

  GPU_OBJS=(
    "$OUT/obj/ec_backend_vulkan.o"
    "$OUT/obj/ec_dit.o" "$OUT/obj/ec_vram_pool.o" "$OUT/obj/ec_models.o" "$OUT/obj/ec_lora.o"
    "$OUT/obj/ec_flux2.o" "$OUT/obj/ec_acestep.o" "$OUT/obj/ec_trellis2.o" "$OUT/obj/ec_wan22.o"
  )

  "$CXX" -O3 -fopenmp "${C_OBJS[@]}" "${GPU_OBJS[@]}" "$OUT/obj/ec_bench.o" \
    -lvulkan -lm -lpthread -ldl -o "$OUT/ec-bench"
  "$CXX" -O3 -fopenmp "${C_OBJS[@]}" "${GPU_OBJS[@]}" "$OUT/obj/ec_studio.o" "$OUT/obj/ec_serve.o" \
    -lvulkan -lm -lpthread -ldl -o "$OUT/eureka-creative-serve"

elif [ "$EC_BACKEND" = "metal" ]; then
  CLANGXX=${CLANGXX:-clang++}
  OMP_FLAGS=()
  if "$CLANGXX" -Xpreprocessor -fopenmp -lomp -x c++ -E - </dev/null >/dev/null 2>&1; then
    OMP_FLAGS=(-Xpreprocessor -fopenmp -lomp)
  fi

  if command -v xcrun >/dev/null 2>&1 && xcrun -sdk macosx metal -v >/dev/null 2>&1; then
    METAL_STD=""
    for cand_std in metal3.1 metal3.0 metal2.4; do
      if xcrun -sdk macosx metal -std="$cand_std" -O3 -c "$SRC/backends/ec_creative.metal" -o "$OUT/obj/ec_creative.air" 2>/dev/null; then
        METAL_STD="$cand_std"
        break
      fi
    done
    if [ -n "$METAL_STD" ]; then
      xcrun -sdk macosx metallib "$OUT/obj/ec_creative.air" -o "$OUT/ec_creative.metallib"
      echo "  [metal] compiled ec_creative.metallib (-std=$METAL_STD)"
    else
      cp "$SRC/backends/ec_creative.metal" "$OUT/ec_creative.metal"
    fi
  else
    cp "$SRC/backends/ec_creative.metal" "$OUT/ec_creative.metal"
  fi

  "$CLANGXX" -O3 -std=c++17 -fobjc-arc -DEC_BACKEND_METAL=1 \
    -DEC_METAL_SHADER_SRC_PATH="\"$SRC/backends/ec_creative.metal\"" \
    "${OMP_FLAGS[@]+"${OMP_FLAGS[@]}"}" -I"$SRC" \
    -c "$SRC/backends/ec_backend_metal.mm" -o "$OUT/obj/ec_backend_metal.o"

  for mf in "${MODEL_CUS[@]}" ec_studio ec_bench; do
    "$CLANGXX" -O3 -std=c++17 -DEC_BACKEND_METAL=1 -Wno-unknown-pragmas \
      "${OMP_FLAGS[@]+"${OMP_FLAGS[@]}"}" -x c++ -I"$SRC" \
      -c "$SRC/${mf}.cu" -o "$OUT/obj/${mf}.o"
  done

  GPU_OBJS=(
    "$OUT/obj/ec_backend_metal.o"
    "$OUT/obj/ec_dit.o" "$OUT/obj/ec_vram_pool.o" "$OUT/obj/ec_models.o" "$OUT/obj/ec_lora.o"
    "$OUT/obj/ec_flux2.o" "$OUT/obj/ec_acestep.o" "$OUT/obj/ec_trellis2.o" "$OUT/obj/ec_wan22.o"
  )

  "$CLANGXX" -O3 "${OMP_FLAGS[@]+"${OMP_FLAGS[@]}"}" "${C_OBJS[@]}" "${GPU_OBJS[@]}" "$OUT/obj/ec_bench.o" \
    -framework Metal -framework Foundation -lm -lpthread -ldl -o "$OUT/ec-bench"
  "$CLANGXX" -O3 "${OMP_FLAGS[@]+"${OMP_FLAGS[@]}"}" "${C_OBJS[@]}" "${GPU_OBJS[@]}" "$OUT/obj/ec_studio.o" "$OUT/obj/ec_serve.o" \
    -framework Metal -framework Foundation -lm -lpthread -ldl -o "$OUT/eureka-creative-serve"
else
  echo "Unknown EC_BACKEND=$EC_BACKEND (expected cuda, vulkan, or metal)" >&2
  exit 1
fi

SIZE_FN() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1"; }
echo "=== Built [$EC_BACKEND] $OUT/ec-bench ($(SIZE_FN "$OUT/ec-bench") bytes) ==="
echo "=== Built [$EC_BACKEND] $OUT/eureka-creative-serve ($(SIZE_FN "$OUT/eureka-creative-serve") bytes) ==="
