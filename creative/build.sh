#!/usr/bin/env bash
# Scratch-only build for 100% Native Eureka Creative Engine (creative/ v3).
# Zero cuBLAS, zero ggml, zero external ML frameworks.
# All build outputs live strictly under OUT (default /tmp/eureka-creative-build).
set -euo pipefail

SRC=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:-/tmp/eureka-creative-build}
mkdir -p "$OUT/obj" "$OUT/artifacts" "$OUT/receipts"

NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
CC=${CC:-gcc}
GENC=(-gencode arch=compute_80,code=sm_80
      -gencode arch=compute_86,code=sm_86
      -gencode arch=compute_89,code=sm_89
      -gencode arch=compute_90,code=sm_90
      -gencode arch=compute_120,code=[sm_120,compute_120])

echo "=== Building 100% Native Eureka Creative Engine v18 Frontier Peak (Z-Image-Turbo 6.15B + FLUX.2 + LTX-2.3 22B Joint A/V + Wan 2.2 + ACE-Step 1.5-XL 4B + TRELLIS.2-4B 512^3 O-Voxel PBR + speridlabs/idol 1M 3DGS) in $OUT ==="
"$CC" -std=c11 -O3 -Wall -Wextra -I"$SRC" -c "$SRC/ec_safetensors.c" -o "$OUT/obj/ec_safetensors.o"
"$CC" -std=c11 -O3 -Wall -Wextra -I"$SRC" -c "$SRC/ec_cowrie.c"      -o "$OUT/obj/ec_cowrie.o"
"$CC" -std=c11 -O3 -Wall -Wextra -I"$SRC" -c "$SRC/ec_flow.c"        -o "$OUT/obj/ec_flow.o"
"$CC" -std=c11 -O3 -Wall -Wextra -I"$SRC" -c "$SRC/ec_media.c"       -o "$OUT/obj/ec_media.o"
"$CC" -std=c11 -O3 -Wall -Wextra -I"$SRC" -c "$SRC/ec_nvenc.c"       -o "$OUT/obj/ec_nvenc.o"
"$CC" -std=c11 -O3 -Wall -Wextra -I"$SRC" -c "$SRC/ec_storage.c"     -o "$OUT/obj/ec_storage.o"
"$CC" -std=c11 -O3 -Wall -Wextra -I"$SRC" -c "$SRC/ec_bpe.c"         -o "$OUT/obj/ec_bpe.o"
"$CC" -std=c11 -O3 -Wall -Wextra -I"$SRC" -c "$SRC/ec_unigram.c"     -o "$OUT/obj/ec_unigram.o"
"$CC" -std=c11 -O3 -Wall -Wextra -I"$SRC" -c "$SRC/ec_recipes.c"     -o "$OUT/obj/ec_recipes.o"
"$CC" -std=c11 -O3 -Wall -Wextra -I"$SRC" -c "$SRC/ec_serve.c"       -o "$OUT/obj/ec_serve.o"

"$NVCC" -O3 -std=c++17 "${GENC[@]}" -I"$SRC" -c "$SRC/ec_kernels.cu"    -o "$OUT/obj/ec_kernels.o"
"$NVCC" -O3 -std=c++17 "${GENC[@]}" -I"$SRC" -c "$SRC/ec_kernels_v2.cu" -o "$OUT/obj/ec_kernels_v2.o"
"$NVCC" -O3 -std=c++17 "${GENC[@]}" -I"$SRC" -c "$SRC/ec_ops.cu"        -o "$OUT/obj/ec_ops.o"
"$NVCC" -O3 -std=c++17 "${GENC[@]}" -I"$SRC" -c "$SRC/ec_dit.cu"        -o "$OUT/obj/ec_dit.o"
"$NVCC" -O3 -std=c++17 "${GENC[@]}" -I"$SRC" -c "$SRC/ec_vram_pool.cu"  -o "$OUT/obj/ec_vram_pool.o"
"$NVCC" -O3 -std=c++17 "${GENC[@]}" -I"$SRC" -c "$SRC/ec_models.cu"     -o "$OUT/obj/ec_models.o"
"$NVCC" -O3 -std=c++17 "${GENC[@]}" -I"$SRC" -c "$SRC/ec_lora.cu"       -o "$OUT/obj/ec_lora.o"
"$NVCC" -O3 -std=c++17 "${GENC[@]}" -I"$SRC" -c "$SRC/ec_flux2.cu"      -o "$OUT/obj/ec_flux2.o"
"$NVCC" -O3 -std=c++17 "${GENC[@]}" -I"$SRC" -c "$SRC/ec_zimage.cu"     -o "$OUT/obj/ec_zimage.o"
"$NVCC" -O3 -std=c++17 "${GENC[@]}" -I"$SRC" -c "$SRC/ec_acestep.cu"    -o "$OUT/obj/ec_acestep.o"
"$NVCC" -O3 -std=c++17 "${GENC[@]}" -I"$SRC" -c "$SRC/ec_trellis2.cu"   -o "$OUT/obj/ec_trellis2.o"
"$NVCC" -O3 -std=c++17 "${GENC[@]}" -I"$SRC" -c "$SRC/ec_idol.cu"       -o "$OUT/obj/ec_idol.o"
"$NVCC" -O3 -std=c++17 "${GENC[@]}" -I"$SRC" -c "$SRC/ec_wan22.cu"      -o "$OUT/obj/ec_wan22.o"
"$NVCC" -O3 -std=c++17 "${GENC[@]}" -I"$SRC" -c "$SRC/ec_ltx2.cu"       -o "$OUT/obj/ec_ltx2.o"
"$NVCC" -O3 -std=c++17 "${GENC[@]}" -I"$SRC" -c "$SRC/ec_studio.cu"     -o "$OUT/obj/ec_studio.o"
"$NVCC" -O3 -std=c++17 "${GENC[@]}" -I"$SRC" -c "$SRC/ec_bench.cu"      -o "$OUT/obj/ec_bench.o"

COMMON_OBJS=(
  "$OUT/obj/ec_safetensors.o"
  "$OUT/obj/ec_cowrie.o"
  "$OUT/obj/ec_flow.o"
  "$OUT/obj/ec_media.o"
  "$OUT/obj/ec_nvenc.o"
  "$OUT/obj/ec_storage.o"
  "$OUT/obj/ec_bpe.o"
  "$OUT/obj/ec_unigram.o"
  "$OUT/obj/ec_recipes.o"
  "$OUT/obj/ec_kernels.o"
  "$OUT/obj/ec_kernels_v2.o"
  "$OUT/obj/ec_ops.o"
  "$OUT/obj/ec_dit.o"
  "$OUT/obj/ec_vram_pool.o"
  "$OUT/obj/ec_models.o"
  "$OUT/obj/ec_lora.o"
  "$OUT/obj/ec_flux2.o"
  "$OUT/obj/ec_zimage.o"
  "$OUT/obj/ec_acestep.o"
  "$OUT/obj/ec_trellis2.o"
  "$OUT/obj/ec_idol.o"
  "$OUT/obj/ec_wan22.o"
  "$OUT/obj/ec_ltx2.o"
)

"$NVCC" -O3 "${GENC[@]}" -cudart static \
  "${COMMON_OBJS[@]}" "$OUT/obj/ec_bench.o" \
  -lm -lrt -lpthread -ldl -o "$OUT/ec-bench"

"$NVCC" -O3 "${GENC[@]}" -cudart static \
  "${COMMON_OBJS[@]}" "$OUT/obj/ec_studio.o" "$OUT/obj/ec_serve.o" \
  -lm -lrt -lpthread -ldl -o "$OUT/eureka-creative-serve"

echo "=== Built $OUT/ec-bench ($(stat -c%s "$OUT/ec-bench") bytes) ==="
echo "=== Built $OUT/eureka-creative-serve ($(stat -c%s "$OUT/eureka-creative-serve") bytes) ==="
