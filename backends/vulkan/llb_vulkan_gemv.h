// backends/vulkan/llb_vulkan_gemv.h — Native Cross-Vendor Vulkan 1.2+ LLM GEMV Backend
// Accelerates tools/qwen35_engine.c LLB_GEMV_ROWS on AMD Radeon, Intel Arc, and NVIDIA GPUs
// via DEVICE_LOCAL Vulkan VRAM weight slabs + Naga-compiled SPIR-V 1.3 shaders.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint64_t off;
  uint64_t bytes;
} llb_vk_tensor_span_t;

// Initialize Vulkan device and upload model weight tensors (from gPack) into <=1 GiB DEVICE_LOCAL slabs.
// Returns 0 if Vulkan GPU acceleration is active, -1 on fallback to CPU workers (never fatal).
int llb_vk_init(const uint8_t *pack_base, size_t pack_bytes,
                const llb_vk_tensor_span_t *spans, int n_spans,
                int max_rows, int max_cols);

// Execute y[0..rows-1] = W[rows, cols] @ x[0..cols-1] on Vulkan GPU if W is resident in VRAM
// and fmt is supported (F32, Q8_0, Q4_K, Q5_K, Q6_K). Returns 1 if handled on GPU, 0 to fall back to CPU.
int llb_vk_gemv(int fmt, const uint8_t *W, int rb, int rows, int cols, const float *x, float *y);

// Human-readable device summary string (empty if not active).
const char *llb_vk_device_info(void);

// Query Vulkan device VRAM (free, total) in bytes. Returns 0 on success, -1 if unavailable.
int llb_vk_mem_info(size_t *free_b, size_t *total_b);

#ifdef __cplusplus
}
#endif
