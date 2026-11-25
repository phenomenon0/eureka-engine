// ec_vram_pool.h — 100% Native Eureka Global VRAM Residency & Async Pinned-RAM Swapper
// Manages multi-modality creative weights (LLM Trunk, DiT Image/Video/3D/Music, VAEs, LoRAs)
// within a hard VRAM budget using pinned host backing stores + dedicated non-blocking DMA stream.

#pragma once

#include "ec_hal.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EC_VRAM_MAX_SLOTS 32

typedef enum {
  EC_SLOT_UNLOADED = 0,
  EC_SLOT_IN_RAM   = 1,  // pinned host RAM, ready for async H2D prefetch
  EC_SLOT_PREFETCH = 2,  // async H2D DMA in flight
  EC_SLOT_RESIDENT = 3,  // live in GPU VRAM
} ec_slot_state_t;

typedef struct {
  char            name[48];
  size_t          nbytes;
  void           *host_pinned;  // cudaHostAlloc backing buffer
  void           *dev_ptr;      // cudaMalloc VRAM buffer (NULL when evicted)
  ec_slot_state_t state;
  uint64_t        last_used_tick;
  int             pinned_in_vram; // 1 = never auto-evict (e.g., active trunk)
  cudaEvent_t     dma_done;
} ec_vram_slot_t;

typedef struct {
  size_t         budget_bytes;
  size_t         used_bytes;
  uint64_t       tick;
  cudaStream_t   dma_stream;
  ec_vram_slot_t slots[EC_VRAM_MAX_SLOTS];
  int            n_slots;
  // Telemetry
  uint64_t       total_evictions;
  uint64_t       total_prefetches;
  uint64_t       bytes_h2d;
  uint64_t       bytes_d2h;
} ec_vram_pool_t;

int   ec_vram_pool_init(ec_vram_pool_t *pool, size_t budget_bytes);
void  ec_vram_pool_free(ec_vram_pool_t *pool);

// Register a model weight slab in pinned host RAM (copies initial data if src != NULL)
int   ec_vram_register(ec_vram_pool_t *pool, const char *name, size_t nbytes, const void *init_data, int pin_vram);

// Kick off asynchronous H2D prefetch onto dma_stream (auto-evicts LRU non-pinned slots if over budget)
int   ec_vram_prefetch_async(ec_vram_pool_t *pool, int slot_id);

// Wait for DMA completion on compute_stream (zero host stall via cudaStreamWaitEvent!) and return dev_ptr
void *ec_vram_acquire(ec_vram_pool_t *pool, int slot_id, cudaStream_t compute_stream);

// Explicitly evict a slot from VRAM back to pinned host RAM
int   ec_vram_evict(ec_vram_pool_t *pool, int slot_id);

#ifdef __cplusplus
}
#endif
