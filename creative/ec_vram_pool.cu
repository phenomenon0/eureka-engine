// ec_vram_pool.cu — 100% Native Eureka VRAM Residency & Async Pinned-RAM DMA Swapper

#include "ec_vram_pool.h"
#include <stdio.h>
#include <string.h>

int ec_vram_pool_init(ec_vram_pool_t *pool, size_t budget_bytes) {
  memset(pool, 0, sizeof *pool);
  pool->budget_bytes = budget_bytes;
  if (cudaStreamCreateWithFlags(&pool->dma_stream, cudaStreamNonBlocking) != cudaSuccess) return -1;
  return 0;
}

void ec_vram_pool_free(ec_vram_pool_t *pool) {
  if (!pool) return;
  if (pool->dma_stream) cudaStreamSynchronize(pool->dma_stream);
  for (int i = 0; i < pool->n_slots; i++) {
    ec_vram_slot_t *s = &pool->slots[i];
    if (s->dev_ptr)     cudaFree(s->dev_ptr);
    if (s->host_pinned) cudaFreeHost(s->host_pinned);
    if (s->dma_done)    cudaEventDestroy(s->dma_done);
  }
  if (pool->dma_stream) cudaStreamDestroy(pool->dma_stream);
  memset(pool, 0, sizeof *pool);
}

int ec_vram_register(ec_vram_pool_t *pool, const char *name, size_t nbytes, const void *init_data, int pin_vram) {
  if (!pool || pool->n_slots >= EC_VRAM_MAX_SLOTS || nbytes == 0) return -1;
  int id = pool->n_slots;
  ec_vram_slot_t *s = &pool->slots[id];
  memset(s, 0, sizeof *s);
  snprintf(s->name, sizeof s->name, "%s", name ? name : "unnamed");
  s->nbytes = nbytes;
  s->pinned_in_vram = pin_vram;
  if (cudaHostAlloc(&s->host_pinned, nbytes, cudaHostAllocDefault) != cudaSuccess) return -1;
  if (cudaEventCreateWithFlags(&s->dma_done, cudaEventDisableTiming) != cudaSuccess) {
    cudaFreeHost(s->host_pinned); s->host_pinned = nullptr; return -1;
  }
  if (init_data) memcpy(s->host_pinned, init_data, nbytes);
  else memset(s->host_pinned, 0, nbytes);
  s->state = EC_SLOT_IN_RAM;
  pool->n_slots++;
  return id;
}

int ec_vram_evict(ec_vram_pool_t *pool, int slot_id) {
  if (!pool || slot_id < 0 || slot_id >= pool->n_slots) return -1;
  ec_vram_slot_t *s = &pool->slots[slot_id];
  if (s->state == EC_SLOT_IN_RAM || !s->dev_ptr) return 0;
  // Ensure any in-flight DMA on this slot completed before freeing VRAM
  cudaEventSynchronize(s->dma_done);
  cudaFree(s->dev_ptr);
  s->dev_ptr = nullptr;
  pool->used_bytes = (pool->used_bytes >= s->nbytes) ? (pool->used_bytes - s->nbytes) : 0;
  s->state = EC_SLOT_IN_RAM;
  pool->total_evictions++;
  return 0;
}

static int evict_lru_for(ec_vram_pool_t *pool, size_t need, int protect_id) {
  while (pool->used_bytes + need > pool->budget_bytes) {
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < pool->n_slots; i++) {
      ec_vram_slot_t *s = &pool->slots[i];
      if (i == protect_id || s->pinned_in_vram || !s->dev_ptr) continue;
      if (s->last_used_tick < oldest) {
        oldest = s->last_used_tick;
        victim = i;
      }
    }
    if (victim < 0) return -1; // cannot free enough VRAM
    if (ec_vram_evict(pool, victim) != 0) return -1;
  }
  return 0;
}

int ec_vram_prefetch_async(ec_vram_pool_t *pool, int slot_id) {
  if (!pool || slot_id < 0 || slot_id >= pool->n_slots) return -1;
  ec_vram_slot_t *s = &pool->slots[slot_id];
  s->last_used_tick = ++pool->tick;
  if (s->state == EC_SLOT_RESIDENT || s->state == EC_SLOT_PREFETCH) return 0;

  if (evict_lru_for(pool, s->nbytes, slot_id) != 0) return -1;
  if (cudaMalloc(&s->dev_ptr, s->nbytes) != cudaSuccess) return -1;
  pool->used_bytes += s->nbytes;

  if (cudaMemcpyAsync(s->dev_ptr, s->host_pinned, s->nbytes, cudaMemcpyHostToDevice, pool->dma_stream) != cudaSuccess) {
    cudaFree(s->dev_ptr); s->dev_ptr = nullptr; pool->used_bytes -= s->nbytes; return -1;
  }
  cudaEventRecord(s->dma_done, pool->dma_stream);
  s->state = EC_SLOT_PREFETCH;
  pool->total_prefetches++;
  pool->bytes_h2d += s->nbytes;
  return 0;
}

void *ec_vram_acquire(ec_vram_pool_t *pool, int slot_id, cudaStream_t compute_stream) {
  if (!pool || slot_id < 0 || slot_id >= pool->n_slots) return nullptr;
  ec_vram_slot_t *s = &pool->slots[slot_id];
  if (s->state == EC_SLOT_IN_RAM) {
    if (ec_vram_prefetch_async(pool, slot_id) != 0) return nullptr;
  }
  if (s->state == EC_SLOT_PREFETCH) {
    // Non-blocking GPU-side stream fence (compute_stream waits on dma_done without CPU stall!)
    cudaStreamWaitEvent(compute_stream, s->dma_done, 0);
    s->state = EC_SLOT_RESIDENT;
  }
  s->last_used_tick = ++pool->tick;
  return s->dev_ptr;
}
