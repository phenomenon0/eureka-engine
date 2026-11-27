// ec_backend_vulkan.c — High-Throughput Cross-Vendor Linux & Windows Vulkan 1.2/1.3+ Turbo Compute Backend
// Runs natively on AMD Radeon (RADV/Windows), Intel Arc (ANV/Windows), NVIDIA Vulkan, Qualcomm, and Lavapipe.
// Architecture v3 (Turbo Path T3 + T4 + T5):
//   - T3: Full GPU-resident glue shaders (RMSNorm/LayerNorm+AdaLN, GatedRes, SwiGLU, SiLU, FlowEuler,
//         FLUX.2 PackQKV+RoPE & SingleCatQ8, NHWC GroupNorm, Upsample2x, VAE MidAttn D=512, Qwen3 RoPE+GQA)
//         -> zero mid-step PCIe host readbacks across FLUX.2 Text Encoder, DiT, and 2D VAE!
//   - T4: Runtime-probed VK_KHR_cooperative_matrix (16x16x16 BF16/FP16->F32 & 16x16x32 INT8->I32)
//         for GEMM (BF16, FP16, INT8, Q8_G64), Implicit-GEMM 3x3 Conv2D, and Block-Tiled FlashAttention-2
//         (Br=32xBc=32), with automatic Tier-0 SPIR-V 1.3 fallback on GPUs without coopmat!
//   - T5: Disk-backed VkPipelineCache, 512-slot batched command buffer, and zero-copy pinned-VkBuffer DMA.

#define _POSIX_C_SOURCE 200809L
#ifndef EC_BACKEND_VULKAN
  #define EC_BACKEND_VULKAN 1
#endif

#include "../ec_hal.h"
#include "ec_ops_common.h"
#include "ec_vulkan_spv.h"
#include "ec_vulkan_glue_spv.h"
#include "ec_vulkan_coop_spv.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vulkan/vulkan.h>

#ifdef _WIN32
#include <direct.h>
#define ec_vk_mkdir(p) _mkdir(p)
#else
#define ec_vk_mkdir(p) mkdir((p), 0755)
#endif

typedef struct {
  uint32_t p0, p1, p2, p3;
  uint32_t p4, p5, p6, p7;
  uint32_t o0, o1, o2, o3;
  uint32_t o4, o5, pad0, pad1;
  float    f0, f1, f2, f3;
} ec_vk_params_t;

typedef enum {
  // Tier 0 portable SPIR-V 1.3 heavy compute baselines
  VK_PIPE_GEMM_BF16 = 0,
  VK_PIPE_GEMM_F16,
  VK_PIPE_QUANT_Q8,
  VK_PIPE_GEMM_Q8,
  VK_PIPE_FA2_BF16,
  VK_PIPE_FA2_F16,
  VK_PIPE_CONV3X3,
  VK_PIPE_PACK16X2,
  VK_PIPE_GEMM_I8,
  // T3 portable SPIR-V 1.3 GPU-resident glue shaders
  VK_PIPE_NORM_ADALN,
  VK_PIPE_GATED_RES,
  VK_PIPE_ELEMWISE,
  VK_PIPE_FLUX2_PACK_QKV_ROPE,
  VK_PIPE_FLUX2_SINGLE_CAT_Q8,
  VK_PIPE_NHWC_GN_STATS,
  VK_PIPE_NHWC_GN_APPLY,
  VK_PIPE_UPSAMPLE2X_NHWC,
  VK_PIPE_VAE_MID_ATTN,
  VK_PIPE_QWEN3_QK_ROPE,
  VK_PIPE_QWEN3_GQA_ATTN,
  VK_PIPE_GROUPNORM_SILU,
  // T4 runtime-probed VK_KHR_cooperative_matrix (SPIR-V 1.6)
  VK_PIPE_COOP_GEMM_BF16,
  VK_PIPE_COOP_GEMM_F16,
  VK_PIPE_COOP_GEMM_I8,
  VK_PIPE_COOP_GEMM_Q8,
  VK_PIPE_COOP_CONV3X3,
  VK_PIPE_COOP_FA2_F16,
  VK_PIPE_COOP_FA2_BF16,
  VK_PIPE_COUNT
} ec_vk_pipe_id_t;

typedef struct {
  uintptr_t      base;
  size_t         size;
  VkBuffer       host_buf;
  VkDeviceMemory host_mem;
  void          *mapped;
  VkBuffer       dev_buf;
  VkDeviceMemory dev_mem;
  int            has_dev_mirror;
  int            is_pinned_only; // 1 for ec_hal_host_alloc_raw zero-copy DMA staging buffers
  int            host_dirty;
  size_t         dirty_lo;
  size_t         dirty_hi;
  int            dev_dirty;
  int            in_flight_h2d;
} ec_vk_alloc_t;

#define EC_VK_MAX_ALLOCS    8192
#define EC_VK_FREE_CACHE    16
#define EC_VK_FREE_CAP_B    (256ULL << 20)
#define EC_VK_BATCH_SLOTS   512
#define EC_VK_UBO_STRIDE    256
#define EC_VK_MAX_BATCH_WGS 262144u

static struct {
  pthread_mutex_t                  mu;
  int                              init_done;
  int                              init_ok;
  VkInstance                       instance;
  VkPhysicalDevice                 phys;
  VkDevice                         dev;
  VkQueue                          queue;
  uint32_t                         qfam;
  VkPhysicalDeviceProperties       props;
  VkPhysicalDeviceMemoryProperties mem_props;
  size_t                           vram_total;
  size_t                           vram_used;
  int                              is_unified_mem;
  int                              use_dev_mirror;

  // T4 capability flags
  int                              has_vk13;
  int                              has_sub_ctrl;
  int                              has_coop_f16;
  int                              has_coop_bf16;
  int                              has_coop_i8;
  uint32_t                         sub_size;

  // T5 pipeline cache
  VkPipelineCache                  pipe_cache;
  char                             cache_path[512];
  int                              cache_dirty;

  VkCommandPool                    cmd_pool;
  VkCommandBuffer                  cmd;
  VkFence                          fence;
  VkDescriptorSetLayout            dsl;
  VkPipelineLayout                 pipe_layout;
  VkDescriptorPool                 desc_pool;
  VkDescriptorSet                  desc_sets[EC_VK_BATCH_SLOTS];
  VkShaderModule                   shaders[VK_PIPE_COUNT];
  VkPipeline                       pipes[VK_PIPE_COUNT];

  VkBuffer                         dummy_buf;
  VkDeviceMemory                   dummy_mem;
  VkBuffer                         ubo_buf;
  VkDeviceMemory                   ubo_mem;
  uint8_t                         *ubo_mapped;

  int                              batch_open;
  int                              batch_ops;
  uint32_t                         batch_wgs;

  float                           *tmp_y_f32;
  size_t                           tmp_y_cap;

  ec_vk_alloc_t                    allocs[EC_VK_MAX_ALLOCS];
  int                              n_allocs;
  ec_vk_alloc_t                    free_cache[EC_VK_FREE_CACHE];
  int                              n_free_cache;
  size_t                           free_cache_bytes;
} g_vk = { .mu = PTHREAD_MUTEX_INITIALIZER };

const char *ec_hal_backend_name(void) { return "vulkan"; }

static int find_mem_type(uint32_t type_bits, VkMemoryPropertyFlags req_flags, VkMemoryPropertyFlags avoid_flags) {
  for (uint32_t i = 0; i < g_vk.mem_props.memoryTypeCount; i++) {
    if (!(type_bits & (1u << i))) continue;
    VkMemoryPropertyFlags f = g_vk.mem_props.memoryTypes[i].propertyFlags;
    if ((f & req_flags) == req_flags && (f & avoid_flags) == 0) return (int)i;
  }
  return -1;
}

static int create_host_vk_buf(size_t nbytes, VkBufferUsageFlags usage, int prefer_unified,
                              VkBuffer *out_buf, VkDeviceMemory *out_mem, void **out_map, int *out_is_unified) {
  size_t aligned = (nbytes + 255ULL) & ~255ULL;
  if (aligned < 256) aligned = 256;
  VkBufferCreateInfo bci = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = aligned, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  if (vkCreateBuffer(g_vk.dev, &bci, NULL, out_buf) != VK_SUCCESS) return -1;
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(g_vk.dev, *out_buf, &req);

  int mt = -1, is_uni = 0;
  if (prefer_unified) {
    mt = find_mem_type(req.memoryTypeBits,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
    if (mt >= 0) is_uni = 1;
  }
  if (mt < 0) {
    mt = find_mem_type(req.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                       VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
  }
  if (mt < 0) {
    mt = find_mem_type(req.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
  }
  if (mt < 0) { vkDestroyBuffer(g_vk.dev, *out_buf, NULL); return -1; }

  VkMemoryAllocateInfo mai = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize = req.size, .memoryTypeIndex = (uint32_t)mt,
  };
  if (vkAllocateMemory(g_vk.dev, &mai, NULL, out_mem) != VK_SUCCESS) {
    if (is_uni) {
      is_uni = 0;
      int mt2 = find_mem_type(req.memoryTypeBits,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      if (mt2 < 0) mt2 = find_mem_type(req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
      if (mt2 < 0 || (mai.memoryTypeIndex = (uint32_t)mt2, vkAllocateMemory(g_vk.dev, &mai, NULL, out_mem) != VK_SUCCESS)) {
        vkDestroyBuffer(g_vk.dev, *out_buf, NULL);
        return -1;
      }
    } else {
      vkDestroyBuffer(g_vk.dev, *out_buf, NULL);
      return -1;
    }
  }
  if (vkBindBufferMemory(g_vk.dev, *out_buf, *out_mem, 0) != VK_SUCCESS ||
      vkMapMemory(g_vk.dev, *out_mem, 0, aligned, 0, out_map) != VK_SUCCESS) {
    vkFreeMemory(g_vk.dev, *out_mem, NULL);
    vkDestroyBuffer(g_vk.dev, *out_buf, NULL);
    return -1;
  }
  if (out_is_unified) *out_is_unified = is_uni;
  return 0;
}

static int create_dev_local_vk_buf(size_t nbytes, VkBufferUsageFlags usage,
                                   VkBuffer *out_buf, VkDeviceMemory *out_mem) {
  size_t aligned = (nbytes + 255ULL) & ~255ULL;
  if (aligned < 256) aligned = 256;
  VkBufferCreateInfo bci = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = aligned, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  if (vkCreateBuffer(g_vk.dev, &bci, NULL, out_buf) != VK_SUCCESS) return -1;
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(g_vk.dev, *out_buf, &req);

  int mt = find_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  if (mt < 0) mt = find_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
  if (mt < 0) { vkDestroyBuffer(g_vk.dev, *out_buf, NULL); return -1; }

  VkMemoryAllocateInfo mai = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize = req.size, .memoryTypeIndex = (uint32_t)mt,
  };
  if (vkAllocateMemory(g_vk.dev, &mai, NULL, out_mem) != VK_SUCCESS) {
    vkDestroyBuffer(g_vk.dev, *out_buf, NULL);
    return -1;
  }
  if (vkBindBufferMemory(g_vk.dev, *out_buf, *out_mem, 0) != VK_SUCCESS) {
    vkFreeMemory(g_vk.dev, *out_mem, NULL);
    vkDestroyBuffer(g_vk.dev, *out_buf, NULL);
    return -1;
  }
  return 0;
}

static int str_icontains(const char *hay, const char *needle) {
  if (!hay || !needle || !needle[0]) return 0;
  size_t nl = strlen(needle);
  for (const char *p = hay; *p; p++) {
    size_t i = 0;
    while (i < nl && p[i]) {
      char a = p[i] >= 'A' && p[i] <= 'Z' ? (char)(p[i] + 32) : p[i];
      char b = needle[i] >= 'A' && needle[i] <= 'Z' ? (char)(needle[i] + 32) : needle[i];
      if (a != b) break;
      i++;
    }
    if (i == nl) return 1;
  }
  return 0;
}

// ---- T5: Disk-backed VkPipelineCache ----
static void ec_vk_init_pipe_cache_locked(void) {
  char dir[400] = {0};
#ifdef _WIN32
  const char *base = getenv("LOCALAPPDATA");
  if (!base || !base[0]) base = ".";
  snprintf(dir, sizeof dir, "%s\\Eureka", base);
  ec_vk_mkdir(dir);
  snprintf(dir, sizeof dir, "%s\\Eureka\\cache", base);
  ec_vk_mkdir(dir);
  snprintf(g_vk.cache_path, sizeof g_vk.cache_path, "%s\\vk_ec_%04x_%04x_%08x.bin",
           dir, g_vk.props.vendorID, g_vk.props.deviceID, g_vk.props.driverVersion);
#else
  const char *xdg = getenv("XDG_CACHE_HOME");
  const char *home = getenv("HOME");
  if (xdg && xdg[0]) {
    snprintf(dir, sizeof dir, "%s/eureka", xdg);
    ec_vk_mkdir(xdg);
    ec_vk_mkdir(dir);
  } else if (home && home[0]) {
    snprintf(dir, sizeof dir, "%s/.cache", home);
    ec_vk_mkdir(dir);
    snprintf(dir, sizeof dir, "%s/.cache/eureka", home);
    ec_vk_mkdir(dir);
  } else {
    snprintf(dir, sizeof dir, "/tmp/eureka-vk-cache");
    ec_vk_mkdir(dir);
  }
  snprintf(g_vk.cache_path, sizeof g_vk.cache_path, "%s/vk_ec_%04x_%04x_%08x.bin",
           dir, g_vk.props.vendorID, g_vk.props.deviceID, g_vk.props.driverVersion);
#endif

  void *blob = NULL;
  size_t blob_sz = 0;
  FILE *f = fopen(g_vk.cache_path, "rb");
  if (f) {
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz > 32 && sz < (64L << 20)) {
      blob = malloc((size_t)sz);
      if (blob && fread(blob, 1, (size_t)sz, f) == (size_t)sz) {
        const uint32_t *hdr = (const uint32_t *)blob;
        if (hdr[2] == g_vk.props.vendorID && hdr[3] == g_vk.props.deviceID &&
            memcmp((const uint8_t *)blob + 16, g_vk.props.pipelineCacheUUID, VK_UUID_SIZE) == 0) {
          blob_sz = (size_t)sz;
        }
      }
    }
    fclose(f);
  }

  VkPipelineCacheCreateInfo pcci = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
    .initialDataSize = blob_sz,
    .pInitialData = blob_sz ? blob : NULL,
  };
  if (vkCreatePipelineCache(g_vk.dev, &pcci, NULL, &g_vk.pipe_cache) != VK_SUCCESS) {
    g_vk.pipe_cache = VK_NULL_HANDLE;
  }
  free(blob);
}

static void ec_vk_save_pipe_cache_locked(void) {
  if (!g_vk.pipe_cache || !g_vk.cache_path[0] || !g_vk.cache_dirty) return;
  size_t sz = 0;
  if (vkGetPipelineCacheData(g_vk.dev, g_vk.pipe_cache, &sz, NULL) != VK_SUCCESS || sz == 0 || sz > (64ULL << 20))
    return;
  void *buf = malloc(sz);
  if (!buf) return;
  if (vkGetPipelineCacheData(g_vk.dev, g_vk.pipe_cache, &sz, buf) == VK_SUCCESS) {
    char tmp[540];
    snprintf(tmp, sizeof tmp, "%s.tmp", g_vk.cache_path);
    FILE *f = fopen(tmp, "wb");
    if (f) {
      if (fwrite(buf, 1, sz, f) == sz) {
        fclose(f);
        rename(tmp, g_vk.cache_path);
        g_vk.cache_dirty = 0;
      } else {
        fclose(f);
        remove(tmp);
      }
    }
  }
  free(buf);
}

// ===========================================================================
// Batched Command Buffer & Lazy-Coherent VRAM Mirror Engine
// ===========================================================================
static void vk_batch_begin_locked(void) {
  if (g_vk.batch_open) return;
  vkResetCommandBuffer(g_vk.cmd, 0);
  VkCommandBufferBeginInfo bi = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  vkBeginCommandBuffer(g_vk.cmd, &bi);
  g_vk.batch_open = 1;
  g_vk.batch_ops = 0;
  g_vk.batch_wgs = 0;
}

static void vk_cmd_barrier_locked(VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage,
                                  VkAccessFlags src_acc, VkAccessFlags dst_acc) {
  VkMemoryBarrier mb = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .srcAccessMask = src_acc, .dstAccessMask = dst_acc,
  };
  vkCmdPipelineBarrier(g_vk.cmd, src_stage, dst_stage, 0, 1, &mb, 0, NULL, 0, NULL);
}

static int vk_batch_flush_locked(int pull_dev_dirty) {
  int need_pull = 0;
  if (pull_dev_dirty && g_vk.use_dev_mirror) {
    for (int i = 0; i < g_vk.n_allocs; i++) {
      if (g_vk.allocs[i].has_dev_mirror && g_vk.allocs[i].dev_dirty) { need_pull = 1; break; }
    }
  }
  if (!g_vk.batch_open && !need_pull) return 0;

  vk_batch_begin_locked();
  if (need_pull) {
    vk_cmd_barrier_locked(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_TRANSFER_READ_BIT);
    for (int i = 0; i < g_vk.n_allocs; i++) {
      ec_vk_alloc_t *a = &g_vk.allocs[i];
      if (a->has_dev_mirror && a->dev_dirty) {
        VkBufferCopy cp = { .srcOffset = 0, .dstOffset = 0, .size = a->size };
        vkCmdCopyBuffer(g_vk.cmd, a->dev_buf, a->host_buf, 1, &cp);
        a->dev_dirty = 0;
      }
    }
    vk_cmd_barrier_locked(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                          VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
  } else {
    vk_cmd_barrier_locked(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  }
  vkEndCommandBuffer(g_vk.cmd);

  vkResetFences(g_vk.dev, 1, &g_vk.fence);
  VkSubmitInfo si = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &g_vk.cmd,
  };
  VkResult r1 = vkQueueSubmit(g_vk.queue, 1, &si, g_vk.fence);
  VkResult r2 = (r1 == VK_SUCCESS) ? vkWaitForFences(g_vk.dev, 1, &g_vk.fence, VK_TRUE, UINT64_MAX) : r1;

  g_vk.batch_open = 0;
  g_vk.batch_ops = 0;
  g_vk.batch_wgs = 0;
  for (int i = 0; i < g_vk.n_allocs; i++) g_vk.allocs[i].in_flight_h2d = 0;
  if (g_vk.cache_dirty) ec_vk_save_pipe_cache_locked();
  return (r1 == VK_SUCCESS && r2 == VK_SUCCESS) ? 0 : -1;
}

static ec_vk_alloc_t *find_alloc_locked(const void *ptr) {
  if (!ptr || g_vk.n_allocs == 0) return NULL;
  uintptr_t addr = (uintptr_t)ptr;
  int lo = 0, hi = g_vk.n_allocs - 1;
  while (lo <= hi) {
    int mid = (lo + hi) >> 1;
    ec_vk_alloc_t *a = &g_vk.allocs[mid];
    if (addr < a->base) hi = mid - 1;
    else if (addr >= a->base + a->size) lo = mid + 1;
    else return a;
  }
  return NULL;
}

static void mark_host_dirty_range_locked(ec_vk_alloc_t *a, size_t lo, size_t hi) {
  if (!a || !a->has_dev_mirror) return;
  if (lo > a->size) lo = a->size;
  if (hi > a->size) hi = a->size;
  if (hi <= lo) return;
  if (!a->host_dirty) {
    a->host_dirty = 1; a->dirty_lo = lo; a->dirty_hi = hi;
  } else {
    if (lo < a->dirty_lo) a->dirty_lo = lo;
    if (hi > a->dirty_hi) a->dirty_hi = hi;
  }
}

static void ensure_dev_up_to_date_locked(ec_vk_alloc_t *a) {
  if (!a || !a->has_dev_mirror || !a->host_dirty) return;
  size_t lo = a->dirty_lo & ~3ULL;
  size_t hi = (a->dirty_hi + 3ULL) & ~3ULL;
  if (hi > a->size) hi = a->size;
  if (hi <= lo) { a->host_dirty = 0; return; }
  vk_batch_begin_locked();
  vk_cmd_barrier_locked(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT);
  VkBufferCopy cp = { .srcOffset = lo, .dstOffset = lo, .size = hi - lo };
  vkCmdCopyBuffer(g_vk.cmd, a->host_buf, a->dev_buf, 1, &cp);
  vk_cmd_barrier_locked(VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
  a->host_dirty = 0; a->dirty_lo = 0; a->dirty_hi = 0; a->in_flight_h2d = 1;
}

static void sync_single_alloc_to_host_locked(ec_vk_alloc_t *a, size_t lo, size_t hi, int clear_dirty) {
  if (!a || !a->has_dev_mirror || !a->dev_dirty) {
    if (g_vk.batch_open) vk_batch_flush_locked(0);
    return;
  }
  lo &= ~3ULL;
  hi = (hi + 3ULL) & ~3ULL;
  if (hi > a->size) hi = a->size;
  if (hi <= lo) return;
  vk_batch_begin_locked();
  vk_cmd_barrier_locked(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_ACCESS_TRANSFER_READ_BIT);
  VkBufferCopy cp = { .srcOffset = lo, .dstOffset = lo, .size = hi - lo };
  vkCmdCopyBuffer(g_vk.cmd, a->dev_buf, a->host_buf, 1, &cp);
  vk_cmd_barrier_locked(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
  if (clear_dirty) a->dev_dirty = 0;
  vk_batch_flush_locked(0);
}

static int ec_dev_has_ext(VkPhysicalDevice phys, const char *name) {
  uint32_t cnt = 0;
  vkEnumerateDeviceExtensionProperties(phys, NULL, &cnt, NULL);
  if (cnt == 0) return 0;
  VkExtensionProperties *exts = (VkExtensionProperties *)malloc((size_t)cnt * sizeof *exts);
  if (!exts) return 0;
  vkEnumerateDeviceExtensionProperties(phys, NULL, &cnt, exts);
  int found = 0;
  for (uint32_t i = 0; i < cnt; i++) {
    if (!strcmp(exts[i].extensionName, name)) { found = 1; break; }
  }
  free(exts);
  return found;
}

int ec_hal_init(void) {
  pthread_mutex_lock(&g_vk.mu);
  if (g_vk.init_done) {
    int ok = g_vk.init_ok;
    pthread_mutex_unlock(&g_vk.mu);
    return ok ? 0 : -1;
  }
  g_vk.init_done = 1;

  uint32_t inst_ver = VK_API_VERSION_1_2;
  PFN_vkEnumerateInstanceVersion fn_ver = (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(NULL, "vkEnumerateInstanceVersion");
  if (fn_ver) {
    uint32_t v = 0;
    if (fn_ver(&v) == VK_SUCCESS && v >= VK_API_VERSION_1_3) inst_ver = VK_API_VERSION_1_3;
  }

  VkApplicationInfo app = {
    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .pApplicationName = "eureka-creative-vulkan",
    .applicationVersion = VK_MAKE_VERSION(3, 0, 0),
    .pEngineName = "eureka-creative-turbo",
    .engineVersion = VK_MAKE_VERSION(3, 0, 0),
    .apiVersion = inst_ver,
  };
  VkInstanceCreateInfo ici = {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
  };
  if (vkCreateInstance(&ici, NULL, &g_vk.instance) != VK_SUCCESS) {
    pthread_mutex_unlock(&g_vk.mu);
    return -1;
  }

  uint32_t n_phys = 0;
  vkEnumeratePhysicalDevices(g_vk.instance, &n_phys, NULL);
  if (n_phys == 0 || n_phys > 32) { pthread_mutex_unlock(&g_vk.mu); return -1; }
  VkPhysicalDevice devs[32];
  vkEnumeratePhysicalDevices(g_vk.instance, &n_phys, devs);

  const char *want = getenv("EC_VULKAN_DEVICE");
  int chosen = -1, best_score = -1;
  for (uint32_t i = 0; i < n_phys; i++) {
    VkPhysicalDeviceProperties pr;
    vkGetPhysicalDeviceProperties(devs[i], &pr);
    if (want && want[0]) {
      char idx_s[16];
      snprintf(idx_s, sizeof idx_s, "%u", i);
      if (!strcmp(want, idx_s) || str_icontains(pr.deviceName, want)) { chosen = (int)i; break; }
    }
    int score = 10;
    if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score = 1000;
    else if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score = 500;
    else if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) score = 200;
    if (score > best_score) { best_score = score; chosen = (int)i; }
  }
  if (chosen < 0) chosen = 0;
  g_vk.phys = devs[chosen];
  vkGetPhysicalDeviceProperties(g_vk.phys, &g_vk.props);
  vkGetPhysicalDeviceMemoryProperties(g_vk.phys, &g_vk.mem_props);

  g_vk.vram_total = 0;
  for (uint32_t i = 0; i < g_vk.mem_props.memoryHeapCount; i++) {
    if (g_vk.mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      if (g_vk.mem_props.memoryHeaps[i].size > g_vk.vram_total)
        g_vk.vram_total = (size_t)g_vk.mem_props.memoryHeaps[i].size;
    }
  }
  if (g_vk.vram_total == 0 && g_vk.mem_props.memoryHeapCount > 0)
    g_vk.vram_total = (size_t)g_vk.mem_props.memoryHeaps[0].size;

  g_vk.is_unified_mem = 0;
  for (uint32_t i = 0; i < g_vk.mem_props.memoryTypeCount; i++) {
    VkMemoryPropertyFlags f = g_vk.mem_props.memoryTypes[i].propertyFlags;
    VkMemoryPropertyFlags uni = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    if ((f & uni) == uni) { g_vk.is_unified_mem = 1; break; }
  }
  const char *no_mir = getenv("EC_VULKAN_NO_MIRROR");
  g_vk.use_dev_mirror = (!g_vk.is_unified_mem && !(no_mir && no_mir[0] == '1')) ? 1 : 0;

  // ---- T4: Probe Vulkan 1.3 + Subgroup + 8/16-bit Storage + VK_KHR_cooperative_matrix + BF16 ----
  g_vk.has_vk13 = (inst_ver >= VK_API_VERSION_1_3 && g_vk.props.apiVersion >= VK_API_VERSION_1_3);
  const char *no_cm = getenv("EC_VULKAN_NO_COOPMAT");
  int allow_cm = !(no_cm && no_cm[0] == '1') && g_vk.has_vk13;

  VkPhysicalDeviceSubgroupSizeControlProperties sub_ctrl_props = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
  };
  VkPhysicalDeviceSubgroupProperties sub_props = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    .pNext = g_vk.has_vk13 ? &sub_ctrl_props : NULL,
  };
  VkPhysicalDeviceProperties2 props2 = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
    .pNext = &sub_props,
  };
  vkGetPhysicalDeviceProperties2(g_vk.phys, &props2);
  g_vk.sub_size = sub_props.subgroupSize;

  VkPhysicalDeviceVulkan11Features f11 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
  VkPhysicalDeviceVulkan12Features f12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &f11 };
  VkPhysicalDeviceVulkan13Features f13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .pNext = &f12 };
  void *feat_chain = g_vk.has_vk13 ? (void *)&f13 : NULL;

  int ext_coop = 0, ext_bf16 = 0;
#ifdef VK_KHR_cooperative_matrix
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR f_cm = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
  };
  if (allow_cm && ec_dev_has_ext(g_vk.phys, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
    ext_coop = 1;
    f_cm.pNext = feat_chain;
    feat_chain = &f_cm;
  }
#endif
#ifdef VK_KHR_shader_bfloat16
  VkPhysicalDeviceShaderBfloat16FeaturesKHR f_bf16 = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
  };
  if (ext_coop && ec_dev_has_ext(g_vk.phys, VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME)) {
    ext_bf16 = 1;
    f_bf16.pNext = feat_chain;
    feat_chain = &f_bf16;
  }
#endif

  if (g_vk.has_vk13) {
    VkPhysicalDeviceFeatures2 f2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = feat_chain,
    };
    vkGetPhysicalDeviceFeatures2(g_vk.phys, &f2);
    g_vk.has_sub_ctrl = (f13.subgroupSizeControl &&
                         (sub_ctrl_props.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
                         sub_ctrl_props.minSubgroupSize <= 32 && sub_ctrl_props.maxSubgroupSize >= 32) ? 1 : 0;
  }

  int can_wave32 = (g_vk.has_sub_ctrl || g_vk.sub_size == 32u);
  int base_coop_ok = (ext_coop && can_wave32 &&
                      f11.storageBuffer16BitAccess && f12.storageBuffer8BitAccess &&
                      f12.shaderFloat16 && f12.shaderInt8 &&
                      g_vk.props.limits.maxComputeSharedMemorySize >= 49152u);

#ifdef VK_KHR_cooperative_matrix
  if (base_coop_ok && f_cm.cooperativeMatrix) {
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR fn_cm =
      (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(g_vk.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (fn_cm) {
      uint32_t cnt = 0;
      fn_cm(g_vk.phys, &cnt, NULL);
      if (cnt > 0 && cnt < 256) {
        VkCooperativeMatrixPropertiesKHR *cmps = (VkCooperativeMatrixPropertiesKHR *)calloc(cnt, sizeof *cmps);
        if (cmps) {
          for (uint32_t k = 0; k < cnt; k++) cmps[k].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
          fn_cm(g_vk.phys, &cnt, cmps);
          for (uint32_t k = 0; k < cnt; k++) {
            if (cmps[k].scope != VK_SCOPE_SUBGROUP_KHR || cmps[k].saturatingAccumulation) continue;
            if (cmps[k].MSize == 16 && cmps[k].NSize == 16 && cmps[k].KSize == 16 &&
                cmps[k].AType == VK_COMPONENT_TYPE_FLOAT16_KHR && cmps[k].BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                cmps[k].CType == VK_COMPONENT_TYPE_FLOAT32_KHR && cmps[k].ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
              g_vk.has_coop_f16 = 1;
            }
            if (cmps[k].MSize == 16 && cmps[k].NSize == 16 && cmps[k].KSize == 32 &&
                cmps[k].AType == VK_COMPONENT_TYPE_SINT8_KHR && cmps[k].BType == VK_COMPONENT_TYPE_SINT8_KHR &&
                cmps[k].CType == VK_COMPONENT_TYPE_SINT32_KHR && cmps[k].ResultType == VK_COMPONENT_TYPE_SINT32_KHR) {
              g_vk.has_coop_i8 = 1;
            }
#ifdef VK_KHR_shader_bfloat16
            if (ext_bf16 && f_bf16.shaderBFloat16Type && f_bf16.shaderBFloat16CooperativeMatrix &&
                cmps[k].MSize == 16 && cmps[k].NSize == 16 && cmps[k].KSize == 16 &&
                cmps[k].AType == VK_COMPONENT_TYPE_BFLOAT16_KHR && cmps[k].BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                cmps[k].CType == VK_COMPONENT_TYPE_FLOAT32_KHR && cmps[k].ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
              g_vk.has_coop_bf16 = 1;
            }
#endif
          }
          free(cmps);
        }
      }
    }
  }
#endif

  // On integrated RDNA2 GPUs (which expose software/low-CU F16 coopmat without BF16/S8x32),
  // only enable coopmat on discrete GPUs or when at least BF16 or S8x32 hardware matrix units exist
  // unless user explicitly overrides.
  if (g_vk.props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && !g_vk.has_coop_bf16 && !g_vk.has_coop_i8) {
    g_vk.has_coop_f16 = 0;
  }

  uint32_t n_qf = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys, &n_qf, NULL);
  VkQueueFamilyProperties qfs[32];
  if (n_qf > 32) n_qf = 32;
  vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys, &n_qf, qfs);
  g_vk.qfam = 0;
  for (uint32_t i = 0; i < n_qf; i++) {
    if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_vk.qfam = i; break; }
  }

  float qprio = 1.0f;
  VkDeviceQueueCreateInfo qci = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .queueFamilyIndex = g_vk.qfam, .queueCount = 1, .pQueuePriorities = &qprio,
  };

  const char *en_exts[4];
  uint32_t n_en_exts = 0;
  VkPhysicalDeviceVulkan11Features en_f11 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
  VkPhysicalDeviceVulkan12Features en_f12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &en_f11 };
  VkPhysicalDeviceVulkan13Features en_f13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .pNext = &en_f12 };
  void *en_pnext = NULL;

  int any_coop = (g_vk.has_coop_f16 || g_vk.has_coop_bf16 || g_vk.has_coop_i8);
  if (g_vk.has_vk13 && any_coop) {
    en_f11.storageBuffer16BitAccess = VK_TRUE;
    en_f12.storageBuffer8BitAccess = VK_TRUE;
    en_f12.shaderFloat16 = VK_TRUE;
    en_f12.shaderInt8 = VK_TRUE;
    en_f13.subgroupSizeControl = g_vk.has_sub_ctrl ? VK_TRUE : VK_FALSE;
    en_pnext = &en_f13;
  }

#ifdef VK_KHR_cooperative_matrix
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR en_cm = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
    .cooperativeMatrix = VK_TRUE,
  };
  if (any_coop) {
    en_exts[n_en_exts++] = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
    en_cm.pNext = en_pnext;
    en_pnext = &en_cm;
  }
#endif
#ifdef VK_KHR_shader_bfloat16
  VkPhysicalDeviceShaderBfloat16FeaturesKHR en_bf16 = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
    .shaderBFloat16Type = VK_TRUE,
    .shaderBFloat16CooperativeMatrix = VK_TRUE,
  };
  if (g_vk.has_coop_bf16) {
    en_exts[n_en_exts++] = VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME;
    en_bf16.pNext = en_pnext;
    en_pnext = &en_bf16;
  }
#endif

  VkDeviceCreateInfo dci = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .pNext = en_pnext,
    .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
    .enabledExtensionCount = n_en_exts, .ppEnabledExtensionNames = n_en_exts ? en_exts : NULL,
  };
  if (vkCreateDevice(g_vk.phys, &dci, NULL, &g_vk.dev) != VK_SUCCESS) {
    // Graceful retry with bare Vulkan 1.2 device (guaranteed portable Tier-0!)
    g_vk.has_coop_f16 = g_vk.has_coop_bf16 = g_vk.has_coop_i8 = 0;
    VkDeviceCreateInfo dci_min = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
    };
    if (vkCreateDevice(g_vk.phys, &dci_min, NULL, &g_vk.dev) != VK_SUCCESS) {
      pthread_mutex_unlock(&g_vk.mu);
      return -1;
    }
  }
  vkGetDeviceQueue(g_vk.dev, g_vk.qfam, 0, &g_vk.queue);

  ec_vk_init_pipe_cache_locked();

  VkCommandPoolCreateInfo cpci = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = g_vk.qfam,
  };
  vkCreateCommandPool(g_vk.dev, &cpci, NULL, &g_vk.cmd_pool);
  VkCommandBufferAllocateInfo cbai = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = g_vk.cmd_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
  };
  vkAllocateCommandBuffers(g_vk.dev, &cbai, &g_vk.cmd);
  VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
  vkCreateFence(g_vk.dev, &fci, NULL, &g_vk.fence);

  void *dummy_map = NULL;
  create_host_vk_buf(256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1, &g_vk.dummy_buf, &g_vk.dummy_mem, &dummy_map, NULL);
  if (dummy_map) memset(dummy_map, 0, 256);
  void *ubo_map = NULL;
  create_host_vk_buf((size_t)EC_VK_BATCH_SLOTS * EC_VK_UBO_STRIDE,
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 0, &g_vk.ubo_buf, &g_vk.ubo_mem, &ubo_map, NULL);
  g_vk.ubo_mapped = (uint8_t *)ubo_map;

  VkDescriptorSetLayoutBinding binds[7];
  memset(binds, 0, sizeof binds);
  binds[0].binding = 0; binds[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; binds[0].descriptorCount = 1; binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  for (int i = 1; i <= 6; i++) {
    binds[i].binding = (uint32_t)i;
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslci = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 7, .pBindings = binds,
  };
  vkCreateDescriptorSetLayout(g_vk.dev, &dslci, NULL, &g_vk.dsl);
  VkPipelineLayoutCreateInfo plci = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &g_vk.dsl,
  };
  vkCreatePipelineLayout(g_vk.dev, &plci, NULL, &g_vk.pipe_layout);

  VkDescriptorPoolSize dps[2] = {
    { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, EC_VK_BATCH_SLOTS },
    { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, EC_VK_BATCH_SLOTS * 6 },
  };
  VkDescriptorPoolCreateInfo dpci = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .maxSets = EC_VK_BATCH_SLOTS, .poolSizeCount = 2, .pPoolSizes = dps,
  };
  vkCreateDescriptorPool(g_vk.dev, &dpci, NULL, &g_vk.desc_pool);
  VkDescriptorSetLayout dsls[EC_VK_BATCH_SLOTS];
  for (int i = 0; i < EC_VK_BATCH_SLOTS; i++) dsls[i] = g_vk.dsl;
  VkDescriptorSetAllocateInfo dsai = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = g_vk.desc_pool, .descriptorSetCount = EC_VK_BATCH_SLOTS, .pSetLayouts = dsls,
  };
  vkAllocateDescriptorSets(g_vk.dev, &dsai, g_vk.desc_sets);

  g_vk.init_ok = 1;
  pthread_mutex_unlock(&g_vk.mu);
  return 0;
}

static int vk_ensure_pipe_locked(ec_vk_pipe_id_t pid) {
  if (g_vk.pipes[pid] != VK_NULL_HANDLE) return 0;
  static const struct { const uint32_t *code; size_t bytes; const char *entry; int pin_sub32; } specs[VK_PIPE_COUNT] = {
    [VK_PIPE_GEMM_BF16]           = { ec_spv_vk_gemm_bf16_f32,        sizeof ec_spv_vk_gemm_bf16_f32,        "vk_gemm_bf16_f32",        0 },
    [VK_PIPE_GEMM_F16]            = { ec_spv_vk_gemm_f16_f32,         sizeof ec_spv_vk_gemm_f16_f32,         "vk_gemm_f16_f32",         0 },
    [VK_PIPE_QUANT_Q8]            = { ec_spv_vk_quantize_bf16_q8g64,  sizeof ec_spv_vk_quantize_bf16_q8g64,  "vk_quantize_bf16_q8g64",  0 },
    [VK_PIPE_GEMM_Q8]             = { ec_spv_vk_gemm_q8g64_f32,       sizeof ec_spv_vk_gemm_q8g64_f32,       "vk_gemm_q8g64_f32",       0 },
    [VK_PIPE_FA2_BF16]            = { ec_spv_vk_fa2_bf16,             sizeof ec_spv_vk_fa2_bf16,             "vk_fa2_bf16",             0 },
    [VK_PIPE_FA2_F16]             = { ec_spv_vk_fa2_f16,              sizeof ec_spv_vk_fa2_f16,              "vk_fa2_f16",              0 },
    [VK_PIPE_CONV3X3]             = { ec_spv_vk_conv2d_3x3_nhwc_bf16, sizeof ec_spv_vk_conv2d_3x3_nhwc_bf16, "vk_conv2d_3x3_nhwc_bf16", 0 },
    [VK_PIPE_PACK16X2]            = { ec_spv_vk_pack_f32_to_16x2,     sizeof ec_spv_vk_pack_f32_to_16x2,     "vk_pack_f32_to_16x2",     0 },
    [VK_PIPE_GEMM_I8]             = { ec_spv_vk_gemm_i8_f32,          sizeof ec_spv_vk_gemm_i8_f32,          "vk_gemm_i8_f32",          0 },
    // T3 GPU glue shaders (GLSL -> SPIR-V 1.3, entry "main")
    [VK_PIPE_NORM_ADALN]          = { ec_spv_norm_adaln,              sizeof ec_spv_norm_adaln,              "main",                    0 },
    [VK_PIPE_GATED_RES]           = { ec_spv_gated_res,               sizeof ec_spv_gated_res,               "main",                    0 },
    [VK_PIPE_ELEMWISE]            = { ec_spv_elemwise,                sizeof ec_spv_elemwise,                "main",                    0 },
    [VK_PIPE_FLUX2_PACK_QKV_ROPE] = { ec_spv_flux2_pack_qkv_rope,     sizeof ec_spv_flux2_pack_qkv_rope,     "main",                    0 },
    [VK_PIPE_FLUX2_SINGLE_CAT_Q8] = { ec_spv_flux2_single_cat_q8g64,  sizeof ec_spv_flux2_single_cat_q8g64,  "main",                    0 },
    [VK_PIPE_NHWC_GN_STATS]       = { ec_spv_nhwc_gn_stats,           sizeof ec_spv_nhwc_gn_stats,           "main",                    0 },
    [VK_PIPE_NHWC_GN_APPLY]       = { ec_spv_nhwc_gn_apply_bf16,      sizeof ec_spv_nhwc_gn_apply_bf16,      "main",                    0 },
    [VK_PIPE_UPSAMPLE2X_NHWC]     = { ec_spv_upsample2x_nhwc_bf16,    sizeof ec_spv_upsample2x_nhwc_bf16,    "main",                    0 },
    [VK_PIPE_VAE_MID_ATTN]        = { ec_spv_vae_mid_attn_d512,       sizeof ec_spv_vae_mid_attn_d512,       "main",                    0 },
    [VK_PIPE_QWEN3_QK_ROPE]       = { ec_spv_qwen3_qk_rope,           sizeof ec_spv_qwen3_qk_rope,           "main",                    0 },
    [VK_PIPE_QWEN3_GQA_ATTN]      = { ec_spv_qwen3_gqa_attn,          sizeof ec_spv_qwen3_gqa_attn,          "main",                    0 },
    [VK_PIPE_GROUPNORM_SILU]      = { ec_spv_groupnorm_silu,          sizeof ec_spv_groupnorm_silu,          "main",                    0 },
    // T4 Cooperative Matrix shaders (GLSL -> SPIR-V 1.6, entry "main", wave32)
    [VK_PIPE_COOP_GEMM_BF16]      = { ec_spv_coop_gemm_bf16,          sizeof ec_spv_coop_gemm_bf16,          "main",                    1 },
    [VK_PIPE_COOP_GEMM_F16]       = { ec_spv_coop_gemm_f16,           sizeof ec_spv_coop_gemm_f16,           "main",                    1 },
    [VK_PIPE_COOP_GEMM_I8]        = { ec_spv_coop_gemm_i8,            sizeof ec_spv_coop_gemm_i8,            "main",                    1 },
    [VK_PIPE_COOP_GEMM_Q8]        = { ec_spv_coop_gemm_q8g64,         sizeof ec_spv_coop_gemm_q8g64,         "main",                    1 },
    [VK_PIPE_COOP_CONV3X3]        = { ec_spv_coop_conv3x3_bf16,       sizeof ec_spv_coop_conv3x3_bf16,       "main",                    1 },
    [VK_PIPE_COOP_FA2_F16]        = { ec_spv_coop_fa2_f16,            sizeof ec_spv_coop_fa2_f16,            "main",                    1 },
    [VK_PIPE_COOP_FA2_BF16]       = { ec_spv_coop_fa2_bf16,           sizeof ec_spv_coop_fa2_bf16,           "main",                    1 },
  };
  VkShaderModuleCreateInfo smci = {
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = specs[pid].bytes, .pCode = specs[pid].code,
  };
  if (vkCreateShaderModule(g_vk.dev, &smci, NULL, &g_vk.shaders[pid]) != VK_SUCCESS) return -1;

  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sub_ci = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
    .requiredSubgroupSize = 32u,
  };
  int pin = specs[pid].pin_sub32 && g_vk.has_sub_ctrl;
  VkComputePipelineCreateInfo cp = {
    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = pin ? &sub_ci : NULL,
      .flags = pin ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : 0,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = g_vk.shaders[pid], .pName = specs[pid].entry,
    },
    .layout = g_vk.pipe_layout,
  };
  if (vkCreateComputePipelines(g_vk.dev, g_vk.pipe_cache, 1, &cp, NULL, &g_vk.pipes[pid]) != VK_SUCCESS) return -1;
  g_vk.cache_dirty = 1;
  return 0;
}

static int lookup_vk_ptr_locked(const void *ptr, size_t elem_bytes,
                                VkDescriptorBufferInfo *out_dbi, uint32_t *out_elem_off, ec_vk_alloc_t **out_alloc) {
  if (out_alloc) *out_alloc = NULL;
  if (!ptr) {
    out_dbi->buffer = g_vk.dummy_buf;
    out_dbi->offset = 0;
    out_dbi->range  = VK_WHOLE_SIZE;
    *out_elem_off   = 0;
    return 0;
  }
  ec_vk_alloc_t *a = find_alloc_locked(ptr);
  if (!a) return -1;
  size_t byte_off = (size_t)((uintptr_t)ptr - a->base);
  size_t align = (size_t)g_vk.props.limits.minStorageBufferOffsetAlignment;
  if (align < 256) align = 256;
  size_t desc_off = byte_off & ~(align - 1ULL);
  size_t rem_bytes = byte_off - desc_off;
  out_dbi->buffer = a->dev_buf;
  out_dbi->offset = (VkDeviceSize)desc_off;
  out_dbi->range  = VK_WHOLE_SIZE;
  *out_elem_off   = (uint32_t)(rem_bytes / elem_bytes);
  if (out_alloc) *out_alloc = a;
  return 0;
}

static int pipe_writes_b6(ec_vk_pipe_id_t pid) {
  return pid == VK_PIPE_QUANT_Q8 || pid == VK_PIPE_PACK16X2 ||
         pid == VK_PIPE_NORM_ADALN || pid == VK_PIPE_ELEMWISE ||
         pid == VK_PIPE_FLUX2_PACK_QKV_ROPE || pid == VK_PIPE_FLUX2_SINGLE_CAT_Q8 ||
         pid == VK_PIPE_NHWC_GN_APPLY || pid == VK_PIPE_UPSAMPLE2X_NHWC ||
         pid == VK_PIPE_VAE_MID_ATTN || pid == VK_PIPE_QWEN3_GQA_ATTN;
}

static int vk_dispatch_locked(
    ec_vk_pipe_id_t pid, const ec_vk_params_t *params,
    const void *b1, size_t eb1, const void *b2, size_t eb2,
    const void *b3, size_t eb3, const void *b4, size_t eb4,
    void *b5, size_t eb5, void *b6, size_t eb6,
    uint32_t gx, uint32_t gy, uint32_t gz) {
  if (gx == 0 || gy == 0 || gz == 0) return 0;
  if (vk_ensure_pipe_locked(pid) != 0) return -1;
  ec_vk_params_t p = *params;
  VkDescriptorBufferInfo dbis[7];
  ec_vk_alloc_t *al[7] = { NULL };
  if (lookup_vk_ptr_locked(b1, eb1, &dbis[1], &p.o0, &al[1]) != 0) return -1;
  if (lookup_vk_ptr_locked(b2, eb2, &dbis[2], &p.o1, &al[2]) != 0) return -1;
  if (lookup_vk_ptr_locked(b3, eb3, &dbis[3], &p.o2, &al[3]) != 0) return -1;
  if (lookup_vk_ptr_locked(b4, eb4, &dbis[4], &p.o3, &al[4]) != 0) return -1;
  if (lookup_vk_ptr_locked(b5, eb5, &dbis[5], &p.o4, &al[5]) != 0) return -1;
  if (lookup_vk_ptr_locked(b6, eb6, &dbis[6], &p.o5, &al[6]) != 0) return -1;

  for (int i = 1; i <= 6; i++) {
    if (al[i]) ensure_dev_up_to_date_locked(al[i]);
  }

  const int is_igpu = (g_vk.props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
  const uint32_t max_wg_chunk = (is_igpu && (pid == VK_PIPE_FA2_F16 || pid == VK_PIPE_FA2_BF16)) ? 4096u :
                                (is_igpu ? 8192u : 32768u);
  const uint32_t max_batch_wgs = is_igpu ? 8192u : EC_VK_MAX_BATCH_WGS;
  uint32_t step_x = gx > max_wg_chunk ? max_wg_chunk : gx;
  uint32_t step_y = max_wg_chunk / (step_x > 0 ? step_x : 1u);
  if (step_y == 0) step_y = 1u;
  if (step_y > gy) step_y = gy;

  for (uint32_t oy = 0; oy < gy; oy += step_y) {
    uint32_t cy = (oy + step_y <= gy) ? step_y : (gy - oy);
    for (uint32_t ox = 0; ox < gx; ox += step_x) {
      uint32_t cx = (ox + step_x <= gx) ? step_x : (gx - ox);
      uint32_t chunk_wgs = cx * cy * gz;
      if (g_vk.batch_ops >= EC_VK_BATCH_SLOTS ||
          (g_vk.batch_open && g_vk.batch_wgs + chunk_wgs > max_batch_wgs)) {
        if (vk_batch_flush_locked(0) != 0) return -1;
      }

      int slot = g_vk.batch_ops;
      size_t ubo_off = (size_t)slot * EC_VK_UBO_STRIDE;
      p.pad0 = ox;
      p.pad1 = oy;
      memcpy(g_vk.ubo_mapped + ubo_off, &p, sizeof p);

      dbis[0].buffer = g_vk.ubo_buf;
      dbis[0].offset = (VkDeviceSize)ubo_off;
      dbis[0].range  = sizeof(ec_vk_params_t);

      VkWriteDescriptorSet writes[7];
      memset(writes, 0, sizeof writes);
      for (int i = 0; i < 7; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = g_vk.desc_sets[slot];
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = (i == 0) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbis[i];
      }
      vkUpdateDescriptorSets(g_vk.dev, 7, writes, 0, NULL);

      vk_batch_begin_locked();
      vkCmdBindPipeline(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pipes[pid]);
      vkCmdBindDescriptorSets(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pipe_layout,
                              0, 1, &g_vk.desc_sets[slot], 0, NULL);
      vkCmdDispatch(g_vk.cmd, cx, cy, gz);
      vk_cmd_barrier_locked(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_ACCESS_SHADER_WRITE_BIT,
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

      g_vk.batch_ops++;
      g_vk.batch_wgs += chunk_wgs;
    }
  }

  if (al[5] && al[5]->has_dev_mirror) al[5]->dev_dirty = 1;
  if (pipe_writes_b6(pid) && al[6] && al[6]->has_dev_mirror) al[6]->dev_dirty = 1;
  return 0;
}

cudaError_t ec_hal_get_device_props(cudaDeviceProp *prop, int device) {
  (void)device;
  if (ec_hal_init() != 0 || !prop) return cudaErrorNotSupported;
  memset(prop, 0, sizeof *prop);
  const char *cm_tag = (g_vk.has_coop_bf16 || g_vk.has_coop_f16 || g_vk.has_coop_i8) ? "+CoopMat" : "";
  snprintf(prop->name, sizeof prop->name, "%s (Vulkan%s%s)", g_vk.props.deviceName,
           g_vk.use_dev_mirror ? " VRAM" : " Unified", cm_tag);
  prop->totalGlobalMem = g_vk.vram_total;
  prop->major = (int)VK_API_VERSION_MAJOR(g_vk.props.apiVersion);
  prop->minor = (int)VK_API_VERSION_MINOR(g_vk.props.apiVersion);
  prop->multiProcessorCount = 32;
  prop->unifiedAddressing = 1;
  return cudaSuccess;
}

cudaError_t ec_hal_mem_get_info(size_t *free_b, size_t *total_b) {
  if (ec_hal_init() != 0) return cudaErrorNotSupported;
  pthread_mutex_lock(&g_vk.mu);
  size_t tot = g_vk.vram_total, used = g_vk.vram_used;
  pthread_mutex_unlock(&g_vk.mu);
  if (total_b) *total_b = tot;
  if (free_b)  *free_b  = (tot > used) ? (tot - used) : (512ULL << 20);
  return cudaSuccess;
}

static void insert_alloc_locked(ec_vk_alloc_t entry) {
  int pos = 0;
  while (pos < g_vk.n_allocs && g_vk.allocs[pos].base < entry.base) pos++;
  memmove(&g_vk.allocs[pos + 1], &g_vk.allocs[pos], (size_t)(g_vk.n_allocs - pos) * sizeof(ec_vk_alloc_t));
  g_vk.allocs[pos] = entry;
  g_vk.n_allocs++;
}

static void destroy_alloc_resources_locked(const ec_vk_alloc_t *a) {
  if (a->has_dev_mirror) {
    vkDestroyBuffer(g_vk.dev, a->dev_buf, NULL);
    vkFreeMemory(g_vk.dev, a->dev_mem, NULL);
  }
  vkUnmapMemory(g_vk.dev, a->host_mem);
  vkDestroyBuffer(g_vk.dev, a->host_buf, NULL);
  vkFreeMemory(g_vk.dev, a->host_mem, NULL);
}

static void drain_free_cache_locked(void) {
  for (int i = 0; i < g_vk.n_free_cache; i++) {
    destroy_alloc_resources_locked(&g_vk.free_cache[i]);
  }
  g_vk.n_free_cache = 0;
  g_vk.free_cache_bytes = 0;
}

cudaError_t ec_hal_malloc_raw(void **ptr, size_t nbytes) {
  if (!ptr) return cudaErrorMemoryAllocation;
  *ptr = NULL;
  if (nbytes == 0) nbytes = 256;
  if (ec_hal_init() != 0) return cudaErrorMemoryAllocation;

  pthread_mutex_lock(&g_vk.mu);
  if (g_vk.n_allocs >= EC_VK_MAX_ALLOCS) { pthread_mutex_unlock(&g_vk.mu); return cudaErrorMemoryAllocation; }
  size_t aligned = (nbytes + 255ULL) & ~255ULL;

  // T5: Exact-size reuse from VRAM caching allocator (zero vkAllocateMemory / host page-fault stall!)
  for (int i = g_vk.n_free_cache - 1; i >= 0; i--) {
    if (g_vk.free_cache[i].size == aligned) {
      ec_vk_alloc_t a = g_vk.free_cache[i];
      g_vk.free_cache_bytes -= a.size;
      g_vk.free_cache[i] = g_vk.free_cache[--g_vk.n_free_cache];
      a.host_dirty = 0; a.dirty_lo = 0; a.dirty_hi = 0;
      a.dev_dirty = 0; a.in_flight_h2d = 0;
      insert_alloc_locked(a);
      g_vk.vram_used += aligned;
      *ptr = a.mapped;
      pthread_mutex_unlock(&g_vk.mu);
      return cudaSuccess;
    }
  }

  VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  VkBuffer host_buf = VK_NULL_HANDLE;
  VkDeviceMemory host_mem = VK_NULL_HANDLE;
  void *mapped = NULL;
  int is_uni = 0;
  if (create_host_vk_buf(aligned, usage, !g_vk.use_dev_mirror, &host_buf, &host_mem, &mapped, &is_uni) != 0) {
    drain_free_cache_locked();
    if (create_host_vk_buf(aligned, usage, !g_vk.use_dev_mirror, &host_buf, &host_mem, &mapped, &is_uni) != 0) {
      pthread_mutex_unlock(&g_vk.mu);
      return cudaErrorMemoryAllocation;
    }
  }

  VkBuffer dev_buf = host_buf;
  VkDeviceMemory dev_mem = VK_NULL_HANDLE;
  int has_mirror = 0;
  if (g_vk.use_dev_mirror && !is_uni) {
    if (create_dev_local_vk_buf(aligned, usage, &dev_buf, &dev_mem) == 0) has_mirror = 1;
    else {
      drain_free_cache_locked();
      if (create_dev_local_vk_buf(aligned, usage, &dev_buf, &dev_mem) == 0) has_mirror = 1;
      else { dev_buf = host_buf; dev_mem = VK_NULL_HANDLE; has_mirror = 0; }
    }
  }

  insert_alloc_locked((ec_vk_alloc_t){
    .base = (uintptr_t)mapped, .size = aligned,
    .host_buf = host_buf, .host_mem = host_mem, .mapped = mapped,
    .dev_buf = dev_buf, .dev_mem = dev_mem,
    .has_dev_mirror = has_mirror, .is_pinned_only = 0,
  });
  g_vk.vram_used += aligned;
  *ptr = mapped;
  pthread_mutex_unlock(&g_vk.mu);
  return cudaSuccess;
}

cudaError_t ec_hal_free_raw(void *ptr) {
  if (!ptr) return cudaSuccess;
  pthread_mutex_lock(&g_vk.mu);
  if (g_vk.batch_open) vk_batch_flush_locked(0);
  uintptr_t addr = (uintptr_t)ptr;
  for (int i = 0; i < g_vk.n_allocs; i++) {
    if (g_vk.allocs[i].base == addr) {
      ec_vk_alloc_t a = g_vk.allocs[i];
      if (!a.is_pinned_only && g_vk.vram_used >= a.size) g_vk.vram_used -= a.size;
      memmove(&g_vk.allocs[i], &g_vk.allocs[i + 1], (size_t)(g_vk.n_allocs - i - 1) * sizeof(ec_vk_alloc_t));
      g_vk.n_allocs--;
      if (!a.is_pinned_only && a.size <= (128ULL << 20) &&
          g_vk.n_free_cache < EC_VK_FREE_CACHE && g_vk.free_cache_bytes + a.size <= EC_VK_FREE_CAP_B) {
        g_vk.free_cache[g_vk.n_free_cache++] = a;
        g_vk.free_cache_bytes += a.size;
      } else {
        destroy_alloc_resources_locked(&a);
      }
      break;
    }
  }
  pthread_mutex_unlock(&g_vk.mu);
  return cudaSuccess;
}

void *ec_hal_mmap_bind(const void *host_ptr, size_t nbytes) {
  (void)host_ptr; (void)nbytes; return NULL;
}

// T5: Pinned host memory backed by HOST_CACHED Vulkan staging buffer -> zero-memcpy PCIe DMA!
cudaError_t ec_hal_host_alloc_raw(void **ptr, size_t nbytes, unsigned int flags) {
  (void)flags;
  if (!ptr) return cudaErrorMemoryAllocation;
  *ptr = NULL;
  if (nbytes == 0) nbytes = 256;
  if (ec_hal_init() == 0) {
    pthread_mutex_lock(&g_vk.mu);
    if (g_vk.n_allocs < EC_VK_MAX_ALLOCS) {
      size_t aligned = (nbytes + 255ULL) & ~255ULL;
      VkBuffer buf = VK_NULL_HANDLE;
      VkDeviceMemory mem = VK_NULL_HANDLE;
      void *mapped = NULL;
      VkBufferUsageFlags u = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      if (create_host_vk_buf(aligned, u, 0, &buf, &mem, &mapped, NULL) == 0) {
        insert_alloc_locked((ec_vk_alloc_t){
          .base = (uintptr_t)mapped, .size = aligned,
          .host_buf = buf, .host_mem = mem, .mapped = mapped,
          .dev_buf = buf, .dev_mem = VK_NULL_HANDLE,
          .has_dev_mirror = 0, .is_pinned_only = 1,
        });
        *ptr = mapped;
        pthread_mutex_unlock(&g_vk.mu);
        return cudaSuccess;
      }
    }
    pthread_mutex_unlock(&g_vk.mu);
  }
  void *p = NULL;
  if (posix_memalign(&p, 64, (nbytes + 63ULL) & ~63ULL) != 0 || !p) return cudaErrorMemoryAllocation;
  *ptr = p;
  return cudaSuccess;
}

cudaError_t ec_hal_host_free_raw(void *ptr) {
  if (!ptr) return cudaSuccess;
  pthread_mutex_lock(&g_vk.mu);
  ec_vk_alloc_t *a = find_alloc_locked(ptr);
  pthread_mutex_unlock(&g_vk.mu);
  if (a) return ec_hal_free_raw(ptr);
  free(ptr);
  return cudaSuccess;
}

void ec_hal_sync_ptr_for_cpu(const void *ptr) {
  if (!ptr || !g_vk.init_ok) return;
  pthread_mutex_lock(&g_vk.mu);
  ec_vk_alloc_t *a = find_alloc_locked(ptr);
  if (a && a->has_dev_mirror && a->dev_dirty) sync_single_alloc_to_host_locked(a, 0, a->size, 1);
  else if (g_vk.batch_open) vk_batch_flush_locked(0);
  pthread_mutex_unlock(&g_vk.mu);
}

void ec_hal_mark_host_dirty(const void *ptr) {
  if (!ptr || !g_vk.init_ok || !g_vk.use_dev_mirror) return;
  pthread_mutex_lock(&g_vk.mu);
  ec_vk_alloc_t *a = find_alloc_locked(ptr);
  if (a && a->has_dev_mirror) { mark_host_dirty_range_locked(a, 0, a->size); a->dev_dirty = 0; }
  pthread_mutex_unlock(&g_vk.mu);
}

cudaError_t ec_hal_memcpy_async(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
  (void)kind; (void)stream;
  if (count == 0 || !dst || !src || dst == src) return cudaSuccess;
  if (!g_vk.init_ok) { memcpy(dst, src, count); return cudaSuccess; }

  pthread_mutex_lock(&g_vk.mu);
  ec_vk_alloc_t *da = find_alloc_locked(dst);
  ec_vk_alloc_t *sa = find_alloc_locked(src);

  // T5 Fast-Path 1: Direct zero-memcpy PCIe DMA from pinned VkBuffer -> DEVICE_LOCAL VRAM!
  if (da && da->has_dev_mirror && sa && sa->is_pinned_only) {
    size_t d_off = (size_t)((uintptr_t)dst - da->base);
    size_t s_off = (size_t)((uintptr_t)src - sa->base);
    if (da->host_dirty) {
      if (d_off == 0 && count == da->size) { da->host_dirty = 0; da->dirty_lo = da->dirty_hi = 0; }
      else ensure_dev_up_to_date_locked(da);
    }
    vk_batch_begin_locked();
    vk_cmd_barrier_locked(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferCopy cp = { .srcOffset = s_off, .dstOffset = d_off, .size = count };
    vkCmdCopyBuffer(g_vk.cmd, sa->host_buf, da->dev_buf, 1, &cp);
    vk_cmd_barrier_locked(VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
    da->dev_dirty = 1;
    pthread_mutex_unlock(&g_vk.mu);
    return cudaSuccess;
  }

  // T5 Fast-Path 2: Direct zero-memcpy PCIe DMA from DEVICE_LOCAL VRAM -> pinned VkBuffer!
  if (sa && sa->has_dev_mirror && da && da->is_pinned_only) {
    size_t d_off = (size_t)((uintptr_t)dst - da->base);
    size_t s_off = (size_t)((uintptr_t)src - sa->base);
    ensure_dev_up_to_date_locked(sa);
    vk_batch_begin_locked();
    vk_cmd_barrier_locked(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_TRANSFER_READ_BIT);
    VkBufferCopy cp = { .srcOffset = s_off, .dstOffset = d_off, .size = count };
    vkCmdCopyBuffer(g_vk.cmd, sa->dev_buf, da->host_buf, 1, &cp);
    vk_cmd_barrier_locked(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                          VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    pthread_mutex_unlock(&g_vk.mu);
    return cudaSuccess;
  }

  // GPU-to-GPU copy inside DEVICE_LOCAL VRAM
  if (da && da->has_dev_mirror && sa && sa->has_dev_mirror) {
    size_t d_off = (size_t)((uintptr_t)dst - da->base);
    size_t s_off = (size_t)((uintptr_t)src - sa->base);
    ensure_dev_up_to_date_locked(sa);
    if (da->host_dirty) {
      if (d_off == 0 && count == da->size) { da->host_dirty = 0; da->dirty_lo = da->dirty_hi = 0; }
      else ensure_dev_up_to_date_locked(da);
    }
    vk_batch_begin_locked();
    vk_cmd_barrier_locked(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferCopy cp = { .srcOffset = s_off, .dstOffset = d_off, .size = count };
    vkCmdCopyBuffer(g_vk.cmd, sa->dev_buf, da->dev_buf, 1, &cp);
    vk_cmd_barrier_locked(VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
    da->dev_dirty = 1;
    pthread_mutex_unlock(&g_vk.mu);
    return cudaSuccess;
  }

  if (sa && sa->has_dev_mirror) {
    size_t s_off = (size_t)((uintptr_t)src - sa->base);
    if (sa->dev_dirty) {
      int full = (s_off == 0 && count == sa->size);
      sync_single_alloc_to_host_locked(sa, s_off, s_off + count, full);
    } else if (g_vk.batch_open) {
      vk_batch_flush_locked(0);
    }
    memcpy(dst, src, count);
    pthread_mutex_unlock(&g_vk.mu);
    return cudaSuccess;
  }

  if (da && da->has_dev_mirror) {
    size_t d_off = (size_t)((uintptr_t)dst - da->base);
    int full = (d_off == 0 && count == da->size);
    if (da->dev_dirty && !full) sync_single_alloc_to_host_locked(da, 0, da->size, 1);
    else if (da->in_flight_h2d) vk_batch_flush_locked(0);
    memcpy(dst, src, count);
    if (full) da->dev_dirty = 0;
    mark_host_dirty_range_locked(da, d_off, d_off + count);
    pthread_mutex_unlock(&g_vk.mu);
    return cudaSuccess;
  }

  if ((sa || da) && g_vk.batch_open) vk_batch_flush_locked(0);
  memcpy(dst, src, count);
  pthread_mutex_unlock(&g_vk.mu);
  return cudaSuccess;
}

cudaError_t ec_hal_memset_async(void *dst, int value, size_t count, cudaStream_t stream) {
  (void)stream;
  if (count == 0 || !dst) return cudaSuccess;
  if (!g_vk.init_ok) { memset(dst, value, count); return cudaSuccess; }

  pthread_mutex_lock(&g_vk.mu);
  ec_vk_alloc_t *da = find_alloc_locked(dst);
  if (da && da->has_dev_mirror) {
    size_t d_off = (size_t)((uintptr_t)dst - da->base);
    if ((d_off & 3ULL) == 0 && (count & 3ULL) == 0) {
      if (da->host_dirty) {
        if (d_off == 0 && count == da->size) { da->host_dirty = 0; da->dirty_lo = da->dirty_hi = 0; }
        else ensure_dev_up_to_date_locked(da);
      }
      uint32_t b = (uint32_t)(value & 0xff);
      uint32_t w = b | (b << 8) | (b << 16) | (b << 24);
      vk_batch_begin_locked();
      vk_cmd_barrier_locked(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                            VK_ACCESS_TRANSFER_WRITE_BIT);
      vkCmdFillBuffer(g_vk.cmd, da->dev_buf, d_off, count, w);
      vk_cmd_barrier_locked(VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_ACCESS_TRANSFER_WRITE_BIT,
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
      da->dev_dirty = 1;
      pthread_mutex_unlock(&g_vk.mu);
      return cudaSuccess;
    }
    int full = (d_off == 0 && count == da->size);
    if (da->dev_dirty && !full) sync_single_alloc_to_host_locked(da, 0, da->size, 1);
    else if (da->in_flight_h2d) vk_batch_flush_locked(0);
    memset(dst, value, count);
    if (full) da->dev_dirty = 0;
    mark_host_dirty_range_locked(da, d_off, d_off + count);
    pthread_mutex_unlock(&g_vk.mu);
    return cudaSuccess;
  }
  if (da && g_vk.batch_open) vk_batch_flush_locked(0);
  memset(dst, value, count);
  pthread_mutex_unlock(&g_vk.mu);
  return cudaSuccess;
}

cudaError_t ec_hal_stream_create(cudaStream_t *pStream, unsigned int flags) {
  (void)flags;
  if (ec_hal_init() != 0) return cudaErrorNotSupported;
  if (pStream) *pStream = (cudaStream_t)(uintptr_t)1;
  return cudaSuccess;
}
cudaError_t ec_hal_stream_destroy(cudaStream_t s) { (void)s; return cudaSuccess; }
cudaError_t ec_hal_stream_sync(cudaStream_t s) {
  (void)s;
  if (!g_vk.init_ok) return cudaSuccess;
  pthread_mutex_lock(&g_vk.mu);
  int rc = vk_batch_flush_locked(0);
  pthread_mutex_unlock(&g_vk.mu);
  return rc == 0 ? cudaSuccess : cudaErrorNotSupported;
}
void ec_hal_sync_all_for_cpu(cudaStream_t s) {
  (void)s;
  if (!g_vk.init_ok) return;
  pthread_mutex_lock(&g_vk.mu);
  vk_batch_flush_locked(1);
  pthread_mutex_unlock(&g_vk.mu);
}
cudaError_t ec_hal_device_sync(void) { return ec_hal_stream_sync(NULL); }

typedef struct { struct timespec ts; } ec_vk_ev_t;
cudaError_t ec_hal_event_create(cudaEvent_t *e, unsigned int flags) {
  (void)flags;
  ec_vk_ev_t *ev = (ec_vk_ev_t *)calloc(1, sizeof *ev);
  if (!ev) return cudaErrorMemoryAllocation;
  clock_gettime(CLOCK_MONOTONIC, &ev->ts);
  *e = (cudaEvent_t)ev;
  return cudaSuccess;
}
cudaError_t ec_hal_event_destroy(cudaEvent_t e) { free(e); return cudaSuccess; }
cudaError_t ec_hal_event_record(cudaEvent_t e, cudaStream_t s) {
  (void)s;
  if (g_vk.init_ok) {
    pthread_mutex_lock(&g_vk.mu);
    vk_batch_flush_locked(0);
    pthread_mutex_unlock(&g_vk.mu);
  }
  if (e) clock_gettime(CLOCK_MONOTONIC, &((ec_vk_ev_t *)e)->ts);
  return cudaSuccess;
}
cudaError_t ec_hal_event_sync(cudaEvent_t e) {
  (void)e;
  if (g_vk.init_ok) {
    pthread_mutex_lock(&g_vk.mu);
    vk_batch_flush_locked(0);
    pthread_mutex_unlock(&g_vk.mu);
  }
  return cudaSuccess;
}
cudaError_t ec_hal_event_elapsed(float *ms, cudaEvent_t a, cudaEvent_t b) {
  if (!ms || !a || !b) return cudaErrorNotSupported;
  ec_vk_ev_t *ea = (ec_vk_ev_t *)a, *eb = (ec_vk_ev_t *)b;
  *ms = (float)((double)(eb->ts.tv_sec - ea->ts.tv_sec) * 1e3 + (double)(eb->ts.tv_nsec - ea->ts.tv_nsec) * 1e-6);
  return cudaSuccess;
}
cudaError_t ec_hal_stream_wait_event(cudaStream_t s, cudaEvent_t e, unsigned int f) {
  (void)s; (void)e; (void)f; return cudaSuccess;
}

static float *ensure_vk_tmp_f32_locked(size_t n_elems) {
  size_t need = n_elems * sizeof(float);
  if (need <= g_vk.tmp_y_cap && g_vk.tmp_y_f32) return g_vk.tmp_y_f32;
  pthread_mutex_unlock(&g_vk.mu);
  if (g_vk.tmp_y_f32) ec_hal_free_raw(g_vk.tmp_y_f32);
  void *p = NULL;
  ec_hal_malloc_raw(&p, need);
  pthread_mutex_lock(&g_vk.mu);
  g_vk.tmp_y_f32 = (float *)p;
  g_vk.tmp_y_cap = p ? need : 0;
  return g_vk.tmp_y_f32;
}

static void pack_f32_to_bf16_host_or_gpu_locked(const float *src, uint16_t *dst, size_t n) {
  if (!src || !dst || n == 0) return;
  if ((((uintptr_t)dst) & 3u) == 0 && (n & 1u) == 0 && n >= 2) {
    size_t pairs = n >> 1;
    ec_vk_params_t pp = { .p0 = (uint32_t)n, .p1 = 0 };
    if (vk_dispatch_locked(VK_PIPE_PACK16X2, &pp, NULL, 4, NULL, 4, src, 4, NULL, 4, NULL, 4, dst, 4,
                           (uint32_t)((pairs + 255) / 256), 1, 1) == 0) return;
  }
  vk_batch_flush_locked(1);
  for (size_t i = 0; i < n; i++) dst[i] = ec_cmn_f32_to_bf16(src[i]);
  ec_vk_alloc_t *da = find_alloc_locked(dst);
  if (da && da->has_dev_mirror) { mark_host_dirty_range_locked(da, 0, da->size); da->dev_dirty = 0; }
}

static void pack_f32_to_f16_host_or_gpu_locked(const float *src, __half *dst, size_t n) {
  if (!src || !dst || n == 0) return;
  if ((((uintptr_t)dst) & 3u) == 0 && (n & 1u) == 0 && n >= 2) {
    size_t pairs = n >> 1;
    ec_vk_params_t pp = { .p0 = (uint32_t)n, .p1 = 2 };
    if (vk_dispatch_locked(VK_PIPE_PACK16X2, &pp, NULL, 4, NULL, 4, src, 4, NULL, 4, NULL, 4, dst, 4,
                           (uint32_t)((pairs + 255) / 256), 1, 1) == 0) return;
  }
  vk_batch_flush_locked(1);
  for (size_t i = 0; i < n; i++) dst[i] = __float2half_rn(src[i]);
  ec_vk_alloc_t *da = find_alloc_locked(dst);
  if (da && da->has_dev_mirror) { mark_host_dirty_range_locked(da, 0, da->size); da->dev_dirty = 0; }
}

// ===========================================================================
// T4 Heavy Operators (Cooperative Matrix TensorCore/WMMA/XMX + SPIR-V 1.3 Fallback)
// ===========================================================================
int ec_ops_gemm_bf16(
    const uint16_t *X, const uint16_t *W, const float *bias,
    float *Y_f32, uint16_t *Y_bf16, int M, int N, int K, cudaStream_t stream) {
  (void)stream;
  if (!X || !W || M <= 0 || N <= 0 || K <= 0 || (!Y_f32 && !Y_bf16)) return -1;
  if (ec_hal_init() != 0) return -1;
  pthread_mutex_lock(&g_vk.mu);
  float *dst_f32 = Y_f32 ? Y_f32 : ensure_vk_tmp_f32_locked((size_t)M * N);
  if (!dst_f32) { pthread_mutex_unlock(&g_vk.mu); return -1; }
  ec_vk_params_t p = { .p0 = (uint32_t)M, .p1 = (uint32_t)N, .p2 = (uint32_t)K, .p3 = bias ? 1u : 0u };
  int rc;
  if (g_vk.has_coop_bf16 && (K & 7) == 0 && (((uintptr_t)X) & 15u) == 0 && (((uintptr_t)W) & 15u) == 0) {
    rc = vk_dispatch_locked(VK_PIPE_COOP_GEMM_BF16, &p, X, 16, W, 16, bias, 4, NULL, 4, dst_f32, 4, NULL, 4,
                            (uint32_t)((N + 127) / 128), (uint32_t)((M + 127) / 128), 1);
  } else {
    rc = vk_dispatch_locked(VK_PIPE_GEMM_BF16, &p, X, 2, W, 2, bias, 4, NULL, 4, dst_f32, 4, NULL, 4,
                            (uint32_t)((N + 63) / 64), (uint32_t)((M + 63) / 64), 1);
  }
  if (rc == 0 && Y_bf16) pack_f32_to_bf16_host_or_gpu_locked(dst_f32, Y_bf16, (size_t)M * N);
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}

int ec_launch_gemm_f16(
    const __half *X, const __half *W, const float *bias, const float *res,
    float *Y, __half *Y_f16, int M, int N, int K, int epilogue, cudaStream_t stream) {
  (void)stream;
  if (!X || !W || M <= 0 || N <= 0 || K <= 0 || (!Y && !Y_f16)) return -1;
  if (ec_hal_init() != 0) return -1;
  pthread_mutex_lock(&g_vk.mu);
  float *dst_f32 = Y ? Y : ensure_vk_tmp_f32_locked((size_t)M * N);
  if (!dst_f32) { pthread_mutex_unlock(&g_vk.mu); return -1; }
  ec_vk_params_t p = {
    .p0 = (uint32_t)M, .p1 = (uint32_t)N, .p2 = (uint32_t)K,
    .p3 = (uint32_t)epilogue, .p4 = bias ? 1u : 0u, .p5 = res ? 1u : 0u,
  };
  int rc;
  if (g_vk.has_coop_f16 && (K & 7) == 0 && (((uintptr_t)X) & 15u) == 0 && (((uintptr_t)W) & 15u) == 0) {
    rc = vk_dispatch_locked(VK_PIPE_COOP_GEMM_F16, &p, X, 16, W, 16, bias, 4, res, 4, dst_f32, 4, NULL, 4,
                            (uint32_t)((N + 127) / 128), (uint32_t)((M + 127) / 128), 1);
  } else {
    rc = vk_dispatch_locked(VK_PIPE_GEMM_F16, &p, X, 2, W, 2, bias, 4, res, 4, dst_f32, 4, NULL, 4,
                            (uint32_t)((N + 63) / 64), (uint32_t)((M + 63) / 64), 1);
  }
  if (rc == 0 && Y_f16) pack_f32_to_f16_host_or_gpu_locked(dst_f32, Y_f16, (size_t)M * N);
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}

int ec_ops_quantize_bf16_q8g64(
    const uint16_t *src_bf16, int8_t *dst_q8, float *dst_sc,
    int rows, int cols, cudaStream_t stream) {
  (void)stream;
  if (!src_bf16 || !dst_q8 || !dst_sc || rows <= 0 || cols <= 0 || (cols & 63)) return -1;
  if (ec_hal_init() != 0) return -1;
  uint32_t total_g = (uint32_t)rows * (uint32_t)(cols >> 6);
  pthread_mutex_lock(&g_vk.mu);
  ec_vk_params_t p = { .p0 = total_g };
  int rc = vk_dispatch_locked(VK_PIPE_QUANT_Q8, &p, src_bf16, 2, NULL, 4, NULL, 4, NULL, 4, dst_sc, 4, dst_q8, 4,
                              (total_g + 63u) / 64u, 1, 1);
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}

int ec_ops_gemm_q8g64(
    const int8_t *X_q8, const float *x_sc,
    const int8_t *W_q8, const float *w_sc,
    const float *bias, float *Y_f32, uint16_t *Y_bf16,
    int M, int N, int K, cudaStream_t stream) {
  (void)stream;
  if (!X_q8 || !x_sc || !W_q8 || !w_sc || M <= 0 || N <= 0 || (K & 63) || (!Y_f32 && !Y_bf16)) return -1;
  if (ec_hal_init() != 0) return -1;
  pthread_mutex_lock(&g_vk.mu);
  float *dst_f32 = Y_f32 ? Y_f32 : ensure_vk_tmp_f32_locked((size_t)M * N);
  if (!dst_f32) { pthread_mutex_unlock(&g_vk.mu); return -1; }
  ec_vk_params_t p = { .p0 = (uint32_t)M, .p1 = (uint32_t)N, .p2 = (uint32_t)K, .p3 = bias ? 1u : 0u };
  int rc;
  if (g_vk.has_coop_i8 && (((uintptr_t)X_q8) & 15u) == 0 && (((uintptr_t)W_q8) & 15u) == 0) {
    rc = vk_dispatch_locked(VK_PIPE_COOP_GEMM_Q8, &p, X_q8, 16, W_q8, 16, x_sc, 4, w_sc, 4, dst_f32, 4, (void *)bias, 4,
                            (uint32_t)((N + 63) / 64), (uint32_t)((M + 127) / 128), 1);
  } else {
    rc = vk_dispatch_locked(VK_PIPE_GEMM_Q8, &p, X_q8, 1, W_q8, 1, x_sc, 4, w_sc, 4, dst_f32, 4, (void *)bias, 4,
                            (uint32_t)((N + 63) / 64), (uint32_t)((M + 63) / 64), 1);
  }
  if (rc == 0 && Y_bf16) pack_f32_to_bf16_host_or_gpu_locked(dst_f32, Y_bf16, (size_t)M * N);
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}

int ec_launch_gemm_i8(
    const int8_t *X_i8, const float *x_sc,
    const int8_t *W_i8, const float *w_sc,
    const float *bias, float *Y,
    int M, int N, int K, cudaStream_t stream) {
  (void)stream;
  if (!X_i8 || !x_sc || !W_i8 || !w_sc || !Y || M <= 0 || N <= 0 || K <= 0) return -1;
  if (ec_hal_init() != 0) return -1;
  pthread_mutex_lock(&g_vk.mu);
  ec_vk_params_t p = { .p0 = (uint32_t)M, .p1 = (uint32_t)N, .p2 = (uint32_t)K, .p3 = bias ? 1u : 0u };
  int rc;
  if (g_vk.has_coop_i8 && (K & 15) == 0 && (((uintptr_t)X_i8) & 15u) == 0 && (((uintptr_t)W_i8) & 15u) == 0) {
    rc = vk_dispatch_locked(VK_PIPE_COOP_GEMM_I8, &p, X_i8, 16, W_i8, 16, x_sc, 4, w_sc, 4, Y, 4, (void *)bias, 4,
                            (uint32_t)((N + 127) / 128), (uint32_t)((M + 127) / 128), 1);
  } else {
    rc = vk_dispatch_locked(VK_PIPE_GEMM_I8, &p, X_i8, 1, W_i8, 1, x_sc, 4, w_sc, 4, Y, 4, (void *)bias, 4,
                            (uint32_t)((N + 63) / 64), (uint32_t)((M + 63) / 64), 1);
  }
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}

int ec_ops_fa2_bf16(
    const uint16_t *Q, const uint16_t *K, const uint16_t *V,
    float *O_f32, uint16_t *O_bf16,
    int Lq, int Lkv, int Lq_pad, int Lkv_pad,
    int Hq, int Hkv, int D, float scale, int window, cudaStream_t stream) {
  (void)stream;
  if (!Q || !K || !V || Lq <= 0 || Lkv <= 0 || Hq <= 0 || Hkv <= 0 || (Hq % Hkv) || (D != 64 && D != 128)) return -1;
  if (ec_hal_init() != 0) return -1;
  pthread_mutex_lock(&g_vk.mu);
  size_t out_elems = (size_t)Lq * Hq * D;
  float *dst_f32 = O_f32 ? O_f32 : ensure_vk_tmp_f32_locked(out_elems);
  if (!dst_f32) { pthread_mutex_unlock(&g_vk.mu); return -1; }
  ec_vk_params_t p = {
    .p0 = (uint32_t)Lq, .p1 = (uint32_t)Lkv, .p2 = (uint32_t)Lq_pad, .p3 = (uint32_t)Lkv_pad,
    .p4 = (uint32_t)Hq, .p5 = (uint32_t)Hkv, .p6 = (uint32_t)D, .p7 = (uint32_t)(window > 0 ? window : 0),
    .f0 = scale,
  };
  int rc;
  if (g_vk.has_coop_bf16 && (((uintptr_t)Q) & 15u) == 0 && (((uintptr_t)K) & 15u) == 0 && (((uintptr_t)V) & 15u) == 0) {
    rc = vk_dispatch_locked(VK_PIPE_COOP_FA2_BF16, &p, Q, 16, K, 16, NULL, 4, NULL, 4, dst_f32, 4, (void *)V, 16,
                            (uint32_t)((Lq + 63) / 64), (uint32_t)Hq, 1);
  } else {
    rc = vk_dispatch_locked(VK_PIPE_FA2_BF16, &p, Q, 2, K, 2, NULL, 4, NULL, 4, dst_f32, 4, (void *)V, 2,
                            (uint32_t)Lq * (uint32_t)Hq, 1, 1);
  }
  if (rc == 0 && O_bf16) pack_f32_to_bf16_host_or_gpu_locked(dst_f32, O_bf16, out_elems);
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}

int ec_launch_fa2_f16(
    const __half *Q, const __half *K, const __half *V,
    float *O, __half *O_f16,
    int n_q, int n_kv, int n_q_pad, int n_kv_pad,
    int n_heads, int head_dim, float scale, int block_causal_size, cudaStream_t stream) {
  (void)stream;
  if (!Q || !K || !V || n_q <= 0 || n_kv <= 0 || n_heads <= 0 || (head_dim != 64 && head_dim != 128)) return -1;
  if (ec_hal_init() != 0) return -1;
  pthread_mutex_lock(&g_vk.mu);
  size_t out_elems = (size_t)n_q * n_heads * head_dim;
  float *dst_f32 = O ? O : ensure_vk_tmp_f32_locked(out_elems);
  if (!dst_f32) { pthread_mutex_unlock(&g_vk.mu); return -1; }
  ec_vk_params_t p = {
    .p0 = (uint32_t)n_q, .p1 = (uint32_t)n_kv, .p2 = (uint32_t)n_q_pad, .p3 = (uint32_t)n_kv_pad,
    .p4 = (uint32_t)n_heads, .p5 = (uint32_t)head_dim, .p6 = (uint32_t)(block_causal_size > 0 ? block_causal_size : 0),
    .f0 = scale,
  };
  int rc;
  if (g_vk.has_coop_f16 && (((uintptr_t)Q) & 15u) == 0 && (((uintptr_t)K) & 15u) == 0 && (((uintptr_t)V) & 15u) == 0) {
    rc = vk_dispatch_locked(VK_PIPE_COOP_FA2_F16, &p, Q, 16, K, 16, NULL, 4, NULL, 4, dst_f32, 4, (void *)V, 16,
                            (uint32_t)((n_q + 63) / 64), (uint32_t)n_heads, 1);
  } else {
    rc = vk_dispatch_locked(VK_PIPE_FA2_F16, &p, Q, 2, K, 2, NULL, 4, NULL, 4, dst_f32, 4, (void *)V, 2,
                            (uint32_t)n_q * (uint32_t)n_heads, 1, 1);
  }
  if (rc == 0 && O_f16) pack_f32_to_f16_host_or_gpu_locked(dst_f32, O_f16, out_elems);
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}

int ec_ops_conv2d_3x3_nhwc_bf16(
    const uint16_t *X_nhwc, const uint16_t *W_ohwi,
    const float *bias, const float *res_f32,
    float *Y_f32, uint16_t *Y_bf16,
    int H, int W, int Cin, int Cout, cudaStream_t stream) {
  (void)stream;
  if (!X_nhwc || !W_ohwi || H <= 0 || W <= 0 || (Cin & 31) != 0 || (Cout & 63) != 0 || (!Y_f32 && !Y_bf16)) return -1;
  if (ec_hal_init() != 0) return -1;
  pthread_mutex_lock(&g_vk.mu);
  size_t M = (size_t)H * W;
  float *dst_f32 = Y_f32 ? Y_f32 : ensure_vk_tmp_f32_locked(M * Cout);
  if (!dst_f32) { pthread_mutex_unlock(&g_vk.mu); return -1; }
  ec_vk_params_t p = {
    .p0 = (uint32_t)H, .p1 = (uint32_t)W, .p2 = (uint32_t)Cin, .p3 = (uint32_t)Cout,
    .p4 = bias ? 1u : 0u, .p5 = res_f32 ? 1u : 0u,
  };
  int use_coop = g_vk.has_coop_bf16 && (((uintptr_t)X_nhwc) & 15u) == 0 && (((uintptr_t)W_ohwi) & 15u) == 0;
  ec_vk_pipe_id_t pid = use_coop ? VK_PIPE_COOP_CONV3X3 : VK_PIPE_CONV3X3;
  size_t eb = use_coop ? 16 : 2;
  int rc = vk_dispatch_locked(pid, &p, X_nhwc, eb, W_ohwi, eb, bias, 4, res_f32, 4, dst_f32, 4, NULL, 4,
                              (uint32_t)((Cout + 63) / 64), (uint32_t)((M + 63) / 64), 1);
  if (rc == 0 && Y_bf16) pack_f32_to_bf16_host_or_gpu_locked(dst_f32, Y_bf16, M * Cout);
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}

// ===========================================================================
// T3 GPU-Resident Glue Operators (Zero Mid-Step PCIe Host Syncs!)
// ===========================================================================
int ec_ops_rmsnorm_adaln(
    const float *x, const float *w, const float *scale, const float *shift,
    float *y_f32, uint16_t *y_bf16, int rows, int dim, float eps, cudaStream_t stream) {
  (void)stream;
  if (!x || rows <= 0 || dim <= 0) return -1;
  if (ec_hal_init() == 0 && (!y_bf16 || ((dim & 1) == 0 && (((uintptr_t)y_bf16) & 3u) == 0))) {
    pthread_mutex_lock(&g_vk.mu);
    uint32_t fl = (w ? 1u : 0u) | (scale ? 2u : 0u) | (shift ? 4u : 0u) |
                  (y_f32 ? 8u : 0u) | (y_bf16 ? 16u : 0u);
    ec_vk_params_t p = { .p0 = (uint32_t)rows, .p1 = (uint32_t)dim, .p2 = 0u, .p3 = fl, .f0 = eps };
    int rc = vk_dispatch_locked(VK_PIPE_NORM_ADALN, &p, x, 4, w, 4, scale, 4, shift, 4, y_f32, 4, y_bf16, 4,
                                (uint32_t)rows, 1, 1);
    pthread_mutex_unlock(&g_vk.mu);
    if (rc == 0) return 0;
  }
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(r, rows)
    const float *xr = x + (long)r * dim;
    float ss = 0.0f;
    for (int c = 0; c < dim; c++) ss += xr[c] * xr[c];
    float inv = rsqrtf(ss / (float)dim + eps);
    float *yf = y_f32 ? (y_f32 + (long)r * dim) : NULL;
    uint16_t *yb = y_bf16 ? (y_bf16 + (long)r * dim) : NULL;
    for (int c = 0; c < dim; c++) {
      float v = xr[c] * inv * (w ? w[c] : 1.0f);
      if (scale) v = v * (1.0f + scale[c]);
      if (shift) v += shift[c];
      if (yf) yf[c] = v;
      if (yb) yb[c] = ec_cmn_f32_to_bf16(v);
    }
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(y_f32);
  ec_hal_mark_host_dirty(y_bf16);
  return 0;
}

int ec_ops_layernorm_adaln(
    const float *x, const float *w, const float *b, const float *scale, const float *shift,
    float *y_f32, uint16_t *y_bf16, int rows, int dim, float eps, cudaStream_t stream) {
  (void)stream;
  if (!x || rows <= 0 || dim <= 0) return -1;
  if (!b && ec_hal_init() == 0 && (!y_bf16 || ((dim & 1) == 0 && (((uintptr_t)y_bf16) & 3u) == 0))) {
    pthread_mutex_lock(&g_vk.mu);
    uint32_t fl = (w ? 1u : 0u) | (scale ? 2u : 0u) | (shift ? 4u : 0u) |
                  (y_f32 ? 8u : 0u) | (y_bf16 ? 16u : 0u);
    ec_vk_params_t p = { .p0 = (uint32_t)rows, .p1 = (uint32_t)dim, .p2 = 1u, .p3 = fl, .f0 = eps };
    int rc = vk_dispatch_locked(VK_PIPE_NORM_ADALN, &p, x, 4, w, 4, scale, 4, shift, 4, y_f32, 4, y_bf16, 4,
                                (uint32_t)rows, 1, 1);
    pthread_mutex_unlock(&g_vk.mu);
    if (rc == 0) return 0;
  }
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(r, rows)
    const float *xr = x + (long)r * dim;
    float s1 = 0.0f, s2 = 0.0f;
    for (int c = 0; c < dim; c++) { float v = xr[c]; s1 += v; s2 += v * v; }
    float mean = s1 / (float)dim;
    float var = s2 / (float)dim - mean * mean;
    if (var < 0.0f) var = 0.0f;
    float inv = rsqrtf(var + eps);
    float *yf = y_f32 ? (y_f32 + (long)r * dim) : NULL;
    uint16_t *yb = y_bf16 ? (y_bf16 + (long)r * dim) : NULL;
    for (int c = 0; c < dim; c++) {
      float v = (xr[c] - mean) * inv;
      if (w) v *= w[c];
      if (b) v += b[c];
      if (scale) v = v * (1.0f + scale[c]);
      if (shift) v += shift[c];
      if (yf) yf[c] = v;
      if (yb) yb[c] = ec_cmn_f32_to_bf16(v);
    }
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(y_f32);
  ec_hal_mark_host_dirty(y_bf16);
  return 0;
}

int ec_launch_adaln_zero(
    const float *x, const float *norm_w, const float *shift, const float *scale,
    float *y, __half *y_f16, int n_rows, int dim, float eps, int norm_mode,
    cudaStream_t stream) {
  (void)stream;
  if (n_rows <= 0 || dim <= 0) return -1;
  if (ec_hal_init() == 0 && (!y_f16 || ((dim & 1) == 0 && (((uintptr_t)y_f16) & 3u) == 0))) {
    pthread_mutex_lock(&g_vk.mu);
    uint32_t mode = (norm_mode == 0) ? 1u : 0u; // norm_mode 0 = LayerNorm, 1/2 = RMSNorm
    const float *w_use = (norm_mode == 2) ? norm_w : NULL;
    uint32_t fl = (w_use ? 1u : 0u) | (scale ? 2u : 0u) | (shift ? 4u : 0u) |
                  (y ? 8u : 0u) | (y_f16 ? 32u : 0u);
    ec_vk_params_t p = { .p0 = (uint32_t)n_rows, .p1 = (uint32_t)dim, .p2 = mode, .p3 = fl, .f0 = eps };
    int rc = vk_dispatch_locked(VK_PIPE_NORM_ADALN, &p, x, 4, w_use, 4, scale, 4, shift, 4, y, 4, y_f16, 4,
                                (uint32_t)n_rows, 1, 1);
    pthread_mutex_unlock(&g_vk.mu);
    if (rc == 0) return 0;
  }
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(r, n_rows)
    const float *xr = x + (long)r * dim;
    float mean = 0.0f;
    if (norm_mode == 0) {
      float s = 0.0f;
      for (int i = 0; i < dim; i++) s += xr[i];
      mean = s / (float)dim;
    }
    float v = 0.0f;
    for (int i = 0; i < dim; i++) { float d = xr[i] - mean; v += d * d; }
    float inv = rsqrtf(v / (float)dim + eps);
    float *yr = y + (long)r * dim;
    __half *yh = y_f16 ? (y_f16 + (long)r * dim) : NULL;
    for (int i = 0; i < dim; i++) {
      float nrm = (xr[i] - mean) * inv;
      if (norm_mode == 2 && norm_w) nrm *= norm_w[i];
      float out = nrm * (scale ? (1.0f + scale[i]) : 1.0f) + (shift ? shift[i] : 0.0f);
      yr[i] = out;
      if (yh) yh[i] = __float2half_rn(out);
    }
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(y);
  ec_hal_mark_host_dirty(y_f16);
  return 0;
}

int ec_launch_gated_res(float *x, const float *h, const float *gate, int n_rows, int dim, cudaStream_t stream) {
  (void)stream;
  long total = (long)n_rows * dim;
  if (total <= 0) return -1;
  if (ec_hal_init() == 0) {
    pthread_mutex_lock(&g_vk.mu);
    ec_vk_params_t p = { .p0 = (uint32_t)n_rows, .p1 = (uint32_t)dim };
    int rc = vk_dispatch_locked(VK_PIPE_GATED_RES, &p, NULL, 4, NULL, 4, h, 4, gate, 4, x, 4, NULL, 4,
                                (uint32_t)((total + 255) / 256), 1, 1);
    pthread_mutex_unlock(&g_vk.mu);
    if (rc == 0) return 0;
  }
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(i, total) x[i] += gate[i % dim] * h[i]; EC_PAR_FOR_END
  ec_hal_mark_host_dirty(x);
  return 0;
}

int ec_ops_silu_f32_to_bf16(const float *src, uint16_t *dst, long n, cudaStream_t stream) {
  (void)stream;
  if (n <= 0) return -1;
  if (ec_hal_init() == 0 && (n & 1L) == 0 && (((uintptr_t)dst) & 3u) == 0) {
    pthread_mutex_lock(&g_vk.mu);
    ec_vk_params_t p = { .p0 = (uint32_t)n, .p1 = 1u };
    int rc = vk_dispatch_locked(VK_PIPE_ELEMWISE, &p, NULL, 4, NULL, 4, src, 4, NULL, 4, NULL, 4, dst, 4,
                                (uint32_t)(((n >> 1) + 255) / 256), 1, 1);
    pthread_mutex_unlock(&g_vk.mu);
    if (rc == 0) return 0;
  }
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(i, n) dst[i] = ec_cmn_f32_to_bf16(ec_cmn_silu(src[i])); EC_PAR_FOR_END
  ec_hal_mark_host_dirty(dst);
  return 0;
}

int ec_ops_dequant_q8g64_bf16(
    const int8_t *src_q8, const float *src_sc, uint16_t *dst_bf16,
    int rows, int cols, cudaStream_t stream) {
  (void)stream;
  if (!src_q8 || !src_sc || !dst_bf16 || rows <= 0 || cols <= 0 || (cols & 63)) return -1;
  if (ec_hal_init() == 0 && (((uintptr_t)dst_bf16) & 3u) == 0) {
    uint32_t pairs = (uint32_t)((long)rows * cols >> 1);
    pthread_mutex_lock(&g_vk.mu);
    ec_vk_params_t p = { .p0 = pairs, .p1 = 6u };
    int rc = vk_dispatch_locked(VK_PIPE_ELEMWISE, &p, src_q8, 1, NULL, 4, src_sc, 4, NULL, 4, NULL, 4, dst_bf16, 4,
                                (pairs + 255u) / 256u, 1, 1);
    pthread_mutex_unlock(&g_vk.mu);
    if (rc == 0) return 0;
  }
  ec_hal_sync_all_for_cpu(stream);
  const long total_g = (long)rows * (cols >> 6);
  EC_PAR_FOR_BEGIN(g, total_g)
    const float sc = src_sc[g];
    const int8_t *sq = src_q8 + g * 64;
    uint16_t *db = dst_bf16 + g * 64;
    for (int i = 0; i < 64; i++) db[i] = ec_cmn_f32_to_bf16((float)sq[i] * sc);
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(dst_bf16);
  return 0;
}

int ec_launch_swiglu_mul(const float *gate, const float *up, float *out, __half *out_f16, int n_elems, cudaStream_t stream) {
  (void)stream;
  if (n_elems <= 0) return -1;
  if (ec_hal_init() == 0 && (!out_f16 || ((n_elems & 1) == 0 && (((uintptr_t)out_f16) & 3u) == 0))) {
    pthread_mutex_lock(&g_vk.mu);
    ec_vk_params_t p = { .p0 = (uint32_t)n_elems, .p1 = 2u, .p2 = out_f16 ? 1u : 0u };
    uint32_t threads = out_f16 ? (uint32_t)(n_elems >> 1) : (uint32_t)n_elems;
    int rc = vk_dispatch_locked(VK_PIPE_ELEMWISE, &p, NULL, 4, NULL, 4, gate, 4, up, 4, out, 4, out_f16, 4,
                                (threads + 255u) / 256u, 1, 1);
    pthread_mutex_unlock(&g_vk.mu);
    if (rc == 0) return 0;
  }
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(i, n_elems)
    float v = ec_cmn_silu(gate[i]) * up[i];
    out[i] = v;
    if (out_f16) out_f16[i] = __float2half_rn(v);
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(out);
  ec_hal_mark_host_dirty(out_f16);
  return 0;
}

int ec_launch_flow_cfg_euler(
    const float *x, const float *v_cond, const float *v_uncond,
    float *x_next, int n_elems, float dt, float cfg_scale, cudaStream_t stream) {
  (void)stream;
  if (n_elems <= 0) return -1;
  if (ec_hal_init() == 0) {
    pthread_mutex_lock(&g_vk.mu);
    ec_vk_params_t p = { .p0 = (uint32_t)n_elems, .p1 = 5u, .p2 = v_uncond ? 1u : 0u, .f0 = dt, .f1 = cfg_scale };
    int rc = vk_dispatch_locked(VK_PIPE_ELEMWISE, &p, x, 4, NULL, 4, v_cond, 4, v_uncond, 4, x_next, 4, NULL, 4,
                                ((uint32_t)n_elems + 255u) / 256u, 1, 1);
    pthread_mutex_unlock(&g_vk.mu);
    if (rc == 0) return 0;
  }
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(i, n_elems)
    float v = v_uncond ? (v_uncond[i] + cfg_scale * (v_cond[i] - v_uncond[i])) : v_cond[i];
    x_next[i] = x[i] + dt * v;
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(x_next);
  return 0;
}

int ec_launch_groupnorm_silu(
    const float *x, const float *gamma, const float *beta,
    float *y, int C, int spatial, int num_groups, float eps, int apply_silu,
    cudaStream_t stream) {
  (void)stream;
  if (C <= 0 || spatial <= 0 || num_groups <= 0 || (C % num_groups)) return -1;
  if (ec_hal_init() == 0) {
    pthread_mutex_lock(&g_vk.mu);
    uint32_t fl = (gamma ? 1u : 0u) | (beta ? 2u : 0u) | (apply_silu ? 4u : 0u);
    ec_vk_params_t p = { .p0 = (uint32_t)C, .p1 = (uint32_t)spatial, .p2 = (uint32_t)num_groups, .p3 = fl, .f0 = eps };
    int rc = vk_dispatch_locked(VK_PIPE_GROUPNORM_SILU, &p, x, 4, NULL, 4, gamma, 4, beta, 4, y, 4, NULL, 4,
                                (uint32_t)num_groups, 1, 1);
    pthread_mutex_unlock(&g_vk.mu);
    if (rc == 0) return 0;
  }
  ec_hal_sync_all_for_cpu(stream);
  const int cpg = C / num_groups;
  const long n = (long)cpg * spatial;
  EC_PAR_FOR_BEGIN(g, num_groups)
    int c0 = g * cpg;
    const float *xg = x + (long)c0 * spatial;
    float *yg = y + (long)c0 * spatial;
    double s = 0.0;
    for (long i = 0; i < n; i++) s += xg[i];
    float mean = (float)(s / (double)n);
    double v = 0.0;
    for (long i = 0; i < n; i++) { double d = xg[i] - mean; v += d * d; }
    float inv = rsqrtf((float)(v / (double)n) + eps);
    for (long i = 0; i < n; i++) {
      int c = c0 + (int)(i / spatial);
      float val = (xg[i] - mean) * inv;
      if (gamma) val *= gamma[c];
      if (beta)  val += beta[c];
      if (apply_silu) val = ec_cmn_silu(val);
      yg[i] = val;
    }
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(y);
  return 0;
}

// ===========================================================================
// T3 Model-Specific Vulkan Hooks (called from ec_model_kernels_portable.h & ec_flux2.cu)
// ===========================================================================
int ec_vk_k_flux2_pack_qkv_rope(
    const float *qkv, int row_stride, const float *q_norm_w, const float *k_norm_w,
    const float *rope_cos, const float *rope_sin,
    __half *Q_out, __half *K_out, __half *V_out,
    int n_tok, int tok_off, int n_pad, int H, int D, float eps) {
  if (ec_hal_init() != 0 || D > 128 || (D & 1)) return -1;
  pthread_mutex_lock(&g_vk.mu);
  uint32_t pairs = (uint32_t)n_tok * (uint32_t)H;
  ec_vk_params_t pq = { (uint32_t)n_tok, (uint32_t)tok_off, (uint32_t)n_pad, (uint32_t)H, (uint32_t)D, (uint32_t)row_stride, 0u, 0u, 0,0,0,0,0,0,0,0, eps,0,0,0 };
  int rc = vk_dispatch_locked(VK_PIPE_FLUX2_PACK_QKV_ROPE, &pq, qkv, 4, q_norm_w, 4, rope_cos, 4, rope_sin, 4, NULL, 4, Q_out, 4, pairs, 1, 1);
  if (rc == 0) {
    pq.p6 = 1u;
    rc = vk_dispatch_locked(VK_PIPE_FLUX2_PACK_QKV_ROPE, &pq, qkv, 4, k_norm_w, 4, rope_cos, 4, rope_sin, 4, NULL, 4, K_out, 4, pairs, 1, 1);
  }
  if (rc == 0) {
    pq.p6 = 2u;
    rc = vk_dispatch_locked(VK_PIPE_FLUX2_PACK_QKV_ROPE, &pq, qkv, 4, NULL, 4, NULL, 4, NULL, 4, NULL, 4, V_out, 4, pairs, 1, 1);
  }
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}

int ec_vk_k_flux2_single_cat_q8g64(
    const float *attn_out, const float *lin1, int8_t *dst_q8, float *dst_sc,
    int N_tot, int E, int F) {
  if (ec_hal_init() != 0 || ((E + F) & 63)) return -1;
  pthread_mutex_lock(&g_vk.mu);
  uint32_t total_g = (uint32_t)N_tot * (uint32_t)((E + F) >> 6);
  ec_vk_params_t p = { .p0 = (uint32_t)N_tot, .p1 = (uint32_t)E, .p2 = (uint32_t)F };
  int rc = vk_dispatch_locked(VK_PIPE_FLUX2_SINGLE_CAT_Q8, &p, NULL, 4, NULL, 4, attn_out, 4, lin1, 4, dst_sc, 4, dst_q8, 4, total_g, 1, 1);
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}

int ec_vk_k_nhwc_gn_stats(const float *x_nhwc, float *stats64, int M, int C) {
  if (ec_hal_init() != 0 || (C & 31)) return -1;
  pthread_mutex_lock(&g_vk.mu);
  ec_vk_params_t p = { .p0 = (uint32_t)M, .p1 = (uint32_t)C };
  int rc = vk_dispatch_locked(VK_PIPE_NHWC_GN_STATS, &p, NULL, 4, NULL, 4, x_nhwc, 4, NULL, 4, stats64, 4, NULL, 4, 32u, 1, 1);
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}

int ec_vk_k_nhwc_gn_apply_bf16(
    const float *x_nhwc, const float *stats64, const float *gamma, const float *beta,
    uint16_t *y_bf16, int M, int C, float eps, int apply_silu) {
  if (ec_hal_init() != 0 || (C & 31) || (((uintptr_t)y_bf16) & 3u)) return -1;
  pthread_mutex_lock(&g_vk.mu);
  uint32_t pairs = (uint32_t)((long)M * C >> 1);
  ec_vk_params_t p = { .p0 = (uint32_t)M, .p1 = (uint32_t)C, .p2 = (uint32_t)apply_silu, .f0 = eps };
  int rc = vk_dispatch_locked(VK_PIPE_NHWC_GN_APPLY, &p, x_nhwc, 4, stats64, 4, gamma, 4, beta, 4, NULL, 4, y_bf16, 4,
                              (pairs + 255u) / 256u, 1, 1);
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}

int ec_vk_k_upsample2x_nhwc_bf16(const float *x_nhwc, uint16_t *y_bf16, int H, int W, int C) {
  if (ec_hal_init() != 0 || (C & 1) || (((uintptr_t)y_bf16) & 3u)) return -1;
  pthread_mutex_lock(&g_vk.mu);
  uint32_t pairs = (uint32_t)(4L * H * W * (C >> 1));
  ec_vk_params_t p = { .p0 = (uint32_t)H, .p1 = (uint32_t)W, .p2 = (uint32_t)C };
  int rc = vk_dispatch_locked(VK_PIPE_UPSAMPLE2X_NHWC, &p, NULL, 4, NULL, 4, x_nhwc, 4, NULL, 4, NULL, 4, y_bf16, 4,
                              (pairs + 255u) / 256u, 1, 1);
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}

int ec_vk_k_vae_mid_attn(const void *Q, const void *K, const void *V, uint16_t *O_bf16,
                         int S, int C, float scale, int is_bf16) {
  if (ec_hal_init() != 0 || C > 512 || (C & 1) || (((uintptr_t)O_bf16) & 3u)) return -1;
  if (is_bf16 && (((uintptr_t)V) & 3u)) return -1;
  pthread_mutex_lock(&g_vk.mu);
  ec_vk_params_t p = { .p0 = (uint32_t)S, .p1 = (uint32_t)C, .p2 = (uint32_t)is_bf16, .f0 = scale };
  size_t eb = is_bf16 ? 2 : 4;
  int rc = vk_dispatch_locked(VK_PIPE_VAE_MID_ATTN, &p, Q, eb, K, eb, V, 4, NULL, 4, NULL, 4, O_bf16, 4,
                              (uint32_t)S, 1, 1);
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}

int ec_vk_k_qwen3_qk_rope(
    const float *q_in, const float *k_in, const float *v_in,
    const float *q_norm_w, const float *k_norm_w,
    float *q_out, float *k_out, float *v_out,
    int L, int Hq, int Hkv, int D, float theta, float eps) {
  if (ec_hal_init() != 0 || D > 128 || (D & 1)) return -1;
  pthread_mutex_lock(&g_vk.mu);
  ec_vk_params_t p = { .p0 = (uint32_t)L, .p1 = (uint32_t)Hq, .p2 = (uint32_t)Hkv, .p3 = (uint32_t)D, .p4 = 0u, .f0 = theta, .f1 = eps };
  int rc = vk_dispatch_locked(VK_PIPE_QWEN3_QK_ROPE, &p, q_in, 4, q_norm_w, 4, NULL, 4, NULL, 4, q_out, 4, NULL, 4, (uint32_t)(L * Hq), 1, 1);
  if (rc == 0) {
    p.p4 = 1u;
    rc = vk_dispatch_locked(VK_PIPE_QWEN3_QK_ROPE, &p, k_in, 4, k_norm_w, 4, NULL, 4, NULL, 4, k_out, 4, NULL, 4, (uint32_t)(L * Hkv), 1, 1);
  }
  if (rc == 0) {
    p.p4 = 2u;
    rc = vk_dispatch_locked(VK_PIPE_QWEN3_QK_ROPE, &p, v_in, 4, NULL, 4, NULL, 4, NULL, 4, v_out, 4, NULL, 4, (uint32_t)(L * Hkv), 1, 1);
  }
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}

int ec_vk_k_qwen3_gqa_attn(
    const float *Q, const float *K, const float *V, const uint8_t *mask,
    uint16_t *O_bf16, int L, int Hq, int Hkv, int D, float scale) {
  if (ec_hal_init() != 0 || D > 128 || (D & 1) || (((uintptr_t)O_bf16) & 3u) || (((uintptr_t)mask) & 3u)) return -1;
  pthread_mutex_lock(&g_vk.mu);
  ec_vk_params_t p = { .p0 = (uint32_t)L, .p1 = (uint32_t)Hq, .p2 = (uint32_t)Hkv, .p3 = (uint32_t)D, .f0 = scale };
  int rc = vk_dispatch_locked(VK_PIPE_QWEN3_GQA_ATTN, &p, Q, 4, K, 4, V, 4, mask, 4, NULL, 4, O_bf16, 4,
                              (uint32_t)(L * Hq), 1, 1);
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}

int ec_vk_f32_to_bf16(const float *src, uint16_t *dst, long n) {
  if (ec_hal_init() != 0 || n <= 0 || (n & 1L) || (((uintptr_t)dst) & 3u)) return -1;
  pthread_mutex_lock(&g_vk.mu);
  ec_vk_params_t p = { .p0 = (uint32_t)n, .p1 = 0u };
  int rc = vk_dispatch_locked(VK_PIPE_PACK16X2, &p, NULL, 4, NULL, 4, src, 4, NULL, 4, NULL, 4, dst, 4,
                              (uint32_t)(((n >> 1) + 255) / 256), 1, 1);
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}

int ec_vk_add_inplace(float *dst, const float *src, long n) {
  if (ec_hal_init() != 0 || n <= 0) return -1;
  pthread_mutex_lock(&g_vk.mu);
  ec_vk_params_t p = { .p0 = (uint32_t)n, .p1 = 0u };
  int rc = vk_dispatch_locked(VK_PIPE_ELEMWISE, &p, NULL, 4, NULL, 4, src, 4, NULL, 4, dst, 4, NULL, 4,
                              (uint32_t)((n + 255) / 256), 1, 1);
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}

int ec_vk_swiglu_pair_bf16(const float *gate, const float *up, uint16_t *out, long n) {
  if (ec_hal_init() != 0 || n <= 0 || (n & 1L) || (((uintptr_t)out) & 3u)) return -1;
  pthread_mutex_lock(&g_vk.mu);
  ec_vk_params_t p = { .p0 = (uint32_t)n, .p1 = 3u };
  int rc = vk_dispatch_locked(VK_PIPE_ELEMWISE, &p, NULL, 4, NULL, 4, gate, 4, up, 4, NULL, 4, out, 4,
                              (uint32_t)(((n >> 1) + 255) / 256), 1, 1);
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}

int ec_vk_flux2_swiglu_bf16(const float *in2h, uint16_t *out_h, int rows, int H) {
  if (ec_hal_init() != 0 || rows <= 0 || H <= 0 || (H & 1) || (((uintptr_t)out_h) & 3u)) return -1;
  pthread_mutex_lock(&g_vk.mu);
  uint32_t pairs = (uint32_t)((long)rows * (H >> 1));
  ec_vk_params_t p = { .p0 = pairs, .p1 = 4u, .p2 = (uint32_t)H };
  int rc = vk_dispatch_locked(VK_PIPE_ELEMWISE, &p, NULL, 4, NULL, 4, in2h, 4, NULL, 4, NULL, 4, out_h, 4,
                              (pairs + 255u) / 256u, 1, 1);
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}
