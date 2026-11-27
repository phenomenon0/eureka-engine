// backends/vulkan/llb_vulkan_gemv.c — Native Cross-Vendor Vulkan 1.2/1.3+ Turbo LLM Engine
// Architecture:
//   - T1: Full GPU-resident single-submit forward graph (static per-slab descriptor sets + 64B push constants)
//   - T2: Batched prefill GEMM (BM=8) + wave32 subgroup & VK_KHR_shader_integer_dot_product (dp4a) GEMV
//   - T5: Disk-backed VkPipelineCache + dedicated VK_QUEUE_TRANSFER_BIT weight slab DMA

#define _POSIX_C_SOURCE 200809L
#include "llb_vulkan_gemv.h"
#include "llb_vulkan_spv.h"
#include "llb_vulkan_turbo_spv.h"
#include "llb_vulkan_turbo_ext_spv.h"
#include "../../host/llb_formats.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <vulkan/vulkan.h>

#ifdef _WIN32
#include <direct.h>
#define lvk_mkdir(p) _mkdir(p)
#else
#define lvk_mkdir(p) mkdir((p), 0755)
#endif

typedef struct {
  uint32_t rows, cols, rb, w_off, row_base, p5, p6, p7;
} llb_vk_params_t;

typedef struct {
  uint32_t p0, p1, p2, p3;
  uint32_t p4, p5, p6, p7;
  uint32_t p8, p9, p10, p11;
  float    f0, f1, f2, f3;
} lvk_pc_t;

typedef enum {
  LVK_PIPE_Q4K = 0, LVK_PIPE_Q5K = 1, LVK_PIPE_Q6K = 2, LVK_PIPE_Q8_0 = 3, LVK_PIPE_F32 = 4,
  LVK_PIPE_COUNT = 5
} llb_vk_pipe_t;

typedef enum {
  GP_DEQ_EMBED = 0,
  GP_RMS_NORM,
  GP_QUANT_X_Q8,
  GP_FULL_QK_ROPE_KV,
  GP_GQA_ATTN_GATE,
  GP_LIN_CONV_SILU,
  GP_LIN_DELTA_NET,
  GP_SWIGLU_MUL,
  GP_RES_ADD,
  GP_ARGMAX_TOP2,
  // Decode GEMV (BM=1)
  GP_GEMV_Q4K,
  GP_GEMV_Q5K,
  GP_GEMV_Q6K,
  GP_GEMV_Q8_0,
  GP_GEMV_F32,
  // Batched Prefill GEMM (BM=8)
  GP_GEMM_Q4K,
  GP_GEMM_Q5K,
  GP_GEMM_Q6K,
  GP_GEMM_Q8_0,
  GP_GEMM_F32,
  GP_COUNT
} lvk_gp_t;

typedef struct {
  VkBuffer       buf;
  VkDeviceMemory mem;
  size_t         size;
} llb_vk_slab_t;

typedef struct {
  uint64_t pack_off;
  uint64_t bytes;
  int      slab_idx;
  uint64_t slab_off;
} llb_vk_entry_t;

typedef struct {
  int      slab_idx;
  uint32_t slab_bo;   // byte offset inside slab (< 1 GiB)
  int      rows, cols, rb, fmt;
} lvk_gtensor_t;

#define LVK_MAX_SLABS       64
#define LVK_MAX_ENTRIES     2048
#define LVK_CHUNK_SLOTS     16
#define LVK_UBO_STRIDE      256
#define LVK_MAX_WG_X        32768u
#define LVK_STAGE_BYTES     (32ULL << 20)
#define LVK_PREFILL_CHUNK   32
#define LVK_PREFILL_BM      8

static struct {
  int                               active;
  int                               is_unified;
  int                               has_vk13;
  int                               has_subgroups;
  int                               has_dp4a;
  int                               has_sub_ctrl;
  int                               use_dp4a;
  int                               use_subgroups;
  uint32_t                          sub_size;       // effective subgroup size (32 when pinned)
  uint32_t                          rows_per_wg;    // 128 / sub_size (4 on wave32)
  const uint8_t                    *pack_base;
  size_t                            pack_bytes;

  VkInstance                        instance;
  VkPhysicalDevice                  phys;
  VkDevice                          dev;
  VkQueue                           queue;
  uint32_t                          qfam;
  VkQueue                           xfer_queue;
  uint32_t                          xfer_qfam;
  int                               has_ded_xfer;
  VkPhysicalDeviceProperties        props;
  VkPhysicalDeviceMemoryProperties  mem_props;
  size_t                            vram_total;
  size_t                            vram_used;
  char                              info[512];

  VkPipelineCache                   pipe_cache;
  char                              cache_path[512];
  int                               cache_dirty;

  VkCommandPool                     cmd_pool;
  VkCommandBuffer                   cmd;
  VkFence                           fence;
  VkCommandPool                     xfer_pool;
  VkCommandBuffer                   xfer_cmd;
  VkFence                           xfer_fence;

  // Legacy per-GEMV fallback resources
  VkDescriptorSetLayout             dsl;
  VkPipelineLayout                  pipe_layout;
  VkDescriptorPool                  desc_pool;
  VkDescriptorSet                   desc_sets[LVK_CHUNK_SLOTS];
  VkShaderModule                    shaders[LVK_PIPE_COUNT];
  VkPipeline                        pipes[LVK_PIPE_COUNT];
  VkBuffer                          ubo_buf;
  VkDeviceMemory                    ubo_mem;
  uint8_t                          *ubo_map;
  VkBuffer                          x_host_buf, x_dev_buf;
  VkDeviceMemory                    x_host_mem, x_dev_mem;
  float                            *x_host_map;
  size_t                            x_cap_bytes;
  VkBuffer                          y_host_buf, y_dev_buf;
  VkDeviceMemory                    y_host_mem, y_dev_mem;
  float                            *y_host_map;
  size_t                            y_cap_bytes;

  llb_vk_slab_t                     slabs[LVK_MAX_SLABS];
  int                               n_slabs;
  llb_vk_entry_t                    entries[LVK_MAX_ENTRIES];
  int                               n_entries;

  // ---- T1/T2 Full GPU Graph state ----
  int                               graph_ok;
  llb_vk_model_cfg_t                cfg;
  llb_vk_model_ids_t                ids;
  lvk_gtensor_t                    *gt;
  int                               n_gt;

  VkDescriptorSetLayout             g_dsl;
  VkPipelineLayout                  g_pipe_layout;
  VkDescriptorPool                  g_desc_pool;
  VkDescriptorSet                   g_slab_ds[LVK_MAX_SLABS];
  VkShaderModule                    g_shaders[GP_COUNT];
  VkPipeline                        g_pipes[GP_COUNT];
  int                               g_cur_slab;

  // DEVICE_LOCAL GPU buffers for full graph
  VkBuffer                          act_buf, state_buf, kv_buf, xq8_buf;
  VkDeviceMemory                    act_mem, state_mem, kv_mem, xq8_mem;
  size_t                            act_bytes, state_bytes, kv_bytes, xq8_bytes;

  // Small host-visible transfer buffers (toks/constants upload + logits/top2 readback)
  VkBuffer                          io_host_buf;
  VkDeviceMemory                    io_host_mem;
  uint8_t                          *io_host_map;
  size_t                            io_host_bytes;

  // Fixed float offsets inside act_buf
  uint32_t off_X, off_HH, off_QKV, off_Z, off_BETA, off_ALPHA, off_GATE_PRE, off_Y;
  uint32_t off_QGN, off_QN, off_KB, off_VB, off_ATTN, off_SCR, off_OUT0, off_LOGITS;
  uint32_t off_INV_FREQ, off_TOK_IDS, off_TOP2, off_META_FULL, off_META_CONV, off_META_DELTA;
  uint32_t st_conv_base, kv_v_base;
} g_lvk;

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

static int find_mem_type(uint32_t bits, VkMemoryPropertyFlags req, VkMemoryPropertyFlags avoid) {
  for (uint32_t i = 0; i < g_lvk.mem_props.memoryTypeCount; i++) {
    if (!(bits & (1u << i))) continue;
    VkMemoryPropertyFlags f = g_lvk.mem_props.memoryTypes[i].propertyFlags;
    if ((f & req) == req && (f & avoid) == 0) return (int)i;
  }
  return -1;
}

static int alloc_host_buf(size_t sz, VkBufferUsageFlags usage, int prefer_unified,
                          VkBuffer *buf, VkDeviceMemory *mem, void **map) {
  sz = (sz + 255ULL) & ~255ULL;
  if (sz < 256) sz = 256;
  VkBufferCreateInfo bci = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = sz, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  if (vkCreateBuffer(g_lvk.dev, &bci, NULL, buf) != VK_SUCCESS) return -1;
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(g_lvk.dev, *buf, &req);

  int mt = -1;
  if (prefer_unified) {
    mt = find_mem_type(req.memoryTypeBits,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
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
  if (mt < 0) { vkDestroyBuffer(g_lvk.dev, *buf, NULL); return -1; }

  VkMemoryAllocateInfo mai = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize = req.size, .memoryTypeIndex = (uint32_t)mt,
  };
  if (vkAllocateMemory(g_lvk.dev, &mai, NULL, mem) != VK_SUCCESS ||
      vkBindBufferMemory(g_lvk.dev, *buf, *mem, 0) != VK_SUCCESS ||
      vkMapMemory(g_lvk.dev, *mem, 0, sz, 0, map) != VK_SUCCESS) {
    if (*mem) vkFreeMemory(g_lvk.dev, *mem, NULL);
    vkDestroyBuffer(g_lvk.dev, *buf, NULL);
    *buf = VK_NULL_HANDLE; *mem = VK_NULL_HANDLE;
    return -1;
  }
  return 0;
}

static int alloc_dev_buf(size_t sz, VkBufferUsageFlags usage, VkBuffer *buf, VkDeviceMemory *mem) {
  sz = (sz + 255ULL) & ~255ULL;
  if (sz < 256) sz = 256;
  uint32_t qfams[2] = { g_lvk.qfam, g_lvk.xfer_qfam };
  VkBufferCreateInfo bci = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = sz, .usage = usage,
    .sharingMode = g_lvk.has_ded_xfer ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
    .queueFamilyIndexCount = g_lvk.has_ded_xfer ? 2u : 0u,
    .pQueueFamilyIndices = g_lvk.has_ded_xfer ? qfams : NULL,
  };
  if (vkCreateBuffer(g_lvk.dev, &bci, NULL, buf) != VK_SUCCESS) return -1;
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(g_lvk.dev, *buf, &req);

  int mt = -1;
  if (g_lvk.is_unified) {
    mt = find_mem_type(req.memoryTypeBits,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
  }
  if (mt < 0) mt = find_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  if (mt < 0) mt = find_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
  if (mt < 0) { vkDestroyBuffer(g_lvk.dev, *buf, NULL); return -1; }

  VkMemoryAllocateInfo mai = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize = req.size, .memoryTypeIndex = (uint32_t)mt,
  };
  if (vkAllocateMemory(g_lvk.dev, &mai, NULL, mem) != VK_SUCCESS ||
      vkBindBufferMemory(g_lvk.dev, *buf, *mem, 0) != VK_SUCCESS) {
    if (*mem) vkFreeMemory(g_lvk.dev, *mem, NULL);
    vkDestroyBuffer(g_lvk.dev, *buf, NULL);
    *buf = VK_NULL_HANDLE; *mem = VK_NULL_HANDLE;
    return -1;
  }
  return 0;
}

// ---- T5: Disk-backed VkPipelineCache ----
static void lvk_init_pipe_cache(void) {
  char dir[400] = {0};
#ifdef _WIN32
  const char *base = getenv("LOCALAPPDATA");
  if (!base || !base[0]) base = ".";
  snprintf(dir, sizeof dir, "%s\\Eureka", base);
  lvk_mkdir(dir);
  snprintf(dir, sizeof dir, "%s\\Eureka\\cache", base);
  lvk_mkdir(dir);
  snprintf(g_lvk.cache_path, sizeof g_lvk.cache_path, "%s\\vk_llm_%04x_%04x_%08x.bin",
           dir, g_lvk.props.vendorID, g_lvk.props.deviceID, g_lvk.props.driverVersion);
#else
  const char *xdg = getenv("XDG_CACHE_HOME");
  const char *home = getenv("HOME");
  if (xdg && xdg[0]) {
    snprintf(dir, sizeof dir, "%s/eureka", xdg);
    lvk_mkdir(xdg);
    lvk_mkdir(dir);
  } else if (home && home[0]) {
    snprintf(dir, sizeof dir, "%s/.cache", home);
    lvk_mkdir(dir);
    snprintf(dir, sizeof dir, "%s/.cache/eureka", home);
    lvk_mkdir(dir);
  } else {
    snprintf(dir, sizeof dir, "/tmp/eureka-vk-cache");
    lvk_mkdir(dir);
  }
  snprintf(g_lvk.cache_path, sizeof g_lvk.cache_path, "%s/vk_llm_%04x_%04x_%08x.bin",
           dir, g_lvk.props.vendorID, g_lvk.props.deviceID, g_lvk.props.driverVersion);
#endif

  void *blob = NULL;
  size_t blob_sz = 0;
  FILE *f = fopen(g_lvk.cache_path, "rb");
  if (f) {
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz > 32 && sz < (64L << 20)) {
      blob = malloc((size_t)sz);
      if (blob && fread(blob, 1, (size_t)sz, f) == (size_t)sz) {
        // Validate Vulkan pipeline cache header (vendorID @ +8, deviceID @ +12, UUID @ +16)
        const uint32_t *hdr = (const uint32_t *)blob;
        if (hdr[2] == g_lvk.props.vendorID && hdr[3] == g_lvk.props.deviceID &&
            memcmp((const uint8_t *)blob + 16, g_lvk.props.pipelineCacheUUID, VK_UUID_SIZE) == 0) {
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
  if (vkCreatePipelineCache(g_lvk.dev, &pcci, NULL, &g_lvk.pipe_cache) != VK_SUCCESS) {
    g_lvk.pipe_cache = VK_NULL_HANDLE;
  }
  free(blob);
}

static void lvk_save_pipe_cache(void) {
  if (!g_lvk.pipe_cache || !g_lvk.cache_path[0] || !g_lvk.cache_dirty) return;
  size_t sz = 0;
  if (vkGetPipelineCacheData(g_lvk.dev, g_lvk.pipe_cache, &sz, NULL) != VK_SUCCESS || sz == 0 || sz > (64ULL << 20))
    return;
  void *buf = malloc(sz);
  if (!buf) return;
  if (vkGetPipelineCacheData(g_lvk.dev, g_lvk.pipe_cache, &sz, buf) == VK_SUCCESS) {
    char tmp[540];
    snprintf(tmp, sizeof tmp, "%s.tmp", g_lvk.cache_path);
    FILE *f = fopen(tmp, "wb");
    if (f) {
      if (fwrite(buf, 1, sz, f) == sz) {
        fclose(f);
        rename(tmp, g_lvk.cache_path);
        g_lvk.cache_dirty = 0;
      } else {
        fclose(f);
        remove(tmp);
      }
    }
  }
  free(buf);
}

static int cmp_span(const void *a, const void *b) {
  uint64_t oa = ((const llb_vk_tensor_span_t *)a)->off;
  uint64_t ob = ((const llb_vk_tensor_span_t *)b)->off;
  return (oa < ob) ? -1 : (oa > ob) ? 1 : 0;
}

static int upload_slab_chunked(VkBuffer dst_buf, size_t dst_off, const uint8_t *src, size_t bytes,
                               VkBuffer stage_buf, uint8_t *stage_map) {
  VkCommandBuffer cb = g_lvk.has_ded_xfer ? g_lvk.xfer_cmd : g_lvk.cmd;
  VkQueue q = g_lvk.has_ded_xfer ? g_lvk.xfer_queue : g_lvk.queue;
  VkFence fn = g_lvk.has_ded_xfer ? g_lvk.xfer_fence : g_lvk.fence;
  size_t pos = 0;
  while (pos < bytes) {
    size_t chunk = bytes - pos;
    if (chunk > LVK_STAGE_BYTES) chunk = LVK_STAGE_BYTES;
    memcpy(stage_map, src + pos, chunk);

    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cb, &bi);
    VkBufferCopy cp = { .srcOffset = 0, .dstOffset = dst_off + pos, .size = chunk };
    vkCmdCopyBuffer(cb, stage_buf, dst_buf, 1, &cp);
    vkEndCommandBuffer(cb);

    vkResetFences(g_lvk.dev, 1, &fn);
    VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cb,
    };
    if (vkQueueSubmit(q, 1, &si, fn) != VK_SUCCESS ||
        vkWaitForFences(g_lvk.dev, 1, &fn, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
      return -1;
    }
    pos += chunk;
  }
  return 0;
}

static int dev_has_ext(VkPhysicalDevice phys, const char *name) {
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

int llb_vk_init(const uint8_t *pack_base, size_t pack_bytes,
                const llb_vk_tensor_span_t *spans, int n_spans,
                int max_rows, int max_cols) {
  const char *off_env = getenv("LLB_VULKAN_OFF");
  if (off_env && off_env[0] == '1') return -1;
  if (!pack_base || pack_bytes == 0 || !spans || n_spans <= 0) return -1;

  memset(&g_lvk, 0, sizeof g_lvk);
  g_lvk.pack_base = pack_base;
  g_lvk.pack_bytes = pack_bytes;
  g_lvk.g_cur_slab = -1;

  // Negotiate Vulkan 1.3 if loader supports it, else 1.2
  uint32_t inst_ver = VK_API_VERSION_1_2;
  PFN_vkEnumerateInstanceVersion fn_ver = (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(NULL, "vkEnumerateInstanceVersion");
  if (fn_ver) {
    uint32_t v = 0;
    if (fn_ver(&v) == VK_SUCCESS && v >= VK_API_VERSION_1_3) inst_ver = VK_API_VERSION_1_3;
  }

  VkApplicationInfo app = {
    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .pApplicationName = "eureka-engine-vulkan",
    .applicationVersion = VK_MAKE_VERSION(2, 0, 0),
    .pEngineName = "eureka-llm-turbo",
    .engineVersion = VK_MAKE_VERSION(2, 0, 0),
    .apiVersion = inst_ver,
  };
  VkInstanceCreateInfo ici = {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pApplicationInfo = &app,
  };
  if (vkCreateInstance(&ici, NULL, &g_lvk.instance) != VK_SUCCESS) return -1;

  uint32_t n_phys = 0;
  vkEnumeratePhysicalDevices(g_lvk.instance, &n_phys, NULL);
  if (n_phys == 0 || n_phys > 32) return -1;
  VkPhysicalDevice devs[32];
  vkEnumeratePhysicalDevices(g_lvk.instance, &n_phys, devs);

  const char *want = getenv("LLB_VULKAN_DEVICE");
  if (!want || !want[0]) want = getenv("EC_VULKAN_DEVICE");
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
  g_lvk.phys = devs[chosen];
  vkGetPhysicalDeviceProperties(g_lvk.phys, &g_lvk.props);
  vkGetPhysicalDeviceMemoryProperties(g_lvk.phys, &g_lvk.mem_props);

  if (g_lvk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU && !(want && want[0])) {
    return -1;
  }

  g_lvk.has_vk13 = (inst_ver >= VK_API_VERSION_1_3 && g_lvk.props.apiVersion >= VK_API_VERSION_1_3);

  // Probe subgroup properties & Vulkan 1.3 features (subgroupSizeControl + shaderIntegerDotProduct)
  VkPhysicalDeviceSubgroupSizeControlProperties sub_ctrl_props = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
  };
  VkPhysicalDeviceSubgroupProperties sub_props = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    .pNext = g_lvk.has_vk13 ? &sub_ctrl_props : NULL,
  };
  VkPhysicalDeviceProperties2 props2 = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
    .pNext = &sub_props,
  };
  vkGetPhysicalDeviceProperties2(g_lvk.phys, &props2);

  g_lvk.has_subgroups = (g_lvk.has_vk13 &&
                         (sub_props.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
                         (sub_props.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) &&
                         (sub_props.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT)) ? 1 : 0;

  VkPhysicalDeviceVulkan13Features feat13 = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
  };
  VkPhysicalDeviceFeatures2 feat2 = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    .pNext = g_lvk.has_vk13 ? &feat13 : NULL,
  };
  vkGetPhysicalDeviceFeatures2(g_lvk.phys, &feat2);

  g_lvk.has_dp4a = (g_lvk.has_vk13 && feat13.shaderIntegerDotProduct) ? 1 : 0;
  g_lvk.has_sub_ctrl = (g_lvk.has_vk13 && feat13.subgroupSizeControl &&
                        (sub_ctrl_props.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
                        sub_ctrl_props.minSubgroupSize <= 32 && sub_ctrl_props.maxSubgroupSize >= 32) ? 1 : 0;

  const char *exact_env = getenv("LLB_VULKAN_EXACT");
  int exact_mode = (exact_env && exact_env[0] == '1');
  g_lvk.use_subgroups = g_lvk.has_subgroups;
  g_lvk.use_dp4a = (g_lvk.has_subgroups && g_lvk.has_dp4a && !exact_mode) ? 1 : 0;
  if (g_lvk.use_subgroups) {
    g_lvk.sub_size = g_lvk.has_sub_ctrl ? 32u : sub_props.subgroupSize;
    if (g_lvk.sub_size == 0 || g_lvk.sub_size > 128) g_lvk.sub_size = 32u;
    g_lvk.rows_per_wg = 128u / g_lvk.sub_size;
    if (g_lvk.rows_per_wg == 0) g_lvk.rows_per_wg = 1u;
  } else {
    g_lvk.sub_size = 32u;
    g_lvk.rows_per_wg = 4u;
  }

  for (uint32_t i = 0; i < g_lvk.mem_props.memoryHeapCount; i++) {
    if (g_lvk.mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      if (g_lvk.mem_props.memoryHeaps[i].size > g_lvk.vram_total)
        g_lvk.vram_total = (size_t)g_lvk.mem_props.memoryHeaps[i].size;
    }
  }
  for (uint32_t i = 0; i < g_lvk.mem_props.memoryTypeCount; i++) {
    VkMemoryPropertyFlags f = g_lvk.mem_props.memoryTypes[i].propertyFlags;
    VkMemoryPropertyFlags uni = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    if ((f & uni) == uni) { g_lvk.is_unified = 1; break; }
  }

  // Pick compute queue family + optional dedicated transfer queue family (T5)
  uint32_t n_qf = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(g_lvk.phys, &n_qf, NULL);
  VkQueueFamilyProperties qfs[32];
  if (n_qf > 32) n_qf = 32;
  vkGetPhysicalDeviceQueueFamilyProperties(g_lvk.phys, &n_qf, qfs);
  g_lvk.qfam = 0;
  for (uint32_t i = 0; i < n_qf; i++) {
    if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_lvk.qfam = i; break; }
  }
  g_lvk.xfer_qfam = g_lvk.qfam;
  g_lvk.has_ded_xfer = 0;
  for (uint32_t i = 0; i < n_qf; i++) {
    if ((qfs[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
        !(qfs[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
      g_lvk.xfer_qfam = i;
      g_lvk.has_ded_xfer = 1;
      break;
    }
  }

  float qprio = 1.0f;
  VkDeviceQueueCreateInfo qcis[2];
  uint32_t n_qci = 1;
  qcis[0] = (VkDeviceQueueCreateInfo){
    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .queueFamilyIndex = g_lvk.qfam, .queueCount = 1, .pQueuePriorities = &qprio,
  };
  if (g_lvk.has_ded_xfer) {
    qcis[1] = (VkDeviceQueueCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = g_lvk.xfer_qfam, .queueCount = 1, .pQueuePriorities = &qprio,
    };
    n_qci = 2;
  }

  VkPhysicalDeviceVulkan13Features en_feat13 = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
    .subgroupSizeControl = g_lvk.has_sub_ctrl ? VK_TRUE : VK_FALSE,
    .shaderIntegerDotProduct = g_lvk.has_dp4a ? VK_TRUE : VK_FALSE,
  };
  VkDeviceCreateInfo dci = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .pNext = g_lvk.has_vk13 ? &en_feat13 : NULL,
    .queueCreateInfoCount = n_qci,
    .pQueueCreateInfos = qcis,
  };
  if (vkCreateDevice(g_lvk.phys, &dci, NULL, &g_lvk.dev) != VK_SUCCESS) return -1;
  vkGetDeviceQueue(g_lvk.dev, g_lvk.qfam, 0, &g_lvk.queue);
  if (g_lvk.has_ded_xfer) vkGetDeviceQueue(g_lvk.dev, g_lvk.xfer_qfam, 0, &g_lvk.xfer_queue);

  lvk_init_pipe_cache();

  VkCommandPoolCreateInfo cpci = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = g_lvk.qfam,
  };
  vkCreateCommandPool(g_lvk.dev, &cpci, NULL, &g_lvk.cmd_pool);
  VkCommandBufferAllocateInfo cbai = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = g_lvk.cmd_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
  };
  vkAllocateCommandBuffers(g_lvk.dev, &cbai, &g_lvk.cmd);
  VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
  vkCreateFence(g_lvk.dev, &fci, NULL, &g_lvk.fence);

  if (g_lvk.has_ded_xfer) {
    VkCommandPoolCreateInfo xcpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = g_lvk.xfer_qfam,
    };
    vkCreateCommandPool(g_lvk.dev, &xcpci, NULL, &g_lvk.xfer_pool);
    VkCommandBufferAllocateInfo xcbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = g_lvk.xfer_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
    };
    vkAllocateCommandBuffers(g_lvk.dev, &xcbai, &g_lvk.xfer_cmd);
    vkCreateFence(g_lvk.dev, &fci, NULL, &g_lvk.xfer_fence);
  }

  // Legacy fallback UBO & x/y buffers
  void *um = NULL;
  if (alloc_host_buf((size_t)LVK_CHUNK_SLOTS * LVK_UBO_STRIDE, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     0, &g_lvk.ubo_buf, &g_lvk.ubo_mem, &um) != 0) return -1;
  g_lvk.ubo_map = (uint8_t *)um;
  g_lvk.x_cap_bytes = ((size_t)(max_cols > 4096 ? max_cols : 4096) * sizeof(float) + 4095ULL) & ~4095ULL;
  g_lvk.y_cap_bytes = ((size_t)(max_rows > 65536 ? max_rows : 65536) * sizeof(float) + 4095ULL) & ~4095ULL;
  void *xm = NULL, *ym = NULL;
  if (alloc_host_buf(g_lvk.x_cap_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     g_lvk.is_unified, &g_lvk.x_host_buf, &g_lvk.x_host_mem, &xm) != 0) return -1;
  if (alloc_host_buf(g_lvk.y_cap_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     g_lvk.is_unified, &g_lvk.y_host_buf, &g_lvk.y_host_mem, &ym) != 0) return -1;
  g_lvk.x_host_map = (float *)xm;
  g_lvk.y_host_map = (float *)ym;
  if (g_lvk.is_unified) {
    g_lvk.x_dev_buf = g_lvk.x_host_buf;
    g_lvk.y_dev_buf = g_lvk.y_host_buf;
  } else {
    if (alloc_dev_buf(g_lvk.x_cap_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      &g_lvk.x_dev_buf, &g_lvk.x_dev_mem) != 0) return -1;
    if (alloc_dev_buf(g_lvk.y_cap_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      &g_lvk.y_dev_buf, &g_lvk.y_dev_mem) != 0) return -1;
  }

  VkDescriptorSetLayoutBinding binds[4];
  memset(binds, 0, sizeof binds);
  binds[0].binding = 0; binds[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; binds[0].descriptorCount = 1; binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  for (int i = 1; i <= 3; i++) {
    binds[i].binding = (uint32_t)i;
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslci = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 4, .pBindings = binds,
  };
  vkCreateDescriptorSetLayout(g_lvk.dev, &dslci, NULL, &g_lvk.dsl);
  VkPipelineLayoutCreateInfo plci = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &g_lvk.dsl,
  };
  vkCreatePipelineLayout(g_lvk.dev, &plci, NULL, &g_lvk.pipe_layout);

  VkDescriptorPoolSize dps[2] = {
    { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, LVK_CHUNK_SLOTS },
    { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, LVK_CHUNK_SLOTS * 3 },
  };
  VkDescriptorPoolCreateInfo dpci = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .maxSets = LVK_CHUNK_SLOTS, .poolSizeCount = 2, .pPoolSizes = dps,
  };
  vkCreateDescriptorPool(g_lvk.dev, &dpci, NULL, &g_lvk.desc_pool);
  VkDescriptorSetLayout dsls[LVK_CHUNK_SLOTS];
  for (int i = 0; i < LVK_CHUNK_SLOTS; i++) dsls[i] = g_lvk.dsl;
  VkDescriptorSetAllocateInfo dsai = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = g_lvk.desc_pool, .descriptorSetCount = LVK_CHUNK_SLOTS, .pSetLayouts = dsls,
  };
  vkAllocateDescriptorSets(g_lvk.dev, &dsai, g_lvk.desc_sets);

  // Deduplicate & pack weight tensors into <=1 GiB DEVICE_LOCAL VRAM slabs
  llb_vk_tensor_span_t *sorted = (llb_vk_tensor_span_t *)malloc((size_t)n_spans * sizeof *sorted);
  if (!sorted) return -1;
  memcpy(sorted, spans, (size_t)n_spans * sizeof *sorted);
  qsort(sorted, (size_t)n_spans, sizeof *sorted, cmp_span);

  llb_vk_tensor_span_t *uniq = (llb_vk_tensor_span_t *)malloc((size_t)n_spans * sizeof *uniq);
  int n_uniq = 0;
  for (int i = 0; i < n_spans; i++) {
    if (sorted[i].bytes == 0 || sorted[i].off + sorted[i].bytes > pack_bytes) continue;
    if (n_uniq > 0 && sorted[i].off == uniq[n_uniq - 1].off) {
      if (sorted[i].bytes > uniq[n_uniq - 1].bytes) uniq[n_uniq - 1].bytes = sorted[i].bytes;
      continue;
    }
    if (n_uniq < LVK_MAX_ENTRIES) uniq[n_uniq++] = sorted[i];
  }
  free(sorted);

  size_t max_slab = 1024ULL << 20;
  if (g_lvk.props.limits.maxStorageBufferRange > 0 &&
      (size_t)g_lvk.props.limits.maxStorageBufferRange < max_slab) {
    max_slab = (size_t)g_lvk.props.limits.maxStorageBufferRange & ~255ULL;
  }
  size_t slab_sizes[LVK_MAX_SLABS];
  memset(slab_sizes, 0, sizeof slab_sizes);
  // First pass: pack small/1D/conv tensors (<=1 MiB: all norms, biases, ssm_dt, ssm_a, ssm_conv1d weights)
  // into slab 0 so multi-weight layer ops (qnw+knw, cw+dt+al) never straddle a 1 GiB slab boundary.
  int cur_slab = 0;
  for (int pass = 0; pass < 2; pass++) {
    for (int i = 0; i < n_uniq; i++) {
      int is_small = (uniq[i].bytes <= (1ULL << 20));
      if ((pass == 0 && !is_small) || (pass == 1 && is_small)) continue;
      size_t tsz = ((size_t)uniq[i].bytes + 255ULL) & ~255ULL;
      if (slab_sizes[cur_slab] > 0 && slab_sizes[cur_slab] + tsz > max_slab) {
        cur_slab++;
        if (cur_slab >= LVK_MAX_SLABS) break;
      }
      g_lvk.entries[g_lvk.n_entries++] = (llb_vk_entry_t){
        .pack_off = uniq[i].off, .bytes = uniq[i].bytes,
        .slab_idx = cur_slab, .slab_off = slab_sizes[cur_slab],
      };
      slab_sizes[cur_slab] += tsz;
    }
  }
  int total_slabs = (g_lvk.n_entries > 0) ? (cur_slab + 1) : 0;
  free(uniq);
  // Re-sort entries by pack_off so binary search in find_entry() remains O(log N) exact!
  for (int i = 1; i < g_lvk.n_entries; i++) {
    llb_vk_entry_t key = g_lvk.entries[i];
    int j = i - 1;
    while (j >= 0 && g_lvk.entries[j].pack_off > key.pack_off) {
      g_lvk.entries[j + 1] = g_lvk.entries[j];
      j--;
    }
    g_lvk.entries[j + 1] = key;
  }

  VkBuffer stage_buf = VK_NULL_HANDLE;
  VkDeviceMemory stage_mem = VK_NULL_HANDLE;
  void *stage_map = NULL;
  if (alloc_host_buf(LVK_STAGE_BYTES, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 0,
                     &stage_buf, &stage_mem, &stage_map) != 0) return -1;

  size_t uploaded_bytes = 0;
  int resident_slabs = 0;
  for (int s = 0; s < total_slabs; s++) {
    if (alloc_dev_buf(slab_sizes[s], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      &g_lvk.slabs[s].buf, &g_lvk.slabs[s].mem) != 0) break;
    g_lvk.slabs[s].size = slab_sizes[s];
    g_lvk.vram_used += slab_sizes[s];
    resident_slabs = s + 1;
  }
  g_lvk.n_slabs = resident_slabs;

  int resident_tensors = 0;
  for (int i = 0; i < g_lvk.n_entries; i++) {
    llb_vk_entry_t *e = &g_lvk.entries[i];
    if (e->slab_idx >= g_lvk.n_slabs) continue;
    if (upload_slab_chunked(g_lvk.slabs[e->slab_idx].buf, (size_t)e->slab_off,
                            pack_base + e->pack_off, (size_t)e->bytes,
                            stage_buf, (uint8_t *)stage_map) != 0) {
      e->slab_idx = -1;
      continue;
    }
    uploaded_bytes += (size_t)e->bytes;
    resident_tensors++;
  }

  vkUnmapMemory(g_lvk.dev, stage_mem);
  vkDestroyBuffer(g_lvk.dev, stage_buf, NULL);
  vkFreeMemory(g_lvk.dev, stage_mem, NULL);

  if (resident_tensors == 0) return -1;
  g_lvk.active = 1;
  snprintf(g_lvk.info, sizeof g_lvk.info,
           "%s (Vulkan %u.%u, %s, %s, w%u%s, %.2f GiB VRAM across %d slab%s, %d/%d tensors)",
           g_lvk.props.deviceName,
           VK_API_VERSION_MAJOR(g_lvk.props.apiVersion), VK_API_VERSION_MINOR(g_lvk.props.apiVersion),
           g_lvk.is_unified ? "Unified" : "DEVICE_LOCAL",
           g_lvk.use_dp4a ? "dp4a+subgroup" : (g_lvk.use_subgroups ? "fp32-subgroup" : "portable-spv13"),
           g_lvk.sub_size,
           g_lvk.has_ded_xfer ? "+dma" : "",
           (double)uploaded_bytes / (1024.0 * 1024.0 * 1024.0),
           g_lvk.n_slabs, g_lvk.n_slabs == 1 ? "" : "s",
           resident_tensors, g_lvk.n_entries);
  (void)dev_has_ext;
  return 0;
}

static const llb_vk_entry_t *find_entry(uint64_t pack_off) {
  int lo = 0, hi = g_lvk.n_entries - 1;
  while (lo <= hi) {
    int mid = (lo + hi) >> 1;
    const llb_vk_entry_t *e = &g_lvk.entries[mid];
    if (pack_off < e->pack_off) hi = mid - 1;
    else if (pack_off >= e->pack_off + e->bytes) lo = mid + 1;
    else return e;
  }
  return NULL;
}

static int ensure_legacy_pipe(llb_vk_pipe_t pid) {
  if (g_lvk.pipes[pid] != VK_NULL_HANDLE) return 0;
  static const struct { const uint32_t *code; size_t bytes; const char *entry; } specs[LVK_PIPE_COUNT] = {
    { ec_spv_vk_gemv_q4k,  sizeof(ec_spv_vk_gemv_q4k),  "vk_gemv_q4k"  },
    { ec_spv_vk_gemv_q5k,  sizeof(ec_spv_vk_gemv_q5k),  "vk_gemv_q5k"  },
    { ec_spv_vk_gemv_q6k,  sizeof(ec_spv_vk_gemv_q6k),  "vk_gemv_q6k"  },
    { ec_spv_vk_gemv_q8_0, sizeof(ec_spv_vk_gemv_q8_0), "vk_gemv_q8_0" },
    { ec_spv_vk_gemv_f32,  sizeof(ec_spv_vk_gemv_f32),  "vk_gemv_f32"  },
  };
  VkShaderModuleCreateInfo smci = {
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = specs[pid].bytes, .pCode = specs[pid].code,
  };
  if (vkCreateShaderModule(g_lvk.dev, &smci, NULL, &g_lvk.shaders[pid]) != VK_SUCCESS) return -1;
  VkComputePipelineCreateInfo cp = {
    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = g_lvk.shaders[pid], .pName = specs[pid].entry,
    },
    .layout = g_lvk.pipe_layout,
  };
  if (vkCreateComputePipelines(g_lvk.dev, g_lvk.pipe_cache, 1, &cp, NULL, &g_lvk.pipes[pid]) != VK_SUCCESS) return -1;
  g_lvk.cache_dirty = 1;
  return 0;
}

int llb_vk_gemv(int fmt, const uint8_t *W, int rb, int rows, int cols, const float *x, float *y) {
  if (!g_lvk.active || !W || !x || !y || rows <= 0 || cols <= 0) return 0;
  llb_vk_pipe_t pid;
  switch (fmt) {
    case LLB_FM_Q4_K: pid = LVK_PIPE_Q4K;  if (cols & 255) return 0; break;
    case LLB_FM_Q5_K: pid = LVK_PIPE_Q5K;  if (cols & 255) return 0; break;
    case LLB_FM_Q6_K: pid = LVK_PIPE_Q6K;  if (cols & 255) return 0; break;
    case LLB_FM_Q8_0: pid = LVK_PIPE_Q8_0; if (cols & 31)  return 0; break;
    case LLB_FM_F32:  pid = LVK_PIPE_F32;  break;
    default: return 0;
  }
  if (ensure_legacy_pipe(pid) != 0) return 0;
  if ((uintptr_t)W < (uintptr_t)g_lvk.pack_base) return 0;
  uint64_t pack_off = (uint64_t)((uintptr_t)W - (uintptr_t)g_lvk.pack_base);
  if (pack_off >= g_lvk.pack_bytes) return 0;
  const llb_vk_entry_t *e = find_entry(pack_off);
  if (!e || e->slab_idx < 0 || e->slab_idx >= g_lvk.n_slabs) return 0;

  size_t x_bytes = (size_t)cols * sizeof(float);
  size_t y_bytes = (size_t)rows * sizeof(float);
  if (x_bytes > g_lvk.x_cap_bytes || y_bytes > g_lvk.y_cap_bytes) return 0;

  uint64_t w_slab_off = e->slab_off + (pack_off - e->pack_off);
  size_t align = (size_t)g_lvk.props.limits.minStorageBufferOffsetAlignment;
  if (align < 256) align = 256;
  uint64_t desc_off = w_slab_off & ~(uint64_t)(align - 1ULL);
  uint32_t sub_off = (uint32_t)(w_slab_off - desc_off);
  uint64_t total_w_bytes = (uint64_t)sub_off + (uint64_t)rows * (uint64_t)rb;
  if (total_w_bytes > 0xFFFFFFFFULL) return 0;

  memcpy(g_lvk.x_host_map, x, x_bytes);

  vkResetCommandBuffer(g_lvk.cmd, 0);
  VkCommandBufferBeginInfo bi = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  vkBeginCommandBuffer(g_lvk.cmd, &bi);

  if (!g_lvk.is_unified) {
    VkBufferCopy xcp = { .srcOffset = 0, .dstOffset = 0, .size = x_bytes };
    vkCmdCopyBuffer(g_lvk.cmd, g_lvk.x_host_buf, g_lvk.x_dev_buf, 1, &xcp);
    VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    vkCmdPipelineBarrier(g_lvk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
  }

  vkCmdBindPipeline(g_lvk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_lvk.pipes[pid]);
  uint32_t total_wgs = ((uint32_t)rows + 1u) >> 1;
  uint32_t wg_done = 0;
  int slot = 0;
  while (wg_done < total_wgs && slot < LVK_CHUNK_SLOTS) {
    uint32_t chunk_wgs = total_wgs - wg_done;
    if (chunk_wgs > LVK_MAX_WG_X) chunk_wgs = LVK_MAX_WG_X;
    llb_vk_params_t p = {
      .rows = (uint32_t)rows, .cols = (uint32_t)cols, .rb = (uint32_t)rb,
      .w_off = sub_off, .row_base = wg_done * 2u,
    };
    size_t ubo_off = (size_t)slot * LVK_UBO_STRIDE;
    memcpy(g_lvk.ubo_map + ubo_off, &p, sizeof p);
    VkDescriptorBufferInfo dbis[4] = {
      { g_lvk.ubo_buf, ubo_off, sizeof(llb_vk_params_t) },
      { g_lvk.slabs[e->slab_idx].buf, desc_off, VK_WHOLE_SIZE },
      { g_lvk.x_dev_buf, 0, x_bytes },
      { g_lvk.y_dev_buf, 0, y_bytes },
    };
    VkWriteDescriptorSet writes[4];
    memset(writes, 0, sizeof writes);
    for (int i = 0; i < 4; i++) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = g_lvk.desc_sets[slot];
      writes[i].dstBinding = (uint32_t)i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = (i == 0) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &dbis[i];
    }
    vkUpdateDescriptorSets(g_lvk.dev, 4, writes, 0, NULL);
    vkCmdBindDescriptorSets(g_lvk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_lvk.pipe_layout,
                            0, 1, &g_lvk.desc_sets[slot], 0, NULL);
    vkCmdDispatch(g_lvk.cmd, chunk_wgs, 1, 1);
    wg_done += chunk_wgs;
    slot++;
  }

  if (!g_lvk.is_unified) {
    VkMemoryBarrier mb1 = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
    };
    vkCmdPipelineBarrier(g_lvk.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb1, 0, NULL, 0, NULL);
    VkBufferCopy ycp = { .srcOffset = 0, .dstOffset = 0, .size = y_bytes };
    vkCmdCopyBuffer(g_lvk.cmd, g_lvk.y_dev_buf, g_lvk.y_host_buf, 1, &ycp);
    VkMemoryBarrier mb2 = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(g_lvk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &mb2, 0, NULL, 0, NULL);
  } else {
    VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(g_lvk.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
  }
  vkEndCommandBuffer(g_lvk.cmd);

  vkResetFences(g_lvk.dev, 1, &g_lvk.fence);
  VkSubmitInfo si = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &g_lvk.cmd,
  };
  if (vkQueueSubmit(g_lvk.queue, 1, &si, g_lvk.fence) != VK_SUCCESS ||
      vkWaitForFences(g_lvk.dev, 1, &g_lvk.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
    return 0;
  }
  memcpy(y, g_lvk.y_host_map, y_bytes);
  return 1;
}

// ===========================================================================
// T1 + T2: Full GPU-Resident Single-Submit Graph & Batched Prefill Engine
// ===========================================================================

static void g_barrier(VkPipelineStageFlags src_st, VkPipelineStageFlags dst_st,
                      VkAccessFlags src_ac, VkAccessFlags dst_ac) {
  VkMemoryBarrier mb = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .srcAccessMask = src_ac, .dstAccessMask = dst_ac,
  };
  vkCmdPipelineBarrier(g_lvk.cmd, src_st, dst_st, 0, 1, &mb, 0, NULL, 0, NULL);
}

static void g_comp_barrier(void) {
  g_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

static void g_bind_slab(int slab_idx) {
  if (slab_idx == g_lvk.g_cur_slab) return;
  vkCmdBindDescriptorSets(g_lvk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_lvk.g_pipe_layout,
                          0, 1, &g_lvk.g_slab_ds[slab_idx], 0, NULL);
  g_lvk.g_cur_slab = slab_idx;
}

static int g_create_pipe(lvk_gp_t id, const uint32_t *code, size_t bytes, int pin_sub32) {
  VkShaderModuleCreateInfo smci = {
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = bytes, .pCode = code,
  };
  if (vkCreateShaderModule(g_lvk.dev, &smci, NULL, &g_lvk.g_shaders[id]) != VK_SUCCESS) return -1;

  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sub_ci = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
    .requiredSubgroupSize = 32u,
  };
  VkComputePipelineCreateInfo cp = {
    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = (pin_sub32 && g_lvk.has_sub_ctrl) ? &sub_ci : NULL,
      .flags = (pin_sub32 && g_lvk.has_sub_ctrl) ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : 0,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = g_lvk.g_shaders[id],
      .pName = "main",
    },
    .layout = g_lvk.g_pipe_layout,
  };
  if (vkCreateComputePipelines(g_lvk.dev, g_lvk.pipe_cache, 1, &cp, NULL, &g_lvk.g_pipes[id]) != VK_SUCCESS) {
    return -1;
  }
  g_lvk.cache_dirty = 1;
  return 0;
}

static uint32_t alloc_act_region(uint32_t *cursor, uint32_t n_floats) {
  uint32_t aligned_c = (*cursor + 63u) & ~63u;
  *cursor = aligned_c + ((n_floats + 63u) & ~63u);
  return aligned_c;
}

static int fmt_ok_graph(int fmt) {
  return fmt == LLB_FM_F32 || fmt == LLB_FM_Q8_0 || fmt == LLB_FM_Q4_K ||
         fmt == LLB_FM_Q5_K || fmt == LLB_FM_Q6_K;
}

int llb_vk_graph_init(const llb_vk_model_cfg_t *cfg,
                      const llb_vk_tensor_desc_t *tensors, int n_tensors,
                      const llb_vk_model_ids_t *ids) {
  if (!g_lvk.active || !cfg || !tensors || n_tensors <= 0 || !ids) return -1;
  const char *no_graph = getenv("LLB_VULKAN_NO_GRAPH");
  if (no_graph && no_graph[0] == '1') return -1;
  if (cfg->HDIM > 256 || cfg->SZ > 256 || cfg->KC > 8 || cfg->L <= 0 || cfg->L > 64) return -1;
  if (g_lvk.n_slabs <= 0) return -1;

  g_lvk.cfg = *cfg;
  g_lvk.ids = *ids;
  g_lvk.gt = (lvk_gtensor_t *)calloc((size_t)n_tensors, sizeof(lvk_gtensor_t));
  if (!g_lvk.gt) return -1;
  g_lvk.n_gt = n_tensors;

  for (int i = 0; i < n_tensors; i++) {
    const llb_vk_entry_t *e = find_entry(tensors[i].off);
    if (!e || e->slab_idx < 0 || e->slab_idx >= g_lvk.n_slabs || !fmt_ok_graph(tensors[i].fmt)) {
      free(g_lvk.gt); g_lvk.gt = NULL;
      return -1;
    }
    uint64_t bo = e->slab_off + (tensors[i].off - e->pack_off);
    if (bo + tensors[i].bytes > 0xFFFFFFFFULL) {
      free(g_lvk.gt); g_lvk.gt = NULL;
      return -1;
    }
    g_lvk.gt[i] = (lvk_gtensor_t){
      .slab_idx = e->slab_idx,
      .slab_bo  = (uint32_t)bo,
      .rows     = tensors[i].rows,
      .cols     = tensors[i].cols,
      .rb       = (tensors[i].fmt == LLB_FM_F32) ? (tensors[i].cols * 4) : tensors[i].rb,
      .fmt      = tensors[i].fmt,
    };
  }

  // Verify every multi-weight layer op shares a single slab (guaranteed by <=1 MiB pass-0 packing)
  for (int il = 0; il < cfg->L; il++) {
    const llb_vk_layer_ids_t *ly = &ids->layers[il];
    if (ly->is_full) {
      if (g_lvk.gt[ly->attn_q_norm].slab_idx != g_lvk.gt[ly->attn_k_norm].slab_idx) {
        free(g_lvk.gt); g_lvk.gt = NULL; return -1;
      }
    } else {
      int cs = g_lvk.gt[ly->ssm_conv1d].slab_idx;
      if (g_lvk.gt[ly->ssm_dt_bias].slab_idx != cs || g_lvk.gt[ly->ssm_a].slab_idx != cs) {
        free(g_lvk.gt); g_lvk.gt = NULL; return -1;
      }
    }
  }

  // Compute act_buf float offsets supporting up to LVK_PREFILL_CHUNK tokens simultaneously
  const uint32_t BMAX = LVK_PREFILL_CHUNK;
  uint32_t cur = 0;
  uint32_t D = (uint32_t)cfg->D, H = (uint32_t)cfg->H, KV = (uint32_t)cfg->KV;
  uint32_t HD = (uint32_t)cfg->HDIM, dff = (uint32_t)cfg->dff, V = (uint32_t)cfg->V;
  uint32_t CONVD = (uint32_t)cfg->CONVD, VDC = (uint32_t)cfg->VDC, TR = (uint32_t)cfg->TR;
  uint32_t max_scr = dff > D ? dff : D;
  if (VDC > max_scr) max_scr = VDC;
  uint32_t max_y = CONVD > dff ? CONVD : dff;
  if (D > max_y) max_y = D;
  uint32_t max_out0 = VDC > D ? VDC : D;

  g_lvk.off_X          = alloc_act_region(&cur, BMAX * D);
  g_lvk.off_HH         = alloc_act_region(&cur, BMAX * D);
  g_lvk.off_QKV        = alloc_act_region(&cur, BMAX * CONVD);
  g_lvk.off_Z          = alloc_act_region(&cur, BMAX * VDC);
  g_lvk.off_BETA       = alloc_act_region(&cur, BMAX * TR);
  g_lvk.off_ALPHA      = alloc_act_region(&cur, BMAX * TR);
  g_lvk.off_GATE_PRE   = alloc_act_region(&cur, BMAX * TR);
  g_lvk.off_Y          = alloc_act_region(&cur, BMAX * max_y);
  g_lvk.off_QGN        = alloc_act_region(&cur, BMAX * 2u * H * HD);
  g_lvk.off_QN         = alloc_act_region(&cur, BMAX * H * HD);
  g_lvk.off_KB         = alloc_act_region(&cur, BMAX * KV * HD);
  g_lvk.off_VB         = alloc_act_region(&cur, BMAX * KV * HD);
  g_lvk.off_ATTN       = alloc_act_region(&cur, BMAX * H * HD);
  g_lvk.off_SCR        = alloc_act_region(&cur, BMAX * max_scr);
  g_lvk.off_OUT0       = alloc_act_region(&cur, BMAX * max_out0);
  g_lvk.off_LOGITS     = alloc_act_region(&cur, V);
  g_lvk.off_INV_FREQ   = alloc_act_region(&cur, 256u);
  g_lvk.off_TOK_IDS    = alloc_act_region(&cur, BMAX);
  g_lvk.off_TOP2       = alloc_act_region(&cur, 64u);
  g_lvk.off_META_FULL  = alloc_act_region(&cur, 64u);
  g_lvk.off_META_CONV  = alloc_act_region(&cur, 64u);
  g_lvk.off_META_DELTA = alloc_act_region(&cur, 64u);

  g_lvk.act_bytes = (size_t)cur * sizeof(float);

  size_t gs_elems = (size_t)cfg->L * cfg->TR * cfg->SZ * cfg->SZ;
  size_t gconv_elems = (size_t)cfg->L * (cfg->KC > 1 ? (cfg->KC - 1) : 1) * cfg->CONVD;
  g_lvk.st_conv_base = (uint32_t)((gs_elems + 63ULL) & ~63ULL);
  g_lvk.state_bytes = ((size_t)g_lvk.st_conv_base + gconv_elems + 64ULL) * sizeof(float);

  size_t kv_half = (size_t)cfg->L * cfg->cap_pos * cfg->KV * cfg->HDIM;
  g_lvk.kv_v_base = (uint32_t)((kv_half + 63ULL) & ~63ULL);
  g_lvk.kv_bytes = ((size_t)g_lvk.kv_v_base + kv_half + 64ULL) * sizeof(float);

  uint32_t max_cols = D;
  if (dff > max_cols) max_cols = dff;
  if (H * HD > max_cols) max_cols = H * HD;
  if (VDC > max_cols) max_cols = VDC;
  g_lvk.xq8_bytes = (size_t)BMAX * ((max_cols + 31u) / 32u) * 10u * sizeof(int32_t);
  if (g_lvk.xq8_bytes < 4096) g_lvk.xq8_bytes = 4096;

  g_lvk.io_host_bytes = ((size_t)V * sizeof(float) + 4095ULL) & ~4095ULL;
  if (g_lvk.io_host_bytes < 65536) g_lvk.io_host_bytes = 65536;

  VkBufferUsageFlags dev_u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (alloc_dev_buf(g_lvk.act_bytes,   dev_u, &g_lvk.act_buf,   &g_lvk.act_mem) != 0 ||
      alloc_dev_buf(g_lvk.state_bytes, dev_u, &g_lvk.state_buf, &g_lvk.state_mem) != 0 ||
      alloc_dev_buf(g_lvk.kv_bytes,    dev_u, &g_lvk.kv_buf,    &g_lvk.kv_mem) != 0 ||
      alloc_dev_buf(g_lvk.xq8_bytes,   dev_u, &g_lvk.xq8_buf,   &g_lvk.xq8_mem) != 0) {
    return -1;
  }
  void *iom = NULL;
  if (alloc_host_buf(g_lvk.io_host_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     0, &g_lvk.io_host_buf, &g_lvk.io_host_mem, &iom) != 0) {
    return -1;
  }
  g_lvk.io_host_map = (uint8_t *)iom;
  g_lvk.vram_used += g_lvk.act_bytes + g_lvk.state_bytes + g_lvk.kv_bytes + g_lvk.xq8_bytes;

  // Descriptor set layout (5 storage buffers + 64B push constants)
  VkDescriptorSetLayoutBinding gb[5];
  memset(gb, 0, sizeof gb);
  for (int i = 0; i < 5; i++) {
    gb[i].binding = (uint32_t)i;
    gb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    gb[i].descriptorCount = 1;
    gb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo gdslci = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 5, .pBindings = gb,
  };
  vkCreateDescriptorSetLayout(g_lvk.dev, &gdslci, NULL, &g_lvk.g_dsl);

  VkPushConstantRange pcr = {
    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(lvk_pc_t),
  };
  VkPipelineLayoutCreateInfo gplci = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 1, .pSetLayouts = &g_lvk.g_dsl,
    .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
  };
  vkCreatePipelineLayout(g_lvk.dev, &gplci, NULL, &g_lvk.g_pipe_layout);

  VkDescriptorPoolSize gdps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)g_lvk.n_slabs * 5u };
  VkDescriptorPoolCreateInfo gdpci = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .maxSets = (uint32_t)g_lvk.n_slabs, .poolSizeCount = 1, .pPoolSizes = &gdps,
  };
  vkCreateDescriptorPool(g_lvk.dev, &gdpci, NULL, &g_lvk.g_desc_pool);

  VkDescriptorSetLayout gdsls[LVK_MAX_SLABS];
  for (int s = 0; s < g_lvk.n_slabs; s++) gdsls[s] = g_lvk.g_dsl;
  VkDescriptorSetAllocateInfo gdsai = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = g_lvk.g_desc_pool, .descriptorSetCount = (uint32_t)g_lvk.n_slabs, .pSetLayouts = gdsls,
  };
  vkAllocateDescriptorSets(g_lvk.dev, &gdsai, g_lvk.g_slab_ds);

  for (int s = 0; s < g_lvk.n_slabs; s++) {
    VkDescriptorBufferInfo dbis[5] = {
      { g_lvk.slabs[s].buf, 0, VK_WHOLE_SIZE },
      { g_lvk.act_buf,      0, VK_WHOLE_SIZE },
      { g_lvk.state_buf,    0, VK_WHOLE_SIZE },
      { g_lvk.kv_buf,       0, VK_WHOLE_SIZE },
      { g_lvk.xq8_buf,      0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet ws[5];
    memset(ws, 0, sizeof ws);
    for (int i = 0; i < 5; i++) {
      ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      ws[i].dstSet = g_lvk.g_slab_ds[s];
      ws[i].dstBinding = (uint32_t)i;
      ws[i].descriptorCount = 1;
      ws[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      ws[i].pBufferInfo = &dbis[i];
    }
    vkUpdateDescriptorSets(g_lvk.dev, 5, ws, 0, NULL);
  }

  // Create Tier 0 portable layer-op pipelines (always valid on any Vulkan 1.2+ ICD)
  int ok = 0;
  ok |= g_create_pipe(GP_DEQ_EMBED,       lvk_spv_deq_embed,       sizeof lvk_spv_deq_embed,       0);
  ok |= g_create_pipe(GP_RMS_NORM,        lvk_spv_rms_norm,        sizeof lvk_spv_rms_norm,        0);
  ok |= g_create_pipe(GP_QUANT_X_Q8,      lvk_spv_quant_x_q8_1,    sizeof lvk_spv_quant_x_q8_1,    0);
  ok |= g_create_pipe(GP_FULL_QK_ROPE_KV, lvk_spv_full_qk_rope_kv, sizeof lvk_spv_full_qk_rope_kv, 0);
  ok |= g_create_pipe(GP_GQA_ATTN_GATE,   lvk_spv_gqa_attn_gate,   sizeof lvk_spv_gqa_attn_gate,   0);
  ok |= g_create_pipe(GP_LIN_CONV_SILU,   lvk_spv_lin_conv_silu,   sizeof lvk_spv_lin_conv_silu,   0);
  ok |= g_create_pipe(GP_LIN_DELTA_NET,   lvk_spv_lin_delta_net,   sizeof lvk_spv_lin_delta_net,   0);
  ok |= g_create_pipe(GP_SWIGLU_MUL,      lvk_spv_swiglu_mul,      sizeof lvk_spv_swiglu_mul,      0);
  ok |= g_create_pipe(GP_RES_ADD,         lvk_spv_res_add,         sizeof lvk_spv_res_add,         0);
  ok |= g_create_pipe(GP_ARGMAX_TOP2,     lvk_spv_argmax_top2,     sizeof lvk_spv_argmax_top2,     0);

  // Create GEMV (BM=1) and Prefill GEMM (BM=8) pipelines using best supported tier:
  //   Tier 1A: dp4a + wave32 subgroup | Tier 1B: FP32 + wave32 subgroup | Tier 0: portable SPIR-V 1.3
  if (g_lvk.use_dp4a) {
    ok |= g_create_pipe(GP_GEMV_Q4K,  lvk_spv_gemv_q4k_dp4a, sizeof lvk_spv_gemv_q4k_dp4a, 1);
    ok |= g_create_pipe(GP_GEMV_Q5K,  lvk_spv_gemv_q5k_dp4a, sizeof lvk_spv_gemv_q5k_dp4a, 1);
    ok |= g_create_pipe(GP_GEMV_Q6K,  lvk_spv_gemv_q6k_dp4a, sizeof lvk_spv_gemv_q6k_dp4a, 1);
    ok |= g_create_pipe(GP_GEMV_Q8_0, lvk_spv_gemv_q8_dp4a,  sizeof lvk_spv_gemv_q8_dp4a,  1);
    ok |= g_create_pipe(GP_GEMV_F32,  lvk_spv_gemv_f32_sg,   sizeof lvk_spv_gemv_f32_sg,   1);
    ok |= g_create_pipe(GP_GEMM_Q4K,  lvk_spv_gemm_q4k_dp4a, sizeof lvk_spv_gemm_q4k_dp4a, 1);
    ok |= g_create_pipe(GP_GEMM_Q5K,  lvk_spv_gemm_q5k_dp4a, sizeof lvk_spv_gemm_q5k_dp4a, 1);
    ok |= g_create_pipe(GP_GEMM_Q6K,  lvk_spv_gemm_q6k_dp4a, sizeof lvk_spv_gemm_q6k_dp4a, 1);
    ok |= g_create_pipe(GP_GEMM_Q8_0, lvk_spv_gemm_q8_dp4a,  sizeof lvk_spv_gemm_q8_dp4a,  1);
    ok |= g_create_pipe(GP_GEMM_F32,  lvk_spv_gemm_f32_sg,   sizeof lvk_spv_gemm_f32_sg,   1);
  } else if (g_lvk.use_subgroups) {
    ok |= g_create_pipe(GP_GEMV_Q4K,  lvk_spv_gemv_q4k_sg,   sizeof lvk_spv_gemv_q4k_sg,   1);
    ok |= g_create_pipe(GP_GEMV_Q5K,  lvk_spv_gemv_q5k_sg,   sizeof lvk_spv_gemv_q5k_sg,   1);
    ok |= g_create_pipe(GP_GEMV_Q6K,  lvk_spv_gemv_q6k_sg,   sizeof lvk_spv_gemv_q6k_sg,   1);
    ok |= g_create_pipe(GP_GEMV_Q8_0, lvk_spv_gemv_q8_sg,    sizeof lvk_spv_gemv_q8_sg,    1);
    ok |= g_create_pipe(GP_GEMV_F32,  lvk_spv_gemv_f32_sg,   sizeof lvk_spv_gemv_f32_sg,   1);
    ok |= g_create_pipe(GP_GEMM_Q4K,  lvk_spv_gemm_q4k_sg,   sizeof lvk_spv_gemm_q4k_sg,   1);
    ok |= g_create_pipe(GP_GEMM_Q5K,  lvk_spv_gemm_q5k_sg,   sizeof lvk_spv_gemm_q5k_sg,   1);
    ok |= g_create_pipe(GP_GEMM_Q6K,  lvk_spv_gemm_q6k_sg,   sizeof lvk_spv_gemm_q6k_sg,   1);
    ok |= g_create_pipe(GP_GEMM_Q8_0, lvk_spv_gemm_q8_sg,    sizeof lvk_spv_gemm_q8_sg,    1);
    ok |= g_create_pipe(GP_GEMM_F32,  lvk_spv_gemm_f32_sg,   sizeof lvk_spv_gemm_f32_sg,   1);
  } else {
    ok |= g_create_pipe(GP_GEMV_Q4K,  lvk_spv_gemv_q4k_p,    sizeof lvk_spv_gemv_q4k_p,    0);
    ok |= g_create_pipe(GP_GEMV_Q5K,  lvk_spv_gemv_q5k_p,    sizeof lvk_spv_gemv_q5k_p,    0);
    ok |= g_create_pipe(GP_GEMV_Q6K,  lvk_spv_gemv_q6k_p,    sizeof lvk_spv_gemv_q6k_p,    0);
    ok |= g_create_pipe(GP_GEMV_Q8_0, lvk_spv_gemv_q8_p,     sizeof lvk_spv_gemv_q8_p,     0);
    ok |= g_create_pipe(GP_GEMV_F32,  lvk_spv_gemv_f32_p,    sizeof lvk_spv_gemv_f32_p,    0);
    ok |= g_create_pipe(GP_GEMM_Q4K,  lvk_spv_gemm_q4k_p,    sizeof lvk_spv_gemm_q4k_p,    0);
    ok |= g_create_pipe(GP_GEMM_Q5K,  lvk_spv_gemm_q5k_p,    sizeof lvk_spv_gemm_q5k_p,    0);
    ok |= g_create_pipe(GP_GEMM_Q6K,  lvk_spv_gemm_q6k_p,    sizeof lvk_spv_gemm_q6k_p,    0);
    ok |= g_create_pipe(GP_GEMM_Q8_0, lvk_spv_gemm_q8_p,     sizeof lvk_spv_gemm_q8_p,     0);
    ok |= g_create_pipe(GP_GEMM_F32,  lvk_spv_gemm_f32_p,    sizeof lvk_spv_gemm_f32_p,    0);
  }
  if (ok != 0) return -1;
  lvk_save_pipe_cache();

  // Upload constant tables (double-precision RoPE inv_freq + layout offset metadata) once to act_buf
  float inv_freq[256];
  memset(inv_freq, 0, sizeof inv_freq);
  int half_rd = cfg->ROPEDIM / 2;
  for (int i = 0; i < half_rd && i < 256; i++) {
    inv_freq[i] = (float)pow((double)cfg->ROPE_BASE, -2.0 * (double)i / (double)cfg->ROPEDIM);
  }
  uint32_t meta_full[6] = { g_lvk.off_QGN, g_lvk.off_QN, g_lvk.off_KB, g_lvk.off_VB, g_lvk.kv_v_base, 0 };
  uint32_t meta_conv[5] = { g_lvk.off_QKV, g_lvk.off_Y, g_lvk.off_BETA, g_lvk.off_ALPHA, g_lvk.off_GATE_PRE };
  uint32_t meta_delta[5] = { g_lvk.off_Y, g_lvk.off_Z, g_lvk.off_BETA, g_lvk.off_GATE_PRE, g_lvk.off_OUT0 };

  uint8_t *io = g_lvk.io_host_map;
  memcpy(io + 0,    inv_freq,   sizeof inv_freq);
  memcpy(io + 1024, meta_full,  sizeof meta_full);
  memcpy(io + 1280, meta_conv,  sizeof meta_conv);
  memcpy(io + 1536, meta_delta, sizeof meta_delta);

  vkResetCommandBuffer(g_lvk.cmd, 0);
  VkCommandBufferBeginInfo bi = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  vkBeginCommandBuffer(g_lvk.cmd, &bi);
  VkBufferCopy cps[4] = {
    { 0,    (VkDeviceSize)g_lvk.off_INV_FREQ * 4,   sizeof inv_freq },
    { 1024, (VkDeviceSize)g_lvk.off_META_FULL * 4,  sizeof meta_full },
    { 1280, (VkDeviceSize)g_lvk.off_META_CONV * 4,  sizeof meta_conv },
    { 1536, (VkDeviceSize)g_lvk.off_META_DELTA * 4, sizeof meta_delta },
  };
  vkCmdCopyBuffer(g_lvk.cmd, g_lvk.io_host_buf, g_lvk.act_buf, 4, cps);
  vkCmdFillBuffer(g_lvk.cmd, g_lvk.state_buf, 0, g_lvk.state_bytes, 0);
  vkEndCommandBuffer(g_lvk.cmd);

  vkResetFences(g_lvk.dev, 1, &g_lvk.fence);
  VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &g_lvk.cmd };
  if (vkQueueSubmit(g_lvk.queue, 1, &si, g_lvk.fence) != VK_SUCCESS ||
      vkWaitForFences(g_lvk.dev, 1, &g_lvk.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
    return -1;
  }

  g_lvk.graph_ok = 1;
  return 0;
}

int llb_vk_graph_active(void) { return g_lvk.graph_ok; }

void llb_vk_graph_reset(void) {
  if (!g_lvk.graph_ok) return;
  vkResetCommandBuffer(g_lvk.cmd, 0);
  VkCommandBufferBeginInfo bi = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  vkBeginCommandBuffer(g_lvk.cmd, &bi);
  vkCmdFillBuffer(g_lvk.cmd, g_lvk.state_buf, 0, g_lvk.state_bytes, 0);
  vkEndCommandBuffer(g_lvk.cmd);
  vkResetFences(g_lvk.dev, 1, &g_lvk.fence);
  VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &g_lvk.cmd };
  vkQueueSubmit(g_lvk.queue, 1, &si, g_lvk.fence);
  vkWaitForFences(g_lvk.dev, 1, &g_lvk.fence, VK_TRUE, UINT64_MAX);
}

static void g_cmd_quant_x(uint32_t cols, uint32_t x_off, uint32_t B) {
  if (!g_lvk.use_dp4a) return;
  g_bind_slab(0);
  lvk_pc_t pc = { .p0 = cols, .p1 = x_off, .p2 = B };
  vkCmdBindPipeline(g_lvk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_lvk.g_pipes[GP_QUANT_X_Q8]);
  vkCmdPushConstants(g_lvk.cmd, g_lvk.g_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
  vkCmdDispatch(g_lvk.cmd, cols >> 5u, B, 1);
  g_comp_barrier();
}

static void g_cmd_matmul(int tid, uint32_t x_off, uint32_t y_off, uint32_t B) {
  const lvk_gtensor_t *t = &g_lvk.gt[tid];
  g_bind_slab(t->slab_idx);
  int use_bm = (B > 1u);
  lvk_gp_t pid;
  switch (t->fmt) {
    case LLB_FM_Q4_K: pid = use_bm ? GP_GEMM_Q4K  : GP_GEMV_Q4K;  break;
    case LLB_FM_Q5_K: pid = use_bm ? GP_GEMM_Q5K  : GP_GEMV_Q5K;  break;
    case LLB_FM_Q6_K: pid = use_bm ? GP_GEMM_Q6K  : GP_GEMV_Q6K;  break;
    case LLB_FM_Q8_0: pid = use_bm ? GP_GEMM_Q8_0 : GP_GEMV_Q8_0; break;
    default:          pid = use_bm ? GP_GEMM_F32  : GP_GEMV_F32;  break;
  }
  vkCmdBindPipeline(g_lvk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_lvk.g_pipes[pid]);
  uint32_t rpw = g_lvk.rows_per_wg;
  uint32_t total_wgs = ((uint32_t)t->rows + rpw - 1u) / rpw;
  uint32_t gy = use_bm ? ((B + LVK_PREFILL_BM - 1u) / LVK_PREFILL_BM) : 1u;
  uint32_t wg_done = 0;
  while (wg_done < total_wgs) {
    uint32_t chunk = total_wgs - wg_done;
    if (chunk > LVK_MAX_WG_X) chunk = LVK_MAX_WG_X;
    lvk_pc_t pc = {
      .p0 = (uint32_t)t->rows, .p1 = (uint32_t)t->cols, .p2 = (uint32_t)t->rb,
      .p3 = t->slab_bo, .p4 = wg_done * rpw, .p5 = x_off, .p6 = y_off, .p7 = B,
    };
    vkCmdPushConstants(g_lvk.cmd, g_lvk.g_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(g_lvk.cmd, chunk, gy, 1);
    wg_done += chunk;
  }
}

static void g_cmd_rms_norm(int norm_tid, uint32_t x_off, uint32_t out_off, uint32_t res_off, uint32_t B) {
  const lvk_gtensor_t *w = &g_lvk.gt[norm_tid];
  g_bind_slab(w->slab_idx);
  lvk_pc_t pc = {
    .p0 = (uint32_t)g_lvk.cfg.D, .p1 = w->slab_bo >> 2u,
    .p2 = x_off, .p3 = out_off, .p4 = res_off, .p5 = B,
    .f0 = g_lvk.cfg.EPS,
  };
  vkCmdBindPipeline(g_lvk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_lvk.g_pipes[GP_RMS_NORM]);
  vkCmdPushConstants(g_lvk.cmd, g_lvk.g_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
  vkCmdDispatch(g_lvk.cmd, B, 1, 1);
  g_comp_barrier();
}

static void g_record_layers(uint32_t pos0, uint32_t B) {
  const llb_vk_model_cfg_t *c = &g_lvk.cfg;
  const llb_vk_model_ids_t *m = &g_lvk.ids;

  // 1. Dequantize token_embd rows for toks[0..B-1] into X
  {
    const lvk_gtensor_t *emb = &g_lvk.gt[m->token_embd];
    g_bind_slab(emb->slab_idx);
    lvk_pc_t pc = {
      .p0 = (uint32_t)emb->fmt, .p1 = (uint32_t)c->D, .p2 = (uint32_t)emb->rb,
      .p3 = emb->slab_bo, .p4 = g_lvk.off_X, .p5 = g_lvk.off_TOK_IDS, .p6 = B,
    };
    vkCmdBindPipeline(g_lvk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_lvk.g_pipes[GP_DEQ_EMBED]);
    vkCmdPushConstants(g_lvk.cmd, g_lvk.g_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(g_lvk.cmd, ((uint32_t)c->D + 255u) / 256u, B, 1);
    g_comp_barrier();
  }

  for (int il = 0; il < c->L; il++) {
    const llb_vk_layer_ids_t *ly = &m->layers[il];

    // Pre-attention RMSNorm (fuses previous layer's FFN residual Y into X on il > 0!)
    g_cmd_rms_norm(ly->attn_norm, g_lvk.off_X, g_lvk.off_HH, (il > 0) ? g_lvk.off_Y : 0xffffffffu, B);
    g_cmd_quant_x((uint32_t)c->D, g_lvk.off_HH, B);

    if (ly->is_full) {
      // Full GQA Attention layer: QGN, KB, VB share the same quantized HH!
      g_cmd_matmul(ly->attn_q, g_lvk.off_HH, g_lvk.off_QGN, B);
      g_cmd_matmul(ly->attn_k, g_lvk.off_HH, g_lvk.off_KB,  B);
      g_cmd_matmul(ly->attn_v, g_lvk.off_HH, g_lvk.off_VB,  B);
      g_comp_barrier();

      const lvk_gtensor_t *qnw = &g_lvk.gt[ly->attn_q_norm];
      const lvk_gtensor_t *knw = &g_lvk.gt[ly->attn_k_norm];
      // Note: qnw and knw are small adjacent F32 norm weights in the same layer slab
      g_bind_slab(qnw->slab_idx);
      if (qnw->slab_idx == knw->slab_idx) {
        lvk_pc_t pc = {
          .p0 = (uint32_t)c->H, .p1 = (uint32_t)c->KV, .p2 = (uint32_t)c->HDIM, .p3 = (uint32_t)c->ROPEDIM,
          .p4 = pos0, .p5 = B, .p6 = (uint32_t)il, .p7 = (uint32_t)c->cap_pos,
          .p8 = qnw->slab_bo >> 2u, .p9 = knw->slab_bo >> 2u,
          .p10 = g_lvk.off_INV_FREQ, .p11 = g_lvk.off_META_FULL,
          .f0 = c->EPS,
        };
        vkCmdBindPipeline(g_lvk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_lvk.g_pipes[GP_FULL_QK_ROPE_KV]);
        vkCmdPushConstants(g_lvk.cmd, g_lvk.g_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
        vkCmdDispatch(g_lvk.cmd, (uint32_t)(c->H + c->KV), B, 1);
        g_comp_barrier();
      }

      // Causal online-softmax GQA + output sigmoid gate
      {
        lvk_pc_t pc = {
          .p0 = (uint32_t)c->H, .p1 = (uint32_t)c->KV, .p2 = (uint32_t)c->HDIM,
          .p3 = pos0, .p4 = B, .p5 = (uint32_t)il, .p6 = (uint32_t)c->cap_pos,
          .p7 = g_lvk.off_QGN, .p8 = g_lvk.off_QN, .p9 = g_lvk.off_ATTN, .p10 = g_lvk.kv_v_base,
          .f0 = 1.0f / sqrtf((float)c->HDIM),
        };
        vkCmdBindPipeline(g_lvk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_lvk.g_pipes[GP_GQA_ATTN_GATE]);
        vkCmdPushConstants(g_lvk.cmd, g_lvk.g_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
        vkCmdDispatch(g_lvk.cmd, (uint32_t)c->H, B, 1);
        g_comp_barrier();
      }

      g_cmd_quant_x((uint32_t)(c->H * c->HDIM), g_lvk.off_ATTN, B);
      g_cmd_matmul(ly->attn_output, g_lvk.off_ATTN, g_lvk.off_OUT0, B);
      g_comp_barrier();
    } else {
      // Linear Gated Delta-Net layer: QKV, Z, BETA, ALPHA all share quantized HH!
      g_cmd_matmul(ly->attn_qkv,  g_lvk.off_HH, g_lvk.off_QKV,   B);
      g_cmd_matmul(ly->attn_gate, g_lvk.off_HH, g_lvk.off_Z,     B);
      g_cmd_matmul(ly->ssm_beta,  g_lvk.off_HH, g_lvk.off_BETA,  B);
      g_cmd_matmul(ly->ssm_alpha, g_lvk.off_HH, g_lvk.off_ALPHA, B);
      g_comp_barrier();

      const lvk_gtensor_t *cw = &g_lvk.gt[ly->ssm_conv1d];
      const lvk_gtensor_t *dt = &g_lvk.gt[ly->ssm_dt_bias];
      const lvk_gtensor_t *al = &g_lvk.gt[ly->ssm_a];
      g_bind_slab(cw->slab_idx);
      {
        lvk_pc_t pc = {
          .p0 = (uint32_t)c->CONVD, .p1 = (uint32_t)c->KC, .p2 = (uint32_t)c->TR,
          .p3 = B, .p4 = (uint32_t)il, .p5 = g_lvk.st_conv_base,
          .p6 = cw->slab_bo >> 2u, .p7 = dt->slab_bo >> 2u, .p8 = al->slab_bo >> 2u,
          .p9 = g_lvk.off_META_CONV,
        };
        vkCmdBindPipeline(g_lvk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_lvk.g_pipes[GP_LIN_CONV_SILU]);
        vkCmdPushConstants(g_lvk.cmd, g_lvk.g_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
        vkCmdDispatch(g_lvk.cmd, ((uint32_t)c->CONVD + 255u) / 256u, 1, 1);
        g_comp_barrier();
      }

      const lvk_gtensor_t *sn = &g_lvk.gt[ly->ssm_norm];
      g_bind_slab(sn->slab_idx);
      {
        lvk_pc_t pc = {
          .p0 = (uint32_t)c->TR, .p1 = (uint32_t)c->GK, .p2 = (uint32_t)c->SZ,
          .p3 = (uint32_t)c->KDC, .p4 = (uint32_t)c->VDC, .p5 = (uint32_t)c->CONVD,
          .p6 = B, .p7 = (uint32_t)il, .p8 = sn->slab_bo >> 2u, .p9 = g_lvk.off_META_DELTA,
          .f0 = c->EPS,
        };
        vkCmdBindPipeline(g_lvk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_lvk.g_pipes[GP_LIN_DELTA_NET]);
        vkCmdPushConstants(g_lvk.cmd, g_lvk.g_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
        vkCmdDispatch(g_lvk.cmd, (uint32_t)c->TR, 1, 1);
        g_comp_barrier();
      }

      g_cmd_quant_x((uint32_t)c->VDC, g_lvk.off_OUT0, B);
      g_cmd_matmul(ly->ssm_out, g_lvk.off_OUT0, g_lvk.off_SCR, B);
      g_comp_barrier();
    }

    // Fused Residual Add (X += OUT0/SCR) + Post-Attention RMSNorm -> HH
    uint32_t attn_res_off = ly->is_full ? g_lvk.off_OUT0 : g_lvk.off_SCR;
    g_cmd_rms_norm(ly->post_attn_norm, g_lvk.off_X, g_lvk.off_HH, attn_res_off, B);

    // SwiGLU FFN: gate and up share quantized HH!
    g_cmd_quant_x((uint32_t)c->D, g_lvk.off_HH, B);
    g_cmd_matmul(ly->ffn_gate, g_lvk.off_HH, g_lvk.off_SCR, B);
    g_cmd_matmul(ly->ffn_up,   g_lvk.off_HH, g_lvk.off_Y,   B);
    g_comp_barrier();

    {
      uint32_t tot = B * (uint32_t)c->dff;
      lvk_pc_t pc = { .p0 = tot, .p1 = g_lvk.off_SCR, .p2 = g_lvk.off_Y };
      vkCmdBindPipeline(g_lvk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_lvk.g_pipes[GP_SWIGLU_MUL]);
      vkCmdPushConstants(g_lvk.cmd, g_lvk.g_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
      vkCmdDispatch(g_lvk.cmd, (tot + 255u) / 256u, 1, 1);
      g_comp_barrier();
    }

    g_cmd_quant_x((uint32_t)c->dff, g_lvk.off_SCR, B);
    g_cmd_matmul(ly->ffn_down, g_lvk.off_SCR, g_lvk.off_Y, B);
    g_comp_barrier();
  }

  // Final FFN residual add into X for the last layer (note: if computing logits immediately, could fuse!)
  {
    uint32_t tot = B * (uint32_t)c->D;
    lvk_pc_t pc = { .p0 = tot, .p1 = g_lvk.off_X, .p2 = g_lvk.off_Y };
    vkCmdBindPipeline(g_lvk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_lvk.g_pipes[GP_RES_ADD]);
    vkCmdPushConstants(g_lvk.cmd, g_lvk.g_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(g_lvk.cmd, (tot + 255u) / 256u, 1, 1);
    g_comp_barrier();
  }
}

static void g_record_final_logits(uint32_t last_tok_idx, int want_logits, int want_top2) {
  const llb_vk_model_cfg_t *c = &g_lvk.cfg;
  const llb_vk_model_ids_t *m = &g_lvk.ids;

  uint32_t x_last_off = g_lvk.off_X + last_tok_idx * (uint32_t)c->D;
  g_cmd_rms_norm(m->output_norm, x_last_off, g_lvk.off_HH, 0xffffffffu, 1u);
  g_cmd_quant_x((uint32_t)c->D, g_lvk.off_HH, 1u);
  g_cmd_matmul(m->output_w, g_lvk.off_HH, g_lvk.off_LOGITS, 1u);
  g_comp_barrier();

  if (want_top2) {
    lvk_pc_t pc = { .p0 = (uint32_t)c->V, .p1 = g_lvk.off_LOGITS, .p2 = g_lvk.off_TOP2 };
    vkCmdBindPipeline(g_lvk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_lvk.g_pipes[GP_ARGMAX_TOP2]);
    vkCmdPushConstants(g_lvk.cmd, g_lvk.g_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(g_lvk.cmd, 1, 1, 1);
  }

  g_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
  if (want_logits) {
    VkBufferCopy cp = {
      .srcOffset = (VkDeviceSize)g_lvk.off_LOGITS * 4,
      .dstOffset = 0,
      .size = (VkDeviceSize)c->V * 4,
    };
    vkCmdCopyBuffer(g_lvk.cmd, g_lvk.act_buf, g_lvk.io_host_buf, 1, &cp);
  } else if (want_top2) {
    VkBufferCopy cp = {
      .srcOffset = (VkDeviceSize)g_lvk.off_TOP2 * 4,
      .dstOffset = 0,
      .size = 16,
    };
    vkCmdCopyBuffer(g_lvk.cmd, g_lvk.act_buf, g_lvk.io_host_buf, 1, &cp);
  }
  if (want_logits && want_top2) {
    VkBufferCopy cp = {
      .srcOffset = (VkDeviceSize)g_lvk.off_TOP2 * 4,
      .dstOffset = (VkDeviceSize)c->V * 4,
      .size = 16,
    };
    vkCmdCopyBuffer(g_lvk.cmd, g_lvk.act_buf, g_lvk.io_host_buf, 1, &cp);
  }
  g_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
}

static int g_run_chunk(const int *toks, uint32_t B, uint32_t pos0,
                       int compute_logits, float *logits_out, uint32_t *top2_out) {
  if (!g_lvk.graph_ok || B == 0 || B > LVK_PREFILL_CHUNK) return -1;

  uint32_t u_toks[LVK_PREFILL_CHUNK];
  for (uint32_t i = 0; i < B; i++) u_toks[i] = (uint32_t)toks[i];
  memcpy(g_lvk.io_host_map, u_toks, (size_t)B * 4);

  g_lvk.g_cur_slab = -1;
  vkResetCommandBuffer(g_lvk.cmd, 0);
  VkCommandBufferBeginInfo bi = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  vkBeginCommandBuffer(g_lvk.cmd, &bi);

  VkBufferCopy tcp = {
    .srcOffset = 0,
    .dstOffset = (VkDeviceSize)g_lvk.off_TOK_IDS * 4,
    .size = (VkDeviceSize)B * 4,
  };
  vkCmdCopyBuffer(g_lvk.cmd, g_lvk.io_host_buf, g_lvk.act_buf, 1, &tcp);
  g_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

  g_record_layers(pos0, B);
  if (compute_logits) {
    g_record_final_logits(B - 1u, logits_out != NULL, top2_out != NULL);
  }
  // If B > 1 and !compute_logits, copy last token's X into slot 0 so subsequent llb_vk_graph_final_logits() reads slot 0!
  if (!compute_logits && B > 1u) {
    g_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    VkBufferCopy xcp = {
      .srcOffset = (VkDeviceSize)(g_lvk.off_X + (B - 1u) * (uint32_t)g_lvk.cfg.D) * 4,
      .dstOffset = (VkDeviceSize)g_lvk.off_X * 4,
      .size = (VkDeviceSize)g_lvk.cfg.D * 4,
    };
    vkCmdCopyBuffer(g_lvk.cmd, g_lvk.act_buf, g_lvk.act_buf, 1, &xcp);
  }

  vkEndCommandBuffer(g_lvk.cmd);

  vkResetFences(g_lvk.dev, 1, &g_lvk.fence);
  VkSubmitInfo si = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &g_lvk.cmd,
  };
  if (vkQueueSubmit(g_lvk.queue, 1, &si, g_lvk.fence) != VK_SUCCESS ||
      vkWaitForFences(g_lvk.dev, 1, &g_lvk.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
    return -1;
  }

  if (compute_logits) {
    if (logits_out) {
      memcpy(logits_out, g_lvk.io_host_map, (size_t)g_lvk.cfg.V * 4);
      if (top2_out) memcpy(top2_out, g_lvk.io_host_map + (size_t)g_lvk.cfg.V * 4, 16);
    } else if (top2_out) {
      memcpy(top2_out, g_lvk.io_host_map, 16);
    }
  }
  return 0;
}

int llb_vk_graph_step(int tok, int pos, int compute_logits, float *logits_out, uint32_t *top2_out) {
  return g_run_chunk(&tok, 1u, (uint32_t)pos, compute_logits, logits_out, top2_out);
}

int llb_vk_graph_prefill(const int *toks, int n_tok, int pos0,
                         int compute_last_logits, float *logits_out, uint32_t *top2_out) {
  if (!g_lvk.graph_ok || !toks || n_tok <= 0) return -1;
  int done = 0;
  while (done < n_tok) {
    int chunk = n_tok - done;
    if (chunk > LVK_PREFILL_CHUNK) chunk = LVK_PREFILL_CHUNK;
    int is_last = (done + chunk == n_tok);
    if (g_run_chunk(toks + done, (uint32_t)chunk, (uint32_t)(pos0 + done),
                    is_last ? compute_last_logits : 0,
                    is_last ? logits_out : NULL,
                    is_last ? top2_out : NULL) != 0) {
      return -1;
    }
    done += chunk;
  }
  return 0;
}

int llb_vk_graph_final_logits(float *logits_out, uint32_t *top2_out) {
  if (!g_lvk.graph_ok) return -1;
  g_lvk.g_cur_slab = -1;
  vkResetCommandBuffer(g_lvk.cmd, 0);
  VkCommandBufferBeginInfo bi = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  vkBeginCommandBuffer(g_lvk.cmd, &bi);
  g_record_final_logits(0u, logits_out != NULL, top2_out != NULL);
  vkEndCommandBuffer(g_lvk.cmd);

  vkResetFences(g_lvk.dev, 1, &g_lvk.fence);
  VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &g_lvk.cmd };
  if (vkQueueSubmit(g_lvk.queue, 1, &si, g_lvk.fence) != VK_SUCCESS ||
      vkWaitForFences(g_lvk.dev, 1, &g_lvk.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
    return -1;
  }
  if (logits_out) {
    memcpy(logits_out, g_lvk.io_host_map, (size_t)g_lvk.cfg.V * 4);
    if (top2_out) memcpy(top2_out, g_lvk.io_host_map + (size_t)g_lvk.cfg.V * 4, 16);
  } else if (top2_out) {
    memcpy(top2_out, g_lvk.io_host_map, 16);
  }
  return 0;
}

int llb_vk_graph_read_norm_hidden(float *hh_out) {
  if (!g_lvk.graph_ok || !hh_out) return -1;
  g_lvk.g_cur_slab = -1;
  vkResetCommandBuffer(g_lvk.cmd, 0);
  VkCommandBufferBeginInfo bi = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  vkBeginCommandBuffer(g_lvk.cmd, &bi);
  g_cmd_rms_norm(g_lvk.ids.output_norm, g_lvk.off_X, g_lvk.off_HH, 0xffffffffu, 1u);
  g_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
  VkBufferCopy cp = {
    .srcOffset = (VkDeviceSize)g_lvk.off_HH * 4, .dstOffset = 0, .size = (VkDeviceSize)g_lvk.cfg.D * 4,
  };
  vkCmdCopyBuffer(g_lvk.cmd, g_lvk.act_buf, g_lvk.io_host_buf, 1, &cp);
  g_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
  vkEndCommandBuffer(g_lvk.cmd);

  vkResetFences(g_lvk.dev, 1, &g_lvk.fence);
  VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &g_lvk.cmd };
  if (vkQueueSubmit(g_lvk.queue, 1, &si, g_lvk.fence) != VK_SUCCESS ||
      vkWaitForFences(g_lvk.dev, 1, &g_lvk.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
    return -1;
  }
  memcpy(hh_out, g_lvk.io_host_map, (size_t)g_lvk.cfg.D * 4);
  return 0;
}

const char *llb_vk_device_info(void) { return g_lvk.active ? g_lvk.info : ""; }

int llb_vk_mem_info(size_t *free_b, size_t *total_b) {
  if (!g_lvk.active) return -1;
  if (total_b) *total_b = g_lvk.vram_total;
  if (free_b) *free_b = (g_lvk.vram_total > g_lvk.vram_used) ? (g_lvk.vram_total - g_lvk.vram_used) : 0;
  return 0;
}
