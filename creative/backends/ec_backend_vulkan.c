// ec_backend_vulkan.c — Cross-Vendor Linux & Windows Vulkan 1.2+ SPIR-V Compute Backend
// Runs natively on AMD Radeon (RADV/AMDVLK), Intel Arc (ANV), NVIDIA Vulkan, Qualcomm, and Lavapipe.
// Uses pre-validated SPIR-V 1.3 binary shaders (ec_vulkan_spv.h) + ReBAR/Unified Coherent Memory.

#define _POSIX_C_SOURCE 200809L
#ifndef EC_BACKEND_VULKAN
  #define EC_BACKEND_VULKAN 1
#endif

#include "../ec_hal.h"
#include "ec_ops_common.h"
#include "ec_vulkan_spv.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

typedef struct {
  uint32_t p0, p1, p2, p3;
  uint32_t p4, p5, p6, p7;
  uint32_t o0, o1, o2, o3;
  uint32_t o4, o5, pad0, pad1;
  float    f0, f1, f2, f3;
} ec_vk_params_t;

typedef enum {
  VK_PIPE_GEMM_BF16 = 0,
  VK_PIPE_GEMM_F16  = 1,
  VK_PIPE_QUANT_Q8  = 2,
  VK_PIPE_GEMM_Q8   = 3,
  VK_PIPE_FA2_BF16  = 4,
  VK_PIPE_FA2_F16   = 5,
  VK_PIPE_CONV3X3   = 6,
  VK_PIPE_PACK16X2  = 7,
  VK_PIPE_GEMM_I8   = 8,
  VK_PIPE_COUNT     = 9
} ec_vk_pipe_id_t;

typedef struct {
  uintptr_t      base;
  size_t         size;
  VkBuffer       buf;
  VkDeviceMemory mem;
  void          *mapped;
  int            is_dev_local;
} ec_vk_alloc_t;

#define EC_VK_MAX_ALLOCS 8192

static struct {
  pthread_mutex_t       mu;
  int                   init_done;
  int                   init_ok;
  VkInstance            instance;
  VkPhysicalDevice      phys;
  VkDevice              dev;
  VkQueue               queue;
  uint32_t              qfam;
  VkPhysicalDeviceProperties props;
  VkPhysicalDeviceMemoryProperties mem_props;
  size_t                vram_total;
  size_t                vram_used;
  int                   mem_type_rebar; // DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT
  int                   mem_type_host;  // HOST_VISIBLE | HOST_COHERENT

  VkCommandPool         cmd_pool;
  VkCommandBuffer       cmd;
  VkFence               fence;
  VkDescriptorSetLayout dsl;
  VkPipelineLayout      pipe_layout;
  VkDescriptorPool      desc_pool;
  VkDescriptorSet       desc_set;
  VkShaderModule        shaders[VK_PIPE_COUNT];
  VkPipeline            pipes[VK_PIPE_COUNT];

  // Dummy 256B zero buffer for unused bindings + uniform buffer for Params
  VkBuffer              dummy_buf;
  VkDeviceMemory        dummy_mem;
  VkBuffer              ubo_buf;
  VkDeviceMemory        ubo_mem;
  void                 *ubo_mapped;

  // Scratch F32 staging buffer when caller requests BF16/F16-only output without Y_f32
  float                *tmp_y_f32;
  size_t                tmp_y_cap;

  ec_vk_alloc_t         allocs[EC_VK_MAX_ALLOCS];
  int                   n_allocs;
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

static int create_raw_vk_buf(size_t nbytes, VkBufferUsageFlags usage, int prefer_dev_local,
                             VkBuffer *out_buf, VkDeviceMemory *out_mem, void **out_map, int *out_is_dev) {
  size_t aligned = (nbytes + 255ULL) & ~255ULL;
  if (aligned < 256) aligned = 256;
  VkBufferCreateInfo bci = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = aligned,
    .usage = usage,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  if (vkCreateBuffer(g_vk.dev, &bci, NULL, out_buf) != VK_SUCCESS) return -1;
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(g_vk.dev, *out_buf, &req);

  int mt = -1;
  int is_dev = 0;
  if (prefer_dev_local) {
    // 1st choice: Unified / APU memory with BOTH DEVICE_LOCAL and HOST_CACHED (zero WC PCIe read stalls!)
    mt = find_mem_type(req.memoryTypeBits,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
    if (mt >= 0) is_dev = 1;
  }
  // 2nd choice: Cached coherent memory (L3-cached on CPU at 60+ GB/s AND directly GPU DMA-mapped for STORAGE_BUFFER!)
  // Critical on discrete PCIe GPUs where DEVICE_LOCAL|HOST_VISIBLE BAR1 is uncached Write-Combined (~8 MB/s CPU read).
  if (mt < 0) {
    mt = find_mem_type(req.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                       VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
  }
  if (mt < 0 && prefer_dev_local) {
    mt = find_mem_type(req.memoryTypeBits,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    if (mt >= 0) is_dev = 1;
  }
  if (mt < 0) {
    mt = find_mem_type(req.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
  }
  if (mt < 0) { vkDestroyBuffer(g_vk.dev, *out_buf, NULL); return -1; }

  VkMemoryAllocateInfo mai = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize = req.size,
    .memoryTypeIndex = (uint32_t)mt,
  };
  if (vkAllocateMemory(g_vk.dev, &mai, NULL, out_mem) != VK_SUCCESS) {
    // Fallback from ReBAR heap to system coherent heap if BAR aperture is full!
    if (is_dev) {
      is_dev = 0;
      int mt2 = find_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      if (mt2 < 0) mt2 = find_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
      if (mt2 >= 0) {
        mai.memoryTypeIndex = (uint32_t)mt2;
        if (vkAllocateMemory(g_vk.dev, &mai, NULL, out_mem) != VK_SUCCESS) {
          vkDestroyBuffer(g_vk.dev, *out_buf, NULL);
          return -1;
        }
      } else {
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
  if (out_is_dev) *out_is_dev = is_dev;
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

int ec_hal_init(void) {
  pthread_mutex_lock(&g_vk.mu);
  if (g_vk.init_done) {
    int ok = g_vk.init_ok;
    pthread_mutex_unlock(&g_vk.mu);
    return ok ? 0 : -1;
  }
  g_vk.init_done = 1;

  VkApplicationInfo app = {
    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .pApplicationName = "eureka-creative-vulkan",
    .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
    .pEngineName = "eureka-creative",
    .engineVersion = VK_MAKE_VERSION(1, 6, 0),
    .apiVersion = VK_API_VERSION_1_2,
  };
  VkInstanceCreateInfo ici = {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pApplicationInfo = &app,
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
  int chosen = -1;
  int best_score = -1;
  for (uint32_t i = 0; i < n_phys; i++) {
    VkPhysicalDeviceProperties pr;
    vkGetPhysicalDeviceProperties(devs[i], &pr);
    if (want && want[0]) {
      char idx_s[16];
      snprintf(idx_s, sizeof idx_s, "%u", i);
      if (!strcmp(want, idx_s) || str_icontains(pr.deviceName, want)) {
        chosen = (int)i;
        break;
      }
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
    .queueFamilyIndex = g_vk.qfam,
    .queueCount = 1,
    .pQueuePriorities = &qprio,
  };
  VkDeviceCreateInfo dci = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .queueCreateInfoCount = 1,
    .pQueueCreateInfos = &qci,
  };
  if (vkCreateDevice(g_vk.phys, &dci, NULL, &g_vk.dev) != VK_SUCCESS) {
    pthread_mutex_unlock(&g_vk.mu);
    return -1;
  }
  vkGetDeviceQueue(g_vk.dev, g_vk.qfam, 0, &g_vk.queue);

  VkCommandPoolCreateInfo cpci = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    .queueFamilyIndex = g_vk.qfam,
  };
  vkCreateCommandPool(g_vk.dev, &cpci, NULL, &g_vk.cmd_pool);
  VkCommandBufferAllocateInfo cbai = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = g_vk.cmd_pool,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = 1,
  };
  vkAllocateCommandBuffers(g_vk.dev, &cbai, &g_vk.cmd);
  VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
  vkCreateFence(g_vk.dev, &fci, NULL, &g_vk.fence);

  void *dummy_map = NULL;
  create_raw_vk_buf(256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1, &g_vk.dummy_buf, &g_vk.dummy_mem, &dummy_map, NULL);
  if (dummy_map) memset(dummy_map, 0, 256);
  create_raw_vk_buf(sizeof(ec_vk_params_t), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 0, &g_vk.ubo_buf, &g_vk.ubo_mem, &g_vk.ubo_mapped, NULL);

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
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = 7,
    .pBindings = binds,
  };
  vkCreateDescriptorSetLayout(g_vk.dev, &dslci, NULL, &g_vk.dsl);

  VkPipelineLayoutCreateInfo plci = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 1,
    .pSetLayouts = &g_vk.dsl,
  };
  vkCreatePipelineLayout(g_vk.dev, &plci, NULL, &g_vk.pipe_layout);

  VkDescriptorPoolSize dps[2] = {
    { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4 },
    { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 24 },
  };
  VkDescriptorPoolCreateInfo dpci = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .maxSets = 4,
    .poolSizeCount = 2,
    .pPoolSizes = dps,
  };
  vkCreateDescriptorPool(g_vk.dev, &dpci, NULL, &g_vk.desc_pool);
  VkDescriptorSetAllocateInfo dsai = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = g_vk.desc_pool,
    .descriptorSetCount = 1,
    .pSetLayouts = &g_vk.dsl,
  };
  vkAllocateDescriptorSets(g_vk.dev, &dsai, &g_vk.desc_set);

  const struct { const uint32_t *code; size_t bytes; const char *entry; } specs[VK_PIPE_COUNT] = {
    { ec_spv_vk_gemm_bf16_f32,        sizeof(ec_spv_vk_gemm_bf16_f32),        "vk_gemm_bf16_f32" },
    { ec_spv_vk_gemm_f16_f32,         sizeof(ec_spv_vk_gemm_f16_f32),         "vk_gemm_f16_f32" },
    { ec_spv_vk_quantize_bf16_q8g64,  sizeof(ec_spv_vk_quantize_bf16_q8g64),  "vk_quantize_bf16_q8g64" },
    { ec_spv_vk_gemm_q8g64_f32,       sizeof(ec_spv_vk_gemm_q8g64_f32),       "vk_gemm_q8g64_f32" },
    { ec_spv_vk_fa2_bf16,             sizeof(ec_spv_vk_fa2_bf16),             "vk_fa2_bf16" },
    { ec_spv_vk_fa2_f16,              sizeof(ec_spv_vk_fa2_f16),              "vk_fa2_f16" },
    { ec_spv_vk_conv2d_3x3_nhwc_bf16, sizeof(ec_spv_vk_conv2d_3x3_nhwc_bf16), "vk_conv2d_3x3_nhwc_bf16" },
    { ec_spv_vk_pack_f32_to_16x2,     sizeof(ec_spv_vk_pack_f32_to_16x2),     "vk_pack_f32_to_16x2" },
    { ec_spv_vk_gemm_i8_f32,          sizeof(ec_spv_vk_gemm_i8_f32),          "vk_gemm_i8_f32" },
  };
  for (int i = 0; i < VK_PIPE_COUNT; i++) {
    VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = specs[i].bytes,
      .pCode = specs[i].code,
    };
    if (vkCreateShaderModule(g_vk.dev, &smci, NULL, &g_vk.shaders[i]) != VK_SUCCESS) {
      pthread_mutex_unlock(&g_vk.mu);
      return -1;
    }
    VkComputePipelineCreateInfo cp = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = g_vk.shaders[i],
        .pName = specs[i].entry,
      },
      .layout = g_vk.pipe_layout,
    };
    if (vkCreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &cp, NULL, &g_vk.pipes[i]) != VK_SUCCESS) {
      pthread_mutex_unlock(&g_vk.mu);
      return -1;
    }
  }

  g_vk.init_ok = 1;
  pthread_mutex_unlock(&g_vk.mu);
  return 0;
}

// Lookup allocated VkBuffer containing interior pointer `ptr`, returning 256B-aligned descriptor offset + sub-offset!
static int lookup_vk_ptr(const void *ptr, size_t elem_bytes, VkDescriptorBufferInfo *out_dbi, uint32_t *out_elem_off) {
  if (!ptr) {
    out_dbi->buffer = g_vk.dummy_buf;
    out_dbi->offset = 0;
    out_dbi->range  = VK_WHOLE_SIZE;
    *out_elem_off   = 0;
    return 0;
  }
  uintptr_t addr = (uintptr_t)ptr;
  int lo = 0, hi = g_vk.n_allocs - 1;
  while (lo <= hi) {
    int mid = (lo + hi) >> 1;
    const ec_vk_alloc_t *a = &g_vk.allocs[mid];
    if (addr < a->base) {
      hi = mid - 1;
    } else if (addr >= a->base + a->size) {
      lo = mid + 1;
    } else {
      size_t byte_off = (size_t)(addr - a->base);
      size_t align = (size_t)g_vk.props.limits.minStorageBufferOffsetAlignment;
      if (align < 256) align = 256; // 256B is always a multiple of u32/f32/u16/u8!
      size_t desc_off = byte_off & ~(align - 1ULL);
      size_t rem_bytes = byte_off - desc_off;
      out_dbi->buffer = a->buf;
      out_dbi->offset = (VkDeviceSize)desc_off;
      out_dbi->range  = VK_WHOLE_SIZE;
      *out_elem_off   = (uint32_t)(rem_bytes / elem_bytes);
      return 0;
    }
  }
  return -1;
}

static int vk_dispatch_locked(
    ec_vk_pipe_id_t pid, const ec_vk_params_t *params,
    const void *b1, size_t eb1, const void *b2, size_t eb2,
    const void *b3, size_t eb3, const void *b4, size_t eb4,
    void *b5, size_t eb5, void *b6, size_t eb6,
    uint32_t gx, uint32_t gy, uint32_t gz) {
  if (gx == 0 || gy == 0 || gz == 0) return 0;
  ec_vk_params_t p = *params;
  VkDescriptorBufferInfo dbis[7];
  dbis[0].buffer = g_vk.ubo_buf; dbis[0].offset = 0; dbis[0].range = sizeof(ec_vk_params_t);
  if (lookup_vk_ptr(b1, eb1, &dbis[1], &p.o0) != 0) return -1;
  if (lookup_vk_ptr(b2, eb2, &dbis[2], &p.o1) != 0) return -1;
  if (lookup_vk_ptr(b3, eb3, &dbis[3], &p.o2) != 0) return -1;
  if (lookup_vk_ptr(b4, eb4, &dbis[4], &p.o3) != 0) return -1;
  if (lookup_vk_ptr(b5, eb5, &dbis[5], &p.o4) != 0) return -1;
  if (lookup_vk_ptr(b6, eb6, &dbis[6], &p.o5) != 0) return -1;
  VkWriteDescriptorSet writes[7];
  memset(writes, 0, sizeof writes);
  for (int i = 0; i < 7; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = g_vk.desc_set;
    writes[i].dstBinding = (uint32_t)i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = (i == 0) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &dbis[i];
  }
  vkUpdateDescriptorSets(g_vk.dev, 7, writes, 0, NULL);

  // TDR-safe & maxComputeWorkGroupCount-safe sub-grid chunking:
  // Keeps every individual vkQueueSubmit <= 4096 workgroups so even slow 2-CU integrated iGPUs
  // never trip OS/display 2-second TDR watchdogs or exceed 65535 workgroup limits!
  const uint32_t max_wg_total = 4096u;
  uint32_t step_x = gx;
  if (step_x > max_wg_total) step_x = max_wg_total;
  uint32_t step_y = max_wg_total / (step_x > 0 ? step_x : 1u);
  if (step_y == 0) step_y = 1u;
  if (step_y > gy) step_y = gy;

  for (uint32_t oy = 0; oy < gy; oy += step_y) {
    uint32_t cy = (oy + step_y <= gy) ? step_y : (gy - oy);
    for (uint32_t ox = 0; ox < gx; ox += step_x) {
      uint32_t cx = (ox + step_x <= gx) ? step_x : (gx - ox);
      p.pad0 = ox;
      p.pad1 = oy;
      memcpy(g_vk.ubo_mapped, &p, sizeof p);

      vkResetCommandBuffer(g_vk.cmd, 0);
      VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      };
      vkBeginCommandBuffer(g_vk.cmd, &bi);
      vkCmdBindPipeline(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pipes[pid]);
      vkCmdBindDescriptorSets(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pipe_layout, 0, 1, &g_vk.desc_set, 0, NULL);
      vkCmdDispatch(g_vk.cmd, cx, cy, gz);
      VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
      };
      vkCmdPipelineBarrier(g_vk.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, 1, &mb, 0, NULL, 0, NULL);
      vkEndCommandBuffer(g_vk.cmd);

      vkResetFences(g_vk.dev, 1, &g_vk.fence);
      VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &g_vk.cmd,
      };
      if (vkQueueSubmit(g_vk.queue, 1, &si, g_vk.fence) != VK_SUCCESS) return -1;
      if (vkWaitForFences(g_vk.dev, 1, &g_vk.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) return -1;
    }
  }
  return 0;
}

cudaError_t ec_hal_get_device_props(cudaDeviceProp *prop, int device) {
  (void)device;
  if (ec_hal_init() != 0 || !prop) return cudaErrorNotSupported;
  memset(prop, 0, sizeof *prop);
  snprintf(prop->name, sizeof prop->name, "%s (Vulkan)", g_vk.props.deviceName);
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
  size_t tot = g_vk.vram_total;
  size_t used = g_vk.vram_used;
  pthread_mutex_unlock(&g_vk.mu);
  if (total_b) *total_b = tot;
  if (free_b)  *free_b  = (tot > used) ? (tot - used) : (512ULL << 20);
  return cudaSuccess;
}

cudaError_t ec_hal_malloc_raw(void **ptr, size_t nbytes) {
  if (!ptr) return cudaErrorMemoryAllocation;
  *ptr = NULL;
  if (nbytes == 0) nbytes = 256;
  if (ec_hal_init() != 0) return cudaErrorMemoryAllocation;

  pthread_mutex_lock(&g_vk.mu);
  if (g_vk.n_allocs >= EC_VK_MAX_ALLOCS) {
    pthread_mutex_unlock(&g_vk.mu);
    return cudaErrorMemoryAllocation;
  }
  size_t aligned = (nbytes + 255ULL) & ~255ULL;
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  void *mapped = NULL;
  int is_dev = 0;
  if (create_raw_vk_buf(aligned, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        1, &buf, &mem, &mapped, &is_dev) != 0) {
    pthread_mutex_unlock(&g_vk.mu);
    return cudaErrorMemoryAllocation;
  }
  uintptr_t base = (uintptr_t)mapped;
  int pos = 0;
  while (pos < g_vk.n_allocs && g_vk.allocs[pos].base < base) pos++;
  memmove(&g_vk.allocs[pos + 1], &g_vk.allocs[pos], (size_t)(g_vk.n_allocs - pos) * sizeof(ec_vk_alloc_t));
  g_vk.allocs[pos] = (ec_vk_alloc_t){ .base = base, .size = aligned, .buf = buf, .mem = mem, .mapped = mapped, .is_dev_local = is_dev };
  g_vk.n_allocs++;
  g_vk.vram_used += aligned;
  *ptr = mapped;
  pthread_mutex_unlock(&g_vk.mu);
  return cudaSuccess;
}

cudaError_t ec_hal_free_raw(void *ptr) {
  if (!ptr) return cudaSuccess;
  pthread_mutex_lock(&g_vk.mu);
  uintptr_t addr = (uintptr_t)ptr;
  for (int i = 0; i < g_vk.n_allocs; i++) {
    if (g_vk.allocs[i].base == addr) {
      ec_vk_alloc_t a = g_vk.allocs[i];
      if (g_vk.vram_used >= a.size) g_vk.vram_used -= a.size;
      vkUnmapMemory(g_vk.dev, a.mem);
      vkDestroyBuffer(g_vk.dev, a.buf, NULL);
      vkFreeMemory(g_vk.dev, a.mem, NULL);
      memmove(&g_vk.allocs[i], &g_vk.allocs[i + 1], (size_t)(g_vk.n_allocs - i - 1) * sizeof(ec_vk_alloc_t));
      g_vk.n_allocs--;
      break;
    }
  }
  pthread_mutex_unlock(&g_vk.mu);
  return cudaSuccess;
}

void *ec_hal_mmap_bind(const void *host_ptr, size_t nbytes) {
  (void)host_ptr; (void)nbytes;
  return NULL;
}

cudaError_t ec_hal_host_alloc_raw(void **ptr, size_t nbytes, unsigned int flags) {
  (void)flags;
  return ec_hal_malloc_raw(ptr, nbytes);
}
cudaError_t ec_hal_host_free_raw(void *ptr) { return ec_hal_free_raw(ptr); }

cudaError_t ec_hal_memcpy_async(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
  (void)kind; (void)stream;
  if (count > 0 && dst && src && dst != src) memcpy(dst, src, count);
  return cudaSuccess;
}
cudaError_t ec_hal_memset_async(void *dst, int value, size_t count, cudaStream_t stream) {
  (void)stream;
  if (count > 0 && dst) memset(dst, value, count);
  return cudaSuccess;
}
cudaError_t ec_hal_stream_create(cudaStream_t *pStream, unsigned int flags) {
  (void)flags;
  if (ec_hal_init() != 0) return cudaErrorNotSupported;
  if (pStream) *pStream = (cudaStream_t)(uintptr_t)1;
  return cudaSuccess;
}
cudaError_t ec_hal_stream_destroy(cudaStream_t s) { (void)s; return cudaSuccess; }
cudaError_t ec_hal_stream_sync(cudaStream_t s) { (void)s; return cudaSuccess; }
cudaError_t ec_hal_device_sync(void) { return cudaSuccess; }

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
  if (e) clock_gettime(CLOCK_MONOTONIC, &((ec_vk_ev_t *)e)->ts);
  return cudaSuccess;
}
cudaError_t ec_hal_event_sync(cudaEvent_t e) { (void)e; return cudaSuccess; }
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
  if ((((uintptr_t)dst) & 3u) == 0 && n >= 512) {
    size_t pairs = n >> 1;
    ec_vk_params_t pp = { .p0 = (uint32_t)(pairs * 2), .p1 = 0 };
    if (vk_dispatch_locked(VK_PIPE_PACK16X2, &pp, NULL, 4, NULL, 4, src, 4, NULL, 4, NULL, 4, dst, 4,
                           (uint32_t)((pairs + 255) / 256), 1, 1) == 0) {
      if (n & 1) dst[n - 1] = ec_cmn_f32_to_bf16(src[n - 1]);
      return;
    }
  }
  for (size_t i = 0; i < n; i++) dst[i] = ec_cmn_f32_to_bf16(src[i]);
}

static void pack_f32_to_f16_host_or_gpu_locked(const float *src, __half *dst, size_t n) {
  if (!src || !dst || n == 0) return;
  if ((((uintptr_t)dst) & 3u) == 0 && n >= 512) {
    size_t pairs = n >> 1;
    ec_vk_params_t pp = { .p0 = (uint32_t)(pairs * 2), .p1 = 2 };
    if (vk_dispatch_locked(VK_PIPE_PACK16X2, &pp, NULL, 4, NULL, 4, src, 4, NULL, 4, NULL, 4, dst, 4,
                           (uint32_t)((pairs + 255) / 256), 1, 1) == 0) {
      if (n & 1) dst[n - 1] = __float2half_rn(src[n - 1]);
      return;
    }
  }
  for (size_t i = 0; i < n; i++) dst[i] = __float2half_rn(src[i]);
}

// ===========================================================================
// Vulkan SPIR-V GPU Dispatches for Core Heavy Creative Operators
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
  int rc = vk_dispatch_locked(VK_PIPE_GEMM_BF16, &p, X, 2, W, 2, bias, 4, NULL, 4, dst_f32, 4, NULL, 4,
                              (uint32_t)((N + 15) / 16), (uint32_t)((M + 15) / 16), 1);
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
  int rc = vk_dispatch_locked(VK_PIPE_GEMM_F16, &p, X, 2, W, 2, bias, 4, res, 4, dst_f32, 4, NULL, 4,
                              (uint32_t)((N + 15) / 16), (uint32_t)((M + 15) / 16), 1);
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
  int rc = vk_dispatch_locked(VK_PIPE_GEMM_Q8, &p, X_q8, 1, W_q8, 1, x_sc, 4, w_sc, 4, dst_f32, 4, (void *)bias, 4,
                              (uint32_t)((N + 15) / 16), (uint32_t)((M + 15) / 16), 1);
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
  int rc = vk_dispatch_locked(VK_PIPE_GEMM_I8, &p, X_i8, 1, W_i8, 1, x_sc, 4, w_sc, 4, Y, 4, (void *)bias, 4,
                              (uint32_t)((N + 15) / 16), (uint32_t)((M + 15) / 16), 1);
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
  uint32_t pairs = (uint32_t)Lq * (uint32_t)Hq;
  int rc = vk_dispatch_locked(VK_PIPE_FA2_BF16, &p, Q, 2, K, 2, NULL, 4, NULL, 4, dst_f32, 4, (void *)V, 2,
                              pairs, 1, 1);
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
  uint32_t pairs = (uint32_t)n_q * (uint32_t)n_heads;
  int rc = vk_dispatch_locked(VK_PIPE_FA2_F16, &p, Q, 2, K, 2, NULL, 4, NULL, 4, dst_f32, 4, (void *)V, 2,
                              pairs, 1, 1);
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
  int rc = vk_dispatch_locked(VK_PIPE_CONV3X3, &p, X_nhwc, 2, W_ohwi, 2, bias, 4, res_f32, 4, dst_f32, 4, NULL, 4,
                              (uint32_t)((Cout + 15) / 16), (uint32_t)((M + 15) / 16), 1);
  if (rc == 0 && Y_bf16) pack_f32_to_bf16_host_or_gpu_locked(dst_f32, Y_bf16, M * Cout);
  pthread_mutex_unlock(&g_vk.mu);
  return rc;
}
