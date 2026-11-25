// backends/vulkan/llb_vulkan_gemv.c — Native Cross-Vendor Vulkan 1.2+ LLM GEMV Engine
// Packs model weight tensors into <=1 GiB pure DEVICE_LOCAL VRAM slabs (or single unified buffers
// on APUs) and accelerates Q4_K / Q5_K / Q6_K / Q8_0 / F32 GEMV for Qwen3.5 (0.8B..27B) via SPIR-V 1.3.

#define _POSIX_C_SOURCE 200809L
#include "llb_vulkan_gemv.h"
#include "llb_vulkan_spv.h"
#include "../../host/llb_formats.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

typedef struct {
  uint32_t rows;
  uint32_t cols;
  uint32_t rb;
  uint32_t w_off;
  uint32_t row_base;
  uint32_t p5, p6, p7;
} llb_vk_params_t;

typedef enum {
  LVK_PIPE_Q4K  = 0,
  LVK_PIPE_Q5K  = 1,
  LVK_PIPE_Q6K  = 2,
  LVK_PIPE_Q8_0 = 3,
  LVK_PIPE_F32  = 4,
  LVK_PIPE_COUNT = 5
} llb_vk_pipe_t;

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

#define LVK_MAX_SLABS   64
#define LVK_MAX_ENTRIES 2048
#define LVK_CHUNK_SLOTS 16
#define LVK_UBO_STRIDE  256
#define LVK_MAX_WG_X    32768u
#define LVK_STAGE_BYTES (32ULL << 20)

static struct {
  int                               active;
  int                               is_unified;
  const uint8_t                    *pack_base;
  size_t                            pack_bytes;
  VkInstance                        instance;
  VkPhysicalDevice                  phys;
  VkDevice                          dev;
  VkQueue                           queue;
  uint32_t                          qfam;
  VkPhysicalDeviceProperties        props;
  VkPhysicalDeviceMemoryProperties  mem_props;
  size_t                            vram_total;
  size_t                            vram_used;
  char                              info[256];

  VkCommandPool                     cmd_pool;
  VkCommandBuffer                   cmd;
  VkFence                           fence;
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
  VkBufferCreateInfo bci = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = sz, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
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

static int cmp_span(const void *a, const void *b) {
  uint64_t oa = ((const llb_vk_tensor_span_t *)a)->off;
  uint64_t ob = ((const llb_vk_tensor_span_t *)b)->off;
  return (oa < ob) ? -1 : (oa > ob) ? 1 : 0;
}

static int upload_slab_chunked(VkBuffer dst_buf, size_t dst_off, const uint8_t *src, size_t bytes,
                               VkBuffer stage_buf, uint8_t *stage_map) {
  size_t pos = 0;
  while (pos < bytes) {
    size_t chunk = bytes - pos;
    if (chunk > LVK_STAGE_BYTES) chunk = LVK_STAGE_BYTES;
    memcpy(stage_map, src + pos, chunk);

    vkResetCommandBuffer(g_lvk.cmd, 0);
    VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(g_lvk.cmd, &bi);
    VkBufferCopy cp = { .srcOffset = 0, .dstOffset = dst_off + pos, .size = chunk };
    vkCmdCopyBuffer(g_lvk.cmd, stage_buf, dst_buf, 1, &cp);
    vkEndCommandBuffer(g_lvk.cmd);

    vkResetFences(g_lvk.dev, 1, &g_lvk.fence);
    VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &g_lvk.cmd,
    };
    if (vkQueueSubmit(g_lvk.queue, 1, &si, g_lvk.fence) != VK_SUCCESS ||
        vkWaitForFences(g_lvk.dev, 1, &g_lvk.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
      return -1;
    }
    pos += chunk;
  }
  return 0;
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

  VkApplicationInfo app = {
    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .pApplicationName = "eureka-engine-vulkan",
    .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
    .pEngineName = "eureka-llm",
    .engineVersion = VK_MAKE_VERSION(1, 0, 0),
    .apiVersion = VK_API_VERSION_1_2,
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

  // Skip software CPU Vulkan (llvmpipe/lavapipe) unless explicitly requested via LLB_VULKAN_DEVICE
  if (g_lvk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU && !(want && want[0])) {
    return -1;
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

  uint32_t n_qf = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(g_lvk.phys, &n_qf, NULL);
  VkQueueFamilyProperties qfs[32];
  if (n_qf > 32) n_qf = 32;
  vkGetPhysicalDeviceQueueFamilyProperties(g_lvk.phys, &n_qf, qfs);
  g_lvk.qfam = 0;
  for (uint32_t i = 0; i < n_qf; i++) {
    if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_lvk.qfam = i; break; }
  }

  float qprio = 1.0f;
  VkDeviceQueueCreateInfo qci = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .queueFamilyIndex = g_lvk.qfam, .queueCount = 1, .pQueuePriorities = &qprio,
  };
  VkDeviceCreateInfo dci = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
  };
  if (vkCreateDevice(g_lvk.phys, &dci, NULL, &g_lvk.dev) != VK_SUCCESS) return -1;
  vkGetDeviceQueue(g_lvk.dev, g_lvk.qfam, 0, &g_lvk.queue);

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

  void *um = NULL;
  if (alloc_host_buf((size_t)LVK_CHUNK_SLOTS * LVK_UBO_STRIDE, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     0, &g_lvk.ubo_buf, &g_lvk.ubo_mem, &um) != 0) return -1;
  g_lvk.ubo_map = (uint8_t *)um;

  // Allocate persistent x and y vectors (DEVICE_LOCAL on discrete GPUs, single unified buffer on APUs)
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

  // Descriptor layout & pipelines (0: UBO, 1: W, 2: x, 3: y)
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
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = 4, .pBindings = binds,
  };
  vkCreateDescriptorSetLayout(g_lvk.dev, &dslci, NULL, &g_lvk.dsl);
  VkPipelineLayoutCreateInfo plci = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 1, .pSetLayouts = &g_lvk.dsl,
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

  const struct { const uint32_t *code; size_t bytes; const char *entry; } specs[LVK_PIPE_COUNT] = {
    { ec_spv_vk_gemv_q4k,  sizeof(ec_spv_vk_gemv_q4k),  "vk_gemv_q4k"  },
    { ec_spv_vk_gemv_q5k,  sizeof(ec_spv_vk_gemv_q5k),  "vk_gemv_q5k"  },
    { ec_spv_vk_gemv_q6k,  sizeof(ec_spv_vk_gemv_q6k),  "vk_gemv_q6k"  },
    { ec_spv_vk_gemv_q8_0, sizeof(ec_spv_vk_gemv_q8_0), "vk_gemv_q8_0" },
    { ec_spv_vk_gemv_f32,  sizeof(ec_spv_vk_gemv_f32),  "vk_gemv_f32"  },
  };
  for (int i = 0; i < LVK_PIPE_COUNT; i++) {
    VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = specs[i].bytes, .pCode = specs[i].code,
    };
    if (vkCreateShaderModule(g_lvk.dev, &smci, NULL, &g_lvk.shaders[i]) != VK_SUCCESS) return -1;
    VkComputePipelineCreateInfo cp = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = g_lvk.shaders[i], .pName = specs[i].entry,
      },
      .layout = g_lvk.pipe_layout,
    };
    if (vkCreateComputePipelines(g_lvk.dev, VK_NULL_HANDLE, 1, &cp, NULL, &g_lvk.pipes[i]) != VK_SUCCESS) return -1;
  }

  // Deduplicate & sort tensor spans by pack_off, then pack into <=1 GiB DEVICE_LOCAL VRAM slabs!
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

  size_t max_slab = 1024ULL << 20; // 1 GiB per slab (always <= maxStorageBufferRange & u32 safe)
  if (g_lvk.props.limits.maxStorageBufferRange > 0 &&
      (size_t)g_lvk.props.limits.maxStorageBufferRange < max_slab) {
    max_slab = (size_t)g_lvk.props.limits.maxStorageBufferRange & ~255ULL;
  }
  size_t slab_sizes[LVK_MAX_SLABS];
  memset(slab_sizes, 0, sizeof slab_sizes);
  int cur_slab = 0;
  for (int i = 0; i < n_uniq; i++) {
    size_t tsz = ((size_t)uniq[i].bytes + 255ULL) & ~255ULL;
    if (slab_sizes[cur_slab] > 0 && slab_sizes[cur_slab] + tsz > max_slab) {
      cur_slab++;
      if (cur_slab >= LVK_MAX_SLABS) break;
    }
    g_lvk.entries[g_lvk.n_entries++] = (llb_vk_entry_t){
      .pack_off = uniq[i].off,
      .bytes    = uniq[i].bytes,
      .slab_idx = cur_slab,
      .slab_off = slab_sizes[cur_slab],
    };
    slab_sizes[cur_slab] += tsz;
  }
  int total_slabs = (g_lvk.n_entries > 0) ? (cur_slab + 1) : 0;
  free(uniq);

  // Allocate a temporary 32 MiB host staging buffer to DMA weight slabs into pure DEVICE_LOCAL VRAM
  VkBuffer stage_buf = VK_NULL_HANDLE;
  VkDeviceMemory stage_mem = VK_NULL_HANDLE;
  void *stage_map = NULL;
  if (alloc_host_buf(LVK_STAGE_BYTES, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 0,
                     &stage_buf, &stage_mem, &stage_map) != 0) return -1;

  size_t uploaded_bytes = 0;
  int resident_slabs = 0;
  for (int s = 0; s < total_slabs; s++) {
    if (alloc_dev_buf(slab_sizes[s], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      &g_lvk.slabs[s].buf, &g_lvk.slabs[s].mem) != 0) {
      break; // Partial VRAM residency: tensors in higher slabs transparently fall back to CPU!
    }
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
  snprintf(g_lvk.info, sizeof g_lvk.info, "%s (Vulkan %u.%u, %s, %.2f GiB VRAM resident across %d slab%s, %d/%d tensors)",
           g_lvk.props.deviceName,
           VK_API_VERSION_MAJOR(g_lvk.props.apiVersion), VK_API_VERSION_MINOR(g_lvk.props.apiVersion),
           g_lvk.is_unified ? "Unified" : "DEVICE_LOCAL",
           (double)uploaded_bytes / (1024.0 * 1024.0 * 1024.0),
           g_lvk.n_slabs, g_lvk.n_slabs == 1 ? "" : "s",
           resident_tensors, g_lvk.n_entries);
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
  if (total_w_bytes > 0xFFFFFFFFULL) return 0; // Guard u32 offset inside shader

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

  uint32_t total_wgs = ((uint32_t)rows + 1u) >> 1; // 2 rows per 64-thread workgroup
  uint32_t wg_done = 0;
  int slot = 0;
  while (wg_done < total_wgs && slot < LVK_CHUNK_SLOTS) {
    uint32_t chunk_wgs = total_wgs - wg_done;
    if (chunk_wgs > LVK_MAX_WG_X) chunk_wgs = LVK_MAX_WG_X;

    llb_vk_params_t p = {
      .rows = (uint32_t)rows,
      .cols = (uint32_t)cols,
      .rb = (uint32_t)rb,
      .w_off = sub_off,
      .row_base = wg_done * 2u,
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
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
    };
    vkCmdPipelineBarrier(g_lvk.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb1, 0, NULL, 0, NULL);
    VkBufferCopy ycp = { .srcOffset = 0, .dstOffset = 0, .size = y_bytes };
    vkCmdCopyBuffer(g_lvk.cmd, g_lvk.y_dev_buf, g_lvk.y_host_buf, 1, &ycp);
    VkMemoryBarrier mb2 = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(g_lvk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &mb2, 0, NULL, 0, NULL);
  } else {
    VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(g_lvk.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
  }
  vkEndCommandBuffer(g_lvk.cmd);

  vkResetFences(g_lvk.dev, 1, &g_lvk.fence);
  VkSubmitInfo si = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .commandBufferCount = 1, .pCommandBuffers = &g_lvk.cmd,
  };
  if (vkQueueSubmit(g_lvk.queue, 1, &si, g_lvk.fence) != VK_SUCCESS ||
      vkWaitForFences(g_lvk.dev, 1, &g_lvk.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
    return 0;
  }

  memcpy(y, g_lvk.y_host_map, y_bytes);
  return 1;
}

const char *llb_vk_device_info(void) { return g_lvk.active ? g_lvk.info : ""; }

int llb_vk_mem_info(size_t *free_b, size_t *total_b) {
  if (!g_lvk.active) return -1;
  if (total_b) *total_b = g_lvk.vram_total;
  if (free_b) *free_b = (g_lvk.vram_total > g_lvk.vram_used) ? (g_lvk.vram_total - g_lvk.vram_used) : 0;
  return 0;
}
