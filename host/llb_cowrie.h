// llb_cowrie.h — Native Cowrie v1.2 16 KB-aligned SHA-NI / ARMv8 SHA2 Merkle StatePage
// persistence & zero-copy mmap restore for Eureka Engine (hybrid GDN SSM + KV).
// Parity with vLLM CowrieKVConnectorV1 & llama.cpp LlamaPrefixStateCache:
//   - Private 0700 state directory + O_EXCL 0600 atomic StatePages
//   - Unified 64-stride cowrie_model_fingerprint (<2 ms across multi-GB GGUF)
//   - runtime_id = "eureka-llm", ssm_dtype = 2 (exact F32 GDN SSM), optional HMAC-SHA256
//   - Longest-prefix chunk scan (max_k -> pfx_C) across cold process restarts
#ifndef LLB_COWRIE_H
#define LLB_COWRIE_H

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

typedef struct CowrieStatePageHandle CowrieStatePageHandle;

typedef struct {
  int            loaded;
  void*          dl;
  const char*   (*err_name)(int);
  int           (*sha256_hex)(const uint8_t*, size_t, char*);
  int           (*model_fp)(const char*, char*);
  uint8_t       (*has_shani)(void);
  int           (*sp_save)(const char*, const char*, const char*, uint32_t, uint8_t, uint8_t, const uint8_t*, size_t);
  int           (*sp_save_ext)(const char*, const char*, const char*, const char*, uint16_t, uint32_t, uint8_t, uint8_t, const uint8_t*, size_t);
  int           (*sp_open_checked)(const char*, const char*, int, uint8_t, CowrieStatePageHandle**);
  int           (*sp_open_full)(const char*, const char*, const char*, int, int, uint8_t, CowrieStatePageHandle**);
  const uint8_t* (*sp_data)(const CowrieStatePageHandle*);
  size_t        (*sp_len)(const CowrieStatePageHandle*);
  void          (*sp_close)(CowrieStatePageHandle*);
} CowrieApi;

static const char* LLB_COWRIE_RUNTIME_ID = "eureka-llm";
static CowrieApi cowrie_api;
static const char* cowrie_dir_cached;
static int cowrie_dir_checked;
static char cowrie_model_sha[65];
static int cowrie_model_sha_ok;
static u32 cowrie_nl;

// Armed restore handle across layers 0..cowrie_nl-1 at pos == cowrie_arm_K
static CowrieStatePageHandle* cowrie_arm_h;
static u32 cowrie_arm_K;
static size_t cowrie_arm_layer_off;
static double cowrie_arm_t0;
static char cowrie_arm_path[1024];

// Active save accumulator across layers 0..cowrie_nl-1 at pos == cowrie_save_K
static u32 cowrie_save_K;
static uint8_t* cowrie_save_buf;
static size_t cowrie_save_cap;
static size_t cowrie_save_len;
static size_t cowrie_save_meta_off;
static u32 cowrie_save_layers_done;
static char cowrie_save_prefix_sha[65];
static double cowrie_save_t0;

typedef struct {
  char magic[4];      // "EUSP"
  u32  version;       // 1
  u32  nl;
  u32  pos;
  u32  kv_type;       // Eureka llb_kv_type (0=f32, 1=f16, 2=q8)
  u32  d, nh, nkv, hd;
  u32  lkh, lvh, sd;
} EurekaStateHdr;

static double cowrie_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static bool cowrie_init_api(void) {
  if (cowrie_api.loaded != 0) {
    return cowrie_api.loaded > 0;
  }
  const char* cands[5] = {
    getenv("COWRIE_LIB"),
    "/tmp/cowrie-cargo-target/release/libcowrie_core.so",
    "/home/omen/.local/lib/libcowrie_core.so",
    "libcowrie_core.so",
    NULL
  };
  void* dl = NULL;
  for (int i = 0; cands[i] != NULL; i += 1) {
    if (cands[i][0] == 0) continue;
    dl = dlopen(cands[i], RTLD_LAZY | RTLD_LOCAL);
    if (dl != NULL) break;
  }
  if (!dl) {
    fprintf(stderr, "cowrie: libcowrie_core.so not found (set COWRIE_LIB); StatePage persistence off\n");
    cowrie_api.loaded = -1;
    return false;
  }
  cowrie_api.dl              = dl;
  cowrie_api.err_name        = (const char* (*)(int))dlsym(dl, "cowrie_err_name");
  cowrie_api.sha256_hex      = (int (*)(const uint8_t*, size_t, char*))dlsym(dl, "cowrie_sha256_hex");
  cowrie_api.model_fp        = (int (*)(const char*, char*))dlsym(dl, "cowrie_model_fingerprint");
  cowrie_api.has_shani       = (uint8_t (*)(void))dlsym(dl, "cowrie_has_shani");
  cowrie_api.sp_save         = (int (*)(const char*, const char*, const char*, uint32_t, uint8_t, uint8_t, const uint8_t*, size_t))dlsym(dl, "cowrie_statepage_save");
  cowrie_api.sp_save_ext     = (int (*)(const char*, const char*, const char*, const char*, uint16_t, uint32_t, uint8_t, uint8_t, const uint8_t*, size_t))dlsym(dl, "cowrie_statepage_save_ext");
  cowrie_api.sp_open_checked = (int (*)(const char*, const char*, int, uint8_t, CowrieStatePageHandle**))dlsym(dl, "cowrie_statepage_open_checked");
  cowrie_api.sp_open_full    = (int (*)(const char*, const char*, const char*, int, int, uint8_t, CowrieStatePageHandle**))dlsym(dl, "cowrie_statepage_open_full");
  cowrie_api.sp_data         = (const uint8_t* (*)(const CowrieStatePageHandle*))dlsym(dl, "cowrie_statepage_data");
  cowrie_api.sp_len          = (size_t (*)(const CowrieStatePageHandle*))dlsym(dl, "cowrie_statepage_len");
  cowrie_api.sp_close        = (void (*)(CowrieStatePageHandle*))dlsym(dl, "cowrie_statepage_close");
  if (!cowrie_api.err_name || !cowrie_api.sha256_hex || !cowrie_api.sp_save
      || !cowrie_api.sp_open_checked || !cowrie_api.sp_data || !cowrie_api.sp_len || !cowrie_api.sp_close) {
    fprintf(stderr, "cowrie: incomplete libcowrie_core.so ABI\n");
    cowrie_api.loaded = -1;
    return false;
  }
  cowrie_api.loaded = 1;
  fprintf(stderr, "cowrie: loaded libcowrie_core.so v1.2 (HW-SHA256=%u)\n",
    cowrie_api.has_shani ? (unsigned)cowrie_api.has_shani() : 0u);
  return true;
}

static const char* cowrie_state_dir(void) {
  if (!cowrie_dir_checked) {
    cowrie_dir_checked = 1;
    const char* d = getenv("LLB_COWRIE_STATE_DIR");
    if (d && d[0] != 0) {
      if (cowrie_init_api()) {
        mkdir(d, 0700);
        chmod(d, 0700);
        cowrie_dir_cached = d;
      }
    }
  }
  return cowrie_dir_cached;
}

static uint8_t cowrie_env_kv_quant(void) {
  const char* kve = getenv("LLB_KV_TYPE");
  if (!kve || !kve[0] || strcmp(kve, "f32") == 0) return 3;
  if (strcmp(kve, "f16") == 0 || strcmp(kve, "auto") == 0) return 0;
  if (strcmp(kve, "q8") == 0 || strcmp(kve, "q8_0") == 0) return 2;
  return 3;
}

static uint8_t cowrie_live_kv_quant(void) {
  return llb_kv_type == 1 ? 0 : llb_kv_type == 2 ? 2 : 3;
}

static const char* cowrie_get_model_sha(void) {
  if (cowrie_model_sha_ok) return cowrie_model_sha;
  const char* gp = getenv("LLB_GGUF");
  if (!gp || !gp[0]) {
    memset(cowrie_model_sha, '0', 64);
    cowrie_model_sha[64] = 0;
    cowrie_model_sha_ok = 1;
    return cowrie_model_sha;
  }
  if (cowrie_api.model_fp && cowrie_api.model_fp(gp, cowrie_model_sha) == 0) {
    cowrie_model_sha_ok = 1;
    return cowrie_model_sha;
  }
  cowrie_api.sha256_hex((const uint8_t*)gp, strlen(gp), cowrie_model_sha);
  cowrie_model_sha_ok = 1;
  return cowrie_model_sha;
}

static unsigned cowrie_pfx_try_load(const u32* ids, u32 n) {
  const char* dir = cowrie_state_dir();
  if (!dir || n <= pfx_C) return 0;
  if (cowrie_arm_h != NULL) {
    cowrie_api.sp_close(cowrie_arm_h);
    cowrie_arm_h = NULL;
    cowrie_arm_K = 0;
  }
  const char* msha = cowrie_get_model_sha();
  int kvq = (int)cowrie_env_kv_quant();
  u32 max_k = ((n - 1) / pfx_C) * pfx_C;
  for (u32 K = max_k; K >= pfx_C; K = (K > pfx_C ? K - pfx_C : 0)) {
    char psha[65];
    cowrie_api.sha256_hex((const uint8_t*)ids, (size_t)K * sizeof(u32), psha);
    char path[1024];
    snprintf(path, sizeof path, "%s/%.16s_k%u.state.cowrie", dir, psha, K);
    struct stat st;
    if (stat(path, &st) != 0) {
      if (K == pfx_C) break;
      continue;
    }
    double t0 = cowrie_now_ms();
    CowrieStatePageHandle* h = NULL;
    int rc = cowrie_api.sp_open_full
        ? cowrie_api.sp_open_full(path, msha, LLB_COWRIE_RUNTIME_ID, kvq, 2 /* F32 SSM */, 1, &h)
        : cowrie_api.sp_open_checked(path, msha, kvq, 1, &h);
    if (rc != 0) {
      fprintf(stderr, "cowrie: rejected %s: %s (rc=%d)\n", path, cowrie_api.err_name(rc), rc);
      if (K == pfx_C) break;
      continue;
    }
    const uint8_t* data = cowrie_api.sp_data(h);
    size_t dlen = cowrie_api.sp_len(h);
    if (dlen < sizeof(EurekaStateHdr) + (size_t)K * sizeof(u32)) {
      cowrie_api.sp_close(h);
      if (K == pfx_C) break;
      continue;
    }
    EurekaStateHdr hdr;
    memcpy(&hdr, data, sizeof hdr);
    if (memcmp(hdr.magic, "EUSP", 4) != 0 || hdr.version != 1 || hdr.pos != K) {
      cowrie_api.sp_close(h);
      if (K == pfx_C) break;
      continue;
    }
    const uint8_t* stored_ids = data + sizeof(EurekaStateHdr);
    if (memcmp(stored_ids, ids, (size_t)K * sizeof(u32)) != 0) {
      cowrie_api.sp_close(h);
      if (K == pfx_C) break;
      continue;
    }
    cowrie_arm_h = h;
    cowrie_arm_K = K;
    cowrie_arm_t0 = t0;
    cowrie_arm_layer_off = sizeof(EurekaStateHdr) + (size_t)K * sizeof(u32) + hdr.nl + (size_t)hdr.nl * sizeof(u32);
    snprintf(cowrie_arm_path, sizeof cowrie_arm_path, "%s", path);
    return K;
  }
  return 0;
}

static void cowrie_pfx_plan_save(const u32* ids, u32 n, u32 restored_start) {
  const char* dir = cowrie_state_dir();
  cowrie_save_K = 0;
  cowrie_save_layers_done = 0;
  if (!dir || n <= pfx_C) return;
  u32 K = ((n - 1) / pfx_C) * pfx_C;
  if (K < pfx_C || K <= restored_start) return;
  cowrie_api.sha256_hex((const uint8_t*)ids, (size_t)K * sizeof(u32), cowrie_save_prefix_sha);
  char path[1024];
  snprintf(path, sizeof path, "%s/%.16s_k%u.state.cowrie", dir, cowrie_save_prefix_sha, K);
  struct stat st;
  if (stat(path, &st) == 0) return;
  cowrie_save_K = K;
}

static void cowrie_restore_layer(u32 i, bool full, u32 pos, LlbShape sh) {
  if (!cowrie_arm_h || pos != cowrie_arm_K) return;
  const uint8_t* data = cowrie_api.sp_data(cowrie_arm_h);
  size_t dlen = cowrie_api.sp_len(cowrie_arm_h);
  EurekaStateHdr hdr;
  memcpy(&hdr, data, sizeof hdr);
  if (hdr.nl != cowrie_nl || hdr.kv_type != llb_kv_type
      || hdr.d != sh.d || hdr.nh != sh.nh || hdr.nkv != sh.nkv || hdr.hd != sh.hd
      || hdr.lkh != sh.lkh || hdr.lvh != sh.lvh || hdr.sd != sh.sd) {
    err_fail("cowrie: StatePage shape or kv_type mismatch");
  }
  const uint8_t* is_full_arr = data + sizeof(EurekaStateHdr) + (size_t)hdr.pos * sizeof(u32);
  const u32* r_arr = (const u32*)(is_full_arr + hdr.nl);
  if ((is_full_arr[i] != 0) != full) {
    err_fail("cowrie: StatePage layer type mismatch");
  }

  if (full) {
    pass_kv(i, pos, sh);
    size_t rbytes = llb_kv_row_bytes((size_t)sh.nkv * sh.hd);
    size_t kv_bytes = (size_t)hdr.pos * rbytes;
    if (cowrie_arm_layer_off + 2 * kv_bytes > dlen) err_fail("cowrie: StatePage truncated (KV)");
    if (!llb_write(pass_kc[i], data + cowrie_arm_layer_off, kv_bytes)
        || !llb_write(pass_vc[i], data + cowrie_arm_layer_off + kv_bytes, kv_bytes)) {
      err_fail("cowrie: StatePage DMA restore failed (KV)");
    }
    cowrie_arm_layer_off += 2 * kv_bytes;
  } else {
    u32 cv = 2 * sh.lkh * sh.sd + sh.lvh * sh.sd;
    size_t nb = 4 * (size_t)cv * 4;
    size_t ns = (size_t)sh.lvh * sh.sd * sh.sd * 4;
    if (!pass_on[i]) {
      pass_b[i] = pass_zero(4 * (size_t)cv);
      pass_S[i] = pass_zero((size_t)sh.lvh * sh.sd * sh.sd);
    }
    if (cowrie_arm_layer_off + nb + ns > dlen) err_fail("cowrie: StatePage truncated (GDN)");
    if (!llb_write(pass_b[i], data + cowrie_arm_layer_off, nb)
        || !llb_write(pass_S[i], data + cowrie_arm_layer_off + nb, ns)) {
      err_fail("cowrie: StatePage DMA restore failed (GDN)");
    }
    pass_r[i] = r_arr[i];
    cowrie_arm_layer_off += nb + ns;
  }
  pass_on[i] = true;
  pass_pos[i] = pos - 1;

  if (i + 1 == cowrie_nl) {
    llb_sync();
    double ms = cowrie_now_ms() - cowrie_arm_t0;
    fprintf(stderr, "cowrie: restored K=%u (%.2f MB, kv=%u) in %.2f ms from %s (SHA-NI + HMAC verified)\n",
      cowrie_arm_K, (double)dlen / 1e6, (unsigned)cowrie_live_kv_quant(), ms, cowrie_arm_path);
    cowrie_api.sp_close(cowrie_arm_h);
    cowrie_arm_h = NULL;
    cowrie_arm_K = 0;
  }
}

static void cowrie_save_layer(u32 i, bool full, u32 pos, LlbShape sh) {
  if (cowrie_save_K == 0 || pos != cowrie_save_K || !pass_on[i]) return;
  if (i == 0) {
    cowrie_save_t0 = cowrie_now_ms();
    llb_sync();
    u32 cv = 2 * sh.lkh * sh.sd + sh.lvh * sh.sd;
    size_t nb = 4 * (size_t)cv * 4;
    size_t ns = (size_t)sh.lvh * sh.sd * sh.sd * 4;
    size_t kv_bytes = (size_t)pos * llb_kv_row_bytes((size_t)sh.nkv * sh.hd);
    size_t hdr_and_meta = sizeof(EurekaStateHdr) + (size_t)pos * sizeof(u32) + cowrie_nl + (size_t)cowrie_nl * sizeof(u32);
    size_t total = hdr_and_meta;
    for (u32 l = 0; l < cowrie_nl; l += 1) {
      total += llb_full(l) ? (2 * kv_bytes) : (nb + ns);
    }
    if (total > cowrie_save_cap) {
      uint8_t* p = realloc(cowrie_save_buf, total);
      if (!p) err_fail("cowrie: OOM allocating StatePage save buffer");
      cowrie_save_buf = p;
      cowrie_save_cap = total;
    }
    EurekaStateHdr hdr;
    memcpy(hdr.magic, "EUSP", 4);
    hdr.version = 1;
    hdr.nl = cowrie_nl;
    hdr.pos = pos;
    hdr.kv_type = llb_kv_type;
    hdr.d = sh.d; hdr.nh = sh.nh; hdr.nkv = sh.nkv; hdr.hd = sh.hd;
    hdr.lkh = sh.lkh; hdr.lvh = sh.lvh; hdr.sd = sh.sd;
    memcpy(cowrie_save_buf, &hdr, sizeof hdr);
    memcpy(cowrie_save_buf + sizeof hdr, pfx_ids, (size_t)pos * sizeof(u32));
    cowrie_save_meta_off = sizeof hdr + (size_t)pos * sizeof(u32);
    memset(cowrie_save_buf + cowrie_save_meta_off, 0, cowrie_nl + (size_t)cowrie_nl * sizeof(u32));
    cowrie_save_len = hdr_and_meta;
    cowrie_save_layers_done = 0;
  }

  uint8_t* is_full_arr = cowrie_save_buf + cowrie_save_meta_off;
  u32* r_arr = (u32*)(is_full_arr + cowrie_nl);
  is_full_arr[i] = full ? 1 : 0;
  r_arr[i] = full ? 0 : pass_r[i];

  if (full) {
    size_t kv_bytes = (size_t)pos * llb_kv_row_bytes((size_t)sh.nkv * sh.hd);
    if (!llb_read(cowrie_save_buf + cowrie_save_len, pass_kc[i], kv_bytes)
        || !llb_read(cowrie_save_buf + cowrie_save_len + kv_bytes, pass_vc[i], kv_bytes)) {
      err_fail("cowrie: StatePage VRAM read failed (KV)");
    }
    cowrie_save_len += 2 * kv_bytes;
  } else {
    u32 cv = 2 * sh.lkh * sh.sd + sh.lvh * sh.sd;
    size_t nb = 4 * (size_t)cv * 4;
    size_t ns = (size_t)sh.lvh * sh.sd * sh.sd * 4;
    if (!llb_read(cowrie_save_buf + cowrie_save_len, pass_b[i], nb)
        || !llb_read(cowrie_save_buf + cowrie_save_len + nb, pass_S[i], ns)) {
      err_fail("cowrie: StatePage VRAM read failed (GDN)");
    }
    cowrie_save_len += nb + ns;
  }
  cowrie_save_layers_done += 1;

  if (i + 1 == cowrie_nl && cowrie_save_layers_done == cowrie_nl) {
    const char* dir = cowrie_state_dir();
    char path[1024];
    snprintf(path, sizeof path, "%s/%.16s_k%u.state.cowrie", dir, cowrie_save_prefix_sha, cowrie_save_K);
    int rc = cowrie_api.sp_save_ext
        ? cowrie_api.sp_save_ext(
            path, cowrie_get_model_sha(), cowrie_save_prefix_sha,
            LLB_COWRIE_RUNTIME_ID, 1,
            cowrie_save_K, cowrie_live_kv_quant(), 2 /* exact F32 GDN SSM */,
            cowrie_save_buf, cowrie_save_len)
        : cowrie_api.sp_save(
            path, cowrie_get_model_sha(), cowrie_save_prefix_sha,
            cowrie_save_K, cowrie_live_kv_quant(), 2,
            cowrie_save_buf, cowrie_save_len);
    double ms = cowrie_now_ms() - cowrie_save_t0;
    if (rc != 0) {
      fprintf(stderr, "cowrie: save failed for %s: %s (rc=%d)\n", path, cowrie_api.err_name(rc), rc);
    } else {
      fprintf(stderr, "cowrie: saved K=%u (%.2f MB, kv=%u) in %.2f ms -> %s (16KB SHA-NI + HMAC)\n",
        cowrie_save_K, (double)cowrie_save_len / 1e6, (unsigned)cowrie_live_kv_quant(), ms, path);
    }
    cowrie_save_K = 0;
  }
}

#endif // LLB_COWRIE_H
