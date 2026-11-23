// ec_cowrie.c — Optional Zero-Build-Dep Tessera / Cowrie v2 Bridge for Eureka Creative Engine
// Pure portable C11 using dlopen("libcowrie_core.so") only; clean fallback when library absent.

#define _POSIX_C_SOURCE 200809L
#include "ec_cowrie.h"

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

// ===========================================================================
// 1. Exact libcowrie_core.so C-ABI Function Pointers (cowrie-core/src/ffi.rs)
// ===========================================================================
typedef struct {
  uint8_t *data;
  size_t   len;
  size_t   cap;
} cw_buf_t;

typedef struct CowrieStatePageHandle cw_sp_handle_t;

typedef void           (*fn_buf_free_t)(cw_buf_t);
typedef const char    *(*fn_err_name_t)(int);
typedef uint8_t        (*fn_has_shani_t)(void);
typedef int            (*fn_sha256_hex_t)(const uint8_t *, size_t, char *);
typedef int            (*fn_encode_json_t)(const uint8_t *, size_t, uint8_t, uint8_t, cw_buf_t *);
typedef int            (*fn_decode_json_t)(const uint8_t *, size_t, cw_buf_t *);
typedef int            (*fn_frame_verify_t)(const uint8_t *, size_t, uint8_t *, uint8_t *, uint8_t *);
typedef int            (*fn_project_lens_t)(const uint8_t *, size_t, uint8_t, size_t, cw_buf_t *);
typedef uint8_t        (*fn_propagate_auth_t)(uint8_t, const uint8_t *, size_t);
typedef uint8_t        (*fn_is_tainted_t)(uint8_t);
typedef int            (*fn_sp_save_t)(const char *, const char *, const char *, uint32_t, uint8_t, uint8_t, const uint8_t *, size_t);
typedef int            (*fn_sp_open_chk_t)(const char *, const char *, int, uint8_t, cw_sp_handle_t **);
typedef const uint8_t *(*fn_sp_data_t)(const cw_sp_handle_t *);
typedef size_t         (*fn_sp_len_t)(const cw_sp_handle_t *);
typedef void           (*fn_sp_close_t)(cw_sp_handle_t *);

typedef struct {
  void               *dl;
  int                 loaded;
  int                 shani;
  fn_buf_free_t       buf_free;
  fn_err_name_t       err_name;
  fn_has_shani_t      has_shani;
  fn_sha256_hex_t     sha256_hex;
  fn_encode_json_t    encode_json;
  fn_decode_json_t    decode_json;
  fn_frame_verify_t   frame_verify;
  fn_project_lens_t   project_lens;
  fn_propagate_auth_t propagate_auth;
  fn_is_tainted_t     is_tainted;
  fn_sp_save_t        sp_save;
  fn_sp_open_chk_t    sp_open_checked;
  fn_sp_data_t        sp_data;
  fn_sp_len_t         sp_len;
  fn_sp_close_t       sp_close;
} cw_api_t;

static cw_api_t        g_cw;
static pthread_once_t  g_cw_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_cw_mu   = PTHREAD_MUTEX_INITIALIZER;

static _Atomic uint64_t g_sp_saves      = 0;
static _Atomic uint64_t g_sp_hits       = 0;
static _Atomic uint64_t g_sp_misses     = 0;
static _Atomic uint64_t g_sp_mismatches = 0;

static char g_cache_dir[320] = "/tmp/eureka-creative-build/artifacts/cowrie_cache";

#define EC_TRUST_MAX 256
typedef struct {
  int     used;
  char    cid_hex[69];
  char    handle[24];
  char    source[64];
  char    label[96];
  uint8_t authority;
  uint8_t epistemic;
  int     tainted;
  size_t  raw_bytes;
  size_t  cowr_bytes;
  char    r0[160];
  char    r1[1024];
} trust_rec_t;

static trust_rec_t g_trust[EC_TRUST_MAX];
static int         g_trust_next  = 0;
static int         g_trust_count = 0;

static void cw_load_once(void) {
  memset(&g_cw, 0, sizeof g_cw);
  const char *env_cache = getenv("EC_COWRIE_CACHE_DIR");
  if (env_cache && env_cache[0]) {
    snprintf(g_cache_dir, sizeof g_cache_dir, "%s", env_cache);
  }

  const char *env_lib = getenv("COWRIE_LIB");
  const char *cands[5];
  int nc = 0;
  if (env_lib && env_lib[0]) cands[nc++] = env_lib;
  cands[nc++] = "/tmp/cowrie-cargo-target/release/libcowrie_core.so";
  cands[nc++] = "/home/omen/.local/lib/libcowrie_core.so";
  cands[nc++] = "/usr/local/lib/libcowrie_core.so";
  cands[nc++] = "libcowrie_core.so";

  void *h = NULL;
  for (int i = 0; i < nc && !h; i++) {
    h = dlopen(cands[i], RTLD_NOW | RTLD_LOCAL);
  }
  if (!h) return;

  g_cw.dl              = h;
  g_cw.buf_free        = (fn_buf_free_t)dlsym(h, "cowrie_buf_free");
  g_cw.err_name        = (fn_err_name_t)dlsym(h, "cowrie_err_name");
  g_cw.has_shani       = (fn_has_shani_t)dlsym(h, "cowrie_has_shani");
  g_cw.sha256_hex      = (fn_sha256_hex_t)dlsym(h, "cowrie_sha256_hex");
  g_cw.encode_json     = (fn_encode_json_t)dlsym(h, "cowrie_encode_json");
  g_cw.decode_json     = (fn_decode_json_t)dlsym(h, "cowrie_decode_to_json");
  g_cw.frame_verify    = (fn_frame_verify_t)dlsym(h, "cowrie_frame_verify");
  g_cw.project_lens    = (fn_project_lens_t)dlsym(h, "cowrie_project_lens");
  g_cw.propagate_auth  = (fn_propagate_auth_t)dlsym(h, "cowrie_propagate_authority");
  g_cw.is_tainted      = (fn_is_tainted_t)dlsym(h, "cowrie_is_tainted");
  g_cw.sp_save         = (fn_sp_save_t)dlsym(h, "cowrie_statepage_save");
  g_cw.sp_open_checked = (fn_sp_open_chk_t)dlsym(h, "cowrie_statepage_open_checked");
  g_cw.sp_data         = (fn_sp_data_t)dlsym(h, "cowrie_statepage_data");
  g_cw.sp_len          = (fn_sp_len_t)dlsym(h, "cowrie_statepage_len");
  g_cw.sp_close        = (fn_sp_close_t)dlsym(h, "cowrie_statepage_close");

  if (!g_cw.buf_free || !g_cw.sha256_hex || !g_cw.encode_json || !g_cw.frame_verify ||
      !g_cw.project_lens || !g_cw.propagate_auth || !g_cw.is_tainted ||
      !g_cw.sp_save || !g_cw.sp_open_checked || !g_cw.sp_data || !g_cw.sp_len || !g_cw.sp_close) {
    dlclose(h);
    memset(&g_cw, 0, sizeof g_cw);
    return;
  }
  g_cw.loaded = 1;
  g_cw.shani  = g_cw.has_shani ? (int)g_cw.has_shani() : 0;
}

int ec_cowrie_init(void) {
  pthread_once(&g_cw_once, cw_load_once);
  return g_cw.loaded ? 0 : -1;
}

int ec_cowrie_available(void) {
  pthread_once(&g_cw_once, cw_load_once);
  return g_cw.loaded;
}

int ec_cowrie_has_shani(void) {
  pthread_once(&g_cw_once, cw_load_once);
  return g_cw.loaded ? g_cw.shani : 0;
}

static void cw_mkdir_p(const char *path) {
  char tmp[384];
  snprintf(tmp, sizeof tmp, "%s", path);
  for (char *p = tmp + 1; *p; p++) {
    if (*p == '/') {
      *p = '\0';
      mkdir(tmp, 0755);
      *p = '/';
    }
  }
  mkdir(tmp, 0755);
}

void ec_cowrie_set_cache_dir(const char *dir) {
  pthread_once(&g_cw_once, cw_load_once);
  const char *env_cache = getenv("EC_COWRIE_CACHE_DIR");
  const char *use = (env_cache && env_cache[0]) ? env_cache : dir;
  if (!use || !use[0]) return;
  pthread_mutex_lock(&g_cw_mu);
  snprintf(g_cache_dir, sizeof g_cache_dir, "%s", use);
  cw_mkdir_p(g_cache_dir);
  pthread_mutex_unlock(&g_cw_mu);
}

const char *ec_cowrie_get_cache_dir(void) {
  pthread_once(&g_cw_once, cw_load_once);
  return g_cache_dir;
}

void ec_cowrie_get_stats(ec_cowrie_stats_t *out) {
  if (!out) return;
  memset(out, 0, sizeof *out);
  pthread_once(&g_cw_once, cw_load_once);
  out->enabled       = g_cw.loaded;
  out->shani         = g_cw.shani;
  out->sp_saves      = atomic_load(&g_sp_saves);
  out->sp_hits       = atomic_load(&g_sp_hits);
  out->sp_misses     = atomic_load(&g_sp_misses);
  out->sp_mismatches = atomic_load(&g_sp_mismatches);
  pthread_mutex_lock(&g_cw_mu);
  out->trust_entries = g_trust_count;
  snprintf(out->cache_dir, sizeof out->cache_dir, "%s", g_cache_dir);
  pthread_mutex_unlock(&g_cw_mu);
}

// ===========================================================================
// 2. Hashing & Sub-Millisecond Model Contract Fingerprinting
// ===========================================================================
static void cw_bytes_to_hex(const uint8_t *in, size_t n, char *out) {
  static const char H[] = "0123456789abcdef";
  for (size_t i = 0; i < n; i++) {
    out[2 * i]     = H[(in[i] >> 4) & 0xFu];
    out[2 * i + 1] = H[in[i] & 0xFu];
  }
  out[2 * n] = '\0';
}

// Deterministic 64-hex FNV-1a fallback when libcowrie_core.so is absent
static void cw_fallback_hex64(const void *data, size_t len, char out_hex_65[65]) {
  uint64_t h[4] = {
    14695981039346656037ULL, 0x9E3779B97F4A7C15ULL,
    0xBF58476D1CE4E5B9ULL,   0x94D049BB133111EBULL
  };
  const uint8_t *p = (const uint8_t *)data;
  for (size_t i = 0; i < len; i++) {
    for (int k = 0; k < 4; k++) {
      h[k] ^= (uint64_t)p[i] ^ ((uint64_t)k << 8);
      h[k] *= 1099511628211ULL;
    }
  }
  snprintf(out_hex_65, 65, "%016llx%016llx%016llx%016llx",
           (unsigned long long)h[0], (unsigned long long)h[1],
           (unsigned long long)h[2], (unsigned long long)h[3]);
}

int ec_cowrie_sha256_hex(const void *data, size_t len, char out_hex_65[65]) {
  if (!out_hex_65) return -1;
  pthread_once(&g_cw_once, cw_load_once);
  if (g_cw.loaded && g_cw.sha256_hex) {
    static const uint8_t dummy = 0;
    const uint8_t *ptr = (data && len > 0) ? (const uint8_t *)data : &dummy;
    if (g_cw.sha256_hex(ptr, len, out_hex_65) == 0) return 0;
  }
  cw_fallback_hex64(data ? data : "", len, out_hex_65);
  return 0;
}

int ec_cowrie_file_fingerprint(const char *path, char out_hex_65[65]) {
  if (!out_hex_65) return -1;
  out_hex_65[0] = '\0';
  if (!path || !path[0]) {
    return ec_cowrie_sha256_hex("empty", 5, out_hex_65);
  }
  struct stat st;
  if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) {
    return ec_cowrie_sha256_hex(path, strlen(path), out_hex_65);
  }
  FILE *f = fopen(path, "rb");
  if (!f) {
    return ec_cowrie_sha256_hex(path, strlen(path), out_hex_65);
  }
  const size_t CHUNK = 65536;
  uint8_t *buf = (uint8_t *)malloc(8 + 2 * CHUNK);
  if (!buf) {
    fclose(f);
    return ec_cowrie_sha256_hex(path, strlen(path), out_hex_65);
  }
  uint64_t fsz = (uint64_t)st.st_size;
  memcpy(buf, &fsz, 8);
  size_t pos = 8;
  size_t n1 = fread(buf + pos, 1, CHUNK, f);
  pos += n1;
  if (fsz > CHUNK) {
    long tail_off = (long)(fsz - CHUNK);
    if (fseek(f, tail_off, SEEK_SET) == 0) {
      size_t n2 = fread(buf + pos, 1, CHUNK, f);
      pos += n2;
    }
  }
  fclose(f);
  int rc = ec_cowrie_sha256_hex(buf, pos, out_hex_65);
  free(buf);
  return rc;
}

int ec_cowrie_composite_sha(const char *const *parts, int n_parts, char out_hex_65[65]) {
  char joined[2048];
  size_t pos = 0;
  joined[0] = '\0';
  for (int i = 0; i < n_parts && pos + 8 < sizeof joined; i++) {
    const char *s = (parts && parts[i]) ? parts[i] : "";
    int n = snprintf(joined + pos, sizeof joined - pos, "%s%s", i ? "|" : "", s);
    if (n > 0) pos += (size_t)n;
  }
  return ec_cowrie_sha256_hex(joined, pos, out_hex_65);
}

// ===========================================================================
// 3. Phase A: 16 KB-Aligned SHA-NI StatePage Persistence
// ===========================================================================
static void cw_sp_build_path(const char *ns, const char *doc_cid, char *out_path, size_t cap) {
  pthread_mutex_lock(&g_cw_mu);
  char dir[320];
  snprintf(dir, sizeof dir, "%s", g_cache_dir);
  pthread_mutex_unlock(&g_cw_mu);
  snprintf(out_path, cap, "%s/%s_%.24s.state.cowrie", dir, ns ? ns : "sp", doc_cid);
}

int ec_cowrie_sp_save_file(
    const char *path,
    const char *model_sha_64,
    const char *doc_cid_64,
    uint32_t ctx_tokens, uint8_t kv_quant,
    const void *state_data, size_t state_len) {
  pthread_once(&g_cw_once, cw_load_once);
  if (!g_cw.loaded || !path || !model_sha_64 || !doc_cid_64 || !state_data || state_len == 0) {
    return -1;
  }
  int rc = g_cw.sp_save(path, model_sha_64, doc_cid_64, ctx_tokens, kv_quant, 0,
                        (const uint8_t *)state_data, state_len);
  if (rc == 0) {
    atomic_fetch_add(&g_sp_saves, 1);
    return 0;
  }
  return -rc;
}

int ec_cowrie_sp_load_file(
    const char *path,
    const char *expected_model_sha_64,
    int expected_kv_quant,
    void *out_buf, size_t max_len, size_t *out_actual_len) {
  pthread_once(&g_cw_once, cw_load_once);
  if (!g_cw.loaded || !path || !out_buf) return -1;
  cw_sp_handle_t *h = NULL;
  int rc = g_cw.sp_open_checked(path, expected_model_sha_64, expected_kv_quant, 1, &h);
  if (rc != 0 || !h) {
    if (rc == EC_COWRIE_ERR_MODEL_MISMATCH ||
        rc == EC_COWRIE_ERR_STATEPAGE_HASH_MISMATCH ||
        rc == EC_COWRIE_ERR_KV_QUANT_MISMATCH) {
      atomic_fetch_add(&g_sp_mismatches, 1);
    } else {
      atomic_fetch_add(&g_sp_misses, 1);
    }
    return rc > 0 ? -rc : -1;
  }
  size_t slen = g_cw.sp_len(h);
  const uint8_t *sptr = g_cw.sp_data(h);
  if (out_actual_len) *out_actual_len = slen;
  if (!sptr || slen == 0 || slen > max_len) {
    g_cw.sp_close(h);
    atomic_fetch_add(&g_sp_misses, 1);
    return -1;
  }
  memcpy(out_buf, sptr, slen);
  g_cw.sp_close(h);
  atomic_fetch_add(&g_sp_hits, 1);
  return 0;
}

int ec_cowrie_sp_save(
    const char *ns,
    const char *model_sha_64,
    const void *key_data, size_t key_len,
    uint32_t ctx_tokens, uint8_t kv_quant,
    const void *state_data, size_t state_len) {
  if (!ec_cowrie_available() || !model_sha_64 || strlen(model_sha_64) != 64 || !state_data || state_len == 0) {
    return -1;
  }
  pthread_mutex_lock(&g_cw_mu);
  cw_mkdir_p(g_cache_dir);
  pthread_mutex_unlock(&g_cw_mu);

  char doc_cid[65];
  ec_cowrie_sha256_hex(key_data, key_len, doc_cid);
  char path[512];
  cw_sp_build_path(ns, doc_cid, path, sizeof path);
  return ec_cowrie_sp_save_file(path, model_sha_64, doc_cid, ctx_tokens, kv_quant, state_data, state_len);
}

int ec_cowrie_sp_load(
    const char *ns,
    const char *expected_model_sha_64,
    const void *key_data, size_t key_len,
    int expected_kv_quant,
    void *out_buf, size_t expected_len) {
  if (!ec_cowrie_available() || !expected_model_sha_64 || strlen(expected_model_sha_64) != 64 ||
      !out_buf || expected_len == 0) {
    return -1;
  }
  char doc_cid[65];
  ec_cowrie_sha256_hex(key_data, key_len, doc_cid);
  char path[512];
  cw_sp_build_path(ns, doc_cid, path, sizeof path);

  struct stat st;
  if (stat(path, &st) != 0) {
    atomic_fetch_add(&g_sp_misses, 1);
    return -1;
  }
  size_t actual = 0;
  int rc = ec_cowrie_sp_load_file(path, expected_model_sha_64, expected_kv_quant, out_buf, expected_len, &actual);
  if (rc != 0) return rc;
  if (actual != expected_len) return -1;
  return 0;
}

int ec_cowrie_sp_purge_prefix(const char *ns_prefix) {
  pthread_once(&g_cw_once, cw_load_once);
  pthread_mutex_lock(&g_cw_mu);
  char dir[320];
  snprintf(dir, sizeof dir, "%s", g_cache_dir);
  pthread_mutex_unlock(&g_cw_mu);

  DIR *d = opendir(dir);
  if (!d) return 0;
  size_t plen = ns_prefix ? strlen(ns_prefix) : 0;
  int removed = 0;
  struct dirent *de;
  while ((de = readdir(d)) != NULL) {
    if (de->d_name[0] == '.') continue;
    if (!strstr(de->d_name, ".state.cowrie")) continue;
    if (plen > 0 && strncmp(de->d_name, ns_prefix, plen) != 0) continue;
    char full[600];
    snprintf(full, sizeof full, "%s/%s", dir, de->d_name);
    if (unlink(full) == 0) removed++;
  }
  closedir(d);
  return removed;
}

// ===========================================================================
// 4. Phase B: Trust Membrane, Creative Trust Gate & R0/R1/R2 Lenses
// ===========================================================================
static void cw_json_esc(const char *src, char *dst, size_t cap) {
  size_t j = 0;
  for (size_t i = 0; src && src[i] && j + 6 < cap; i++) {
    unsigned char c = (unsigned char)src[i];
    if (c == '"' || c == '\\') { dst[j++] = '\\'; dst[j++] = (char)c; }
    else if (c == '\n') { dst[j++] = '\\'; dst[j++] = 'n'; }
    else if (c == '\r') { dst[j++] = '\\'; dst[j++] = 'r'; }
    else if (c == '\t') { dst[j++] = '\\'; dst[j++] = 't'; }
    else if (c >= 0x20) dst[j++] = (char)c;
  }
  dst[j] = '\0';
}

static void cw_extract_handle(const char *r0, const char *cid_hex, char out_handle[24]) {
  if (r0) {
    const char *at = strchr(r0, '@');
    if (at) {
      size_t j = 0;
      while (at[j] && at[j] != ' ' && at[j] != '\n' && at[j] != '\r' &&
             at[j] != '\t' && at[j] != ')' && at[j] != ']' && at[j] != '|' && j + 1 < 24) {
        out_handle[j] = at[j];
        j++;
      }
      out_handle[j] = '\0';
      if (j >= 2) return;
    }
  }
  snprintf(out_handle, 24, "@%.8s", (cid_hex && cid_hex[0]) ? cid_hex : "00000000");
}

uint8_t ec_cowrie_propagate_authority(uint8_t base_auth, const uint8_t *parents, size_t n_parents) {
  pthread_once(&g_cw_once, cw_load_once);
  if (g_cw.loaded && g_cw.propagate_auth) {
    return g_cw.propagate_auth(base_auth, parents, n_parents);
  }
  uint8_t eff = base_auth;
  if (eff == EC_COWRIE_AUTH_UNTRUSTED_EXT) eff |= EC_COWRIE_BIT_TAINTED;
  for (size_t i = 0; i < n_parents; i++) {
    if ((parents[i] & EC_COWRIE_BIT_TAINTED) || (parents[i] & 0x0Fu) == EC_COWRIE_AUTH_UNTRUSTED_EXT) {
      eff |= EC_COWRIE_BIT_TAINTED;
    }
  }
  return eff;
}

int ec_cowrie_is_tainted(uint8_t authority) {
  pthread_once(&g_cw_once, cw_load_once);
  if (g_cw.loaded && g_cw.is_tainted) {
    return (int)g_cw.is_tainted(authority);
  }
  return ((authority & EC_COWRIE_BIT_TAINTED) || ((authority & 0x0Fu) == EC_COWRIE_AUTH_UNTRUSTED_EXT)) ? 1 : 0;
}

static trust_rec_t *trust_find_locked(const char *id) {
  if (!id || !id[0]) return NULL;
  size_t ilen = strlen(id);
  for (int i = 0; i < EC_TRUST_MAX; i++) {
    if (!g_trust[i].used) continue;
    if (!strcmp(g_trust[i].handle, id) || !strcmp(g_trust[i].cid_hex, id)) {
      return &g_trust[i];
    }
    if (ilen >= 8 && !strncmp(g_trust[i].cid_hex, id, ilen)) {
      return &g_trust[i];
    }
  }
  return NULL;
}

int ec_cowrie_canonical_json_cid(
    const char *json, size_t json_len, uint8_t authority,
    char out_cid_hex[69], size_t *out_cowr_len) {
  if (!json || !out_cid_hex) return -1;
  out_cid_hex[0] = '\0';
  if (out_cowr_len) *out_cowr_len = 0;
  if (!ec_cowrie_available()) return -1;

  cw_buf_t frame = {0};
  int rc = g_cw.encode_json((const uint8_t *)json, json_len, authority, EC_COWRIE_EPI_OBSERVED, &frame);
  if (rc != 0 || !frame.data || frame.len == 0) {
    if (frame.data) g_cw.buf_free(frame);
    return rc > 0 ? -rc : -1;
  }
  uint8_t v_auth = 0, v_epi = 0, cid34[34] = {0};
  int vrc = g_cw.frame_verify(frame.data, frame.len, &v_auth, &v_epi, cid34);
  if (vrc == 0) {
    cw_bytes_to_hex(cid34, 34, out_cid_hex);
    if (out_cowr_len) *out_cowr_len = frame.len;
  }
  g_cw.buf_free(frame);
  return vrc == 0 ? 0 : -vrc;
}

int ec_cowrie_trust_register_json(
    const char *json, size_t json_len,
    uint8_t base_authority,
    const char *source, const char *label,
    const char *const *parent_ids, int n_parents,
    ec_cowrie_trust_meta_t *out_meta) {
  if (out_meta) memset(out_meta, 0, sizeof *out_meta);
  if (!json || json_len == 0) return -1;

  // Gather parent authorities under lock for sticky BIT_TAINTED propagation
  uint8_t pauths[32];
  size_t np = 0;
  pthread_mutex_lock(&g_cw_mu);
  for (int i = 0; i < n_parents && np < 32; i++) {
    trust_rec_t *pr = trust_find_locked(parent_ids[i]);
    if (pr) pauths[np++] = pr->authority;
  }
  pthread_mutex_unlock(&g_cw_mu);

  uint8_t eff_auth = ec_cowrie_propagate_authority(base_authority, pauths, np);
  int tainted = ec_cowrie_is_tainted(eff_auth);

  ec_cowrie_trust_meta_t meta;
  memset(&meta, 0, sizeof meta);
  meta.authority = eff_auth;
  meta.epistemic = EC_COWRIE_EPI_OBSERVED;
  meta.tainted   = tainted;
  meta.raw_bytes = json_len;

  if (ec_cowrie_available()) {
    cw_buf_t frame = {0};
    int rc = g_cw.encode_json((const uint8_t *)json, json_len, eff_auth, EC_COWRIE_EPI_OBSERVED, &frame);
    if (rc == 0 && frame.data && frame.len > 0) {
      uint8_t v_auth = 0, v_epi = 0, cid34[34] = {0};
      if (g_cw.frame_verify(frame.data, frame.len, &v_auth, &v_epi, cid34) == 0) {
        meta.ok         = 1;
        meta.authority  = v_auth;
        meta.epistemic  = v_epi;
        meta.tainted    = ec_cowrie_is_tainted(v_auth);
        meta.cowr_bytes = frame.len;
        cw_bytes_to_hex(cid34, 34, meta.cid_hex);

        cw_buf_t b0 = {0}, b1 = {0};
        if (g_cw.project_lens(frame.data, frame.len, 0, 0, &b0) == 0 && b0.data && b0.len > 0) {
          size_t cp = b0.len < sizeof(meta.r0) - 1 ? b0.len : sizeof(meta.r0) - 1;
          memcpy(meta.r0, b0.data, cp);
          meta.r0[cp] = '\0';
        }
        if (b0.data) g_cw.buf_free(b0);

        if (g_cw.project_lens(frame.data, frame.len, 1, 0, &b1) == 0 && b1.data && b1.len > 0) {
          size_t cp = b1.len < sizeof(meta.r1) - 1 ? b1.len : sizeof(meta.r1) - 1;
          memcpy(meta.r1, b1.data, cp);
          meta.r1[cp] = '\0';
        }
        if (b1.data) g_cw.buf_free(b1);

        cw_extract_handle(meta.r0, meta.cid_hex, meta.handle);
      }
    }
    if (frame.data) g_cw.buf_free(frame);
  }

  if (!meta.ok) {
    // Graceful fallback without libcowrie_core.so (still enforces Trust Membrane!)
    char h64[65];
    ec_cowrie_sha256_hex(json, json_len, h64);
    snprintf(meta.cid_hex, sizeof meta.cid_hex, "1220%s", h64);
    snprintf(meta.handle, sizeof meta.handle, "@%.6s", h64);
    snprintf(meta.r0, sizeof meta.r0, "%s auth=0x%02x %zuB", meta.handle, eff_auth, json_len);
    meta.cowr_bytes = json_len;
    meta.ok = 1;
  }

  // Upsert into Creative Trust Store
  pthread_mutex_lock(&g_cw_mu);
  trust_rec_t *dst = trust_find_locked(meta.cid_hex);
  if (!dst) {
    dst = &g_trust[g_trust_next % EC_TRUST_MAX];
    if (!dst->used) g_trust_count++;
    g_trust_next = (g_trust_next + 1) % EC_TRUST_MAX;
  }
  dst->used       = 1;
  dst->authority  = meta.authority;
  dst->epistemic  = meta.epistemic;
  dst->tainted    = meta.tainted;
  dst->raw_bytes  = meta.raw_bytes;
  dst->cowr_bytes = meta.cowr_bytes;
  snprintf(dst->cid_hex, sizeof dst->cid_hex, "%s", meta.cid_hex);
  snprintf(dst->handle,  sizeof dst->handle,  "%s", meta.handle);
  snprintf(dst->source,  sizeof dst->source,  "%s", source ? source : "external");
  snprintf(dst->label,   sizeof dst->label,   "%s", label  ? label  : "workflow");
  snprintf(dst->r0,      sizeof dst->r0,      "%s", meta.r0);
  snprintf(dst->r1,      sizeof dst->r1,      "%s", meta.r1);
  pthread_mutex_unlock(&g_cw_mu);

  if (out_meta) *out_meta = meta;
  return 0;
}

static int cw_is_readonly_action(const char *action) {
  if (!action || !action[0]) return 0;
  return (!strcmp(action, "read_only") ||
          !strcmp(action, "read_only_safe") ||
          !strcmp(action, "search") ||
          !strcmp(action, "preview") ||
          !strcmp(action, "inspect")) ? 1 : 0;
}

int ec_cowrie_trust_gate_json(
    const char *action,
    const char *const *arg_ids, int n_args,
    int owner_override,
    char *out_json, size_t out_cap) {
  const char *act = (action && action[0]) ? action : "unknown";
  uint8_t pauths[32];
  size_t np = 0;
  char tainted_src[96] = {0};
  int unknown_args = 0;

  pthread_mutex_lock(&g_cw_mu);
  for (int i = 0; i < n_args && np < 32; i++) {
    if (!arg_ids || !arg_ids[i] || !arg_ids[i][0]) continue;
    trust_rec_t *r = trust_find_locked(arg_ids[i]);
    if (r) {
      pauths[np++] = r->authority;
      if (r->tainted && !tainted_src[0]) {
        snprintf(tainted_src, sizeof tainted_src, "%s (%s)", r->handle, r->source);
      }
    } else {
      unknown_args++;
    }
  }
  pthread_mutex_unlock(&g_cw_mu);

  uint8_t eff_auth = ec_cowrie_propagate_authority(EC_COWRIE_AUTH_VERIFIED_TOOL, pauths, np);
  int tainted = ec_cowrie_is_tainted(eff_auth);
  int ro = cw_is_readonly_action(act);

  char e_act[128], e_src[160];
  cw_json_esc(act, e_act, sizeof e_act);
  cw_json_esc(tainted_src, e_src, sizeof e_src);

  if (tainted && !ro && !owner_override) {
    snprintf(out_json, out_cap,
        "{\"ok\":false,\"allowed\":false,\"error\":\"ERR_TAINTED_EXECUTION_BLOCKED\","
        "\"code\":%d,\"action\":\"%s\",\"effective_authority\":%u,\"tainted\":true,"
        "\"tainted_source\":\"%s\",\"checked_args\":%zu,\"unknown_args\":%d,"
        "\"owner_override_required\":true}",
        EC_COWRIE_ERR_TAINTED_EXEC_BLOCKED, e_act, (unsigned)eff_auth,
        e_src, np, unknown_args);
    return 403;
  }

  snprintf(out_json, out_cap,
      "{\"ok\":true,\"allowed\":true,\"action\":\"%s\",\"effective_authority\":%u,"
      "\"tainted\":%s,\"tainted_source\":\"%s\",\"read_only\":%s,\"owner_override\":%s,"
      "\"checked_args\":%zu,\"unknown_args\":%d}",
      e_act, (unsigned)eff_auth, tainted ? "true" : "false", e_src,
      ro ? "true" : "false", owner_override ? "true" : "false",
      np, unknown_args);
  return 200;
}

int ec_cowrie_trust_triage_json(char *out_json, size_t out_cap) {
  if (!out_json || out_cap < 64) return -1;
  pthread_mutex_lock(&g_cw_mu);
  int pos = snprintf(out_json, out_cap,
      "{\"ok\":true,\"cowrie_enabled\":%s,\"cowrie_shani\":%s,\"count\":%d,\"entries\":[",
      ec_cowrie_available() ? "true" : "false",
      ec_cowrie_has_shani() ? "true" : "false",
      g_trust_count);
  int emitted = 0;
  for (int i = 0; i < EC_TRUST_MAX && (size_t)pos + 512 < out_cap; i++) {
    const trust_rec_t *r = &g_trust[i];
    if (!r->used) continue;
    char e_src[96], e_lbl[128], e_r0[256];
    cw_json_esc(r->source, e_src, sizeof e_src);
    cw_json_esc(r->label,  e_lbl, sizeof e_lbl);
    cw_json_esc(r->r0,     e_r0,  sizeof e_r0);
    pos += snprintf(out_json + pos, out_cap - (size_t)pos,
        "%s{\"handle\":\"%s\",\"cid\":\"%s\",\"authority\":%u,\"tainted\":%s,"
        "\"source\":\"%s\",\"label\":\"%s\",\"raw_bytes\":%zu,\"cowr_bytes\":%zu,\"r0\":\"%s\"}",
        emitted ? "," : "", r->handle, r->cid_hex, (unsigned)r->authority,
        r->tainted ? "true" : "false", e_src, e_lbl, r->raw_bytes, r->cowr_bytes, e_r0);
    emitted++;
  }
  pos += snprintf(out_json + pos, out_cap - (size_t)pos, "]}");
  pthread_mutex_unlock(&g_cw_mu);
  return pos;
}

int ec_cowrie_lens_response_json(
    const char *target,
    const char *raw_json, size_t raw_len,
    uint8_t authority, int tier,
    char *out_json, size_t out_cap) {
  if (!target || !raw_json || !out_json || out_cap < 256) return -1;
  if (tier < 0) tier = 0;
  if (tier > 2) tier = 2;
  const char *tier_name = (tier == 0) ? "r0" : (tier == 1 ? "r1" : "r2");

  char cid_hex[69] = {0};
  char handle[24]  = {0};
  size_t cowr_len  = raw_len;
  size_t lens_len  = raw_len;
  char *lens_esc   = NULL;

  if (ec_cowrie_available()) {
    cw_buf_t frame = {0};
    int rc = g_cw.encode_json((const uint8_t *)raw_json, raw_len, authority, EC_COWRIE_EPI_OBSERVED, &frame);
    if (rc == 0 && frame.data && frame.len > 0) {
      cowr_len = frame.len;
      uint8_t va = 0, ve = 0, cid34[34] = {0};
      if (g_cw.frame_verify(frame.data, frame.len, &va, &ve, cid34) == 0) {
        cw_bytes_to_hex(cid34, 34, cid_hex);
      }
      cw_buf_t r0b = {0};
      if (g_cw.project_lens(frame.data, frame.len, 0, 0, &r0b) == 0 && r0b.data && r0b.len > 0) {
        char r0tmp[160];
        size_t cp = r0b.len < sizeof(r0tmp) - 1 ? r0b.len : sizeof(r0tmp) - 1;
        memcpy(r0tmp, r0b.data, cp); r0tmp[cp] = '\0';
        cw_extract_handle(r0tmp, cid_hex, handle);
      } else {
        cw_extract_handle(NULL, cid_hex, handle);
      }
      if (r0b.data) g_cw.buf_free(r0b);

      cw_buf_t lb = {0};
      if (g_cw.project_lens(frame.data, frame.len, (uint8_t)tier, 0, &lb) == 0 && lb.data) {
        lens_len = lb.len;
        char *lstr = (char *)malloc(lb.len + 1);
        size_t ecap = lb.len * 2 + 64;
        if (ecap > out_cap - 512) ecap = out_cap - 512;
        lens_esc = (char *)malloc(ecap);
        if (lstr && lens_esc) {
          memcpy(lstr, lb.data, lb.len);
          lstr[lb.len] = '\0';
          cw_json_esc(lstr, lens_esc, ecap);
        }
        free(lstr);
      }
      if (lb.data) g_cw.buf_free(lb);
    }
    if (frame.data) g_cw.buf_free(frame);
  }

  if (!lens_esc) {
    // Fallback without libcowrie_core.so
    char h64[65];
    ec_cowrie_sha256_hex(raw_json, raw_len, h64);
    snprintf(cid_hex, sizeof cid_hex, "1220%s", h64);
    snprintf(handle, sizeof handle, "@%.6s", h64);
    size_t ecap = raw_len * 2 + 64;
    if (ecap > out_cap - 512) ecap = out_cap - 512;
    lens_esc = (char *)malloc(ecap);
    if (!lens_esc) return -1;
    cw_json_esc(raw_json, lens_esc, ecap);
    lens_len = raw_len;
  }

  double red_pct = (raw_len > 0) ? (100.0 * (1.0 - ((double)lens_len / (double)raw_len))) : 0.0;
  int n = snprintf(out_json, out_cap,
      "{\"ok\":true,\"target\":\"%s\",\"tier\":\"%s\","
      "\"cowrie_enabled\":%s,\"cowrie_shani\":%s,"
      "\"cid\":\"%s\",\"handle\":\"%s\","
      "\"raw_bytes\":%zu,\"cowr_bytes\":%zu,\"lens_bytes\":%zu,"
      "\"reduction_pct\":%.1f,\"lens\":\"%s\"}",
      target, tier_name,
      ec_cowrie_available() ? "true" : "false",
      ec_cowrie_has_shani() ? "true" : "false",
      cid_hex, handle, raw_len, cowr_len, lens_len, red_pct, lens_esc);
  free(lens_esc);
  return n;
}
