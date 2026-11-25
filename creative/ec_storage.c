// ec_storage.c — Multi-Drive & Network-Drive Creative Model Vault Manager
#define _POSIX_C_SOURCE 200809L
#include "ec_storage.h"
#include <ctype.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

const char *ec_asset_kind_name(ec_asset_kind_t k) {
  switch (k) {
    case EC_ASSET_CHECKPOINT: return "checkpoint";
    case EC_ASSET_LORA:       return "lora";
    case EC_ASSET_VAE:        return "vae";
    case EC_ASSET_TEXT_ENC:   return "text_encoder";
    case EC_ASSET_TOKENIZER:  return "tokenizer";
    default:                  return "other";
  }
}

static int ci_contains(const char *hay, const char *needle) {
  if (!hay || !needle || !*needle) return 0;
  size_t nl = strlen(needle);
  for (const char *p = hay; *p; p++) {
    if (strncasecmp(p, needle, nl) == 0) return 1;
  }
  return 0;
}

static void detect_mount_info(const char *path, char *fs_out, size_t fs_cap, int *is_net) {
  snprintf(fs_out, fs_cap, "local");
  *is_net = 0;
  if (ci_contains(path, "/gvfs/") || ci_contains(path, "/smb/") ||
      ci_contains(path, "/nfs/")  || ci_contains(path, "/net/")) {
    *is_net = 1;
  }
#ifdef __APPLE__
  if (strncmp(path, "/Volumes/", 9) == 0) {
    snprintf(fs_out, fs_cap, "volume");
  }
#endif
  FILE *fp = fopen("/proc/mounts", "r");
  if (!fp) return;
  char line[1024];
  size_t best_len = 0;
  char best_fs[64] = "local";
  while (fgets(line, sizeof line, fp)) {
    char dev[256], mnt[256], fst[64];
    if (sscanf(line, "%255s %255s %63s", dev, mnt, fst) != 3) continue;
    size_t ml = strlen(mnt);
    if (ml > best_len && strncmp(path, mnt, ml) == 0 && (path[ml] == '/' || path[ml] == '\0' || ml == 1)) {
      best_len = ml;
      snprintf(best_fs, sizeof best_fs, "%s", fst);
    }
  }
  fclose(fp);
  snprintf(fs_out, fs_cap, "%s", best_fs);
  if (ci_contains(best_fs, "nfs") || ci_contains(best_fs, "cifs") ||
      ci_contains(best_fs, "smb") || ci_contains(best_fs, "sshfs") ||
      ci_contains(best_fs, "gvfs") || ci_contains(best_fs, "davfs") ||
      ci_contains(best_fs, "rclone") || ci_contains(best_fs, "s3fs")) {
    *is_net = 1;
  }
}

static void refresh_root_stats(ec_storage_root_t *r) {
  struct stat st;
  r->exists = (stat(r->path, &st) == 0 && S_ISDIR(st.st_mode)) ? 1 : 0;
  r->writable = (r->exists && access(r->path, W_OK) == 0) ? 1 : 0;
  detect_mount_info(r->path, r->fs_type, sizeof r->fs_type, &r->is_network);
  if (r->exists) {
    struct statvfs vfs;
    if (statvfs(r->path, &vfs) == 0) {
      double tot = (double)vfs.f_blocks * (double)vfs.f_frsize;
      double avail = (double)vfs.f_bavail * (double)vfs.f_frsize;
      r->total_gb = tot / (1024.0 * 1024.0 * 1024.0);
      r->free_gb = avail / (1024.0 * 1024.0 * 1024.0);
      r->used_pct = (tot > 0.0) ? (100.0 * (1.0 - avail / tot)) : 0.0;
    }
  } else {
    r->total_gb = r->free_gb = r->used_pct = 0.0;
  }
}

static void save_roots(const ec_storage_vault_t *v) {
  if (!v->config_path[0]) return;
  FILE *fp = fopen(v->config_path, "wb");
  if (!fp) return;
  fprintf(fp, "[\n");
  for (int i = 0; i < v->n_roots; i++) {
    fprintf(fp, "  {\"path\":\"%s\",\"label\":\"%s\"}%s\n",
            v->roots[i].path, v->roots[i].label, (i + 1 < v->n_roots) ? "," : "");
  }
  fprintf(fp, "]\n");
  fclose(fp);
}

static void load_saved_roots(ec_storage_vault_t *v) {
  if (!v->config_path[0]) return;
  FILE *fp = fopen(v->config_path, "rb");
  if (!fp) return;
  char buf[8192];
  size_t nr = fread(buf, 1, sizeof buf - 1, fp);
  fclose(fp);
  buf[nr] = '\0';
  const char *p = buf;
  while ((p = strstr(p, "\"path\"")) != NULL) {
    p = strchr(p, ':'); if (!p) break;
    while (*p && *p != '"') p++;
    if (!*p) break;
    p++;
    char path[512] = {0}, label[64] = "Custom Drive";
    size_t i = 0;
    while (*p && *p != '"' && i + 1 < sizeof path) path[i++] = *p++;
    const char *lp = strstr(p, "\"label\"");
    if (lp && lp < p + 128) {
      lp = strchr(lp, ':');
      if (lp) {
        while (*lp && *lp != '"') lp++;
        if (*lp == '"') {
          lp++;
          size_t j = 0;
          while (*lp && *lp != '"' && j + 1 < sizeof label) label[j++] = *lp++;
          label[j] = '\0';
        }
      }
    }
    if (path[0]) ec_storage_add_root(v, path, label, 0);
  }
}

int ec_storage_add_root(ec_storage_vault_t *v, const char *path, const char *label, int persist) {
  if (!v || !path || !path[0]) return -1;
  // Normalize trailing slash
  char clean[512];
  snprintf(clean, sizeof clean, "%s", path);
  size_t len = strlen(clean);
  while (len > 1 && clean[len - 1] == '/') clean[--len] = '\0';

  for (int i = 0; i < v->n_roots; i++) {
    if (strcmp(v->roots[i].path, clean) == 0) {
      refresh_root_stats(&v->roots[i]);
      return i;
    }
  }
  if (v->n_roots >= EC_STORAGE_MAX_ROOTS) return -1;
  ec_storage_root_t *r = &v->roots[v->n_roots++];
  memset(r, 0, sizeof *r);
  snprintf(r->path, sizeof r->path, "%s", clean);
  snprintf(r->label, sizeof r->label, "%s", (label && label[0]) ? label : "Storage Root");
  refresh_root_stats(r);
  if (persist) save_roots(v);
  return v->n_roots - 1;
}

int ec_storage_remove_root(ec_storage_vault_t *v, const char *path) {
  if (!v || !path) return -1;
  for (int i = 0; i < v->n_roots; i++) {
    if (strcmp(v->roots[i].path, path) == 0) {
      for (int j = i; j + 1 < v->n_roots; j++) v->roots[j] = v->roots[j + 1];
      v->n_roots--;
      save_roots(v);
      return 0;
    }
  }
  return -1;
}

static ec_asset_kind_t classify_asset(const char *fullpath, const char *fname, uint64_t bytes) {
  if (ci_contains(fname, "tokenizer") && ci_contains(fname, ".json")) return EC_ASSET_TOKENIZER;
  if (ci_contains(fullpath, "/loras/") || ci_contains(fname, "lora") || ci_contains(fname, "lightning")) return EC_ASSET_LORA;
  if (ci_contains(fullpath, "/vae/") || ci_contains(fname, "vae") || strcasecmp(fname, "ae.safetensors") == 0) return EC_ASSET_VAE;
  if (ci_contains(fullpath, "/text_encoders/") || ci_contains(fullpath, "/clip/") ||
      ci_contains(fname, "qwen_3_") || ci_contains(fname, "qwen3_") ||
      ci_contains(fname, "qwen_2.5") || ci_contains(fname, "umt5") || ci_contains(fname, "t5xxl")) {
    return EC_ASSET_TEXT_ENC;
  }
  if (bytes > 50ULL * 1024ULL * 1024ULL) return EC_ASSET_CHECKPOINT;
  return EC_ASSET_OTHER;
}

static void scan_dir_rec(ec_storage_vault_t *v, const ec_storage_root_t *root, const char *dir, int depth) {
  if (depth > 3 || v->n_assets >= EC_STORAGE_MAX_ASSETS) return;
  DIR *d = opendir(dir);
  if (!d) return;
  struct dirent *de;
  while ((de = readdir(d)) != NULL && v->n_assets < EC_STORAGE_MAX_ASSETS) {
    if (de->d_name[0] == '.') continue;
    char full[512];
    snprintf(full, sizeof full, "%s/%s", dir, de->d_name);
    struct stat st;
    if (stat(full, &st) != 0) continue;
    if (S_ISDIR(st.st_mode)) {
      scan_dir_rec(v, root, full, depth + 1);
      continue;
    }
    if (!S_ISREG(st.st_mode)) continue;
    size_t nl = strlen(de->d_name);
    int is_st = (nl > 12 && strcasecmp(de->d_name + nl - 12, ".safetensors") == 0);
    int is_tok = (nl > 14 && ci_contains(de->d_name, "tokenizer") && strcasecmp(de->d_name + nl - 5, ".json") == 0);
    if (!is_st && !is_tok) continue;

    // Deduplicate by exact path
    int dup = 0;
    for (int i = 0; i < v->n_assets; i++) {
      if (strcmp(v->assets[i].path, full) == 0) { dup = 1; break; }
    }
    if (dup) continue;

    ec_storage_asset_t *a = &v->assets[v->n_assets++];
    memset(a, 0, sizeof *a);
    snprintf(a->path, sizeof a->path, "%s", full);
    snprintf(a->name, sizeof a->name, "%s", de->d_name);
    snprintf(a->root_label, sizeof a->root_label, "%s", root->label);
    size_t idl = is_st ? (nl - 12) : (nl - 5);
    if (idl >= sizeof a->id) idl = sizeof a->id - 1;
    memcpy(a->id, de->d_name, idl);
    a->id[idl] = '\0';
    a->bytes = (uint64_t)st.st_size;
    a->is_network = root->is_network;
    a->kind = classify_asset(full, de->d_name, a->bytes);
  }
  closedir(d);
}

void ec_storage_init(ec_storage_vault_t *v, const char *primary_models_dir, const char *config_path) {
  memset(v, 0, sizeof *v);
  if (config_path && config_path[0]) snprintf(v->config_path, sizeof v->config_path, "%s", config_path);
  if (primary_models_dir && primary_models_dir[0]) {
    ec_storage_add_root(v, primary_models_dir, "Primary Local Vault", 0);
  }
  // Always auto-discover Eureka's default user creative models directory (~/.local/share/models/creative)
  const char *home = getenv("HOME");
  if (home && home[0]) {
    char def_vault[512];
    snprintf(def_vault, sizeof def_vault, "%s/.local/share/models/creative", home);
    struct stat st_h;
    if (stat(def_vault, &st_h) == 0 && S_ISDIR(st_h.st_mode)) {
      ec_storage_add_root(v, def_vault, "Eureka Creative Vault (NVMe)", 0);
    }
  }
  // Auto-discover existing secondary / ComfyUI / external storage vaults if present
  const char *auto_roots[][2] = {
    { "/mnt/storage/ai-models/comfyui/models", "Storage HDD (ComfyUI)" },
    { "/mnt/storage/pc-archive/comfyui-models", "Storage HDD (Archive)" },
    { "/mnt/storage/maestro-ckpts", "Storage HDD (Maestro)" },
    { "/mnt/external", "External / Network Mount" },
  };
  for (size_t i = 0; i < sizeof auto_roots / sizeof auto_roots[0]; i++) {
    struct stat st;
    if (stat(auto_roots[i][0], &st) == 0 && S_ISDIR(st.st_mode)) {
      ec_storage_add_root(v, auto_roots[i][0], auto_roots[i][1], 0);
    }
  }
  // Also allow EC_STORAGE_ROOTS env var (colon-separated paths, e.g. NFS/SMB mounts)
  const char *env_roots = getenv("EC_STORAGE_ROOTS");
  if (env_roots && env_roots[0]) {
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s", env_roots);
    char *tok = strtok(tmp, ":");
    while (tok) {
      if (tok[0]) ec_storage_add_root(v, tok, "Network/Env Drive", 0);
      tok = strtok(NULL, ":");
    }
  }
  load_saved_roots(v);
  ec_storage_scan(v);
}

int ec_storage_scan(ec_storage_vault_t *v) {
  if (!v) return -1;
  v->n_assets = 0;
  for (int i = 0; i < v->n_roots; i++) {
    refresh_root_stats(&v->roots[i]);
    if (v->roots[i].exists) scan_dir_rec(v, &v->roots[i], v->roots[i].path, 0);
  }
  return v->n_assets;
}

const ec_storage_root_t *ec_storage_best_root(const ec_storage_vault_t *v, double min_free_gb) {
  if (!v || v->n_roots == 0) return NULL;
  // Prefer local writable root with >= min_free_gb; otherwise writable root (including network drive!) with maximum free space
  const ec_storage_root_t *best = NULL;
  for (int i = 0; i < v->n_roots; i++) {
    const ec_storage_root_t *r = &v->roots[i];
    if (!r->exists || !r->writable) continue;
    if (!r->is_network && r->free_gb >= min_free_gb) {
      if (!best || r->free_gb > best->free_gb) best = r;
    }
  }
  if (best) return best;
  for (int i = 0; i < v->n_roots; i++) {
    const ec_storage_root_t *r = &v->roots[i];
    if (!r->exists || !r->writable) continue;
    if (!best || r->free_gb > best->free_gb) best = r;
  }
  return best;
}

static void canon_alnum(const char *src, char *dst, size_t cap) {
  size_t j = 0;
  for (size_t i = 0; src && src[i] && j + 1 < cap; i++) {
    unsigned char c = (unsigned char)src[i];
    if (isalnum(c)) dst[j++] = (char)tolower(c);
  }
  dst[j] = '\0';
}

const ec_storage_asset_t *ec_storage_find(const ec_storage_vault_t *v, ec_asset_kind_t kind, const char *query) {
  if (!v || !query || !query[0]) return NULL;
  // Strip .safetensors / .ckpt / .json extension & directory prefix for fuzzy matching
  const char *qbase = strrchr(query, '/');
  qbase = qbase ? (qbase + 1) : query;
  const char *qbs = strrchr(qbase, '\\');
  if (qbs) qbase = qbs + 1;
  char qstem[256];
  snprintf(qstem, sizeof qstem, "%s", qbase);
  char *dot = strrchr(qstem, '.');
  if (dot && (strcasecmp(dot, ".safetensors") == 0 || strcasecmp(dot, ".ckpt") == 0 || strcasecmp(dot, ".json") == 0)) {
    *dot = '\0';
  }
  char qcan[256];
  canon_alnum(qstem, qcan, sizeof qcan);

  // 1. Exact path or exact id/name match
  for (int i = 0; i < v->n_assets; i++) {
    const ec_storage_asset_t *a = &v->assets[i];
    if (kind != EC_ASSET_OTHER && a->kind != kind) continue;
    if (strcmp(a->path, query) == 0 || strcasecmp(a->id, query) == 0 ||
        strcasecmp(a->name, query) == 0 || strcasecmp(a->id, qstem) == 0)
      return a;
  }
  // 2. Case-insensitive substring match
  for (int i = 0; i < v->n_assets; i++) {
    const ec_storage_asset_t *a = &v->assets[i];
    if (kind != EC_ASSET_OTHER && a->kind != kind) continue;
    if (ci_contains(a->name, qstem) || ci_contains(a->id, qstem)) return a;
  }
  // 3. Canonical alphanumeric match (handles flux-2-klein-4b <-> flux2_klein_4b_bf16!)
  if (strlen(qcan) >= 4) {
    for (int i = 0; i < v->n_assets; i++) {
      const ec_storage_asset_t *a = &v->assets[i];
      if (kind != EC_ASSET_OTHER && a->kind != kind) continue;
      char acan[256];
      canon_alnum(a->id, acan, sizeof acan);
      if (strstr(acan, qcan) || strstr(qcan, acan)) return a;
    }
  }
  // 4. Model-family alias fallback (handles ComfyUI AIO / quantized / naming-variant templates)
  if (strlen(qcan) >= 4) {
    const char *alias = NULL;
    if (strstr(qcan, "acestep")) {
      alias = (kind == EC_ASSET_VAE) ? "acestep_v15_vae" : "acestep_v15_turbo";
    } else if (strstr(qcan, "flux2") && (strstr(qcan, "klein") || strstr(qcan, "4b") || strstr(qcan, "vae"))) {
      alias = (kind == EC_ASSET_VAE || strstr(qcan, "vae")) ? "flux2_klein_vae" : "flux2_klein_4b";
    } else if ((strstr(qcan, "wan22") || (strstr(qcan, "wan") && strstr(qcan, "5b")))) {
      alias = (kind == EC_ASSET_VAE || strstr(qcan, "vae")) ? "wan2.2_vae" : "wan2.2_ti2v_5b";
    } else if (strstr(qcan, "trellis2") || strstr(qcan, "trellis")) {
      alias = (kind == EC_ASSET_VAE || strstr(qcan, "dec")) ? "trellis2_ss_dec" : "trellis2_ss_flow";
    } else if (strstr(qcan, "dino") && strstr(qcan, "v3")) {
      alias = "dinov3_vitl16";
    } else if (strstr(qcan, "umt5")) {
      alias = "umt5_xxl";
    } else if (strstr(qcan, "qwen3") && strstr(qcan, "4b")) {
      alias = "qwen_3_4b";
    }
    if (alias) {
      for (int i = 0; i < v->n_assets; i++) {
        const ec_storage_asset_t *a = &v->assets[i];
        if (kind != EC_ASSET_OTHER && a->kind != kind) continue;
        if (ci_contains(a->id, alias) || ci_contains(a->name, alias)) return a;
      }
    }
  }
  return NULL;
}

int ec_storage_to_json(const ec_storage_vault_t *v, char *buf, size_t cap) {
  if (!v || !buf || cap < 64) return -1;
  const ec_storage_root_t *best = ec_storage_best_root(v, 15.0);
  int n = snprintf(buf, cap, "{\"recommended_download_root\":\"%s\",\"roots\":[",
                   best ? best->path : "");
  for (int i = 0; i < v->n_roots && (size_t)n < cap - 256; i++) {
    const ec_storage_root_t *r = &v->roots[i];
    n += snprintf(buf + n, cap - (size_t)n,
        "%s{\"path\":\"%s\",\"label\":\"%s\",\"fs_type\":\"%s\",\"is_network\":%s,"
        "\"exists\":%s,\"writable\":%s,\"total_gb\":%.2f,\"free_gb\":%.2f,\"used_pct\":%.1f}",
        i ? "," : "", r->path, r->label, r->fs_type,
        r->is_network ? "true" : "false", r->exists ? "true" : "false", r->writable ? "true" : "false",
        r->total_gb, r->free_gb, r->used_pct);
  }
  n += snprintf(buf + n, cap - (size_t)n, "],\"asset_count\":%d,\"assets\":[", v->n_assets);
  for (int i = 0; i < v->n_assets && (size_t)n < cap - 300; i++) {
    const ec_storage_asset_t *a = &v->assets[i];
    n += snprintf(buf + n, cap - (size_t)n,
        "%s{\"id\":\"%s\",\"name\":\"%s\",\"kind\":\"%s\",\"path\":\"%s\","
        "\"root\":\"%s\",\"bytes\":%llu,\"is_network\":%s}",
        i ? "," : "", a->id, a->name, ec_asset_kind_name(a->kind), a->path,
        a->root_label, (unsigned long long)a->bytes, a->is_network ? "true" : "false");
  }
  n += snprintf(buf + n, cap - (size_t)n, "]}");
  return n;
}
