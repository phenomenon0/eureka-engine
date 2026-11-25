// ec_safetensors.c — Hardened Zero-Dependency Safetensors Parser & Writer
// Zero external JSON libs; strict overflow/bounds validation before returning any pointer.

#define _POSIX_C_SOURCE 200809L
#include "ec_safetensors.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static void set_err(char *err, size_t cap, const char *msg) {
  if (err && cap > 0) snprintf(err, cap, "%s", msg);
}

static int parse_dtype(const char *s, ec_dtype_t *out) {
  if (!strcmp(s, "F32"))    { *out = EC_DTYPE_F32; return 0; }
  if (!strcmp(s, "F16"))    { *out = EC_DTYPE_F16; return 0; }
  if (!strcmp(s, "BF16"))   { *out = EC_DTYPE_BF16; return 0; }
  if (!strcmp(s, "F8_E4M3")){ *out = EC_DTYPE_FP8_E4M3; return 0; }
  if (!strcmp(s, "I8"))     { *out = EC_DTYPE_I8; return 0; }
  if (!strcmp(s, "U8"))     { *out = EC_DTYPE_U8; return 0; }
  if (!strcmp(s, "I32"))    { *out = EC_DTYPE_I32; return 0; }
  if (!strcmp(s, "U32"))    { *out = EC_DTYPE_U32; return 0; }
  if (!strcmp(s, "I64"))    { *out = EC_DTYPE_I64; return 0; }
  if (!strcmp(s, "F64"))    { *out = EC_DTYPE_F64; return 0; }
  return -1;
}

static const char *dtype_str(ec_dtype_t dt) {
  switch (dt) {
    case EC_DTYPE_F32:      return "F32";
    case EC_DTYPE_F16:      return "F16";
    case EC_DTYPE_BF16:     return "BF16";
    case EC_DTYPE_FP8_E4M3: return "F8_E4M3";
    case EC_DTYPE_I8:       return "I8";
    case EC_DTYPE_U8:       return "U8";
    case EC_DTYPE_I32:      return "I32";
    case EC_DTYPE_U32:      return "U32";
    default:                return "U8";
  }
}

typedef struct {
  const char *p;
  const char *end;
} js_cur_t;

static void skip_ws(js_cur_t *c) {
  while (c->p < c->end && (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r')) c->p++;
}

static int parse_str(js_cur_t *c, char *buf, size_t cap) {
  skip_ws(c);
  if (c->p >= c->end || *c->p != '"') return -1;
  c->p++;
  size_t n = 0;
  while (c->p < c->end && *c->p != '"') {
    char ch = *c->p++;
    if (ch == '\\') {
      if (c->p >= c->end) return -1;
      ch = *c->p++;
    }
    if (buf && n + 1 < cap) buf[n++] = ch;
  }
  if (c->p >= c->end || *c->p != '"') return -1;
  c->p++;
  if (buf && cap > 0) buf[n] = '\0';
  return 0;
}

static int parse_u64(js_cur_t *c, uint64_t *out) {
  skip_ws(c);
  if (c->p >= c->end || *c->p < '0' || *c->p > '9') return -1;
  uint64_t v = 0;
  while (c->p < c->end && *c->p >= '0' && *c->p <= '9') {
    uint64_t d = (uint64_t)(*c->p++ - '0');
    if (v > (UINT64_MAX - d) / 10ULL) return -1;
    v = v * 10ULL + d;
  }
  *out = v;
  return 0;
}

static int skip_val(js_cur_t *c, int depth) {
  if (depth > 16) return -1;
  skip_ws(c);
  if (c->p >= c->end) return -1;
  if (*c->p == '"') return parse_str(c, NULL, 0);
  if (*c->p == '{') {
    c->p++;
    skip_ws(c);
    if (c->p < c->end && *c->p == '}') { c->p++; return 0; }
    while (c->p < c->end) {
      if (parse_str(c, NULL, 0)) return -1;
      skip_ws(c);
      if (c->p >= c->end || *c->p != ':') return -1;
      c->p++;
      if (skip_val(c, depth + 1)) return -1;
      skip_ws(c);
      if (c->p < c->end && *c->p == ',') { c->p++; continue; }
      if (c->p < c->end && *c->p == '}') { c->p++; return 0; }
      return -1;
    }
    return -1;
  }
  if (*c->p == '[') {
    c->p++;
    skip_ws(c);
    if (c->p < c->end && *c->p == ']') { c->p++; return 0; }
    while (c->p < c->end) {
      if (skip_val(c, depth + 1)) return -1;
      skip_ws(c);
      if (c->p < c->end && *c->p == ',') { c->p++; continue; }
      if (c->p < c->end && *c->p == ']') { c->p++; return 0; }
      return -1;
    }
    return -1;
  }
  while (c->p < c->end && *c->p != ',' && *c->p != '}' && *c->p != ']' && *c->p != ' ' && *c->p != '\n' && *c->p != '\r' && *c->p != '\t')
    c->p++;
  return 0;
}

int ec_st_open(ec_safetensors_t *st, const char *path, char *err, size_t err_cap) {
  memset(st, 0, sizeof *st);
  int fd = open(path, O_RDONLY);
  if (fd < 0) { set_err(err, err_cap, "st: open failed"); return -1; }
  struct stat sb;
  if (fstat(fd, &sb) != 0 || sb.st_size < 10) {
    close(fd); set_err(err, err_cap, "st: file too small"); return -1;
  }
  st->file_size = (size_t)sb.st_size;
  st->map_base = mmap(NULL, st->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (st->map_base == MAP_FAILED) { st->map_base = NULL; set_err(err, err_cap, "st: mmap failed"); return -1; }

  uint64_t hdr_len = 0;
  memcpy(&hdr_len, st->map_base, 8);
  // Hardening bound: reject > 64 MiB headers or headers past EOF
  if (hdr_len == 0 || hdr_len > (64ULL << 20) || hdr_len > (uint64_t)st->file_size - 8ULL) {
    ec_st_close(st); set_err(err, err_cap, "st: invalid header length"); return -1;
  }
  st->payload_off = 8ULL + hdr_len;
  const uint64_t payload_size = (uint64_t)st->file_size - st->payload_off;

  st->entries = (ec_st_entry_t *)calloc(EC_ST_MAX_TENSORS, sizeof(ec_st_entry_t));
  if (!st->entries) { ec_st_close(st); set_err(err, err_cap, "st: OOM"); return -1; }

  js_cur_t c = { (const char *)st->map_base + 8, (const char *)st->map_base + 8 + hdr_len };
  skip_ws(&c);
  if (c.p >= c.end || *c.p != '{') { ec_st_close(st); set_err(err, err_cap, "st: bad JSON root"); return -1; }
  c.p++;
  skip_ws(&c);
  if (c.p < c.end && *c.p == '}') return 0;

  while (c.p < c.end) {
    char key[160];
    if (parse_str(&c, key, sizeof key)) { ec_st_close(st); set_err(err, err_cap, "st: bad key"); return -1; }
    skip_ws(&c);
    if (c.p >= c.end || *c.p != ':') { ec_st_close(st); set_err(err, err_cap, "st: expected colon"); return -1; }
    c.p++;
    if (!strcmp(key, "__metadata__")) {
      if (skip_val(&c, 0)) { ec_st_close(st); set_err(err, err_cap, "st: bad metadata"); return -1; }
    } else {
      if (st->n_tensors >= EC_ST_MAX_TENSORS) { ec_st_close(st); set_err(err, err_cap, "st: too many tensors"); return -1; }
      ec_st_entry_t *e = &st->entries[st->n_tensors];
      memset(e, 0, sizeof *e);
      snprintf(e->name, sizeof e->name, "%s", key);

      skip_ws(&c);
      if (c.p >= c.end || *c.p != '{') { ec_st_close(st); set_err(err, err_cap, "st: bad entry obj"); return -1; }
      c.p++;
      int got_dt = 0, got_sh = 0, got_off = 0;
      uint64_t off_a = 0, off_b = 0;
      while (c.p < c.end) {
        char fkey[32];
        if (parse_str(&c, fkey, sizeof fkey)) { ec_st_close(st); set_err(err, err_cap, "st: bad field key"); return -1; }
        skip_ws(&c);
        if (c.p >= c.end || *c.p != ':') { ec_st_close(st); set_err(err, err_cap, "st: field colon"); return -1; }
        c.p++;
        if (!strcmp(fkey, "dtype")) {
          char dts[16];
          if (parse_str(&c, dts, sizeof dts) || parse_dtype(dts, &e->dtype)) {
            ec_st_close(st); set_err(err, err_cap, "st: unsupported dtype"); return -1;
          }
          got_dt = 1;
        } else if (!strcmp(fkey, "shape")) {
          skip_ws(&c);
          if (c.p >= c.end || *c.p != '[') { ec_st_close(st); set_err(err, err_cap, "st: bad shape"); return -1; }
          c.p++;
          skip_ws(&c);
          e->rank = 0;
          if (c.p < c.end && *c.p == ']') { c.p++; }
          else {
            while (c.p < c.end) {
              uint64_t dim = 0;
              if (parse_u64(&c, &dim) || e->rank >= 5 || dim > INT32_MAX) {
                ec_st_close(st); set_err(err, err_cap, "st: shape dim invalid"); return -1;
              }
              e->shape[e->rank++] = (int64_t)dim;
              skip_ws(&c);
              if (c.p < c.end && *c.p == ',') { c.p++; continue; }
              if (c.p < c.end && *c.p == ']') { c.p++; break; }
              ec_st_close(st); set_err(err, err_cap, "st: shape syntax"); return -1;
            }
          }
          got_sh = 1;
        } else if (!strcmp(fkey, "data_offsets")) {
          skip_ws(&c);
          if (c.p >= c.end || *c.p != '[') { ec_st_close(st); set_err(err, err_cap, "st: offsets"); return -1; }
          c.p++;
          if (parse_u64(&c, &off_a)) { ec_st_close(st); set_err(err, err_cap, "st: off0"); return -1; }
          skip_ws(&c);
          if (c.p >= c.end || *c.p != ',') { ec_st_close(st); set_err(err, err_cap, "st: off comma"); return -1; }
          c.p++;
          if (parse_u64(&c, &off_b)) { ec_st_close(st); set_err(err, err_cap, "st: off1"); return -1; }
          skip_ws(&c);
          if (c.p >= c.end || *c.p != ']') { ec_st_close(st); set_err(err, err_cap, "st: off end"); return -1; }
          c.p++;
          got_off = 1;
        } else {
          if (skip_val(&c, 0)) { ec_st_close(st); set_err(err, err_cap, "st: skip field"); return -1; }
        }
        skip_ws(&c);
        if (c.p < c.end && *c.p == ',') { c.p++; continue; }
        if (c.p < c.end && *c.p == '}') { c.p++; break; }
        ec_st_close(st); set_err(err, err_cap, "st: entry close"); return -1;
      }

      if (!got_dt || !got_sh || !got_off || off_b < off_a || off_b > payload_size) {
        ec_st_close(st); set_err(err, err_cap, "st: tensor offset out of bounds"); return -1;
      }
      // Verify shape * elem_size == (off_b - off_a) with overflow protection
      uint64_t ne = 1;
      for (int d = 0; d < e->rank; d++) {
        if (e->shape[d] > 0 && ne > UINT64_MAX / (uint64_t)e->shape[d]) {
          ec_st_close(st); set_err(err, err_cap, "st: shape overflow"); return -1;
        }
        ne *= (uint64_t)e->shape[d];
      }
      const uint64_t esz = (uint64_t)ec_dtype_size(e->dtype);
      if (esz == 0 || ne > UINT64_MAX / esz || ne * esz != (off_b - off_a)) {
        ec_st_close(st); set_err(err, err_cap, "st: byte count mismatch vs shape"); return -1;
      }
      e->data_off = off_a;
      e->nbytes = off_b - off_a;
      st->n_tensors++;
    }

    skip_ws(&c);
    if (c.p < c.end && *c.p == ',') { c.p++; continue; }
    if (c.p < c.end && *c.p == '}') { c.p++; break; }
    ec_st_close(st); set_err(err, err_cap, "st: root syntax"); return -1;
  }
  return 0;
}

const ec_st_entry_t *ec_st_find(const ec_safetensors_t *st, const char *name) {
  if (!st || !st->entries || !name) return NULL;
  for (uint32_t i = 0; i < st->n_tensors; i++) {
    if (!strcmp(st->entries[i].name, name)) return &st->entries[i];
  }
  return NULL;
}

const void *ec_st_ptr(const ec_safetensors_t *st, const ec_st_entry_t *e) {
  if (!st || !st->map_base || !e) return NULL;
  return (const uint8_t *)st->map_base + st->payload_off + e->data_off;
}

void ec_st_close(ec_safetensors_t *st) {
  if (!st) return;
  if (st->map_base) munmap(st->map_base, st->file_size);
  free(st->entries);
  memset(st, 0, sizeof *st);
}

int ec_st_write_simple(
    const char *path,
    const char *const *names, const ec_dtype_t *dtypes,
    const int32_t *ranks, const int64_t shapes[][5],
    const void *const *datas, const uint64_t *nbytes,
    uint32_t n_tensors) {
  size_t cap = 4096 + (size_t)n_tensors * 256;
  char *hdr = (char *)malloc(cap);
  if (!hdr) return -1;
  size_t pos = 0;
  pos += (size_t)snprintf(hdr + pos, cap - pos, "{");
  uint64_t off = 0;
  for (uint32_t i = 0; i < n_tensors; i++) {
    pos += (size_t)snprintf(hdr + pos, cap - pos, "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[",
                            i ? "," : "", names[i], dtype_str(dtypes[i]));
    for (int32_t d = 0; d < ranks[i]; d++) {
      pos += (size_t)snprintf(hdr + pos, cap - pos, "%s%lld", d ? "," : "", (long long)shapes[i][d]);
    }
    uint64_t end = off + nbytes[i];
    pos += (size_t)snprintf(hdr + pos, cap - pos, "],\"data_offsets\":[%llu,%llu]}",
                            (unsigned long long)off, (unsigned long long)end);
    off = end;
  }
  pos += (size_t)snprintf(hdr + pos, cap - pos, "}");

  FILE *f = fopen(path, "wb");
  if (!f) { free(hdr); return -1; }
  uint64_t hdr_len = (uint64_t)pos;
  int rc = 0;
  if (fwrite(&hdr_len, 8, 1, f) != 1 || fwrite(hdr, 1, pos, f) != pos) rc = -1;
  for (uint32_t i = 0; i < n_tensors && rc == 0; i++) {
    if (nbytes[i] > 0 && fwrite(datas[i], 1, (size_t)nbytes[i], f) != (size_t)nbytes[i]) rc = -1;
  }
  fclose(f);
  free(hdr);
  return rc;
}
