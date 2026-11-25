// ec_safetensors.h — Hardened Zero-Dependency Safetensors + AUX1 Creative Pack Loader
// Strict bounds + integer-overflow checking (matching eureka-engine hardening standards).

#pragma once

#include "ec_types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EC_ST_MAX_TENSORS 16384

typedef struct {
  char       name[160];
  ec_dtype_t dtype;
  int32_t    rank;
  int64_t    shape[5];
  uint64_t   data_off;   // offset from start of binary payload
  uint64_t   nbytes;
} ec_st_entry_t;

typedef struct {
  void          *map_base;
  size_t         file_size;
  uint64_t       payload_off;
  uint32_t       n_tensors;
  ec_st_entry_t *entries;
} ec_safetensors_t;

// Open and validate a .safetensors file via zero-copy mmap. Returns 0 on ok, <0 on error.
int ec_st_open(ec_safetensors_t *st, const char *path, char *err, size_t err_cap);
const ec_st_entry_t *ec_st_find(const ec_safetensors_t *st, const char *name);
const void *ec_st_ptr(const ec_safetensors_t *st, const ec_st_entry_t *e);
void ec_st_close(ec_safetensors_t *st);

// Writer helper (used for creating verified test fixtures & native AUX1/safetensors packs)
int ec_st_write_simple(
    const char *path,
    const char *const *names, const ec_dtype_t *dtypes,
    const int32_t *ranks, const int64_t shapes[][5],
    const void *const *datas, const uint64_t *nbytes,
    uint32_t n_tensors);

#ifdef __cplusplus
}
#endif
