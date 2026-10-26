// head/gguf.h — minimal GGUF v2/v3 reader: metadata scalars/strings and tensor infos.
// Only what the Jev head needs; arrays are skipped (their element type and count are kept).
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

enum { GG_U8, GG_I8, GG_U16, GG_I16, GG_U32, GG_I32, GG_F32, GG_BOOL, GG_STR, GG_ARR, GG_U64, GG_I64, GG_F64 };
enum { GGML_F32 = 0, GGML_F16 = 1, GGML_Q8_0 = 8, GGML_BF16 = 30 };

typedef struct {
  char *key;
  int type;      // GG_*
  double num;    // numeric / bool value
  char *str;     // GG_STR
} gg_kv;

typedef struct {
  char *name;
  int n_dims;
  uint64_t ne[4];  // ggml order: ne[0] = row length
  int type;        // GGML_*
  uint64_t off;    // absolute file offset of the data
} gg_tensor;

typedef struct {
  FILE *f;
  int n_kv, n_t;
  gg_kv *kv;
  gg_tensor *t;
} gguf;

int gguf_open(gguf *g, const char *path, char *err, size_t errn);
void gguf_close(gguf *g);
const gg_kv *gguf_kv(const gguf *g, const char *key);
const gg_tensor *gguf_tensor(const gguf *g, const char *name);
int gguf_read(const gguf *g, const gg_tensor *t, void *dst, size_t n);  // n bytes from the tensor start
size_t ggml_row_bytes(int type, uint64_t n);                             // 0 = unsupported type
float gg_f16(uint16_t h);
float gg_bf16(uint16_t h);
