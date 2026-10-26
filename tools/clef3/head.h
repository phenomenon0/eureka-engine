#pragma once
#include "spans.h"
#include <stdint.h>
typedef struct {
    char *name;
    int rows, cols, type;
    uint64_t offset, bytes;
    float *w;
} Tensor;
typedef struct {
    int fd, nt;
    int d, vocab;   /* output.weight cols (the hidden width the head reads) and rows, from the GGUF */
    uint64_t orow;  /* bytes per output.weight row */
    Tensor t[161];
} Head;
void clef_head_open(Head *h, const char *gguf);
void clef_head_close(Head *h);
float *clef_weight(Head *h, const char *name, int rows, int cols);
void clef_output_row(Head *h, int id, float *out);
void clef_head_run(Head *h, const Record *r, const float *hidden, float *logits);
float *clef_hidden_read(const char *path, int tokens, int d);
int clef_gguf_width(const char *gguf);
