#pragma once
#include "json.h"
#include "../qwen35_tokenizer.h"
#include <setjmp.h>
#define CLEF_MAX_T 16384
#define CLEF_MAX_Q 128
#define CLEF_MAX_O 4096
typedef struct { int start, end; } Span;
typedef struct { char *id; int type, first, count; Span span; } Question;
typedef struct { char *id; Span span; int parent; } Option;
typedef struct {
    int *ids, n, nq, no;
    Question q[CLEF_MAX_Q]; Option o[CLEF_MAX_O];
    sb text, pieces; int roundtrip;
} Record;
void clef_encode(Q35Tokenizer *tok, const jv *req, Record *r, int max_length);
void clef_record_json(const Record *r, sb *out);
void clef_record_free(Record *r);
void clef_fail(const char *message);
/* SERVE-BATCH: while clef_trap is set, clef_fail longjmps there with clef_trap_msg instead of exiting
   (the engine's decide parse refuses the request; the CLI tools leave it NULL = the old exit) */
extern jmp_buf *clef_trap;
extern const char *clef_trap_msg;
