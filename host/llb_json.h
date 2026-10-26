// host/llb_json.h — a minimal JSON DOM (G1): parse, walk, serialise. Shared by the
// JSON-schema converter (llb_grammar) and the Qwen tool-call module (llb_toolcall).
//
// Parse is RFC 8259 strict (no comments, no trailing commas, no NaN), UTF-8 in/out,
// \uXXXX (incl. surrogate pairs) decoded to UTF-8. Objects keep key order and duplicates
// (last wins on lookup). Numbers keep their source lexeme, so a re-serialise is
// lexeme-faithful; llb_json_num reads it as a double.
//
// Serialise (llb_json_dump) follows Python json.dumps(ensure_ascii=False) with the
// default separators ", " and ": " — what the HF chat template's `tojson` emits — or the
// compact ",", ":" when compact = 1. Python re-formats numbers (1.0e2 -> 100.0); we keep
// the lexeme instead (documented divergence; integers and plain decimals agree).
#ifndef LLB_JSON_H
#define LLB_JSON_H

#include <stddef.h>

typedef enum { LLB_J_NULL, LLB_J_BOOL, LLB_J_NUM, LLB_J_STR, LLB_J_ARR, LLB_J_OBJ } LlbJType;

typedef struct LlbJson LlbJson;
struct LlbJson {
  LlbJType t;
  int b;            // bool
  char* s;          // STR: decoded UTF-8 (NUL-terminated, may contain \0 -> use n); NUM: lexeme
  size_t n;         // STR/NUM byte length; ARR/OBJ: number of children
  char** keys;      // OBJ: key per child (decoded UTF-8)
  LlbJson** kids;   // ARR/OBJ children
};

// NULL on error (message in err). The tree is freed with llb_json_free.
LlbJson* llb_json_parse(const char* text, size_t len, char* err, int errlen);
void llb_json_free(LlbJson* j);
// object member (last duplicate wins) or NULL
const LlbJson* llb_json_get(const LlbJson* o, const char* key);
double llb_json_num(const LlbJson* j);
// malloc'd serialisation (caller frees); Python-json style, see the header comment
char* llb_json_dump(const LlbJson* j, int compact);
// append the JSON string literal of bytes s[0..n) to a growable buffer (helper, also used
// by the tool-call renderer). ensure_ascii = 0: non-ASCII passes through as UTF-8.
typedef struct { char* p; size_t n, cap; } LlbBuf;
void llb_buf_put(LlbBuf* b, const char* s, size_t n);
void llb_buf_puts(LlbBuf* b, const char* s);
void llb_json_quote(LlbBuf* b, const char* s, size_t n);

#endif
