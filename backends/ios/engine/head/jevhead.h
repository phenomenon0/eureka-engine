// head/jevhead.h — the Jev (Kev) decision head: request -> per-question prompts, and
// final-normed hidden rows -> {pick, probabilities, confidence}.
//
// The math (SPEC.md §3, llama.cpp 9881906 server-context.cpp:2373-2392, server-decision.cpp:682-790):
//   e_t = W_cls · h_t + b_cls          h_t = result_norm row (n_embd), e_t = [q | k] (2 × n_ptr)
//   s_i = (q_pointer · k_marker_i) / sqrt(n_ptr)      pointer = last prompt token, marker_i = i-th <|box_end|>
//   p_i = exp((s_i − max s) / T) / Σ                    T = temperature.<type>.<bucket> → .<type> → 1.0
// Options keep request insertion order end to end (Kev is order-sensitive; never sort).
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "json.h"

#define JVH_MARKER_TEXT "<|box_end|>"
#define JVH_POINTER_TEXT "<|fim_suffix|>"

enum { JVH_EXACT = 0, JVH_CPU = 1 };  // projection rounding: exact f32 (engine / Metal-like) | ggml-cpu mimic (tests)
enum { JVH_CHOICE = 0, JVH_SCORE = 1, JVH_NOUL = 2 };

typedef struct {
  char arch[64];
  int n_in, n_out, n_ptr;  // n_embd, n_embd_out, n_embd_out / 2
  int wtype;               // GGML type of cls.output.weight
  float *W;                // [n_out][n_in] dequantized
  int8_t *qs;              // Q8_0 only: [n_out][n_in]
  float *qd;               // Q8_0 only: [n_out][n_in / 32]
  float *b;                // [n_out]
  int n_temp;
  char **temp_name;        // suffix after "<arch>.decision.temperature.", e.g. "choice", "choice.3_5"
  float *temp;
} jvh_head;

typedef struct { char *key; const jv *desc; } jvh_opt;
typedef struct {
  char *id;
  int type;
  const jv *instructions;
  jvh_opt *opt;  // insertion order (noul: [false, true])
  size_t n;
} jvh_q;
typedef struct {
  jv *root;
  const jv *state;
  jvh_q *q;      // insertion order
  size_t nq;
} jvh_req;

// head loading. load_gguf checks decision.type == kev and the systemone template; load_prefix reads the
// R1 spike extraction (PREFIX.json + PREFIX.cls.output.{weight,bias}.bin), for the golden replays.
int jvh_load_gguf(jvh_head *h, const char *path, char *err, size_t errn);
int jvh_load_prefix(jvh_head *h, const char *prefix, char *err, size_t errn);
void jvh_free(jvh_head *h);
float jvh_temperature(const jvh_head *h, int qtype, size_t n_options);

// request handling (server-decision.cpp parse_questions / render / decision_kev_text)
int jvh_parse_request(const char *text, size_t len, jvh_req *r, char *err, size_t errn);
void jvh_req_free(jvh_req *r);
char *jvh_render_prompt(const jvh_req *r, const jvh_q *q);  // malloc'd
char *jvh_kev_text(const jv *v);                             // malloc'd, special tokens neutralised

// head math
void jvh_project(const jvh_head *h, const float *row, float *e, int mode);
void jvh_scores(const jvh_head *h, const float *e_ptr, const float *const *e_mk, size_t n, float *s);
void jvh_probs(const float *s, size_t n, float T, double *p);
// answers.<id> object; p = probabilities in option order
void jvh_answer(sbuf *out, const jvh_q *q, const double *p);
// rows [pointer, marker_0..marker_{n-1}] (each n_in floats) -> answer; scores/probs optional outputs
void jvh_answer_rows(sbuf *out, const jvh_head *h, const jvh_q *q, const float *rows, int mode, float *s, double *p);

// llb row files (CLEF-3 llb_hidden_put format: repeated {u32 pos, u32 width, f32[width]})
int jvh_rows_read(const char *path, int width, uint32_t **pos, float **rows, char *err, size_t errn);  // -> count
