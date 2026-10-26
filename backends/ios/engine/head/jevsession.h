// head/jevsession.h — the engine-facing surface of the Jev (Kev) head. Opaque: no JSON or GGUF types leak,
// so it can be included next to CLEF-3's tools/clef3/json.h (which also defines `jv` / `jtype`).
//
// Per request (one /v1/systemone body), the engine glue does:
//   job = jvh_job_new(s, line, len, err)                         parse + render; NULL -> refuse(err)
//   for k in 0..jvh_job_count(job)-1:                            one fresh-state exact prefill per question
//     ids = tokenize(jvh_job_prompt(job, k), parse_special = 1)  no BOS / EOS
//     jvh_job_layout(job, k, ids, n, pos, err)                   -> positions to capture: markers..., pointer
//     prefill ids; capture the final-normed rows (output_norm RMSNorm) at pos[0..n_options]
//     jvh_job_rows(job, k, rows, width, err)                     rows in the same order as pos
//       or jvh_job_rows_file(job, k, hidden_dump_path, err)      straight from the llb_hidden_put file
//   reply = jvh_job_reply(job, model, JVH_EXACT)                 the response line
#pragma once
#include <stddef.h>
#include <stdint.h>

#define JVH_MARKER_ID_KEV 248050  // <|box_end|> in the Qwen3.5 vocab (Kev-4B, Kev-0.8B); verify per tokenizer

typedef struct jvh_session jvh_session;
typedef struct jvh_job jvh_job;

jvh_session *jvh_session_open(const char *gguf, char *err, size_t errn);  // loads cls.output.* + temperatures
void jvh_session_close(jvh_session *s);
int jvh_session_width(const jvh_session *s);                              // n_embd the rows must have

jvh_job *jvh_job_new(jvh_session *s, const char *body, size_t len, char *err, size_t errn);
size_t jvh_job_count(const jvh_job *j);
const char *jvh_job_id(const jvh_job *j, size_t k);
const char *jvh_job_prompt(const jvh_job *j, size_t k);
size_t jvh_job_options(const jvh_job *j, size_t k);
// marker_id: the <|box_end|> id; pos_out has room for options + 1. Returns options + 1, or -1 with
// "unexpected layout of the decision prompt" (server-decision.cpp:494) in err.
int jvh_job_layout(jvh_job *j, size_t k, const int32_t *ids, size_t n_ids, int32_t marker_id, uint32_t *pos_out,
                   char *err, size_t errn);
int jvh_job_rows(jvh_job *j, size_t k, const float *rows, int width, char *err, size_t errn);
// the same from an llb_hidden_put dump ({u32 pos, u32 width, f32[width]}...; host/llb_host.c:140). The dump may hold
// every prompt row (SYS1-DUAL's default, all positions on) or just the layout's; the layout's rows are picked by pos.
int jvh_job_rows_file(jvh_job *j, size_t k, const char *path, char *err, size_t errn);
char *jvh_job_reply(jvh_job *j, const char *model, int mode, char *err, size_t errn);  // malloc'd; mode 0 = exact f32
void jvh_job_free(jvh_job *j);
