// head/jevsession.c — see jevsession.h.
#include "jevsession.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jevhead.h"

struct jvh_session { jvh_head h; };

typedef struct {
  char *prompt;
  uint32_t *pos;       // layout: markers..., pointer (n_options + 1)
  long n_tokens;       // 0 until jvh_job_layout
  float *rows;         // [pointer, marker_0..] (n_options + 1) × width, NULL until jvh_job_rows
} jvh_qstate;

struct jvh_job {
  jvh_session *s;
  jvh_req r;
  jvh_qstate *q;
};

jvh_session *jvh_session_open(const char *gguf, char *err, size_t errn) {
  jvh_session *s = (jvh_session *)calloc(1, sizeof *s);
  if (jvh_load_gguf(&s->h, gguf, err, errn)) { free(s); return NULL; }
  return s;
}
void jvh_session_close(jvh_session *s) {
  if (!s) return;
  jvh_free(&s->h);
  free(s);
}
int jvh_session_width(const jvh_session *s) { return s->h.n_in; }

jvh_job *jvh_job_new(jvh_session *s, const char *body, size_t len, char *err, size_t errn) {
  jvh_job *j = (jvh_job *)calloc(1, sizeof *j);
  j->s = s;
  if (jvh_parse_request(body, len, &j->r, err, errn)) { free(j); return NULL; }
  j->q = (jvh_qstate *)calloc(j->r.nq, sizeof(jvh_qstate));
  for (size_t k = 0; k < j->r.nq; k++) j->q[k].prompt = jvh_render_prompt(&j->r, &j->r.q[k]);
  return j;
}
size_t jvh_job_count(const jvh_job *j) { return j->r.nq; }
const char *jvh_job_id(const jvh_job *j, size_t k) { return j->r.q[k].id; }
const char *jvh_job_prompt(const jvh_job *j, size_t k) { return j->q[k].prompt; }
size_t jvh_job_options(const jvh_job *j, size_t k) { return j->r.q[k].n; }

// server-decision.cpp:486-497: markers = every <|box_end|>, their count must equal the options; pointer = last token
int jvh_job_layout(jvh_job *j, size_t k, const int32_t *ids, size_t n_ids, int32_t marker_id, uint32_t *pos,
                   char *err, size_t errn) {
  const size_t n = j->r.q[k].n;
  size_t m = 0;
  for (size_t i = 0; i < n_ids; i++)
    if (ids[i] == marker_id) {
      if (m < n) pos[m] = (uint32_t)i;
      m++;
    }
  if (m != n || n_ids == 0) {
    snprintf(err, errn, "unexpected layout of the decision prompt");
    return -1;
  }
  pos[n] = (uint32_t)(n_ids - 1);
  j->q[k].n_tokens = (long)n_ids;
  free(j->q[k].pos);
  j->q[k].pos = (uint32_t *)malloc(sizeof(uint32_t) * (n + 1));
  memcpy(j->q[k].pos, pos, sizeof(uint32_t) * (n + 1));
  return (int)n + 1;
}

int jvh_job_rows(jvh_job *j, size_t k, const float *rows, int width, char *err, size_t errn) {
  const jvh_head *h = &j->s->h;
  const size_t n = j->r.q[k].n;
  if (width != h->n_in) { snprintf(err, errn, "row width %d != head n_in %d", width, h->n_in); return -1; }
  if (!j->q[k].n_tokens) { snprintf(err, errn, "jvh_job_rows before jvh_job_layout"); return -1; }
  float *r = (float *)malloc(sizeof(float) * (size_t)width * (n + 1));
  memcpy(r, rows + n * (size_t)width, sizeof(float) * (size_t)width);    // pointer first (jvh_answer_rows order)
  memcpy(r + width, rows, sizeof(float) * (size_t)width * n);
  free(j->q[k].rows);
  j->q[k].rows = r;
  return 0;
}

int jvh_job_rows_file(jvh_job *j, size_t k, const char *path, char *err, size_t errn) {
  const int w = j->s->h.n_in;
  const size_t n = j->r.q[k].n;
  if (!j->q[k].pos) { snprintf(err, errn, "jvh_job_rows_file before jvh_job_layout"); return -1; }
  uint32_t *pos; float *rows;
  int nr = jvh_rows_read(path, w, &pos, &rows, err, errn);
  if (nr < 0) return -1;
  float *pick = (float *)malloc(sizeof(float) * (size_t)w * (n + 1));
  int r = 0, rc = 0;
  for (size_t i = 0; i <= n && !rc; i++) {  // both lists ascend
    while (r < nr && pos[r] < j->q[k].pos[i]) r++;
    if (r == nr || pos[r] != j->q[k].pos[i]) {
      snprintf(err, errn, "%s: no row at position %u", path, j->q[k].pos[i]);
      rc = -1;
    } else memcpy(pick + i * (size_t)w, rows + (size_t)r * w, sizeof(float) * (size_t)w);
  }
  if (!rc && pos[nr - 1] >= (uint32_t)j->q[k].n_tokens) { snprintf(err, errn, "%s: row beyond the prompt", path); rc = -1; }
  if (!rc) rc = jvh_job_rows(j, k, pick, w, err, errn);
  free(pick); free(pos); free(rows);
  return rc;
}

char *jvh_job_reply(jvh_job *j, const char *model, int mode, char *err, size_t errn) {
  sbuf o = {0};
  long ntok = 0;
  sb_puts(&o, "{\"model\":");
  jv_dump_str(&o, model, strlen(model));
  sb_puts(&o, ",\"answers\":{");
  for (size_t k = 0; k < j->r.nq; k++) {
    if (!j->q[k].rows) { free(o.p); snprintf(err, errn, "question %s has no rows", j->r.q[k].id); return NULL; }
    if (k) sb_putc(&o, ',');
    jv_dump_str(&o, j->r.q[k].id, strlen(j->r.q[k].id));
    sb_putc(&o, ':');
    jvh_answer_rows(&o, &j->s->h, &j->r.q[k], j->q[k].rows, mode, NULL, NULL);
    ntok += j->q[k].n_tokens;
  }
  sb_printf(&o, "},\"usage\":{\"input_tokens\":%ld,\"output_tokens\":0}}", ntok);
  return sb_take(&o);
}

void jvh_job_free(jvh_job *j) {
  if (!j) return;
  for (size_t k = 0; k < j->r.nq; k++) { free(j->q[k].prompt); free(j->q[k].rows); free(j->q[k].pos); }
  free(j->q);
  jvh_req_free(&j->r);
  free(j);
}
