/* backends/windows/engine_cpu.c: engine_cpu, the desktop contract's CPU engine, for Windows (XPLAT-WIN).
 *
 * What it is. The forward is tools/qwen35_engine.c, the repo's C CPU twin (the
 * reference the Bend lanes are checked against; on this machine its 4B ids and logit
 * bits equal the macOS reference). The request side is tools/sys1dual/dual.c, the same
 * object the Linux CUDA engine links, so llb-serve drives this binary exactly as it
 * drives llb-engine / sys1mac: dual_next() hands over one request as an argv-like
 * command ("ids... --steps N --pos K [--sample SPEC]"), the engine prints one id per
 * line on stdout (dual.c captures them), then dual_done() answers the request. This
 * file is the loop the Bend engine's srv / Sys1.next / Sys1.done run (engine_qwen35_gpu
 * .bend srv.*, tools/sys1dual/sys1.c), around the twin's forward_token / final_logits.
 *
 * Why not the emitted Bend engine. engine_qwen35_gpu.bend's per-token pass is one
 * host-served bang (llb.pass_warp) whose Bend body is a NaN placeholder ("No CPU body"),
 * and the host chain that serves it (host/llb_host.c) is compiled only with a GPU
 * backend (LLB_LAUNCH). Emitted without BEND_CUDA, llb-engine.c does not compile (sys1.c
 * needs llb_hidden_* from that chain) and would have nothing to run if it did. There is
 * no emitted CPU engine in this repo for any platform; the Android CPU engine is this
 * same C twin (backends/android).
 *
 * argv, as llb-serve passes it after its own `--` (the Bend runtime's form):
 *   engine_cpu [--threads N] [--gpu off|on|<n>GB|<n>MB] [--workers 1] [--] SIDECAR TOK... --steps CTX
 * --threads N   GEMV worker threads (default: every logical CPU, as the Bend runtime's pool)
 * --gpu ...     accepted and ignored with one stderr line: this is the CPU engine
 * SIDECAR       gguf2pack --sidecar-only's dir (manifest.json, table.bin) + llb-tok's tokenizer.bin;
 *               or a full pack dir (pack.bin + index.tsv), which is read instead of LLB_GGUF
 * TOK... --steps CTX   the KV capacity: longest prompt + steps - 1 positions (llb-serve passes "0 --steps CTX")
 * Env: LLB_SYS1=1 (llb-serve sets it): serve dual.c's JSON lines; unset: run TOK... for CTX steps once and
 *      print the ids and wall_ms, as the Bend engine does. LLB_GGUF: the weights (dual.c requires it too).
 *      TMPDIR: dual.c's scratch. LLB_SAMPLE: the default sampler spec. LLB_CPU_EOG_PAD=0: compute every
 *      requested step even after an end-of-generation id (see eog_pad). LLB_CPU_ROWS4=0: the twin's one-row
 *      dot only (dot4_*, same bits; the A/B switch).
 * stdout: dual.c's response lines (and LLB_SYS1_STREAM's token events). stderr: one load line, then a
 *      {"engine_cpu":{...}} line per request.
 */
#include <stdint.h>
#include <stddef.h>

/* the twin's GEMV row loop, split over this file's workers (tools/qwen35_engine.c LLB_GEMV_ROWS) */
static int ecpu_gemv_rows(int fmt, const uint8_t *W, int rb, int rows, int cols, const float *x, float *y);
#define LLB_GEMV_ROWS ecpu_gemv_rows
#define main qwen35_cli_main
#include "../../tools/qwen35_engine.c"
#undef main
#include "../../host/llb_sampler.h"
#include "../../tools/clef3/json.h"
#include <errno.h>
#include <pthread.h>

int dual_next(const char *dir, unsigned ctx, char **cmd, const char **hidden, unsigned *n); /* tools/sys1dual/dual.c */
void dual_done(void);
int ecpu_gguf_load(const char *path, int threads, uint8_t **pack, uint64_t *pack_bytes,
                   int (*each)(void *ctx, const char *name, uint64_t off, uint64_t bytes), void *ctx, char *err);
int ecpu_gguf_has_head(const char *path, char *err);

static void ecpu_die(const char *fmt, const char *a) {
  fprintf(stderr, "engine_cpu: ");
  fprintf(stderr, fmt, a ? a : "");
  fprintf(stderr, "\n");
  exit(1);
}

/* ---------------------------------------------------------------- GEMV workers */

/* A fixed pool: the caller takes chunk 0, the workers the rest; a generation counter and an atomic
 * count of finished chunks, under one SRW-backed mutex and condition variable (the compat pthread). */
typedef struct { int fmt, rb, rows, cols; const uint8_t *W; const float *x; float *y; } GemvJob;
static int pool_n = 1;
static pthread_mutex_t pool_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t pool_go = PTHREAD_COND_INITIALIZER, pool_done = PTHREAD_COND_INITIALIZER;
static GemvJob pool_job;
static unsigned long long pool_gen;
static int pool_left;

/* Four rows at once, for the formats that carry the weights (Q4_K, Q5_K, Q6_K, Q8_0). Each row's sum
 * is the twin's dot_q4k / dot_q5k / dot_q6k / dot_q8 term for term, in the same order, in its own
 * accumulator: the result bits are the twin's (checked: the logit bits below). The twin's dot is one
 * serial float chain, so a core waits on the add latency; four independent chains fill it. */
#define R4 4
static void dot4_q4k(const uint8_t *const *row, int cols, const float *x, float *out) {
  float acc[R4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  for (int b = 0; b < cols / 256; b++) {
    const uint8_t *blk[R4];
    float d[R4], dmin[R4];
    for (int r = 0; r < R4; r++) blk[r] = row[r] + (size_t)b * 144, d[r] = h2f(blk[r]), dmin[r] = h2f(blk[r] + 2);
    for (int g = 0; g < 4; g++) {
      float d1[R4], m1[R4], d2[R4], m2[R4];
      for (int r = 0; r < R4; r++) {
        int sd, sm;
        gsm4(2 * g, blk[r] + 4, &sd, &sm);
        d1[r] = d[r] * (float)sd, m1[r] = dmin[r] * (float)sm;
        gsm4(2 * g + 1, blk[r] + 4, &sd, &sm);
        d2[r] = d[r] * (float)sd, m2[r] = dmin[r] * (float)sm;
      }
      const float *xg = x + b * 256 + g * 64;
      for (int l = 0; l < 32; l++)
        for (int r = 0; r < R4; r++) acc[r] += (d1[r] * (float)(blk[r][16 + g * 32 + l] & 0xF) - m1[r]) * xg[l];
      for (int l = 0; l < 32; l++)
        for (int r = 0; r < R4; r++) acc[r] += (d2[r] * (float)(blk[r][16 + g * 32 + l] >> 4) - m2[r]) * xg[32 + l];
    }
  }
  for (int r = 0; r < R4; r++) out[r] = acc[r];
}
static void dot4_q5k(const uint8_t *const *row, int cols, const float *x, float *out) {
  float acc[R4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  for (int b = 0; b < cols / 256; b++) {
    const uint8_t *blk[R4];
    float d[R4], dmin[R4];
    for (int r = 0; r < R4; r++) blk[r] = row[r] + (size_t)b * 176, d[r] = h2f(blk[r]), dmin[r] = h2f(blk[r] + 2);
    for (int g = 0; g < 4; g++) {
      float d1[R4], m1[R4], d2[R4], m2[R4];
      for (int r = 0; r < R4; r++) {
        int sd, sm;
        gsm4(2 * g, blk[r] + 4, &sd, &sm);
        d1[r] = d[r] * (float)sd, m1[r] = dmin[r] * (float)sm;
        gsm4(2 * g + 1, blk[r] + 4, &sd, &sm);
        d2[r] = d[r] * (float)sd, m2[r] = dmin[r] * (float)sm;
      }
      const float *xg = x + b * 256 + g * 64;
      for (int l = 0; l < 32; l++)
        for (int r = 0; r < R4; r++) {
          const uint8_t *qh = blk[r] + 16, *q = blk[r] + 48 + g * 32;
          acc[r] += (d1[r] * (float)((q[l] & 0xF) + ((qh[l] >> (2 * g)) & 1) * 16) - m1[r]) * xg[l];
        }
      for (int l = 0; l < 32; l++)
        for (int r = 0; r < R4; r++) {
          const uint8_t *qh = blk[r] + 16, *q = blk[r] + 48 + g * 32;
          acc[r] += (d2[r] * (float)((q[l] >> 4) + ((qh[l] >> (2 * g + 1)) & 1) * 16) - m2[r]) * xg[32 + l];
        }
    }
  }
  for (int r = 0; r < R4; r++) out[r] = acc[r];
}
static void dot4_q6k(const uint8_t *const *row, int cols, const float *x, float *out) {
  float acc[R4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  for (int b = 0; b < cols / 256; b++) {
    const uint8_t *blk[R4];
    float d[R4];
    for (int r = 0; r < R4; r++) blk[r] = row[r] + (size_t)b * 210, d[r] = h2f(blk[r] + 208);
    const float *xb = x + b * 256;
    for (int half = 0; half < 2; half++)
      for (int qd = 0; qd < 4; qd++) {
        const float *xq = xb + half * 128 + qd * 32;
        for (int l = 0; l < 32; l++)
          for (int r = 0; r < R4; r++) {
            const uint8_t *qq = blk[r] + half * 64 + (qd & 1) * 32, *qhh = blk[r] + 128 + half * 32;
            const int8_t *sch = (const int8_t *)(blk[r] + 192) + half * 8;
            int nib = (qd < 2) ? (qq[l] & 0xF) : (qq[l] >> 4);
            int code = (nib | (((qhh[l] >> (2 * qd)) & 3) << 4)) - 32;
            float dl = d[r] * (float)sch[2 * qd + (l >> 4)];
            acc[r] += (dl * (float)code) * xq[l];
          }
      }
  }
  for (int r = 0; r < R4; r++) out[r] = acc[r];
}
static void dot4_q8(const uint8_t *const *row, int cols, const float *x, float *out) {
  float acc[R4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  for (int b = 0; b < cols / 32; b++) {
    float dd[R4];
    for (int r = 0; r < R4; r++) dd[r] = h2f(row[r] + cols + 2 * b);
    for (int j = 0; j < 32; j++)
      for (int r = 0; r < R4; r++) acc[r] += ((float)((const i8 *)row[r])[b * 32 + j] * dd[r]) * x[b * 32 + j];
  }
  for (int r = 0; r < R4; r++) out[r] = acc[r];
}

static int ecpu_rows4 = 1; /* LLB_CPU_ROWS4=0: the twin's one-row dot only (the A/B check) */
static void gemv_chunk(const GemvJob *j, int part, int parts) {
  int a = (int)((long long)j->rows * part / parts), b = (int)((long long)j->rows * (part + 1) / parts), r = a;
  void (*d4)(const uint8_t *const *, int, const float *, float *) = !ecpu_rows4 ? NULL
    : j->fmt == 3 ? dot4_q4k : j->fmt == LLB_FM_Q5_K ? dot4_q5k : j->fmt == 4 ? dot4_q6k : j->fmt == 1 ? dot4_q8 : NULL;
  if (d4)
    for (; r + R4 <= b; r += R4) {
      const uint8_t *rows[R4];
      for (int k = 0; k < R4; k++) rows[k] = j->W + (size_t)(r + k) * j->rb;
      d4(rows, j->cols, j->x, j->y + r);
    }
  for (; r < b; r++) j->y[r] = dot_fmt(j->fmt, j->W + (size_t)r * j->rb, j->cols, j->x);
}
static void *pool_worker(void *arg) {
  int me = (int)(intptr_t)arg;
  unsigned long long seen = 0;
  for (;;) {
    pthread_mutex_lock(&pool_mu);
    while (pool_gen == seen) pthread_cond_wait(&pool_go, &pool_mu);
    seen = pool_gen;
    GemvJob j = pool_job;
    pthread_mutex_unlock(&pool_mu);
    gemv_chunk(&j, me, pool_n);
    pthread_mutex_lock(&pool_mu);
    if (--pool_left == 0) pthread_cond_signal(&pool_done);
    pthread_mutex_unlock(&pool_mu);
  }
  return NULL;
}
static void pool_start(int n) {
  pool_n = n < 1 ? 1 : n > 256 ? 256 : n;
  for (int i = 1; i < pool_n; i++) {
    pthread_t t;
    if (pthread_create(&t, NULL, pool_worker, (void *)(intptr_t)i)) { pool_n = i; break; }
    pthread_detach(t);
  }
}
static int ecpu_gemv_rows(int fmt, const uint8_t *W, int rb, int rows, int cols, const float *x, float *y) {
  if (pool_n < 2 || rows < 64) return 0; /* the twin's own loop */
  pthread_mutex_lock(&pool_mu);
  pool_job = (GemvJob){ fmt, rb, rows, cols, W, x, y };
  pool_left = pool_n - 1;
  pool_gen++;
  pthread_cond_broadcast(&pool_go);
  pthread_mutex_unlock(&pool_mu);
  gemv_chunk(&pool_job, 0, pool_n);
  pthread_mutex_lock(&pool_mu);
  while (pool_left) pthread_cond_wait(&pool_done, &pool_mu);
  pthread_mutex_unlock(&pool_mu);
  return 1;
}

/* ---------------------------------------------------------------- the model from a sidecar + GGUF */

/* manifest.json "config" -> the twin's shape globals (the keys and parsing of load_index's #cfg lines) */
static void cfg_from_manifest(const jv *c) {
  for (size_t i = 0; i < c->n; i++) {
    const char *k = c->keys[i];
    const jv *v = c->items[i];
    if (v->type != J_NUM) continue;
    int iv = (int)v->num;
    float fv = strtof(v->raw ? v->raw : "0", NULL);
    if (!strcmp(k, "n_layers")) L = iv;
    else if (!strcmp(k, "n_heads")) H = iv;
    else if (!strcmp(k, "n_kv_heads")) KV = iv;
    else if (!strcmp(k, "d_model")) D = iv;
    else if (!strcmp(k, "head_dim")) HDIM = iv;
    else if (!strcmp(k, "d_ff")) dff = iv;
    else if (!strcmp(k, "vocab_size")) V = iv;
    else if (!strcmp(k, "full_attention_interval")) INTERVAL = iv;
    else if (!strcmp(k, "ssm_conv_kernel")) KC = iv;
    else if (!strcmp(k, "ssm_state_size")) SZ = iv;
    else if (!strcmp(k, "ssm_group_count")) GK = iv;
    else if (!strcmp(k, "ssm_time_step_rank")) TR = iv;
    else if (!strcmp(k, "ssm_inner_size")) VDC = iv;
    else if (!strcmp(k, "linear_key_dim")) KDC = iv;
    else if (!strcmp(k, "linear_value_head_dim")) VHD = iv;
    else if (!strcmp(k, "linear_conv_dim")) CONVD = iv;
    else if (!strcmp(k, "rope_dim")) ROPEDIM = iv;
    else if (!strcmp(k, "rope_freq_base")) ROPE_BASE = fv;
    else if (!strcmp(k, "rms_norm_eps")) EPS = fv;
  }
}

static int ecpu_check_tensor(void *ctx, const char *name, uint64_t off, uint64_t bytes) {
  (void)ctx;
  for (int i = 0; i < gNT; i++)
    if (!strcmp(gT[i].name, name)) return gT[i].off != off || gT[i].bytes != bytes;
  return 0; /* a tensor the manifest leaves out (a CLEF head) is not read by the forward */
}

/* SIDECAR/manifest.json -> gT (the index.tsv rows) and the shape; the weights from LLB_GGUF */
static size_t load_sidecar(const char *dir, int threads) {
  char path[4096];
  snprintf(path, sizeof path, "%s/manifest.json", dir);
  char *text = json_read_file(path);
  if (!text) ecpu_die("cannot read %s", path);
  const char *err = NULL;
  jv *m = json_parse(text, &err);
  if (!m || m->type != J_OBJ) ecpu_die("%s is not a JSON object", path);
  const jv *cfg = json_get(m, "config"), *ts = json_get(m, "tensors");
  if (!cfg || cfg->type != J_OBJ || !ts || ts->type != J_OBJ) ecpu_die("%s has no config / tensors", path);
  const jv *arch = json_get(cfg, "architecture");
  if (!arch || arch->type != J_STR || strcmp(arch->str, "qwen3_5"))
    ecpu_die("%s: the CPU engine runs dense qwen3_5 models only (architecture %s)", arch && arch->type == J_STR ? arch->str : "?");
  cfg_from_manifest(cfg);
  for (size_t i = 0; i < ts->n; i++) {
    const jv *t = ts->items[i], *sh = json_get(t, "shape"), *of = json_get(t, "offset"), *by = json_get(t, "bytes"),
             *st = json_get(t, "src_type"), *fm = json_get(t, "fmt");
    if (!sh || sh->type != J_ARR || sh->n < 1 || sh->n > 2 || !of || !by || !st || st->type != J_STR || !fm)
      ecpu_die("%s: a tensor entry without shape/offset/bytes/src_type/fmt (2-D at most)", ts->keys[i]);
    if (gNT >= (int)(sizeof gT / sizeof gT[0])) ecpu_die("more than %s tensors", "1024");
    TInfo r;
    memset(&r, 0, sizeof r);
    if (strlen(ts->keys[i]) >= sizeof r.name) ecpu_die("tensor name too long: %s", ts->keys[i]);
    strcpy(r.name, ts->keys[i]);
    r.off = (u64)of->num, r.bytes = (u64)by->num;
    r.rows = sh->n == 2 ? (int)sh->items[0]->num : 1;
    r.cols = sh->n == 2 ? (int)sh->items[1]->num : (int)sh->items[0]->num;
    r.is_f32 = !strcmp(st->str, "F32");
    r.fmt = (int)fm->num;
    if (r.fmt != fmt_of_type(st->str)) ecpu_die("%s: fmt does not match its src_type", r.name);
    r.row_bytes = r.is_f32 ? 0 : (int)(r.bytes / (u64)r.rows);
    gT[gNT++] = r;
  }
  if (find("output.weight") < 0) { /* tied embeddings: the head is token_embd (gguf2pack's alias) */
    int e = find("token_embd.weight");
    if (e < 0) ecpu_die("%s: no token_embd.weight", path);
    gT[gNT] = gT[e];
    strcpy(gT[gNT].name, "output.weight");
    gNT++;
  }
  json_free(m);
  free(text);
  const char *gguf = getenv("LLB_GGUF");
  if (!gguf || !*gguf) ecpu_die("set LLB_GGUF to the model's GGUF (the sidecar %s holds no weights)", dir);
  uint8_t *pack = NULL;
  uint64_t bytes = 0;
  char e2[256];
  if (ecpu_gguf_load(gguf, threads, &pack, &bytes, ecpu_check_tensor, NULL, e2)) ecpu_die("%s", e2);
  for (int i = 0; i < gNT; i++)
    if (gT[i].off + gT[i].bytes > bytes) ecpu_die("%s lies past the GGUF's weights: stale sidecar?", gT[i].name);
  gPack = pack;
  return (size_t)bytes;
}

/* a full pack dir (pack.bin + index.tsv), read into RAM: the twin's own loader */
static size_t load_pack(const char *dir) {
  load_index(dir);
  char ppath[4096];
  snprintf(ppath, sizeof ppath, "%s/pack.bin", dir);
  int fd = llb_open_read(ppath);
  if (fd < 0) ecpu_die("cannot open %s", ppath);
  int64_t sz = llb_fdsize(fd);
  uint8_t *buf = sz > 0 ? malloc((size_t)sz) : NULL;
  if (!buf) ecpu_die("cannot hold %s in RAM", ppath);
  for (int64_t off = 0; off < sz;) {
    int64_t got = llb_pread(fd, buf + off, (size_t)(sz - off), off);
    if (got <= 0) ecpu_die("read %s failed", ppath);
    off += got;
  }
  llb_close(fd);
  gPack = buf;
  return (size_t)sz;
}

/* ---------------------------------------------------------------- one request */

static int *hist;      /* hist[p] = the input id at position p of the live sequence */
static int hist_n;     /* positions in hist (prompt + every id handed back) */
static int done_n;     /* positions whose forward is in the state (<= hist_n: eog_pad skips some) */
static int cap_pos;    /* the KV capacity in positions */
static int eog_pad = 1;
static Q35Tokenizer tok;
static LlbSmpParams smp_par;
static LlbSmp *smp;
static int *ids_buf;

static int is_eog(int id) {
  int len;
  const char *s = q35_tok_text(&tok, id, &len);
  return id == tok.eos || (s && ((len == 10 && !memcmp(s, "<|im_end|>", 10)) || (len == 13 && !memcmp(s, "<|endoftext|>", 13))));
}

static void reset_state(void) {
  memset(gS, 0, (size_t)L * TR * SZ * SZ * 4);
  memset(gConv, 0, (size_t)L * (KC - 1) * CONVD * 4);
  done_n = 0;
}
/* bring the state to hold positions [0, upto) of hist */
static void replay_to(int upto) {
  if (upto < done_n) reset_state(); /* the recurrent state cannot step back */
  while (done_n < upto) forward_token(hist[done_n], done_n), done_n++;
}

/* the next id from LOGITS for position p (the Bend host's smp_pick: greedy = the first max) */
static int pick(int p) {
  if (llb_smp_is_argmax(&smp_par)) return argmax_first(LOGITS, V);
  if (!smp) {
    smp = llb_smp_new(&smp_par, V);
    if (!smp) ecpu_die("cannot allocate the sampler%s", NULL);
  }
  llb_smp_history(smp, hist, p);
  return llb_smp_sample(smp, LOGITS, (uint64_t)p);
}

static int set_sampler(const char *spec) {
  char err[160];
  LlbSmpParams p;
  llb_smp_defaults(&p);
  if (spec && *spec && llb_smp_parse(&p, spec, err, sizeof err) != 0) { fprintf(stderr, "%s\n", err); return -1; }
  llb_smp_free(smp);
  smp = NULL;
  smp_par = p;
  return 0;
}

/* cmd = dual.c's command(): "id id ... --steps S --pos K [--sample SPEC]" */
static void run_request(char *cmd, const char *hidden, unsigned n) {
  int steps = 1, pos = 0, k = 0;
  const char *spec = NULL;
  char *save = NULL;
  for (char *w = strtok_r(cmd, " ", &save); w; w = strtok_r(NULL, " ", &save)) {
    if (!strcmp(w, "--steps")) { char *v = strtok_r(NULL, " ", &save); steps = v ? atoi(v) : -1; }
    else if (!strcmp(w, "--pos")) { char *v = strtok_r(NULL, " ", &save); pos = v ? atoi(v) : -1; }
    else if (!strcmp(w, "--sample")) spec = strtok_r(NULL, " ", &save);
    else if (k < (int)n) ids_buf[k++] = atoi(w);
  }
  if (k != (int)n || steps < 1 || pos < 0 || pos > hist_n || pos + (int)n + steps - 1 > cap_pos) {
    fprintf(stderr, "sys1: bad request command\n");
    exit(2); /* as the Bend engine's srv.one: IO.die 2 */
  }
  for (int i = 0; i < k; i++)
    if (ids_buf[i] < 0 || ids_buf[i] >= V) { fprintf(stderr, "sys1: bad request command\n"); exit(2); }
  if (set_sampler(spec ? spec : getenv("LLB_SAMPLE"))) ecpu_die("bad sampler spec %s", spec);
  u64 t0 = now_ms();
  /* the sequence: hist[0, pos) kept (a continuation), then this request's ids */
  if (pos == 0) reset_state();
  replay_to(pos);
  hist_n = pos;
  FILE *hf = NULL;
  if (hidden && !(hf = fopen(hidden, "wb"))) ecpu_die("cannot write the hidden rows %s", hidden);
  int onorm = find("output_norm.weight");
  for (int i = 0; i < k; i++) {
    hist[hist_n++] = ids_buf[i];
    forward_token(ids_buf[i], done_n), done_n++;
    if (hf) { /* CLEF-3: the prompt row after the output RMS norm, {pos, width} + width floats (llb_hidden_put) */
      u32 hd[2] = { (u32)i, (u32)D };
      rms_norm(HH, X, fvec(onorm), D);
      if (fwrite(hd, 4, 2, hf) != 2 || fwrite(HH, 4, (size_t)D, hf) != (size_t)D) ecpu_die("hidden rows: write failed%s", NULL);
    }
  }
  if (hf && fclose(hf)) ecpu_die("hidden rows: close failed%s", NULL);
  u64 t1 = now_ms();
  int gen = 0, padded = 0, eog = 0, cur = 0;
  for (int st = 0; st < steps; st++) {
    if (!eog) {
      final_logits();
      cur = pick(hist_n);
    }
    printf("%d\n", cur);
    fflush(stdout);
    hist[hist_n++] = cur;
    gen++;
    if (st + 1 >= steps) break;
    if (!eog && eog_pad && is_eog(cur)) eog = 1; /* see eog_pad: the rest are copies of this id, not computed */
    if (eog) { padded++; continue; }
    forward_token(cur, done_n), done_n++;
  }
  u64 t2 = now_ms();
  fprintf(stderr, "{\"engine_cpu\":{\"prompt\":%d,\"pos\":%d,\"gen\":%d,\"padded_after_eog\":%d,\"prefill_ms\":%llu,\"decode_ms\":%llu,"
          "\"decode_tok_s\":%.3f}}\n", k, pos, gen, padded, (unsigned long long)(t1 - t0), (unsigned long long)(t2 - t1),
          gen - padded > 1 ? (double)(gen - padded - 1) * 1000.0 / (double)(t2 - t1 ? t2 - t1 : 1) : 0.0);
}

/* ---------------------------------------------------------------- main */

int main(int argc, char **argv) {
  int threads = 0, gpu_note = 0;
  char **pa = malloc(sizeof *pa * (size_t)(argc + 1));
  int pn = 0;
  if (!pa) return 1;
  for (int i = 1; i < argc; i++) {
    const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
    if (!strcmp(a, "--")) { while (i + 1 < argc) pa[pn++] = argv[++i]; }
    else if (!strcmp(a, "--threads")) {
      char *end = NULL;
      long t = v ? strtol(v, &end, 10) : 0;
      if (t < 1 || !end || *end) ecpu_die("expected a thread count of 1 or more after --threads%s", NULL);
      threads = (int)t, i++;
    } else if (!strcmp(a, "--workers")) {
      if (!v || strcmp(v, "1")) ecpu_die("--workers N > 1 forks worker processes; there is no fork on Windows%s", NULL);
      i++;
    } else if (!strcmp(a, "--gpu")) { if (!v) ecpu_die("expected on, off or a size after --gpu%s", NULL); gpu_note = 1, i++; }
    else if (!strcmp(a, "--help")) {
      printf("usage: %s [--threads N] [--] SIDECAR TOK... --steps CTX   (env LLB_SYS1, LLB_GGUF, TMPDIR)\n", argv[0]);
      return 0;
    } else pa[pn++] = argv[i];
  }
  if (pn < 1) ecpu_die("usage: engine_cpu [--threads N] [--] SIDECAR TOK... --steps CTX%s", NULL);
  const char *dir = pa[0];
  int steps = 8, maxlen = 0, run_ids[8192], nrun = 0;
  for (int i = 1; i < pn; i++) {
    if (!strcmp(pa[i], "--steps") && i + 1 < pn) steps = atoi(pa[++i]);
    else if (pa[i][0] == '-' && pa[i][1] == '-') { if (i + 1 < pn && pa[i + 1][0] != '-') i++; } /* --layers/--lpb/...: Bend knobs */
    else if (nrun < 8192) run_ids[nrun++] = atoi(pa[i]);
  }
  maxlen = nrun;
  if (steps < 1 || maxlen < 1) ecpu_die("needs at least one prompt id and --steps >= 1%s", NULL);
  if (threads < 1) threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
  if (gpu_note) fprintf(stderr, "engine_cpu: --gpu ignored: this is the CPU engine\n");
  const char *r4 = getenv("LLB_CPU_ROWS4");
  ecpu_rows4 = !(r4 && !strcmp(r4, "0"));
  const char *ep = getenv("LLB_CPU_EOG_PAD");
  eog_pad = !(ep && !strcmp(ep, "0"));

  u64 t0 = now_ms();
  char probe[4096];
  snprintf(probe, sizeof probe, "%s/index.tsv", dir);
  FILE *pf = fopen(probe, "rb");
  snprintf(probe, sizeof probe, "%s/pack.bin", dir);
  FILE *pb = pf ? fopen(probe, "rb") : NULL;
  size_t psz = 0;
  const char *from;
  if (pf && pb) { fclose(pf), fclose(pb); psz = load_pack(dir); from = "pack.bin"; }
  else {
    if (pf) fclose(pf);
    psz = load_sidecar(dir, threads);
    from = getenv("LLB_GGUF");
  }
  cap_pos = maxlen + steps - 1;
  state_alloc(cap_pos);
  buf_alloc();
  hist = malloc(sizeof *hist * (size_t)(cap_pos + 1)); /* the last id handed back is never forwarded */
  ids_buf = malloc(sizeof *ids_buf * (size_t)cap_pos);
  if (!hist || !ids_buf) ecpu_die("out of memory for %s positions", "the");
  pool_start(threads);
  fprintf(stderr, "engine_cpu: %d tensors, %zu weight bytes from %s in %llu ms; L=%d D=%d H=%d KV=%d hd=%d dff=%d V=%d; "
          "KV capacity %d positions; %d GEMV threads\n", gNT, psz, from, (unsigned long long)(now_ms() - t0), L, D, H, KV,
          HDIM, dff, V, cap_pos, pool_n);

  const char *sys1 = getenv("LLB_SYS1");
  if (!(sys1 && *sys1)) { /* the Bend engine without LLB_SYS1: the given ids, CTX steps, once */
    char cmd[16];
    (void)cmd;
    for (int i = 0; i < nrun; i++) ids_buf[i] = run_ids[i];
    if (set_sampler(getenv("LLB_SAMPLE"))) return 1;
    u64 a = now_ms();
    reset_state();
    hist_n = 0;
    for (int i = 0; i < nrun; i++) hist[hist_n++] = run_ids[i], forward_token(run_ids[i], done_n), done_n++;
    for (int st = 0; st < steps; st++) {
      final_logits();
      int i1, i2; float v1, v2;
      top2(&i1, &v1, &i2, &v2);
      fprintf(stderr, "logits %d %d %u %d %u\n", hist_n - 1, i1, fbits(v1), i2, fbits(v2));
      int cur = pick(hist_n);
      printf("%d\n", cur);
      hist[hist_n++] = cur;
      if (st + 1 < steps) forward_token(cur, done_n), done_n++;
    }
    printf("wall_ms %llu\n", (unsigned long long)(now_ms() - a));
    return 0;
  }
  if (q35_tok_load(&tok, dir)) ecpu_die("%s needs tokenizer.bin (llb-tok extract)", dir);
  /* LLB_SYS1: a stock GGUF has no CLEF decide head, and dual.c dies at start without one unless
   * LLB_SYS1_NOHEAD is set: say so and serve generate only (decide answers dual.c's refusal) */
  const char *gguf = getenv("LLB_GGUF"), *nh = getenv("LLB_SYS1_NOHEAD");
  if (gguf && *gguf && !(nh && *nh)) {
    char e3[256];
    int head = ecpu_gguf_has_head(gguf, e3);
    if (head < 0) ecpu_die("%s", e3);
    if (!head) {
      fprintf(stderr, "engine_cpu: %s has no CLEF decide head: LLB_SYS1_NOHEAD=1 (generate only)\n", gguf);
      setenv("LLB_SYS1_NOHEAD", "1", 1);
    }
  }
  char *cmd = NULL;
  const char *hidden = NULL;
  unsigned n = 0;
  while (dual_next(dir, (unsigned)cap_pos, &cmd, &hidden, &n)) {
    run_request(cmd, hidden, n);
    free(cmd);
    cmd = NULL;
    dual_done();
  }
  return 0;
}

/* ---------------------------------------------------------------- the Bend host's request hooks */
/* dual.c calls these host/llb_host.c functions; that host chain is the GPU engines'. Here each feature is
 * off, which is what llb_host.c answers when its env switch is unset, except the grammar, which is refused. */
int llb_grammar_configure(const char *kind, const char *text, unsigned at) {
  (void)text, (void)at;
  if (!kind) return 0;
  fprintf(stderr, "engine_cpu: %s-constrained decoding is not in the CPU engine (refused)\n", kind);
  return -1;
}
int llb_pfx_on(void) { return 0; }
unsigned llb_pfx_begin(const int *ids, unsigned n, unsigned start, int reuse) { (void)ids, (void)n, (void)reuse; return start; }
int llb_banks(void) { return 1; }
int llb_bank_select(int j) { return j == 0 ? 0 : -1; }
int llb_bx_max(void) { return 1; }
unsigned llb_heap_bump(void) { return 0; }
void llb_bx_plan(const int *const *ids, const unsigned *n, unsigned k) { (void)ids, (void)n, (void)k; }
void llb_bx_select(const int *ids, unsigned n) { (void)ids, (void)n; }
int llb_bx_next_done(void) { return -1; }
void llb_vis_request(const char *rows, unsigned n) { (void)rows, (void)n; }
void llb_vis_request_list(const char *spec) { (void)spec; }
/* host/llb_pf.h IMG-FAST: dual.c marks image requests for the exact prefill; the CPU engine has one prefill */
int llb_pf_hold_req;
/* dual.c's LLB_MEMPROBE line asks libcuda for device memory; the CPU engine has no device (no line) */
int cuMemGetInfo_v2(size_t *free_b, size_t *total_b) { (void)free_b, (void)total_b; return 1; }
