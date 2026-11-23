// ec_serve.c — 100% Native Multi-Threaded Parallel Eureka Creative Studio Server (v2)
// Pure portable C11 (zero CUDA headers, zero external HTTP/JSON libraries).
// Features:
//   - Concurrent pthread-per-connection HTTP/1.1 handling (non-blocking parallel APIs!)
//   - Async Job Queue (POST /v1/jobs, GET /v1/jobs/<id>, GET /v1/jobs/<id>/events SSE, POST /v1/jobs/<id>/cancel)
//   - Full Creative Routes (/v1/creative/health, /v1/images/generations, /v1/3d/generations,
//     /v1/audio/music, /v1/videos/generations [NVENC H.264+AAC MP4], /v1/videos/realtime/burst,
//     /v1/flows/run, /v1/creative/models*, /v1/director/plan, /artifacts/<file>)
//   - Eureka Security Parity: --api-key / EC_API_KEY auth, 127.0.0.1 loopback, --parent-fd lifecycle exit.

#define _POSIX_C_SOURCE 200809L
#include "ec_studio.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int s) { (void)s; g_stop = 1; }

static char g_api_key[128] = {0};

static const char *basename_safe(const char *path) {
  const char *p = strrchr(path, '/');
  return p ? (p + 1) : path;
}

static void send_all(int fd, const void *buf, size_t len) {
  const uint8_t *p = (const uint8_t *)buf;
  while (len > 0) {
    ssize_t w = write(fd, p, len);
    if (w <= 0) { if (errno == EINTR) continue; break; }
    p += w; len -= (size_t)w;
  }
}

static void send_json(int fd, int code, const char *body) {
  const char *reason = code == 200 ? "OK" : code == 202 ? "Accepted" :
                       code == 401 ? "Unauthorized" : code == 404 ? "Not Found" :
                       code == 500 ? "Internal Server Error" : "Bad Request";
  char hdr[512];
  size_t blen = strlen(body);
  int hn = snprintf(hdr, sizeof hdr,
      "HTTP/1.1 %d %s\r\n"
      "Content-Type: application/json\r\n"
      "Access-Control-Allow-Origin: *\r\n"
      "Access-Control-Allow-Headers: Content-Type, Authorization, X-Api-Key\r\n"
      "Content-Length: %zu\r\n"
      "Connection: close\r\n\r\n",
      code, reason, blen);
  send_all(fd, hdr, (size_t)hn);
  send_all(fd, body, blen);
}

static void send_file(int fd, const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) { send_json(fd, 404, "{\"error\":\"artifact not found\"}"); return; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz <= 0 || sz > (128L << 20)) { fclose(f); send_json(fd, 404, "{\"error\":\"bad artifact size\"}"); return; }

  const char *mime = "application/octet-stream";
  if (strstr(path, ".png"))      mime = "image/png";
  else if (strstr(path, ".glb")) mime = "model/gltf-binary";
  else if (strstr(path, ".wav")) mime = "audio/wav";
  else if (strstr(path, ".mp4")) mime = "video/mp4";

  char hdr[512];
  int hn = snprintf(hdr, sizeof hdr,
      "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nAccess-Control-Allow-Origin: *\r\n"
      "Content-Length: %ld\r\nConnection: close\r\n\r\n", mime, sz);
  send_all(fd, hdr, (size_t)hn);
  uint8_t buf[16384];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) send_all(fd, buf, n);
  fclose(f);
}

// Lightweight JSON extractors
static void json_get_str(const char *j, const char *key, char *out, size_t cap, const char *def) {
  snprintf(out, cap, "%s", def);
  char pat[64];
  snprintf(pat, sizeof pat, "\"%s\"", key);
  const char *p = strstr(j, pat);
  if (!p) return;
  p = strchr(p + strlen(pat), ':');
  if (!p) return;
  p++;
  while (*p == ' ' || *p == '\t') p++;
  if (*p != '"') return;
  p++;
  size_t n = 0;
  while (*p && *p != '"' && n + 1 < cap) out[n++] = *p++;
  out[n] = '\0';
}

static double json_get_num(const char *j, const char *key, double def) {
  char pat[64];
  snprintf(pat, sizeof pat, "\"%s\"", key);
  const char *p = strstr(j, pat);
  if (!p) return def;
  p = strchr(p + strlen(pat), ':');
  if (!p) return def;
  p++;
  while (*p == ' ' || *p == '\t') p++;
  char *ep = NULL;
  double v = strtod(p, &ep);
  return (ep != p) ? v : def;
}

static int json_get_bool(const char *j, const char *key, int def) {
  char pat[64];
  snprintf(pat, sizeof pat, "\"%s\"", key);
  const char *p = strstr(j, pat);
  if (!p) return def;
  p = strchr(p + strlen(pat), ':');
  if (!p) return def;
  p++;
  while (*p == ' ' || *p == '\t') p++;
  if (!strncmp(p, "true", 4)) return 1;
  if (!strncmp(p, "false", 5)) return 0;
  return def;
}

// Auth check matching supervisor.mjs (X-Api-Key or Authorization: Bearer <key>)
static int check_auth(const char *raw_req, size_t hdr_len) {
  if (!g_api_key[0]) return 1;
  char *hdr = (char *)malloc(hdr_len + 1);
  if (!hdr) return 0;
  memcpy(hdr, raw_req, hdr_len);
  hdr[hdr_len] = '\0';
  int ok = 0;
  // Case-insensitive header scan line-by-line
  char *line = hdr;
  while (line && *line) {
    char *eol = strstr(line, "\r\n");
    if (eol) *eol = '\0';
    if (!strncasecmp(line, "X-Api-Key:", 10)) {
      const char *v = line + 10; while (*v == ' ') v++;
      if (!strcmp(v, g_api_key)) ok = 1;
    } else if (!strncasecmp(line, "Authorization:", 14)) {
      const char *v = line + 14; while (*v == ' ') v++;
      if (!strncasecmp(v, "Bearer ", 7) && !strcmp(v + 7, g_api_key)) ok = 1;
    }
    if (!eol) break;
    line = eol + 2;
  }
  free(hdr);
  return ok;
}

// ===========================================================================
// Async Parallel Job Manager (Thread-Safe Queue + Live SSE Events + Cancel)
// ===========================================================================
#define EC_MAX_JOBS 64

typedef enum {
  JOB_QUEUED    = 0,
  JOB_RUNNING   = 1,
  JOB_DONE      = 2,
  JOB_CANCELLED = 3,
  JOB_FAILED    = 4,
} job_state_t;

typedef struct {
  int           id;
  char          kind[24];
  char          prompt[256];
  char          tier[16];
  char          stage[32];
  uint32_t      seed;
  int           width, height, steps, frames, bpm, with_audio;
  float         duration_sec;
  job_state_t   state;
  _Atomic int   cancel_req;
  int           cur_step, total_steps, skipped;
  char          result_json[2048];
  ec_studio_t  *st;
} ec_job_t;

static pthread_mutex_t g_jobs_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_jobs_cv = PTHREAD_COND_INITIALIZER;
static ec_job_t        g_jobs[EC_MAX_JOBS];
static int             g_next_job_id = 1;

static const char *job_state_str(job_state_t s) {
  switch (s) {
    case JOB_QUEUED:    return "queued";
    case JOB_RUNNING:   return "running";
    case JOB_DONE:      return "done";
    case JOB_CANCELLED: return "cancelled";
    default:            return "failed";
  }
}

static int job_progress_cb(int step, int total, int skipped, const char *stage, void *ctx) {
  ec_job_t *j = (ec_job_t *)ctx;
  pthread_mutex_lock(&g_jobs_mu);
  j->cur_step = step;
  j->total_steps = total;
  j->skipped = skipped;
  if (stage) snprintf(j->stage, sizeof j->stage, "%s", stage);
  pthread_cond_broadcast(&g_jobs_cv);
  pthread_mutex_unlock(&g_jobs_mu);
  return atomic_load(&j->cancel_req) ? 1 : 0;
}

static void *job_runner_thread(void *arg) {
  ec_job_t *j = (ec_job_t *)arg;
  pthread_mutex_lock(&g_jobs_mu);
  j->state = JOB_RUNNING;
  pthread_cond_broadcast(&g_jobs_cv);
  pthread_mutex_unlock(&g_jobs_mu);

  ec_tier_t tier = !strcmp(j->tier, "exact") ? EC_TIER_EXACT : EC_TIER_FAST;
  ec_studio_result_t r;
  int rc = -1;
  char out[2048] = {0};

  if (!strcmp(j->kind, "image")) {
    rc = ec_studio_gen_image(j->st, j->prompt, j->seed, j->width, j->height, j->steps, tier, job_progress_cb, j, &r);
    if (rc == 0 && !r.cancelled)
      snprintf(out, sizeof out, "{\"artifact_url\":\"/artifacts/%s\",\"width\":%d,\"height\":%d,\"steps_run\":%d,\"steps_skipped\":%d,\"gpu_ms\":%.2f}",
               basename_safe(r.file_main), r.width, r.height, r.steps_run, r.steps_skipped, r.gpu_ms);
  } else if (!strcmp(j->kind, "3d")) {
    rc = ec_studio_gen_3d(j->st, j->prompt, j->seed, 20, tier, job_progress_cb, j, &r);
    if (rc == 0 && !r.cancelled)
      snprintf(out, sizeof out, "{\"glb_url\":\"/artifacts/%s\",\"splat_ply_url\":\"/artifacts/%s\",\"n_verts\":%d,\"n_tris\":%d,\"n_splats\":%d,\"gpu_ms\":%.2f}",
               basename_safe(r.file_main), basename_safe(r.file_aux), r.n_verts, r.n_tris, r.n_splats, r.gpu_ms);
  } else if (!strcmp(j->kind, "music")) {
    rc = ec_studio_gen_music(j->st, j->prompt, j->seed, j->bpm, j->duration_sec, tier, job_progress_cb, j, &r);
    if (rc == 0 && !r.cancelled)
      snprintf(out, sizeof out, "{\"master_wav_url\":\"/artifacts/%s\",\"bpm\":%d,\"gpu_ms\":%.2f}",
               basename_safe(r.file_main), j->bpm, r.gpu_ms);
  } else if (!strcmp(j->kind, "video") || !strcmp(j->kind, "video_av")) {
    int wa = (!strcmp(j->kind, "video_av") || j->with_audio) ? 1 : 0;
    rc = ec_studio_gen_video(j->st, j->prompt, j->seed, j->frames, j->width, j->height, 24, wa, tier, job_progress_cb, j, &r);
    if (rc == 0 && !r.cancelled)
      snprintf(out, sizeof out, "{\"mp4_url\":\"/artifacts/%s\",\"keyframe0_url\":\"/artifacts/%s\",\"encoder\":\"%s\",\"frames\":%d,\"with_audio\":%s,\"gpu_ms\":%.2f}",
               basename_safe(r.file_mp4), basename_safe(r.file_aux), r.encoder_used, r.frames, wa ? "true" : "false", r.gpu_ms);
  } else if (!strcmp(j->kind, "realtime")) {
    rc = ec_studio_realtime_burst(j->st, j->prompt, j->seed, j->frames, "/eureka_rt_job_burst", job_progress_cb, j, &r);
    if (rc == 0 && !r.cancelled)
      snprintf(out, sizeof out, "{\"shm_ring\":\"%s\",\"frames\":%d,\"fps\":%.1f,\"elapsed_ms\":%.2f}",
               r.file_main, r.frames, r.fps, r.gpu_ms);
  }

  pthread_mutex_lock(&g_jobs_mu);
  if (rc == 0 && r.cancelled) {
    j->state = JOB_CANCELLED;
    snprintf(j->result_json, sizeof j->result_json, "{\"cancelled\":true}");
  } else if (rc == 0) {
    j->state = JOB_DONE;
    j->cur_step = j->total_steps > 0 ? j->total_steps : 1;
    snprintf(j->result_json, sizeof j->result_json, "%s", out[0] ? out : "{}");
  } else {
    j->state = JOB_FAILED;
    snprintf(j->result_json, sizeof j->result_json, "{\"error\":\"job failed\"}");
  }
  pthread_cond_broadcast(&g_jobs_cv);
  pthread_mutex_unlock(&g_jobs_mu);
  return NULL;
}

static ec_job_t *find_job_locked(int id) {
  for (int i = 0; i < EC_MAX_JOBS; i++) if (g_jobs[i].id == id) return &g_jobs[i];
  return NULL;
}

static void format_job_json_locked(const ec_job_t *j, char *buf, size_t cap) {
  int pct = (j->state == JOB_DONE) ? 100 : (j->total_steps > 0 ? (j->cur_step * 100 / j->total_steps) : 0);
  snprintf(buf, cap,
      "{\"job_id\":%d,\"kind\":\"%s\",\"state\":\"%s\",\"stage\":\"%s\","
      "\"step\":%d,\"total_steps\":%d,\"skipped\":%d,\"pct\":%d,\"result\":%s}",
      j->id, j->kind, job_state_str(j->state), j->stage,
      j->cur_step, j->total_steps, j->skipped, pct,
      j->result_json[0] ? j->result_json : "null");
}

static void stream_job_sse(int cfd, int job_id) {
  const char *hdr =
      "HTTP/1.1 200 OK\r\n"
      "Content-Type: text/event-stream\r\n"
      "Cache-Control: no-cache\r\n"
      "Access-Control-Allow-Origin: *\r\n"
      "Connection: close\r\n\r\n";
  send_all(cfd, hdr, strlen(hdr));

  int last_step = -1;
  job_state_t last_state = (job_state_t)-1;
  char payload[2560], frame[2700];

  while (!g_stop) {
    pthread_mutex_lock(&g_jobs_mu);
    ec_job_t *j = find_job_locked(job_id);
    if (!j) {
      pthread_mutex_unlock(&g_jobs_mu);
      send_all(cfd, "event: error\ndata: {\"error\":\"job not found\"}\n\n", 45);
      return;
    }
    while (j->cur_step == last_step && j->state == last_state &&
           (j->state == JOB_QUEUED || j->state == JOB_RUNNING) && !g_stop) {
      struct timespec ts;
      clock_gettime(CLOCK_REALTIME, &ts);
      ts.tv_nsec += 200000000L;
      if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
      pthread_cond_timedwait(&g_jobs_cv, &g_jobs_mu, &ts);
    }
    last_step = j->cur_step;
    last_state = j->state;
    format_job_json_locked(j, payload, sizeof payload);
    int done = (j->state == JOB_DONE || j->state == JOB_CANCELLED || j->state == JOB_FAILED);
    const char *ev = done ? (j->state == JOB_DONE ? "done" : job_state_str(j->state)) : "progress";
    pthread_mutex_unlock(&g_jobs_mu);

    int fn = snprintf(frame, sizeof frame, "event: %s\ndata: %s\n\n", ev, payload);
    send_all(cfd, frame, (size_t)fn);
    if (done) return;
  }
}

// ===========================================================================
// Per-Connection Worker Thread (Full Parallel HTTP Handling!)
// ===========================================================================
typedef struct {
  int          cfd;
  ec_studio_t *st;
} conn_arg_t;

static void *client_thread(void *arg) {
  conn_arg_t *ca = (conn_arg_t *)arg;
  int cfd = ca->cfd;
  ec_studio_t *st = ca->st;
  free(ca);

  char *req = (char *)calloc(1, 262144);
  if (!req) { close(cfd); return NULL; }
  size_t got = 0, hdr_end = 0, content_len = 0;
  while (got < 262143) {
    ssize_t r = read(cfd, req + got, 262143 - got);
    if (r <= 0) break;
    got += (size_t)r;
    req[got] = '\0';
    if (!hdr_end) {
      char *e = strstr(req, "\r\n\r\n");
      if (e) {
        hdr_end = (size_t)(e - req) + 4;
        const char *cl = strstr(req, "Content-Length:");
        if (!cl) cl = strstr(req, "content-length:");
        if (cl && cl < e) content_len = (size_t)strtoull(cl + 15, NULL, 10);
      }
    }
    if (hdr_end && got >= hdr_end + content_len) break;
  }
  if (got < 10) { free(req); close(cfd); return NULL; }

  char method[16] = {0}, path[256] = {0};
  if (sscanf(req, "%15s %255s", method, path) != 2) { free(req); close(cfd); return NULL; }
  // Strip query string if present
  char *qmark = strchr(path, '?');
  if (qmark) *qmark = '\0';
  const char *body = hdr_end ? (req + hdr_end) : "";

  if (!strcmp(method, "OPTIONS")) {
    const char *opt =
        "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization, X-Api-Key\r\nConnection: close\r\n\r\n";
    send_all(cfd, opt, strlen(opt));
    free(req); close(cfd); return NULL;
  }

  if (!check_auth(req, hdr_end ? hdr_end : got)) {
    send_json(cfd, 401, "{\"error\":\"unauthorized\"}");
    free(req); close(cfd); return NULL;
  }

  char *resp = (char *)malloc(32768);
  ec_studio_info_t info;
  ec_studio_get_info(st, &info);

  if (!strcmp(method, "GET") && !strcmp(path, "/v1/creative/health")) {
    snprintf(resp, 32768,
        "{\"status\":\"ok\",\"engine\":\"eureka-creative-native-v2\","
        "\"ownership\":\"100%% eureka engine (zero cuBLAS, zero ggml, zero Python)\","
        "\"parallel\":true,\"auth_enabled\":%s,"
        "\"gpu\":\"%s\",\"sm\":\"sm_%d%d\",\"vram_total_gb\":%.2f,"
        "\"vram_pool_used_mb\":%.2f,\"vram_pool_budget_mb\":%.2f,\"models_discovered\":%d,"
        "\"modalities\":[\"image\",\"3d_glb_splat\",\"music_stems\",\"video_nvenc_mp4\",\"video_av_ltx\",\"realtime_video_shm\",\"comfyui_dag\",\"async_jobs_sse\",\"model_registry\",\"director_plan\"]}",
        g_api_key[0] ? "true" : "false",
        info.gpu_name, info.sm_major, info.sm_minor, info.vram_total_gb,
        info.vram_pool_used_mb, info.vram_pool_budget_mb, info.models_discovered);
    send_json(cfd, 200, resp);
  } else if (!strcmp(method, "GET") && !strncmp(path, "/artifacts/", 11)) {
    const char *fname = path + 11;
    if (!fname[0] || strstr(fname, "..") || strchr(fname, '/') || strchr(fname, '\\')) {
      send_json(cfd, 400, "{\"error\":\"invalid artifact filename\"}");
    } else {
      char fpath[512];
      snprintf(fpath, sizeof fpath, "%s/%s", info.artifact_dir, fname);
      send_file(cfd, fpath);
    }
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/jobs")) {
    pthread_mutex_lock(&g_jobs_mu);
    int slot = (g_next_job_id - 1) % EC_MAX_JOBS;
    ec_job_t *j = &g_jobs[slot];
    memset(j, 0, sizeof *j);
    j->id = g_next_job_id++;
    j->st = st;
    j->state = JOB_QUEUED;
    atomic_init(&j->cancel_req, 0);
    json_get_str(body, "kind", j->kind, sizeof j->kind, "image");
    json_get_str(body, "prompt", j->prompt, sizeof j->prompt, "eureka async studio");
    json_get_str(body, "tier", j->tier, sizeof j->tier, "fast");
    j->seed = (uint32_t)json_get_num(body, "seed", 42 + j->id);
    j->width = (int)json_get_num(body, "width", !strcmp(j->kind, "video") || !strcmp(j->kind, "video_av") ? 320 : 256);
    j->height = (int)json_get_num(body, "height", !strcmp(j->kind, "video") || !strcmp(j->kind, "video_av") ? 180 : 256);
    j->steps = (int)json_get_num(body, "steps", 12);
    j->frames = (int)json_get_num(body, "frames", 24);
    j->bpm = (int)json_get_num(body, "bpm", 124);
    j->duration_sec = (float)json_get_num(body, "duration_sec", 2.0);
    j->with_audio = json_get_bool(body, "with_audio", !strcmp(j->kind, "video_av") ? 1 : 0);
    snprintf(j->stage, sizeof j->stage, "queued");
    int jid = j->id;
    pthread_mutex_unlock(&g_jobs_mu);

    pthread_t th;
    pthread_create(&th, NULL, job_runner_thread, j);
    pthread_detach(th);

    snprintf(resp, 32768,
        "{\"ok\":true,\"job_id\":%d,\"status_url\":\"/v1/jobs/%d\","
        "\"events_url\":\"/v1/jobs/%d/events\",\"cancel_url\":\"/v1/jobs/%d/cancel\"}",
        jid, jid, jid, jid);
    send_json(cfd, 202, resp);
  } else if (!strcmp(method, "GET") && !strncmp(path, "/v1/jobs/", 9)) {
    int jid = atoi(path + 9);
    if (strstr(path, "/events")) {
      stream_job_sse(cfd, jid);
    } else {
      pthread_mutex_lock(&g_jobs_mu);
      ec_job_t *j = find_job_locked(jid);
      if (j) { format_job_json_locked(j, resp, 32768); pthread_mutex_unlock(&g_jobs_mu); send_json(cfd, 200, resp); }
      else   { pthread_mutex_unlock(&g_jobs_mu); send_json(cfd, 404, "{\"error\":\"job not found\"}"); }
    }
  } else if (!strcmp(method, "POST") && !strncmp(path, "/v1/jobs/", 9) && strstr(path, "/cancel")) {
    int jid = atoi(path + 9);
    pthread_mutex_lock(&g_jobs_mu);
    ec_job_t *j = find_job_locked(jid);
    if (j) { atomic_store(&j->cancel_req, 1); pthread_cond_broadcast(&g_jobs_cv); }
    pthread_mutex_unlock(&g_jobs_mu);
    send_json(cfd, j ? 200 : 404, j ? "{\"ok\":true,\"cancel_requested\":true}" : "{\"error\":\"job not found\"}");
  } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/creative/models")) {
    ec_studio_models_json(st, resp, 32768);
    send_json(cfd, 200, resp);
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/creative/models/scan")) {
    ec_studio_models_scan(st);
    ec_studio_models_json(st, resp, 32768);
    send_json(cfd, 200, resp);
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/creative/models/load")) {
    char id[64], route[128] = {0}, err[256] = {0};
    double ms = 0.0;
    json_get_str(body, "id", id, sizeof id, "");
    if (ec_studio_model_load(st, id, &ms, route, sizeof route, err, sizeof err) == 0) {
      snprintf(resp, 32768, "{\"ok\":true,\"id\":\"%s\",\"route\":\"%s\",\"probe_gpu_ms\":%.2f}", id, route, ms);
      send_json(cfd, 200, resp);
    } else {
      snprintf(resp, 32768, "{\"error\":\"%s\"}", err[0] ? err : "model load failed");
      send_json(cfd, 400, resp);
    }
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/creative/models/unload")) {
    char id[64]; json_get_str(body, "id", id, sizeof id, "");
    int rc = ec_studio_model_unload(st, id);
    send_json(cfd, rc == 0 ? 200 : 404, rc == 0 ? "{\"ok\":true}" : "{\"error\":\"not loaded\"}");
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/director/plan")) {
    char brief[256];
    json_get_str(body, "brief", brief, sizeof brief, "futuristic bioluminescent studio showcase");
    uint32_t seed = (uint32_t)json_get_num(body, "seed", 2026);
    int bpm = (int)json_get_num(body, "bpm", 124);
    int exec_now = json_get_bool(body, "execute", 0);
    char dag_json[2048];
    snprintf(dag_json, sizeof dag_json,
        "{\"1\":{\"class_type\":\"EurekaImageGen\",\"inputs\":{\"prompt\":\"%s\",\"seed\":%u}},"
        "\"2\":{\"class_type\":\"Eureka3DGen\",\"inputs\":{\"image\":[\"1\",0],\"prompt\":\"%s\",\"seed\":%u}},"
        "\"3\":{\"class_type\":\"EurekaMusicGen\",\"inputs\":{\"prompt\":\"%s soundtrack\",\"seed\":%u,\"bpm\":%d}},"
        "\"4\":{\"class_type\":\"EurekaVideoGen\",\"inputs\":{\"image\":[\"1\",0],\"mesh\":[\"2\",0],\"music\":[\"3\",0],\"prompt\":\"%s\",\"seed\":%u}}}",
        brief, seed, brief, seed, brief, seed, bpm, brief, seed);
    if (exec_now) {
      ec_studio_result_t r; char err[256] = {0};
      if (ec_studio_run_flow_json(st, dag_json, &r, err, sizeof err) == 0) {
        snprintf(resp, 32768,
            "{\"ok\":true,\"executed\":true,\"dag_executed\":%d,\"cache_hits\":%d,"
            "\"image_url\":\"/artifacts/%s\",\"glb_url\":\"/artifacts/%s\","
            "\"music_url\":\"/artifacts/%s\",\"mp4_url\":\"/artifacts/%s\",\"workflow\":%s}",
            r.dag_executed, r.dag_cache_hits,
            basename_safe(r.file_main), basename_safe(r.file_aux),
            basename_safe(r.stem_files[0]), basename_safe(r.file_mp4), dag_json);
        send_json(cfd, 200, resp);
      } else {
        send_json(cfd, 500, "{\"error\":\"director execute failed\"}");
      }
    } else {
      snprintf(resp, 32768, "{\"ok\":true,\"executed\":false,\"workflow\":%s}", dag_json);
      send_json(cfd, 200, resp);
    }
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/images/generations")) {
    char prompt[256], tier_s[16];
    json_get_str(body, "prompt", prompt, sizeof prompt, "eureka studio art");
    json_get_str(body, "tier", tier_s, sizeof tier_s, "fast");
    uint32_t seed = (uint32_t)json_get_num(body, "seed", 42);
    int w = (int)json_get_num(body, "width", 256), h = (int)json_get_num(body, "height", 256);
    int steps = (int)json_get_num(body, "steps", 8);
    ec_tier_t tier = !strcmp(tier_s, "exact") ? EC_TIER_EXACT : EC_TIER_FAST;
    ec_studio_result_t r;
    if (ec_studio_gen_image(st, prompt, seed, w, h, steps, tier, NULL, NULL, &r) == 0) {
      snprintf(resp, 32768,
          "{\"ok\":true,\"artifact_url\":\"/artifacts/%s\",\"width\":%d,\"height\":%d,"
          "\"tier\":\"%s\",\"steps_run\":%d,\"steps_skipped\":%d,\"gpu_ms\":%.2f}",
          basename_safe(r.file_main), r.width, r.height, tier == EC_TIER_FAST ? "fast" : "exact",
          r.steps_run, r.steps_skipped, r.gpu_ms);
      send_json(cfd, 200, resp);
    } else send_json(cfd, 500, "{\"error\":\"image generation failed\"}");
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/3d/generations")) {
    char prompt[256]; json_get_str(body, "prompt", prompt, sizeof prompt, "sculptural torus");
    uint32_t seed = (uint32_t)json_get_num(body, "seed", 77);
    int res_g = (int)json_get_num(body, "grid_res", 20);
    ec_studio_result_t r;
    if (ec_studio_gen_3d(st, prompt, seed, res_g, EC_TIER_FAST, NULL, NULL, &r) == 0) {
      snprintf(resp, 32768,
          "{\"ok\":true,\"glb_url\":\"/artifacts/%s\",\"splat_ply_url\":\"/artifacts/%s\","
          "\"n_verts\":%d,\"n_tris\":%d,\"n_splats\":%d,\"gpu_ms\":%.2f}",
          basename_safe(r.file_main), basename_safe(r.file_aux), r.n_verts, r.n_tris, r.n_splats, r.gpu_ms);
      send_json(cfd, 200, resp);
    } else send_json(cfd, 500, "{\"error\":\"3d generation failed\"}");
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/audio/music")) {
    char prompt[256]; json_get_str(body, "prompt", prompt, sizeof prompt, "ambient electronic pulse");
    uint32_t seed = (uint32_t)json_get_num(body, "seed", 101);
    int bpm = (int)json_get_num(body, "bpm", 122);
    float dur = (float)json_get_num(body, "duration_sec", 3.0);
    ec_studio_result_t r;
    if (ec_studio_gen_music(st, prompt, seed, bpm, dur, EC_TIER_FAST, NULL, NULL, &r) == 0) {
      snprintf(resp, 32768,
          "{\"ok\":true,\"master_wav_url\":\"/artifacts/%s\","
          "\"stems\":{\"drums\":\"/artifacts/%s\",\"bass\":\"/artifacts/%s\","
          "\"harmony\":\"/artifacts/%s\",\"lead\":\"/artifacts/%s\"},"
          "\"bpm\":%d,\"frames\":%d,\"sample_rate\":44100,\"gpu_ms\":%.2f}",
          basename_safe(r.file_main),
          basename_safe(r.stem_files[0]), basename_safe(r.stem_files[1]),
          basename_safe(r.stem_files[2]), basename_safe(r.stem_files[3]),
          bpm, r.frames, r.gpu_ms);
      send_json(cfd, 200, resp);
    } else send_json(cfd, 500, "{\"error\":\"music generation failed\"}");
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/videos/generations")) {
    char prompt[256]; json_get_str(body, "prompt", prompt, sizeof prompt, "cinematic studio motion");
    uint32_t seed = (uint32_t)json_get_num(body, "seed", 303);
    int n_frames = (int)json_get_num(body, "frames", 24);
    int w = (int)json_get_num(body, "width", 320), h = (int)json_get_num(body, "height", 180);
    int fps = (int)json_get_num(body, "fps", 24);
    int with_audio = json_get_bool(body, "with_audio", 1);
    ec_studio_result_t r;
    if (ec_studio_gen_video(st, prompt, seed, n_frames, w, h, fps, with_audio, EC_TIER_FAST, NULL, NULL, &r) == 0) {
      snprintf(resp, 32768,
          "{\"ok\":true,\"mp4_url\":\"/artifacts/%s\",\"keyframe0_url\":\"/artifacts/%s\","
          "\"encoder\":\"%s\",\"with_audio\":%s,\"frames\":%d,\"width\":%d,\"height\":%d,"
          "\"gpu_ms\":%.2f,\"fps\":%.1f}",
          basename_safe(r.file_mp4), basename_safe(r.file_aux), r.encoder_used,
          with_audio ? "true" : "false", r.frames, r.width, r.height, r.gpu_ms, r.fps);
      send_json(cfd, 200, resp);
    } else send_json(cfd, 500, "{\"error\":\"video generation failed\"}");
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/videos/realtime/burst")) {
    char prompt[256]; json_get_str(body, "prompt", prompt, sizeof prompt, "live interactive canvas");
    uint32_t seed = (uint32_t)json_get_num(body, "seed", 404);
    int n_frames = (int)json_get_num(body, "frames", 30);
    ec_studio_result_t r;
    if (ec_studio_realtime_burst(st, prompt, seed, n_frames, "/eureka_rt_http_burst", NULL, NULL, &r) == 0) {
      snprintf(resp, 32768,
          "{\"ok\":true,\"shm_ring\":\"%s\",\"width\":%d,\"height\":%d,"
          "\"frames\":%d,\"elapsed_ms\":%.2f,\"fps\":%.1f}",
          r.file_main, r.width, r.height, r.frames, r.gpu_ms, r.fps);
      send_json(cfd, 200, resp);
    } else send_json(cfd, 500, "{\"error\":\"realtime burst failed\"}");
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/flows/run")) {
    ec_studio_result_t r; char err[256] = {0};
    if (ec_studio_run_flow_json(st, body, &r, err, sizeof err) == 0) {
      snprintf(resp, 32768,
          "{\"ok\":true,\"executed\":%d,\"cache_hits\":%d,\"vram_evictions\":%d,"
          "\"gpu_ms\":%.2f,\"image_url\":\"%s\",\"glb_url\":\"%s\",\"music_url\":\"%s\",\"mp4_url\":\"%s\"}",
          r.dag_executed, r.dag_cache_hits, r.dag_evictions, r.gpu_ms,
          r.file_main[0] ? basename_safe(r.file_main) : "",
          r.file_aux[0]  ? basename_safe(r.file_aux)  : "",
          r.stem_files[0][0] ? basename_safe(r.stem_files[0]) : "",
          r.file_mp4[0] ? basename_safe(r.file_mp4) : "");
      send_json(cfd, 200, resp);
    } else {
      snprintf(resp, 32768, "{\"error\":\"%s\"}", err[0] ? err : "flow failed");
      send_json(cfd, 400, resp);
    }
  } else {
    send_json(cfd, 404, "{\"error\":\"unknown route\"}");
  }

  free(resp);
  free(req);
  close(cfd);
  return NULL;
}

static void *parent_watch_thread(void *arg) {
  int fd = *(int *)arg;
  char b;
  while (read(fd, &b, 1) > 0) {}
  g_stop = 1;
  return NULL;
}

int main(int argc, char **argv) {
  int port = 8099, parent_fd = -1;
  const char *art_dir = "/tmp/eureka-creative-build/artifacts";
  const char *models_dir = getenv("EUREKA_CREATIVE_MODELS");
  const char *env_key = getenv("EC_API_KEY");
  if (env_key && env_key[0]) snprintf(g_api_key, sizeof g_api_key, "%s", env_key);

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--port") && i + 1 < argc)            port = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--artifacts") && i + 1 < argc)  art_dir = argv[++i];
    else if (!strcmp(argv[i], "--models-dir") && i + 1 < argc) models_dir = argv[++i];
    else if (!strcmp(argv[i], "--api-key") && i + 1 < argc)    snprintf(g_api_key, sizeof g_api_key, "%s", argv[++i]);
    else if (!strcmp(argv[i], "--parent-fd") && i + 1 < argc)  parent_fd = atoi(argv[++i]);
  }
  mkdir(art_dir, 0755);

  ec_studio_t *st = ec_studio_create(art_dir, models_dir, 2048);
  if (!st) { fprintf(stderr, "eureka-creative-serve: GPU init failed\n"); return 1; }
  ec_studio_info_t info;
  ec_studio_get_info(st, &info);

  signal(SIGINT, on_sig);
  signal(SIGTERM, on_sig);
  signal(SIGPIPE, SIG_IGN);

  if (parent_fd >= 0) {
    pthread_t pw;
    pthread_create(&pw, NULL, parent_watch_thread, &parent_fd);
    pthread_detach(pw);
  }

  int sfd = socket(AF_INET, SOCK_STREAM, 0);
  int yes = 1;
  setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes);
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof addr);
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons((uint16_t)port);
  if (bind(sfd, (struct sockaddr *)&addr, sizeof addr) != 0 || listen(sfd, 64) != 0) {
    fprintf(stderr, "eureka-creative-serve: bind/listen on 127.0.0.1:%d failed: %s\n", port, strerror(errno));
    ec_studio_destroy(st);
    return 1;
  }
  // Support --port 0 (OS-assigned random loopback port, matching supervisor.mjs!)
  socklen_t alen = sizeof addr;
  getsockname(sfd, (struct sockaddr *)&addr, &alen);
  int actual_port = ntohs(addr.sin_port);

  printf("{\"ready\":true,\"port\":%d,\"gpu\":\"%s\",\"models\":%d}\n",
         actual_port, info.gpu_name, info.models_discovered);
  fflush(stdout);

  while (!g_stop) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(sfd, &rfds);
    struct timeval tv = { 0, 150000 };
    int sel = select(sfd + 1, &rfds, NULL, NULL, &tv);
    if (sel > 0 && FD_ISSET(sfd, &rfds)) {
      int cfd = accept(sfd, NULL, NULL);
      if (cfd >= 0) {
        conn_arg_t *ca = (conn_arg_t *)malloc(sizeof *ca);
        ca->cfd = cfd; ca->st = st;
        pthread_t th;
        pthread_create(&th, NULL, client_thread, ca);
        pthread_detach(th);
      }
    }
  }

  close(sfd);
  ec_studio_destroy(st);
  return 0;
}
