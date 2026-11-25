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
                       code == 401 ? "Unauthorized" : code == 403 ? "Forbidden" :
                       code == 404 ? "Not Found" :
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
  else if (strstr(path, ".jpg") || strstr(path, ".jpeg")) mime = "image/jpeg";
  else if (strstr(path, ".webp")) mime = "image/webp";
  else if (strstr(path, ".glb")) mime = "model/gltf-binary";
  else if (strstr(path, ".wav")) mime = "audio/wav";
  else if (strstr(path, ".mp3")) mime = "audio/mpeg";
  else if (strstr(path, ".mp4")) mime = "video/mp4";

  char hdr[512];
  int hn = snprintf(hdr, sizeof hdr,
      "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nAccess-Control-Allow-Origin: *\r\n"
      "Cache-Control: no-store, max-age=0\r\n"
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

static void query_get_param(const char *q, const char *key, char *out, size_t cap, const char *def) {
  snprintf(out, cap, "%s", def ? def : "");
  if (!q || !q[0] || !key || !key[0]) return;
  size_t klen = strlen(key);
  const char *p = q;
  while (*p) {
    if (!strncmp(p, key, klen) && p[klen] == '=') {
      const char *v = p + klen + 1;
      size_t n = 0;
      while (v[n] && v[n] != '&' && n + 1 < cap) { out[n] = v[n]; n++; }
      out[n] = '\0';
      return;
    }
    const char *amp = strchr(p, '&');
    if (!amp) break;
    p = amp + 1;
  }
}

static int json_get_str_array(const char *j, const char *key, char vals[][96], int max_n) {
  char pat[64];
  snprintf(pat, sizeof pat, "\"%s\"", key);
  const char *p = strstr(j, pat);
  if (!p) return 0;
  p = strchr(p + strlen(pat), '[');
  if (!p) return 0;
  p++;
  int cnt = 0;
  while (*p && *p != ']' && cnt < max_n) {
    while (*p && *p != '"' && *p != ']') p++;
    if (*p != '"') break;
    p++;
    size_t n = 0;
    while (*p && *p != '"' && n + 1 < 96) vals[cnt][n++] = *p++;
    vals[cnt][n] = '\0';
    if (*p == '"') p++;
    if (n > 0) cnt++;
  }
  return cnt;
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
  char          prompt[1024];
  char          neg_prompt[1024];
  char          lyrics[1536];
  char          tier[16];
  char          stage[32];
  char          checkpoint[256];
  ec_studio_lora_t loras[EC_STUDIO_MAX_LORAS];
  int           n_loras;
  int           photoreal;
  int           full_model;
  uint32_t      seed;
  int           width, height, steps, frames, bpm, with_audio, grid_res, fps;
  float         duration_sec, cfg_scale, shift, denoise, repaint_start, repaint_end, rescale_t;
  // v8 Multimodal Inputs
  char          init_image[256];
  char          mask_image[256];
  int           outpaint_pad[4];
  int           tile_upscale;
  char          start_frame[256];
  char          end_frame[256];
  char          camera_motion[32];
  int           hero_keyframe_i2v;
  char          input_images[4][256];
  int           n_input_images;
  int           isolate_fg;
  float         base_color[4];
  char          source_audio[256];
  char          audio_mode[24];
  char          key_scale[32];
  char          time_sig[16];
  char          vocal_mode[24];
  int           stem_mode;
  job_state_t   state;
  _Atomic int   cancel_req;
  int           cur_step, total_steps, skipped;
  int           event_seq;
  char          node_id[32];
  char          node_class[64];
  char          node_status[24];
  double        node_ms;
  char          node_artifact[256];
  int           flow_cache_hits;
  int           flow_executed;
  char         *flow_json;
  char          result_json[32768];
  ec_studio_t  *st;
} ec_job_t;

static void format_stems_json(const ec_studio_result_t *r, char *out, size_t cap) {
  int n = r->n_stems;
  if (n <= 0 && r->stem_files[0][0]) {
    // Legacy 4-stem smoke fallback
    static const char *L4[4] = { "drums", "bass", "harmony", "lead" };
    snprintf(out, cap,
        "{\"%s\":\"/artifacts/%s\",\"%s\":\"/artifacts/%s\",\"%s\":\"/artifacts/%s\",\"%s\":\"/artifacts/%s\"}",
        L4[0], basename_safe(r->stem_files[0]), L4[1], basename_safe(r->stem_files[1]),
        L4[2], basename_safe(r->stem_files[2]), L4[3], basename_safe(r->stem_files[3]));
    return;
  }
  int p = snprintf(out, cap, "{");
  int emitted = 0;
  for (int i = 0; i < n && i < 8 && (size_t)p + 300 < cap; i++) {
    if (!r->stem_files[i][0]) continue;
    const char *nm = r->stem_names[i][0] ? r->stem_names[i] : "stem";
    p += snprintf(out + p, cap - (size_t)p, "%s\"%s\":\"/artifacts/%s\"",
                  emitted ? "," : "", nm, basename_safe(r->stem_files[i]));
    emitted++;
  }
  snprintf(out + p, cap - (size_t)p, "}");
}

static int json_get_path_array(const char *j, const char *key, char vals[][256], int max_n) {
  char pat[64];
  snprintf(pat, sizeof pat, "\"%s\"", key);
  const char *p = strstr(j, pat);
  if (!p) return 0;
  p = strchr(p + strlen(pat), '[');
  if (!p) return 0;
  p++;
  int cnt = 0;
  while (*p && *p != ']' && cnt < max_n) {
    while (*p && *p != '"' && *p != ']') p++;
    if (*p != '"') break;
    p++;
    size_t n = 0;
    while (*p && *p != '"' && n + 1 < 256) vals[cnt][n++] = *p++;
    vals[cnt][n] = '\0';
    if (*p == '"') p++;
    if (n > 0) cnt++;
  }
  return cnt;
}

static int parse_stem_mode(const char *j, int def) {
  char s[32] = {0};
  json_get_str(j, "stem_mode", s, sizeof s, "");
  if (s[0]) {
    if (!strcmp(s, "full") || !strcmp(s, "full_mix") || !strcmp(s, "0")) return 0;
    if (!strcmp(s, "2") || !strcmp(s, "2stem")) return 2;
    if (!strcmp(s, "4") || !strcmp(s, "4stem")) return 4;
    if (!strcmp(s, "8") || !strcmp(s, "8stem")) return 8;
  }
  double n = json_get_num(j, "stem_mode", (double)def);
  int iv = (int)n;
  return (iv == 0 || iv == 2 || iv == 4 || iv == 8) ? iv : def;
}

// Parse JSON array "loras": [ { "path": "...", "strength": 0.8 }, ... ]
static int json_get_loras(const char *j, ec_studio_lora_t *out, int max_n) {
  const char *p = strstr(j, "\"loras\"");
  if (!p) return 0;
  p = strchr(p, '[');
  if (!p) return 0;
  p++;
  int cnt = 0;
  while (*p && *p != ']' && cnt < max_n) {
    const char *obj = strchr(p, '{');
    const char *arr_end = strchr(p, ']');
    if (!obj || (arr_end && obj > arr_end)) break;
    const char *end = strchr(obj, '}');
    if (!end) break;
    char tmp[640];
    size_t len = (size_t)(end - obj + 1);
    if (len >= sizeof tmp) len = sizeof tmp - 1;
    memcpy(tmp, obj, len);
    tmp[len] = '\0';
    memset(&out[cnt], 0, sizeof out[cnt]);
    json_get_str(tmp, "path", out[cnt].path, sizeof out[cnt].path, "");
    if (!out[cnt].path[0]) json_get_str(tmp, "id", out[cnt].path, sizeof out[cnt].path, "");
    if (!out[cnt].path[0]) json_get_str(tmp, "name", out[cnt].path, sizeof out[cnt].path, "");
    out[cnt].strength = (float)json_get_num(tmp, "strength", 0.85);
    if (out[cnt].path[0]) cnt++;
    p = end + 1;
  }
  return cnt;
}

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
  j->event_seq++;
  if (stage) snprintf(j->stage, sizeof j->stage, "%s", stage);
  pthread_cond_broadcast(&g_jobs_cv);
  pthread_mutex_unlock(&g_jobs_mu);
  return atomic_load(&j->cancel_req) ? 1 : 0;
}

static int job_flow_event_cb(
    const char *node_id, const char *class_type, const char *status,
    int step, int total_steps, int cache_hits, int executed,
    double node_ms, const char *artifact_url, void *ctx) {
  ec_job_t *j = (ec_job_t *)ctx;
  pthread_mutex_lock(&g_jobs_mu);
  j->cur_step = step;
  j->total_steps = total_steps > 0 ? total_steps : 1;
  j->flow_cache_hits = cache_hits;
  j->flow_executed = executed;
  j->node_ms = node_ms;
  snprintf(j->node_id, sizeof j->node_id, "%s", node_id ? node_id : "");
  snprintf(j->node_class, sizeof j->node_class, "%s", class_type ? class_type : "");
  snprintf(j->node_status, sizeof j->node_status, "%s", status ? status : "");
  snprintf(j->node_artifact, sizeof j->node_artifact, "%s", artifact_url ? artifact_url : "");
  snprintf(j->stage, sizeof j->stage, "%.14s:%.15s", class_type ? class_type : "flow", status ? status : "run");
  j->event_seq++;
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
  char out[4096] = {0};

  if (!strcmp(j->kind, "image")) {
    ec_studio_image_v8_t iq = {};
    iq.prompt = j->prompt; iq.neg_prompt = j->neg_prompt[0] ? j->neg_prompt : NULL;
    iq.seed = j->seed; iq.width = j->width; iq.height = j->height; iq.steps = j->steps;
    iq.cfg_scale = j->cfg_scale; iq.tier = tier; iq.photoreal = j->photoreal;
    iq.checkpoint_query = j->checkpoint[0] ? j->checkpoint : NULL;
    iq.loras = j->n_loras > 0 ? j->loras : NULL; iq.n_loras = j->n_loras;
    iq.init_image_path = j->init_image[0] ? j->init_image : NULL;
    iq.mask_image_path = j->mask_image[0] ? j->mask_image : NULL;
    iq.denoise = j->denoise; iq.tile_upscale = j->tile_upscale;
    memcpy(iq.outpaint_pad, j->outpaint_pad, sizeof iq.outpaint_pad);
    rc = ec_studio_gen_image_v8(j->st, &iq, job_progress_cb, j, &r);
    if (rc == 0 && !r.cancelled)
      snprintf(out, sizeof out,
          "{\"artifact_url\":\"/artifacts/%s\",\"width\":%d,\"height\":%d,"
          "\"engine_mode\":\"%s\",\"steps_run\":%d,\"steps_skipped\":%d,"
          "\"lora_tensors_fused\":%d,\"text_enc_ms\":%.1f,\"dit_ms\":%.1f,\"vae_ms\":%.1f,\"gpu_ms\":%.2f}",
          basename_safe(r.file_main), r.width, r.height,
          r.engine_mode[0] ? r.engine_mode : "dit_smoke",
          r.steps_run, r.steps_skipped, r.lora_tensors_fused,
          r.text_enc_ms, r.dit_ms, r.vae_ms, r.gpu_ms);
  } else if (!strcmp(j->kind, "3d")) {
    ec_studio_3d_v8_t tq = {};
    tq.prompt = j->prompt; tq.seed = j->seed; tq.grid_res = j->grid_res;
    tq.steps = j->steps > 0 ? j->steps : 12; tq.cfg_scale = j->cfg_scale; tq.rescale_t = j->rescale_t;
    tq.tier = tier; tq.full_model = j->full_model; tq.isolate_fg = j->isolate_fg;
    memcpy(tq.base_color, j->base_color, sizeof tq.base_color);
    for (int i = 0; i < j->n_input_images && i < 4; i++) tq.input_images[i] = j->input_images[i];
    tq.n_input_images = j->n_input_images;
    tq.loras = j->n_loras > 0 ? j->loras : NULL;
    tq.n_loras = j->n_loras;
    rc = ec_studio_gen_3d_v8(j->st, &tq, job_progress_cb, j, &r);
    if (rc == 0 && !r.cancelled)
      snprintf(out, sizeof out,
          "{\"ok\":true,\"glb_url\":\"/artifacts/%s\",\"splat_ply_url\":\"/artifacts/%s\","
          "\"engine_mode\":\"%s\",\"n_verts\":%d,\"n_tris\":%d,\"n_splats\":%d,\"lora_tensors_fused\":%d,"
          "\"text_enc_ms\":%.1f,\"dit_ms\":%.1f,\"vae_ms\":%.1f,\"gpu_ms\":%.2f}",
          basename_safe(r.file_main), basename_safe(r.file_aux),
          r.engine_mode[0] ? r.engine_mode : "dit_smoke",
          r.n_verts, r.n_tris, r.n_splats, r.lora_tensors_fused,
          r.text_enc_ms, r.dit_ms, r.vae_ms, r.gpu_ms);
  } else if (!strcmp(j->kind, "music")) {
    ec_studio_music_v8_t mq = {};
    mq.prompt = j->prompt; mq.lyrics = j->lyrics; mq.seed = j->seed;
    mq.bpm = j->bpm; mq.duration_sec = j->duration_sec; mq.steps = j->steps > 0 ? j->steps : 8;
    mq.shift = j->shift; mq.cfg_scale = j->cfg_scale; mq.stem_mode = j->stem_mode; mq.tier = tier; mq.full_model = j->full_model;
    mq.checkpoint_query = j->checkpoint[0] ? j->checkpoint : NULL;
    mq.key_scale = j->key_scale[0] ? j->key_scale : NULL;
    mq.time_sig = j->time_sig[0] ? j->time_sig : NULL;
    mq.vocal_mode = j->vocal_mode[0] ? j->vocal_mode : NULL;
    mq.audio_mode = j->audio_mode[0] ? j->audio_mode : NULL;
    mq.source_audio_path = j->source_audio[0] ? j->source_audio : NULL;
    mq.denoise = j->denoise; mq.repaint_start_sec = j->repaint_start; mq.repaint_end_sec = j->repaint_end;
    mq.loras = j->n_loras > 0 ? j->loras : NULL; mq.n_loras = j->n_loras;
    rc = ec_studio_gen_music_v8(j->st, &mq, job_progress_cb, j, &r);
    if (rc == 0 && !r.cancelled) {
      char stems_j[2048];
      format_stems_json(&r, stems_j, sizeof stems_j);
      snprintf(out, sizeof out,
          "{\"ok\":true,\"master_wav_url\":\"/artifacts/%s\",\"stems\":%s,\"stem_count\":%d,"
          "\"engine_mode\":\"%s\",\"bpm\":%d,\"frames\":%d,\"sample_rate\":%d,"
          "\"lora_tensors_fused\":%d,\"text_enc_ms\":%.1f,\"dit_ms\":%.1f,\"vae_ms\":%.1f,\"gpu_ms\":%.2f}",
          basename_safe(r.file_main), stems_j, r.n_stems,
          r.engine_mode[0] ? r.engine_mode : "dit_smoke",
          j->bpm, r.frames, r.sample_rate ? r.sample_rate : 44100, r.lora_tensors_fused,
          r.text_enc_ms, r.dit_ms, r.vae_ms, r.gpu_ms);
    }
  } else if (!strcmp(j->kind, "video") || !strcmp(j->kind, "video_av")) {
    int wa = (!strcmp(j->kind, "video_av") || j->with_audio) ? 1 : 0;
    ec_studio_video_v8_t vq = {};
    vq.prompt = j->prompt; vq.neg_prompt = j->neg_prompt[0] ? j->neg_prompt : NULL;
    vq.seed = j->seed; vq.n_frames = j->frames; vq.width = j->width; vq.height = j->height;
    vq.fps = j->fps; vq.steps = j->steps; vq.with_audio = wa; vq.tier = tier; vq.full_model = j->full_model;
    vq.cfg_scale = j->cfg_scale; vq.shift = j->shift; vq.denoise = j->denoise;
    vq.start_frame_path = j->start_frame[0] ? j->start_frame : NULL;
    vq.end_frame_path = j->end_frame[0] ? j->end_frame : NULL;
    vq.camera_motion = j->camera_motion[0] ? j->camera_motion : NULL;
    vq.hero_keyframe_i2v = j->hero_keyframe_i2v;
    vq.checkpoint_query = j->checkpoint[0] ? j->checkpoint : NULL;
    vq.loras = j->n_loras > 0 ? j->loras : NULL; vq.n_loras = j->n_loras;
    rc = ec_studio_gen_video_v8(j->st, &vq, job_progress_cb, j, &r);
    if (rc == 0 && !r.cancelled)
      snprintf(out, sizeof out,
          "{\"ok\":true,\"mp4_url\":\"/artifacts/%s\",\"keyframe0_url\":\"/artifacts/%s\",\"last_frame_url\":\"/artifacts/%s\","
          "\"engine_mode\":\"%s\",\"encoder\":\"%s\",\"width\":%d,\"height\":%d,"
          "\"frames\":%d,\"with_audio\":%s,\"lora_tensors_fused\":%d,\"text_enc_ms\":%.1f,\"dit_ms\":%.1f,\"vae_ms\":%.1f,\"gpu_ms\":%.2f,\"fps\":%.1f}",
          basename_safe(r.file_mp4), basename_safe(r.file_aux),
          basename_safe(r.file_last[0] ? r.file_last : r.file_aux),
          r.engine_mode[0] ? r.engine_mode : "dit_smoke", r.encoder_used,
          r.width, r.height, r.frames, wa ? "true" : "false", r.lora_tensors_fused,
          r.text_enc_ms, r.dit_ms, r.vae_ms, r.gpu_ms, r.fps);
  } else if (!strcmp(j->kind, "realtime")) {
    rc = ec_studio_realtime_burst(j->st, j->prompt, j->seed, j->frames, "/eureka_rt_job_burst", job_progress_cb, j, &r);
    if (rc == 0 && !r.cancelled)
      snprintf(out, sizeof out, "{\"shm_ring\":\"%s\",\"frames\":%d,\"fps\":%.1f,\"elapsed_ms\":%.2f}",
               r.file_main, r.frames, r.fps, r.gpu_ms);
  } else if (!strcmp(j->kind, "flow") && j->flow_json) {
    char *nouts = (char *)malloc(24576);
    char ferr[256] = {0};
    rc = ec_studio_run_flow_ex(j->st, j->flow_json, job_flow_event_cb, j, &r, nouts, 24576, ferr, sizeof ferr);
    free(j->flow_json);
    j->flow_json = NULL;
    if (rc == 0 && !r.cancelled) {
      pthread_mutex_lock(&g_jobs_mu);
      j->state = JOB_DONE;
      j->cur_step = j->total_steps > 0 ? j->total_steps : 1;
      j->flow_cache_hits = r.dag_cache_hits;
      j->flow_executed = r.dag_executed;
      j->event_seq++;
      snprintf(j->result_json, sizeof j->result_json,
          "{\"ok\":true,\"executed\":%d,\"cache_hits\":%d,\"vram_evictions\":%d,\"gpu_ms\":%.2f,"
          "\"image_url\":\"%s%s\",\"glb_url\":\"%s%s\",\"music_url\":\"%s%s\",\"mp4_url\":\"%s%s\","
          "\"node_outputs\":%s}",
          r.dag_executed, r.dag_cache_hits, r.dag_evictions, r.gpu_ms,
          r.file_main[0] ? "/artifacts/" : "", r.file_main[0] ? basename_safe(r.file_main) : "",
          r.file_aux[0]  ? "/artifacts/" : "", r.file_aux[0]  ? basename_safe(r.file_aux)  : "",
          r.stem_files[0][0] ? "/artifacts/" : "", r.stem_files[0][0] ? basename_safe(r.stem_files[0]) : "",
          r.file_mp4[0]  ? "/artifacts/" : "", r.file_mp4[0]  ? basename_safe(r.file_mp4)  : "",
          (nouts && nouts[0] == '{') ? nouts : "{}");
      pthread_cond_broadcast(&g_jobs_cv);
      pthread_mutex_unlock(&g_jobs_mu);
      free(nouts);
      return NULL;
    }
    free(nouts);
  }

  pthread_mutex_lock(&g_jobs_mu);
  if (rc == 0 && r.cancelled) {
    j->state = JOB_CANCELLED;
    j->event_seq++;
    snprintf(j->result_json, sizeof j->result_json, "{\"cancelled\":true}");
  } else if (rc == 0) {
    j->state = JOB_DONE;
    j->cur_step = j->total_steps > 0 ? j->total_steps : 1;
    j->event_seq++;
    snprintf(j->result_json, sizeof j->result_json, "%s", out[0] ? out : "{}");
  } else {
    j->state = JOB_FAILED;
    j->event_seq++;
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
      "\"step\":%d,\"total_steps\":%d,\"skipped\":%d,\"pct\":%d,"
      "\"node_id\":\"%s\",\"class_type\":\"%s\",\"node_status\":\"%s\",\"node_ms\":%.1f,"
      "\"artifact_url\":\"%s\",\"cache_hits\":%d,\"executed\":%d,\"result\":%s}",
      j->id, j->kind, job_state_str(j->state), j->stage,
      j->cur_step, j->total_steps, j->skipped, pct,
      j->node_id, j->node_class, j->node_status, j->node_ms,
      j->node_artifact, j->flow_cache_hits, j->flow_executed,
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

  int last_seq = -1;
  job_state_t last_state = (job_state_t)-1;
  char *payload = (char *)malloc(36864);
  char *frame   = (char *)malloc(37888);
  if (!payload || !frame) { free(payload); free(frame); return; }

  while (!g_stop) {
    pthread_mutex_lock(&g_jobs_mu);
    ec_job_t *j = find_job_locked(job_id);
    if (!j) {
      pthread_mutex_unlock(&g_jobs_mu);
      send_all(cfd, "event: error\ndata: {\"error\":\"job not found\"}\n\n", 45);
      free(payload); free(frame);
      return;
    }
    while (j->event_seq == last_seq && j->state == last_state &&
           (j->state == JOB_QUEUED || j->state == JOB_RUNNING) && !g_stop) {
      struct timespec ts;
      clock_gettime(CLOCK_REALTIME, &ts);
      ts.tv_nsec += 120000000L;
      if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
      pthread_cond_timedwait(&g_jobs_cv, &g_jobs_mu, &ts);
    }
    last_seq = j->event_seq;
    last_state = j->state;
    format_job_json_locked(j, payload, 36864);
    int done = (j->state == JOB_DONE || j->state == JOB_CANCELLED || j->state == JOB_FAILED);
    const char *ev = done ? (j->state == JOB_DONE ? "done" : job_state_str(j->state)) : "progress";
    pthread_mutex_unlock(&g_jobs_mu);

    int fn = snprintf(frame, 37888, "event: %s\ndata: %s\n\n", ev, payload);
    send_all(cfd, frame, (size_t)fn);
    if (done) break;
  }
  free(payload);
  free(frame);
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

  const size_t REQ_CAP = 32 * 1024 * 1024; // 32 MiB (supports WAV/MP3 audio + high-res image & ComfyUI PNG uploads)
  char *req = (char *)calloc(1, REQ_CAP);
  if (!req) { close(cfd); return NULL; }
  size_t got = 0, hdr_end = 0, content_len = 0;
  while (got < REQ_CAP - 1) {
    ssize_t r = read(cfd, req + got, REQ_CAP - 1 - got);
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
  char query[256] = {0};
  char *qmark = strchr(path, '?');
  if (qmark) { snprintf(query, sizeof query, "%s", qmark + 1); *qmark = '\0'; }
  const char *body = hdr_end ? (req + hdr_end) : "";
  size_t body_len = (hdr_end && got >= hdr_end) ? (got - hdr_end) : 0;

  if (!strcmp(method, "OPTIONS")) {
    const char *opt =
        "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization, X-Api-Key\r\nConnection: close\r\n\r\n";
    send_all(cfd, opt, strlen(opt));
    free(req); close(cfd); return NULL;
  }

  if (!check_auth(req, hdr_end ? hdr_end : got)) {
    send_json(cfd, 401, "{\"error\":\"unauthorized\"}");
    free(req); close(cfd); return NULL;
  }

  const size_t RESP_CAP = 262144;
  char *resp = (char *)malloc(RESP_CAP);
  ec_studio_info_t info;
  ec_studio_get_info(st, &info);

  if (!strcmp(method, "GET") && (!strcmp(path, "/v1/creative/health") || !strcmp(path, "/v1/creative/status"))) {
    snprintf(resp, RESP_CAP,
        "{\"status\":\"ok\",\"engine\":\"eureka-creative-native-v17\","
        "\"ownership\":\"100%% eureka engine (zero cuBLAS, zero ggml, zero Python)\","
        "\"parallel\":true,\"auth_enabled\":%s,\"photoreal_ready\":%s,\"zimage_ready\":%s,"
        "\"acestep_ready\":%s,\"trellis2_ready\":%s,\"wan22_ready\":%s,\"ltx2_ready\":%s,"
        "\"cowrie_enabled\":%s,\"cowrie_shani\":%s,"
        "\"cowrie_sp_hits\":%llu,\"cowrie_sp_saves\":%llu,"
        "\"cowrie_sp_mismatches\":%llu,\"cowrie_trust_entries\":%d,"
        "\"gpu\":\"%s\",\"sm\":\"sm_%d%d\",\"vram_total_gb\":%.2f,"
        "\"vram_free_mb\":%.1f,\"engines_resident_mask\":%d,"
        "\"vram_pool_used_mb\":%.2f,\"vram_pool_budget_mb\":%.2f,"
        "\"models_discovered\":%d,\"storage_roots\":%d,\"loras_discovered\":%d,"
        "\"checkpoints_discovered\":%d,\"recipes_count\":%d,"
        "\"modalities\":[\"z_image_turbo_6b\",\"flux2_photoreal\",\"flux2_base_4b\",\"flux2_img2img\",\"flux2_inpaint_outpaint\",\"flux2_tile_upscale_2x\",\"acestep_1.5_xl_4b\",\"acestep_1.5_sft\",\"acestep_1.5_music\",\"acestep_8stem_repaint_extend\",\"studio_master_lufs\",\"trellis2_3d\",\"trellis2_512_ovoxel_pbr\",\"trellis2_color0_pbr\",\"trellis2_photo_multiview\",\"ltx2.3_22b_joint_av\",\"wan2.2_video\",\"wan2.2_hero_i2v\",\"wan2.2_i2v_flf2v\",\"qlora_nf4_fp8_dora_loha_lokr\",\"lora_delta_hotswap_8ms\",\"vae_arena_slab\",\"vram_lease_broker\",\"image\",\"3d_glb_splat\",\"music_stems\",\"video_nvenc_mp4\",\"video_av_ltx\",\"realtime_video_shm\",\"comfyui_dag\",\"comfyui_import\",\"lora_stack\",\"network_storage\",\"prompt_recipes\",\"async_jobs_sse\",\"cowrie_v2_shani\",\"cowrie_statepages\",\"cowrie_trust_membrane\",\"cowrie_lenses\"]}",
        g_api_key[0] ? "true" : "false",
        info.photoreal_ready ? "true" : "false",
        info.zimage_ready    ? "true" : "false",
        info.acestep_ready   ? "true" : "false",
        info.trellis2_ready  ? "true" : "false",
        info.wan22_ready     ? "true" : "false",
        info.ltx2_ready      ? "true" : "false",
        info.cowrie_enabled  ? "true" : "false",
        info.cowrie_shani    ? "true" : "false",
        (unsigned long long)info.cowrie_sp_hits,
        (unsigned long long)info.cowrie_sp_saves,
        (unsigned long long)info.cowrie_sp_mismatches,
        info.cowrie_trust_entries,
        info.gpu_name, info.sm_major, info.sm_minor, info.vram_total_gb,
        info.vram_free_mb, info.engines_resident_mask,
        info.vram_pool_used_mb, info.vram_pool_budget_mb,
        info.models_discovered, info.storage_roots, info.loras_discovered,
        info.checkpoints_discovered, info.recipes_count);
    send_json(cfd, 200, resp);
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/creative/unload")) {
    double unload_ms = 0.0, free_mb = 0.0;
    int rc = ec_studio_unload_all_vram(st, &unload_ms, &free_mb);
    snprintf(resp, 4096,
        "{\"ok\":%s,\"unloaded_ms\":%.2f,\"vram_free_mb\":%.1f,\"vram_total_mb\":%.1f,\"engines_resident_mask\":0}",
        rc == 0 ? "true" : "false", unload_ms, free_mb, info.vram_total_gb * 1024.0);
    send_json(cfd, rc == 0 ? 200 : 500, resp);
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
    json_get_str(body, "neg_prompt", j->neg_prompt, sizeof j->neg_prompt, "");
    json_get_str(body, "lyrics", j->lyrics, sizeof j->lyrics, "");
    json_get_str(body, "tier", j->tier, sizeof j->tier, "fast");
    json_get_str(body, "checkpoint", j->checkpoint, sizeof j->checkpoint, "");
    json_get_str(body, "init_image", j->init_image, sizeof j->init_image, "");
    json_get_str(body, "mask_image", j->mask_image, sizeof j->mask_image, "");
    json_get_str(body, "start_frame", j->start_frame, sizeof j->start_frame, "");
    json_get_str(body, "end_frame", j->end_frame, sizeof j->end_frame, "");
    json_get_str(body, "camera_motion", j->camera_motion, sizeof j->camera_motion, "auto");
    json_get_str(body, "source_audio", j->source_audio, sizeof j->source_audio, "");
    json_get_str(body, "audio_mode", j->audio_mode, sizeof j->audio_mode, "");
    json_get_str(body, "key_scale", j->key_scale, sizeof j->key_scale, "");
    json_get_str(body, "time_sig", j->time_sig, sizeof j->time_sig, "4");
    json_get_str(body, "vocal_mode", j->vocal_mode, sizeof j->vocal_mode, "auto");
    j->n_input_images = json_get_path_array(body, "input_images", j->input_images, 4);
    if (j->n_input_images == 0) {
      char single_img[256] = {0};
      json_get_str(body, "input_image", single_img, sizeof single_img, "");
      if (single_img[0]) { snprintf(j->input_images[0], sizeof j->input_images[0], "%s", single_img); j->n_input_images = 1; }
    }
    j->n_loras = json_get_loras(body, j->loras, EC_STUDIO_MAX_LORAS);
    j->seed = (uint32_t)json_get_num(body, "seed", 42 + j->id);
    int is_vid = (!strcmp(j->kind, "video") || !strcmp(j->kind, "video_av"));
    int req_full = json_get_bool(body, "full_model", -1);
    int def_w = is_vid ? (req_full == 1 ? 832 : 320) : 512;
    int def_h = is_vid ? (req_full == 1 ? 480 : 180) : 512;
    j->width = (int)json_get_num(body, "width", def_w);
    j->height = (int)json_get_num(body, "height", def_h);
    j->grid_res = (int)json_get_num(body, "grid_res", 20);
    j->bpm = (int)json_get_num(body, "bpm", 124);
    j->duration_sec = (float)json_get_num(body, "duration_sec", req_full == 1 ? 10.0 : 2.0);
    j->cfg_scale = (float)json_get_num(body, "cfg_scale", 0.0);
    j->shift = (float)json_get_num(body, "shift", 0.0);
    j->denoise = (float)json_get_num(body, "denoise", 0.0);
    j->repaint_start = (float)json_get_num(body, "repaint_start", 0.0);
    j->repaint_end = (float)json_get_num(body, "repaint_end", 0.0);
    j->rescale_t = (float)json_get_num(body, "rescale_t", 0.0);
    j->tile_upscale = json_get_bool(body, "tile_upscale", 0);
    j->hero_keyframe_i2v = json_get_bool(body, "hero_keyframe_i2v", 0);
    j->isolate_fg = json_get_bool(body, "isolate_fg", 1);
    j->outpaint_pad[0] = (int)json_get_num(body, "outpaint_left", 0);
    j->outpaint_pad[1] = (int)json_get_num(body, "outpaint_top", 0);
    j->outpaint_pad[2] = (int)json_get_num(body, "outpaint_right", 0);
    j->outpaint_pad[3] = (int)json_get_num(body, "outpaint_bottom", 0);
    j->stem_mode = parse_stem_mode(body, 4);
    j->photoreal = json_get_bool(body, "photoreal", (j->width >= 512 || j->height >= 512 || j->checkpoint[0] || j->n_loras > 0 || j->init_image[0]) ? 1 : 0);
    if (req_full >= 0) {
      j->full_model = req_full;
    } else if (!strcmp(j->kind, "3d")) {
      j->full_model = (j->grid_res >= 32 || j->n_input_images > 0) ? 1 : 0;
    } else if (!strcmp(j->kind, "music")) {
      j->full_model = (j->duration_sec >= 5.0f || j->lyrics[0] || j->source_audio[0]) ? 1 : 0;
    } else if (is_vid) {
      j->full_model = (j->width >= 640 || j->height >= 384 || j->start_frame[0] || j->end_frame[0]) ? 1 : 0;
    } else {
      j->full_model = j->photoreal;
    }
    int def_steps = is_vid ? (j->full_model ? 20 : 4) : (!strcmp(j->kind, "3d") ? 12 : (!strcmp(j->kind, "music") ? 8 : (j->photoreal ? 4 : 12)));
    j->steps = (int)json_get_num(body, "steps", def_steps);
    j->frames = (int)json_get_num(body, "frames", (is_vid && j->full_model) ? 17 : 24);
    j->fps = (int)json_get_num(body, "fps", (is_vid && j->full_model) ? 16 : 24);
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
    char prompt[1024], neg_prompt[1024], tier_s[16], ckpt[256], init_img[256], mask_img[256];
    ec_studio_lora_t loras[EC_STUDIO_MAX_LORAS];
    json_get_str(body, "prompt", prompt, sizeof prompt, "A cinematic 35mm photograph, natural light, ultra detailed");
    json_get_str(body, "neg_prompt", neg_prompt, sizeof neg_prompt, "");
    json_get_str(body, "tier", tier_s, sizeof tier_s, "fast");
    json_get_str(body, "checkpoint", ckpt, sizeof ckpt, "");
    json_get_str(body, "init_image", init_img, sizeof init_img, "");
    json_get_str(body, "mask_image", mask_img, sizeof mask_img, "");
    int n_loras = json_get_loras(body, loras, EC_STUDIO_MAX_LORAS);
    uint32_t seed = (uint32_t)json_get_num(body, "seed", 42);
    int w = (int)json_get_num(body, "width", 512), h = (int)json_get_num(body, "height", 512);
    int photoreal = json_get_bool(body, "photoreal", (w >= 512 || h >= 512 || ckpt[0] || n_loras > 0 || init_img[0]) ? 1 : 0);
    int steps = (int)json_get_num(body, "steps", photoreal ? 4 : 8);
    ec_tier_t tier = !strcmp(tier_s, "exact") ? EC_TIER_EXACT : EC_TIER_FAST;
    ec_studio_image_v8_t iq = {0};
    iq.prompt = prompt;
    iq.neg_prompt = neg_prompt[0] ? neg_prompt : NULL;
    iq.seed = seed; iq.width = w; iq.height = h; iq.steps = steps;
    iq.cfg_scale = (float)json_get_num(body, "cfg_scale", 1.0);
    iq.tier = tier; iq.photoreal = photoreal;
    iq.checkpoint_query = ckpt[0] ? ckpt : NULL;
    iq.loras = n_loras > 0 ? loras : NULL; iq.n_loras = n_loras;
    iq.init_image_path = init_img[0] ? init_img : NULL;
    iq.mask_image_path = mask_img[0] ? mask_img : NULL;
    iq.denoise = (float)json_get_num(body, "denoise", init_img[0] ? 0.48 : 1.0);
    iq.outpaint_pad[0] = (int)json_get_num(body, "outpaint_left", 0);
    iq.outpaint_pad[1] = (int)json_get_num(body, "outpaint_top", 0);
    iq.outpaint_pad[2] = (int)json_get_num(body, "outpaint_right", 0);
    iq.outpaint_pad[3] = (int)json_get_num(body, "outpaint_bottom", 0);
    iq.tile_upscale = json_get_bool(body, "tile_upscale", 0);
    ec_studio_result_t r;
    if (ec_studio_gen_image_v8(st, &iq, NULL, NULL, &r) == 0) {
      snprintf(resp, RESP_CAP,
          "{\"ok\":true,\"artifact_url\":\"/artifacts/%s\",\"width\":%d,\"height\":%d,"
          "\"engine_mode\":\"%s\",\"checkpoint\":\"%s\",\"lora_tensors_fused\":%d,"
          "\"tier\":\"%s\",\"steps_run\":%d,\"steps_skipped\":%d,"
          "\"text_enc_ms\":%.1f,\"dit_ms\":%.1f,\"vae_ms\":%.1f,\"gpu_ms\":%.2f}",
          basename_safe(r.file_main), r.width, r.height,
          r.engine_mode[0] ? r.engine_mode : "dit_smoke",
          basename_safe(r.checkpoint_used), r.lora_tensors_fused,
          tier == EC_TIER_FAST ? "fast" : "exact",
          r.steps_run, r.steps_skipped, r.text_enc_ms, r.dit_ms, r.vae_ms, r.gpu_ms);
      send_json(cfd, 200, resp);
    } else send_json(cfd, 500, "{\"error\":\"image generation failed\"}");
  } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/creative/storage")) {
    ec_studio_storage_json(st, resp, RESP_CAP);
    send_json(cfd, 200, resp);
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/creative/storage/roots")) {
    char rpath[512], rlabel[64];
    json_get_str(body, "path", rpath, sizeof rpath, "");
    json_get_str(body, "label", rlabel, sizeof rlabel, "Network / Custom Drive");
    if (rpath[0] && ec_studio_storage_add_root(st, rpath, rlabel) == 0) {
      ec_studio_storage_json(st, resp, RESP_CAP);
      send_json(cfd, 200, resp);
    } else send_json(cfd, 400, "{\"error\":\"invalid storage root path\"}");
  } else if (!strcmp(method, "DELETE") && !strcmp(path, "/v1/creative/storage/roots")) {
    char rpath[512];
    json_get_str(body, "path", rpath, sizeof rpath, "");
    if (rpath[0] && ec_studio_storage_remove_root(st, rpath) == 0) {
      ec_studio_storage_json(st, resp, RESP_CAP);
      send_json(cfd, 200, resp);
    } else send_json(cfd, 404, "{\"error\":\"storage root not found\"}");
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/creative/storage/rescan")) {
    ec_studio_storage_rescan(st);
    ec_studio_storage_json(st, resp, RESP_CAP);
    send_json(cfd, 200, resp);
  } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/creative/loras")) {
    ec_studio_loras_json(st, resp, RESP_CAP);
    send_json(cfd, 200, resp);
  } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/creative/lora-catalog")) {
    ec_studio_lora_catalog_json(st, resp, RESP_CAP);
    send_json(cfd, 200, resp);
  } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/creative/checkpoints")) {
    ec_studio_checkpoints_json(st, resp, RESP_CAP);
    send_json(cfd, 200, resp);
  } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/creative/recipes")) {
    ec_studio_recipes_json(st, resp, RESP_CAP);
    send_json(cfd, 200, resp);
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/creative/recipes")) {
    char id[64], name[96], cat[48], mod[24], prompt[768], cam[128], ckpt[128], lora[128];
    json_get_str(body, "id", id, sizeof id, "");
    json_get_str(body, "name", name, sizeof name, "");
    json_get_str(body, "category", cat, sizeof cat, "Custom");
    json_get_str(body, "modality", mod, sizeof mod, "image");
    json_get_str(body, "prompt", prompt, sizeof prompt, "");
    json_get_str(body, "camera", cam, sizeof cam, "");
    json_get_str(body, "checkpoint", ckpt, sizeof ckpt, "flux2_klein_4b_bf16");
    json_get_str(body, "lora", lora, sizeof lora, "");
    float lstr = (float)json_get_num(body, "lora_strength", 0.8);
    int w = (int)json_get_num(body, "width", 512), h = (int)json_get_num(body, "height", 512);
    int steps = (int)json_get_num(body, "steps", 4);
    uint32_t seed = (uint32_t)json_get_num(body, "seed", 42);
    if (!id[0] && name[0]) snprintf(id, sizeof id, "user-%u", (unsigned)(time(NULL) & 0xFFFFFu));
    if (ec_studio_recipe_save(st, id, name, cat, mod, prompt, cam, ckpt, lora, lstr, w, h, steps, seed) == 0) {
      ec_studio_recipes_json(st, resp, RESP_CAP);
      send_json(cfd, 200, resp);
    } else send_json(cfd, 400, "{\"error\":\"invalid recipe or cannot overwrite builtin\"}");
  } else if (!strcmp(method, "DELETE") && !strncmp(path, "/v1/creative/recipes/", 21)) {
    const char *rid = path + 21;
    if (ec_studio_recipe_delete(st, rid) == 0) send_json(cfd, 200, "{\"ok\":true}");
    else send_json(cfd, 404, "{\"error\":\"user recipe not found\"}");
  } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/creative/prompts")) {
    ec_studio_prompts_json(resp, RESP_CAP);
    send_json(cfd, 200, resp);
  } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/creative/lens")) {
    char target[32], tier_s[16];
    query_get_param(query, "target", target, sizeof target, "vault");
    query_get_param(query, "tier",   tier_s, sizeof tier_s, "r2");
    int tier = (!strcmp(tier_s, "r0") || !strcmp(tier_s, "0")) ? 0 :
               (!strcmp(tier_s, "r1") || !strcmp(tier_s, "1")) ? 1 : 2;
    if (ec_studio_cowrie_lens_json(st, target, tier, resp, RESP_CAP) == 0) {
      send_json(cfd, 200, resp);
    } else {
      send_json(cfd, 400, resp);
    }
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/creative/trust/gate")) {
    char action[96];
    char arg_vals[32][96];
    const char *arg_ptrs[32];
    json_get_str(body, "action", action, sizeof action, "unknown");
    int n_args = json_get_str_array(body, "arg_ids", arg_vals, 32);
    for (int i = 0; i < n_args; i++) arg_ptrs[i] = arg_vals[i];
    int ov = json_get_bool(body, "owner_override", 0);
    int http_code = ec_studio_cowrie_trust_gate_json(st, action, arg_ptrs, n_args, ov, resp, RESP_CAP);
    send_json(cfd, http_code, resp);
  } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/creative/trust/triage")) {
    ec_studio_cowrie_trust_triage_json(st, resp, RESP_CAP);
    send_json(cfd, 200, resp);
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/flows/import")) {
    int is_png = (body_len >= 8 && (uint8_t)body[0] == 137 && body[1] == 'P' && body[2] == 'N' && body[3] == 'G') ? 1 : 0;
    int exec_now = (strstr(query, "execute=1") || strstr(query, "execute=true")) ? 1 : 0;
    char source_tag[48];
    query_get_param(query, "source", source_tag, sizeof source_tag, "external");
    if (ec_studio_import_comfy_ex(st, (const uint8_t *)body, body_len, is_png, exec_now, source_tag, resp, RESP_CAP) == 0) {
      send_json(cfd, 200, resp);
    } else {
      send_json(cfd, 400, resp);
    }
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/3d/generations")) {
    char prompt[1024], in_imgs[4][256], single_img[256];
    ec_studio_lora_t loras[EC_STUDIO_MAX_LORAS];
    json_get_str(body, "prompt", prompt, sizeof prompt, "sculptural matte-white porcelain teapot");
    json_get_str(body, "input_image", single_img, sizeof single_img, "");
    int n_imgs = json_get_path_array(body, "input_images", in_imgs, 4);
    int n_loras = json_get_loras(body, loras, EC_STUDIO_MAX_LORAS);
    if (n_imgs == 0 && single_img[0]) { snprintf(in_imgs[0], sizeof in_imgs[0], "%s", single_img); n_imgs = 1; }
    uint32_t seed = (uint32_t)json_get_num(body, "seed", 77);
    int res_g = (int)json_get_num(body, "grid_res", 20);
    int full_m = json_get_bool(body, "full_model", (res_g >= 32 || n_imgs > 0 || n_loras > 0) ? 1 : 0);
    ec_studio_3d_v8_t tq = {0};
    tq.prompt = prompt; tq.seed = seed; tq.grid_res = res_g;
    tq.steps = (int)json_get_num(body, "steps", 12);
    tq.cfg_scale = (float)json_get_num(body, "cfg_scale", 7.5);
    tq.rescale_t = (float)json_get_num(body, "rescale_t", 5.0);
    tq.isolate_fg = json_get_bool(body, "isolate_fg", 1);
    tq.tier = EC_TIER_FAST; tq.full_model = full_m;
    for (int i = 0; i < n_imgs && i < 4; i++) tq.input_images[i] = in_imgs[i];
    tq.n_input_images = n_imgs;
    tq.loras = n_loras > 0 ? loras : NULL;
    tq.n_loras = n_loras;
    ec_studio_result_t r;
    if (ec_studio_gen_3d_v8(st, &tq, NULL, NULL, &r) == 0) {
      snprintf(resp, RESP_CAP,
          "{\"ok\":true,\"glb_url\":\"/artifacts/%s\",\"splat_ply_url\":\"/artifacts/%s\","
          "\"engine_mode\":\"%s\",\"n_verts\":%d,\"n_tris\":%d,\"n_splats\":%d,\"lora_tensors_fused\":%d,"
          "\"text_enc_ms\":%.1f,\"dit_ms\":%.1f,\"vae_ms\":%.1f,\"gpu_ms\":%.2f}",
          basename_safe(r.file_main), basename_safe(r.file_aux),
          r.engine_mode[0] ? r.engine_mode : "dit_smoke",
          r.n_verts, r.n_tris, r.n_splats, r.lora_tensors_fused,
          r.text_enc_ms, r.dit_ms, r.vae_ms, r.gpu_ms);
      send_json(cfd, 200, resp);
    } else send_json(cfd, 500, "{\"error\":\"3d generation failed\"}");
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/audio/music")) {
    char prompt[1024], lyrics[1536], ckpt[256], key_sc[32], tsig[16], vmode[24], amode[24], src_aud[256];
    ec_studio_lora_t loras[EC_STUDIO_MAX_LORAS];
    json_get_str(body, "prompt", prompt, sizeof prompt, "ambient electronic pulse");
    json_get_str(body, "lyrics", lyrics, sizeof lyrics, "");
    json_get_str(body, "checkpoint", ckpt, sizeof ckpt, "");
    json_get_str(body, "key_scale", key_sc, sizeof key_sc, "");
    json_get_str(body, "time_sig", tsig, sizeof tsig, "4");
    json_get_str(body, "vocal_mode", vmode, sizeof vmode, "auto");
    json_get_str(body, "audio_mode", amode, sizeof amode, "generate");
    json_get_str(body, "source_audio", src_aud, sizeof src_aud, "");
    int n_loras = json_get_loras(body, loras, EC_STUDIO_MAX_LORAS);
    uint32_t seed = (uint32_t)json_get_num(body, "seed", 101);
    int bpm = (int)json_get_num(body, "bpm", 122);
    float dur = (float)json_get_num(body, "duration_sec", 3.0);
    int full_m = json_get_bool(body, "full_model", (dur >= 5.0f || lyrics[0] || src_aud[0] || ckpt[0]) ? 1 : 0);
    ec_studio_music_v8_t mq = {0};
    mq.prompt = prompt; mq.lyrics = lyrics; mq.seed = seed; mq.bpm = bpm;
    mq.duration_sec = dur; mq.steps = (int)json_get_num(body, "steps", 0);
    mq.shift = (float)json_get_num(body, "shift", 3.0);
    mq.cfg_scale = (float)json_get_num(body, "cfg_scale", 0.0);
    mq.checkpoint_query = ckpt[0] ? ckpt : NULL;
    mq.stem_mode = parse_stem_mode(body, 4);
    mq.key_scale = key_sc[0] ? key_sc : NULL;
    mq.time_sig = tsig[0] ? tsig : NULL;
    mq.vocal_mode = vmode[0] ? vmode : NULL;
    mq.audio_mode = amode[0] ? amode : NULL;
    mq.source_audio_path = src_aud[0] ? src_aud : NULL;
    mq.denoise = (float)json_get_num(body, "denoise", 0.65);
    mq.repaint_start_sec = (float)json_get_num(body, "repaint_start", 0.0);
    mq.repaint_end_sec = (float)json_get_num(body, "repaint_end", 0.0);
    mq.loras = n_loras > 0 ? loras : NULL; mq.n_loras = n_loras;
    mq.tier = EC_TIER_FAST; mq.full_model = full_m;
    ec_studio_result_t r;
    if (ec_studio_gen_music_v8(st, &mq, NULL, NULL, &r) == 0) {
      char stems_j[2048];
      format_stems_json(&r, stems_j, sizeof stems_j);
      snprintf(resp, RESP_CAP,
          "{\"ok\":true,\"master_wav_url\":\"/artifacts/%s\",\"stems\":%s,\"stem_count\":%d,"
          "\"engine_mode\":\"%s\",\"bpm\":%d,\"frames\":%d,\"sample_rate\":%d,"
          "\"lora_tensors_fused\":%d,\"text_enc_ms\":%.1f,\"dit_ms\":%.1f,\"vae_ms\":%.1f,\"gpu_ms\":%.2f}",
          basename_safe(r.file_main), stems_j, r.n_stems,
          r.engine_mode[0] ? r.engine_mode : "dit_smoke",
          bpm, r.frames, r.sample_rate ? r.sample_rate : 44100, r.lora_tensors_fused,
          r.text_enc_ms, r.dit_ms, r.vae_ms, r.gpu_ms);
      send_json(cfd, 200, resp);
    } else send_json(cfd, 500, "{\"error\":\"music generation failed\"}");
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/videos/generations")) {
    char prompt[1024], neg_prompt[1024], start_fr[256], end_fr[256], cam[32], ckpt[256];
    ec_studio_lora_t loras[EC_STUDIO_MAX_LORAS];
    json_get_str(body, "prompt", prompt, sizeof prompt, "cinematic studio motion");
    json_get_str(body, "neg_prompt", neg_prompt, sizeof neg_prompt, "");
    json_get_str(body, "start_frame", start_fr, sizeof start_fr, "");
    json_get_str(body, "end_frame", end_fr, sizeof end_fr, "");
    json_get_str(body, "camera_motion", cam, sizeof cam, "");
    json_get_str(body, "checkpoint", ckpt, sizeof ckpt, "");
    if (!ckpt[0]) json_get_str(body, "model", ckpt, sizeof ckpt, "");
    int n_loras = json_get_loras(body, loras, EC_STUDIO_MAX_LORAS);
    uint32_t seed = (uint32_t)json_get_num(body, "seed", 303);
    int req_full = json_get_bool(body, "full_model", -1);
    int w = (int)json_get_num(body, "width", (req_full == 1 || start_fr[0]) ? 832 : 320);
    int h = (int)json_get_num(body, "height", (req_full == 1 || start_fr[0]) ? 480 : 180);
    int full_m = (req_full >= 0) ? req_full : ((w >= 640 || h >= 384 || start_fr[0]) ? 1 : 0);
    int n_frames = (int)json_get_num(body, "frames", full_m ? 17 : 24);
    int fps = (int)json_get_num(body, "fps", full_m ? 16 : 24);
    int steps = (int)json_get_num(body, "steps", full_m ? 20 : 4);
    int with_audio = json_get_bool(body, "with_audio", 1);
    ec_studio_video_v8_t vq = {0};
    vq.prompt = prompt; vq.neg_prompt = neg_prompt[0] ? neg_prompt : NULL;
    vq.seed = seed; vq.n_frames = n_frames; vq.width = w; vq.height = h;
    vq.fps = fps; vq.steps = steps; vq.with_audio = with_audio;
    vq.cfg_scale = (float)json_get_num(body, "cfg_scale", 4.0);
    vq.shift = (float)json_get_num(body, "shift", 0.0);
    vq.denoise = (float)json_get_num(body, "denoise", 1.0);
    vq.start_frame_path = start_fr[0] ? start_fr : NULL;
    vq.end_frame_path = end_fr[0] ? end_fr : NULL;
    vq.camera_motion = cam[0] ? cam : NULL;
    vq.hero_keyframe_i2v = json_get_bool(body, "hero_keyframe_i2v", 0);
    vq.checkpoint_query = ckpt[0] ? ckpt : NULL;
    vq.loras = n_loras > 0 ? loras : NULL; vq.n_loras = n_loras;
    vq.tier = EC_TIER_FAST; vq.full_model = full_m;
    ec_studio_result_t r;
    if (ec_studio_gen_video_v8(st, &vq, NULL, NULL, &r) == 0) {
      snprintf(resp, RESP_CAP,
          "{\"ok\":true,\"mp4_url\":\"/artifacts/%s\",\"keyframe0_url\":\"/artifacts/%s\",\"last_frame_url\":\"/artifacts/%s\","
          "\"engine_mode\":\"%s\",\"encoder\":\"%s\",\"with_audio\":%s,\"frames\":%d,\"width\":%d,\"height\":%d,"
          "\"lora_tensors_fused\":%d,\"text_enc_ms\":%.1f,\"dit_ms\":%.1f,\"vae_ms\":%.1f,\"gpu_ms\":%.2f,\"fps\":%.1f}",
          basename_safe(r.file_mp4), basename_safe(r.file_aux),
          basename_safe(r.file_last[0] ? r.file_last : r.file_aux),
          r.engine_mode[0] ? r.engine_mode : "dit_smoke", r.encoder_used,
          with_audio ? "true" : "false", r.frames, r.width, r.height, r.lora_tensors_fused,
          r.text_enc_ms, r.dit_ms, r.vae_ms, r.gpu_ms, r.fps);
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
  } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/flows/object_info")) {
    ec_studio_flow_object_info_json(st, resp, RESP_CAP);
    send_json(cfd, 200, resp);
  } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/flows/templates")) {
    ec_studio_flow_templates_json(st, resp, RESP_CAP);
    send_json(cfd, 200, resp);
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/flows/clear-cache")) {
    ec_studio_flow_clear_cache(st);
    send_json(cfd, 200, "{\"ok\":true,\"cache_cleared\":true}");
  } else if (!strcmp(method, "POST") && (!strcmp(path, "/v1/flows/upload") || !strcmp(path, "/v1/creative/upload"))) {
    const char *ext = "png";
    if (body_len >= 12 && !memcmp(body, "RIFF", 4) && !memcmp(body + 8, "WAVE", 4)) ext = "wav";
    else if (body_len >= 4 && !memcmp(body, "fLaC", 4)) ext = "flac";
    else if (body_len >= 4 && !memcmp(body, "OggS", 4)) ext = "ogg";
    else if (body_len >= 3 && (!memcmp(body, "ID3", 3) || ((uint8_t)body[0] == 0xFF && ((uint8_t)body[1] & 0xE0) == 0xE0))) ext = "mp3";
    else if (body_len >= 12 && !memcmp(body, "RIFF", 4) && !memcmp(body + 8, "WEBP", 4)) ext = "webp";
    else if (body_len >= 3 && (uint8_t)body[0] == 0xFF && (uint8_t)body[1] == 0xD8) ext = "jpg";
    char fname[96] = {0};
    if (ec_studio_upload_media(st, (const uint8_t *)body, body_len, ext, fname, sizeof fname) == 0) {
      snprintf(resp, 4096, "{\"ok\":true,\"name\":\"%s\",\"ext\":\"%s\",\"url\":\"/artifacts/%s\"}", fname, ext, fname);
      send_json(cfd, 200, resp);
    } else {
      send_json(cfd, 400, "{\"ok\":false,\"error\":\"invalid media upload\"}");
    }
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/flows/queue")) {
    pthread_mutex_lock(&g_jobs_mu);
    int slot = (g_next_job_id - 1) % EC_MAX_JOBS;
    ec_job_t *j = &g_jobs[slot];
    free(j->flow_json);
    memset(j, 0, sizeof *j);
    j->id = g_next_job_id++;
    j->st = st;
    j->state = JOB_QUEUED;
    atomic_init(&j->cancel_req, 0);
    snprintf(j->kind, sizeof j->kind, "flow");
    snprintf(j->stage, sizeof j->stage, "queued");
    j->flow_json = (char *)malloc(body_len + 1);
    if (j->flow_json) { memcpy(j->flow_json, body, body_len); j->flow_json[body_len] = '\0'; }
    int jid = j->id;
    pthread_mutex_unlock(&g_jobs_mu);

    pthread_t th;
    pthread_create(&th, NULL, job_runner_thread, j);
    pthread_detach(th);

    snprintf(resp, 4096,
        "{\"ok\":true,\"job_id\":%d,\"status_url\":\"/v1/jobs/%d\","
        "\"events_url\":\"/v1/jobs/%d/events\",\"cancel_url\":\"/v1/jobs/%d/cancel\"}",
        jid, jid, jid, jid);
    send_json(cfd, 202, resp);
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/flows/run")) {
    ec_studio_result_t r; char err[256] = {0};
    char *nouts = (char *)malloc(24576);
    if (ec_studio_run_flow_ex(st, body, NULL, NULL, &r, nouts, 24576, err, sizeof err) == 0) {
      snprintf(resp, RESP_CAP,
          "{\"ok\":true,\"executed\":%d,\"cache_hits\":%d,\"vram_evictions\":%d,"
          "\"gpu_ms\":%.2f,\"image_url\":\"%s\",\"glb_url\":\"%s\",\"music_url\":\"%s\",\"mp4_url\":\"%s\","
          "\"node_outputs\":%s}",
          r.dag_executed, r.dag_cache_hits, r.dag_evictions, r.gpu_ms,
          r.file_main[0] ? basename_safe(r.file_main) : "",
          r.file_aux[0]  ? basename_safe(r.file_aux)  : "",
          r.stem_files[0][0] ? basename_safe(r.stem_files[0]) : "",
          r.file_mp4[0] ? basename_safe(r.file_mp4) : "",
          (nouts && nouts[0] == '{') ? nouts : "{}");
      send_json(cfd, 200, resp);
    } else {
      snprintf(resp, 4096, "{\"error\":\"%s\"}", err[0] ? err : "flow failed");
      send_json(cfd, 400, resp);
    }
    free(nouts);
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
