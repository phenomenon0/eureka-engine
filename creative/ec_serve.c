// ec_serve.c — 100% Native Zero-Dependency POSIX HTTP/1.1 Eureka Creative Studio Server
// Matches llb-serve's minimal C11 server architecture. Exposes OpenAI-compatible & Studio routes:
//   GET  /v1/creative/health
//   POST /v1/images/generations
//   POST /v1/3d/generations
//   POST /v1/audio/music
//   POST /v1/videos/generations
//   POST /v1/videos/realtime/burst
//   POST /v1/flows/run
//   GET  /artifacts/<filename>

#define _POSIX_C_SOURCE 200809L
#include "ec_studio.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int s) { (void)s; g_stop = 1; }

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
  char hdr[512];
  size_t blen = strlen(body);
  int hn = snprintf(hdr, sizeof hdr,
      "HTTP/1.1 %d %s\r\n"
      "Content-Type: application/json\r\n"
      "Access-Control-Allow-Origin: *\r\n"
      "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
      "Content-Length: %zu\r\n"
      "Connection: close\r\n\r\n",
      code, code == 200 ? "OK" : (code == 404 ? "Not Found" : "Bad Request"), blen);
  send_all(fd, hdr, (size_t)hn);
  send_all(fd, body, blen);
}

static void send_file(int fd, const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) { send_json(fd, 404, "{\"error\":\"artifact not found\"}"); return; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz <= 0 || sz > (64L << 20)) { fclose(f); send_json(fd, 404, "{\"error\":\"bad artifact size\"}"); return; }

  const char *mime = "application/octet-stream";
  if (strstr(path, ".png")) mime = "image/png";
  else if (strstr(path, ".glb")) mime = "model/gltf-binary";
  else if (strstr(path, ".wav")) mime = "audio/wav";
  else if (strstr(path, ".ply")) mime = "application/octet-stream";

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

// Lightweight JSON field extractors for request parameters
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

static void handle_client(int cfd, ec_studio_t *st) {
  char *req = (char *)calloc(1, 262144);
  if (!req) return;
  size_t got = 0;
  size_t hdr_end = 0, content_len = 0;
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
  if (got < 10) { free(req); return; }

  char method[16] = {0}, path[256] = {0};
  if (sscanf(req, "%15s %255s", method, path) != 2) { free(req); return; }
  const char *body = hdr_end ? (req + hdr_end) : "";

  if (!strcmp(method, "OPTIONS")) {
    const char *opt =
        "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\nConnection: close\r\n\r\n";
    send_all(cfd, opt, strlen(opt));
    free(req); return;
  }

  char resp[4096];
  ec_studio_info_t info;
  ec_studio_get_info(st, &info);
  if (!strcmp(method, "GET") && !strcmp(path, "/v1/creative/health")) {
    snprintf(resp, sizeof resp,
        "{\"status\":\"ok\",\"engine\":\"eureka-creative-native-c0\","
        "\"ownership\":\"100%% eureka engine (zero cuBLAS, zero ggml, zero Python)\","
        "\"gpu\":\"%s\",\"sm\":\"sm_%d%d\",\"vram_total_gb\":%.2f,"
        "\"vram_pool_used_mb\":%.2f,\"vram_pool_budget_mb\":%.2f,"
        "\"modalities\":[\"image\",\"3d_glb_splat\",\"music_stems\",\"video\",\"realtime_video_shm\",\"comfyui_dag\"]}",
        info.gpu_name, info.sm_major, info.sm_minor, info.vram_total_gb,
        info.vram_pool_used_mb, info.vram_pool_budget_mb);
    send_json(cfd, 200, resp);
  } else if (!strcmp(method, "GET") && !strncmp(path, "/artifacts/", 11)) {
    const char *fname = path + 11;
    // Strict path-traversal hardening
    if (!fname[0] || strstr(fname, "..") || strchr(fname, '/') || strchr(fname, '\\')) {
      send_json(cfd, 400, "{\"error\":\"invalid artifact filename\"}");
    } else {
      char fpath[512];
      snprintf(fpath, sizeof fpath, "%s/%s", info.artifact_dir, fname);
      send_file(cfd, fpath);
    }
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/images/generations")) {
    char prompt[256], tier_s[16];
    json_get_str(body, "prompt", prompt, sizeof prompt, "eureka studio art");
    json_get_str(body, "tier", tier_s, sizeof tier_s, "fast");
    uint32_t seed = (uint32_t)json_get_num(body, "seed", 42);
    int w = (int)json_get_num(body, "width", 256);
    int h = (int)json_get_num(body, "height", 256);
    int steps = (int)json_get_num(body, "steps", 8);
    ec_tier_t tier = !strcmp(tier_s, "exact") ? EC_TIER_EXACT : EC_TIER_FAST;
    ec_studio_result_t r;
    if (ec_studio_gen_image(st, prompt, seed, w, h, steps, tier, &r) == 0) {
      snprintf(resp, sizeof resp,
          "{\"ok\":true,\"artifact_url\":\"/artifacts/%s\",\"width\":%d,\"height\":%d,"
          "\"tier\":\"%s\",\"steps_run\":%d,\"steps_skipped\":%d,\"gpu_ms\":%.2f}",
          basename_safe(r.file_main), r.width, r.height, tier == EC_TIER_FAST ? "fast" : "exact",
          r.steps_run, r.steps_skipped, r.gpu_ms);
      send_json(cfd, 200, resp);
    } else {
      send_json(cfd, 500, "{\"error\":\"image generation failed\"}");
    }
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/3d/generations")) {
    char prompt[256];
    json_get_str(body, "prompt", prompt, sizeof prompt, "sculptural torus");
    uint32_t seed = (uint32_t)json_get_num(body, "seed", 77);
    int res_g = (int)json_get_num(body, "grid_res", 20);
    ec_studio_result_t r;
    if (ec_studio_gen_3d(st, prompt, seed, res_g, EC_TIER_FAST, &r) == 0) {
      snprintf(resp, sizeof resp,
          "{\"ok\":true,\"glb_url\":\"/artifacts/%s\",\"splat_ply_url\":\"/artifacts/%s\","
          "\"n_verts\":%d,\"n_tris\":%d,\"n_splats\":%d,\"gpu_ms\":%.2f}",
          basename_safe(r.file_main), basename_safe(r.file_aux),
          r.n_verts, r.n_tris, r.n_splats, r.gpu_ms);
      send_json(cfd, 200, resp);
    } else {
      send_json(cfd, 500, "{\"error\":\"3d generation failed\"}");
    }
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/audio/music")) {
    char prompt[256];
    json_get_str(body, "prompt", prompt, sizeof prompt, "ambient electronic pulse");
    uint32_t seed = (uint32_t)json_get_num(body, "seed", 101);
    int bpm = (int)json_get_num(body, "bpm", 122);
    float dur = (float)json_get_num(body, "duration_sec", 3.0);
    ec_studio_result_t r;
    if (ec_studio_gen_music(st, prompt, seed, bpm, dur, EC_TIER_FAST, &r) == 0) {
      snprintf(resp, sizeof resp,
          "{\"ok\":true,\"master_wav_url\":\"/artifacts/%s\","
          "\"stems\":{\"drums\":\"/artifacts/%s\",\"bass\":\"/artifacts/%s\","
          "\"harmony\":\"/artifacts/%s\",\"lead\":\"/artifacts/%s\"},"
          "\"bpm\":%d,\"frames\":%d,\"sample_rate\":44100,\"gpu_ms\":%.2f}",
          basename_safe(r.file_main),
          basename_safe(r.stem_files[0]), basename_safe(r.stem_files[1]),
          basename_safe(r.stem_files[2]), basename_safe(r.stem_files[3]),
          bpm, r.frames, r.gpu_ms);
      send_json(cfd, 200, resp);
    } else {
      send_json(cfd, 500, "{\"error\":\"music generation failed\"}");
    }
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/videos/generations")) {
    char prompt[256];
    json_get_str(body, "prompt", prompt, sizeof prompt, "cinematic studio motion");
    uint32_t seed = (uint32_t)json_get_num(body, "seed", 303);
    int n_frames = (int)json_get_num(body, "frames", 4);
    ec_studio_result_t r;
    if (ec_studio_gen_video(st, prompt, seed, n_frames, 256, 144, EC_TIER_FAST, &r) == 0) {
      snprintf(resp, sizeof resp,
          "{\"ok\":true,\"keyframe0_url\":\"/artifacts/%s\",\"frames\":%d,"
          "\"width\":%d,\"height\":%d,\"gpu_ms\":%.2f,\"fps\":%.1f}",
          basename_safe(r.file_main), r.frames, r.width, r.height, r.gpu_ms, r.fps);
      send_json(cfd, 200, resp);
    } else {
      send_json(cfd, 500, "{\"error\":\"video generation failed\"}");
    }
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/videos/realtime/burst")) {
    char prompt[256];
    json_get_str(body, "prompt", prompt, sizeof prompt, "live interactive canvas");
    uint32_t seed = (uint32_t)json_get_num(body, "seed", 404);
    int n_frames = (int)json_get_num(body, "frames", 30);
    ec_studio_result_t r;
    if (ec_studio_realtime_burst(st, prompt, seed, n_frames, "/eureka_rt_http_burst", &r) == 0) {
      snprintf(resp, sizeof resp,
          "{\"ok\":true,\"shm_ring\":\"%s\",\"width\":%d,\"height\":%d,"
          "\"frames\":%d,\"elapsed_ms\":%.2f,\"fps\":%.1f}",
          r.file_main, r.width, r.height, r.frames, r.gpu_ms, r.fps);
      send_json(cfd, 200, resp);
    } else {
      send_json(cfd, 500, "{\"error\":\"realtime burst failed\"}");
    }
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/flows/run")) {
    ec_studio_result_t r;
    char err[256] = {0};
    if (ec_studio_run_flow_json(st, body, &r, err, sizeof err) == 0) {
      snprintf(resp, sizeof resp,
          "{\"ok\":true,\"executed\":%d,\"cache_hits\":%d,\"vram_evictions\":%d,"
          "\"gpu_ms\":%.2f,\"image_url\":\"%s\",\"glb_url\":\"%s\",\"music_url\":\"%s\"}",
          r.dag_executed, r.dag_cache_hits, r.dag_evictions, r.gpu_ms,
          r.file_main[0] ? basename_safe(r.file_main) : "",
          r.file_aux[0]  ? basename_safe(r.file_aux)  : "",
          r.stem_files[0][0] ? basename_safe(r.stem_files[0]) : "");
      send_json(cfd, 200, resp);
    } else {
      snprintf(resp, sizeof resp, "{\"error\":\"%s\"}", err[0] ? err : "flow failed");
      send_json(cfd, 400, resp);
    }
  } else {
    send_json(cfd, 404, "{\"error\":\"unknown route\"}");
  }
  free(req);
}

int main(int argc, char **argv) {
  int port = 8099;
  const char *art_dir = "/tmp/eureka-creative-build/artifacts";
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--artifacts") && i + 1 < argc) art_dir = argv[++i];
  }
  mkdir(art_dir, 0755);

  ec_studio_t *st = ec_studio_create(art_dir, 2048);
  if (!st) {
    fprintf(stderr, "eureka-creative-serve: GPU init failed\n");
    return 1;
  }
  ec_studio_info_t info;
  ec_studio_get_info(st, &info);

  signal(SIGINT, on_sig);
  signal(SIGTERM, on_sig);
  signal(SIGPIPE, SIG_IGN);

  int sfd = socket(AF_INET, SOCK_STREAM, 0);
  int yes = 1;
  setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes);
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof addr);
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons((uint16_t)port);
  if (bind(sfd, (struct sockaddr *)&addr, sizeof addr) != 0 || listen(sfd, 16) != 0) {
    fprintf(stderr, "eureka-creative-serve: bind/listen on 127.0.0.1:%d failed: %s\n", port, strerror(errno));
    ec_studio_destroy(st);
    return 1;
  }

  printf("eureka-creative-serve listening on http://127.0.0.1:%d (GPU: %s, artifacts: %s)\n",
         port, info.gpu_name, art_dir);
  fflush(stdout);

  while (!g_stop) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(sfd, &rfds);
    struct timeval tv = { 0, 200000 };
    int sel = select(sfd + 1, &rfds, NULL, NULL, &tv);
    if (sel > 0 && FD_ISSET(sfd, &rfds)) {
      int cfd = accept(sfd, NULL, NULL);
      if (cfd >= 0) {
        handle_client(cfd, st);
        close(cfd);
      }
    }
  }

  close(sfd);
  ec_studio_destroy(st);
  return 0;
}
