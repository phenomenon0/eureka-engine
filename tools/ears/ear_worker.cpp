// ear_worker.cpp: llb-ear, the served ear (EARS-CANON). The resident child of `llb-serve --ear DIR`, the way the
// Kokoro worker is the child of --tts: one JSON line in, one JSON line out, one request at a time.
//   llb-ear worker --llm Qwen3-ASR-0.6B-Q8_0.gguf --mmproj mmproj-Qwen3-ASR-0.6B-bf16.gguf [--llm-cpu]
//     stderr: "ear: ready ..." once the tower (CUDA, aut.cu) and the decoder are loaded (each ONCE, ears_open)
//     in : {"op":"transcribe","path":WAV,"sha256":HEX?,"n":MAX_IDS?,"rows_out":PATH?}  |  {"op":"ping"}
//     out: {"ok":true,"text":..,"language":..,"raw":..,"ids":[..],"top1":["%a"..],"top2":["%a"..],"rows_sha256":..,
//           "n_windows":..,"n_audio_tokens":..,"audio_s":..,"timings":{..}}  |  {"ok":false,"status":400|500,"error":..}
//   llb-ear mel WAV GOLDEN_DIR       host mel vs the goldens' pre.wN.f32 (prints "k/n byte-equal"; rc 0 iff all)
// The engine half of docs/reference/multimodal-content.md §6 for audio: the staged file's sha256 is checked before anything is decoded.
#include "ears.h"
extern "C" {
#include "../sys1dual/mm.h"
}
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

static void reply_err(int status, const char *msg) {
  sb o = {0};
  sb_fmt(&o, "{\"ok\":false,\"status\":%d,\"error\":", status);
  json_quote(&o, msg);
  sb_addc(&o, '}');
  printf("%s\n", o.p);
  fflush(stdout);
  free(o.p);
}

static bool read_file(const char *p, std::vector<unsigned char> &b) {
  FILE *f = fopen(p, "rb");
  if (!f) return false;
  unsigned char buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + n);
  fclose(f);
  return true;
}

// input errors (the clip itself) are the client's: 400; anything else is ours: 500
static int err_status(const char *e) {
  static const char *client[] = {"ears: not a wav", "ears: need 16 kHz", "ears: unsupported wav", "ears: no data",
                                 "ears: no audio windows", "ears: cannot open", NULL};
  for (int i = 0; client[i]; i++)
    if (!strncmp(e, client[i], strlen(client[i]))) return 400;
  return 500;
}

static void transcribe(ears_ctx *c, const jv *rq) {
  const jv *p = json_get(rq, "path"), *sh = json_get(rq, "sha256"), *n = json_get(rq, "n"), *ro = json_get(rq, "rows_out");
  if (!p || p->type != J_STR) return reply_err(400, "transcribe needs a string path");
  int max_ids = n && n->type == J_NUM && n->num >= 1 ? (int)n->num : 1024;
  if (max_ids > 2048) max_ids = 2048;
  std::vector<unsigned char> bytes;
  if (!read_file(p->str, bytes)) return reply_err(400, "ear: cannot read the staged audio file");
  if (sh && sh->type == J_STR) {
    char hex[65];
    mm_sha256_hex(bytes.data(), bytes.size(), hex);
    if (strcmp(hex, sh->str)) return reply_err(400, "ear: the staged audio file does not match its sha256");
  }
  const char *mime = mm_sniff_audio(bytes.data(), bytes.size());
  if (!mime || strcmp(mime, "wav"))
    return reply_err(400, "the ear takes 16 kHz mono WAV (PCM16 or float32); convert mp3/flac/ogg to wav first");
  std::vector<int32_t> ids(max_ids);
  std::vector<float> t1(max_ids), t2(max_ids);
  std::string rows = ro && ro->type == J_STR ? ro->str : std::string("/dev/shm/llb-ear-rows-") + std::to_string(getpid()) + ".f32";
  ears_stats st;
  char err[512] = {0};
  const int k = ears_run(c, p->str, max_ids, rows.c_str(), ids.data(), t1.data(), t2.data(), max_ids, &st, err);
  std::vector<unsigned char> rb;
  const bool have_rows = read_file(rows.c_str(), rb);
  if (!(ro && ro->type == J_STR)) unlink(rows.c_str());
  if (k < 0) return reply_err(err_status(err), err);
  char rsha[65] = "";
  if (have_rows) mm_sha256_hex(rb.data(), rb.size(), rsha);
  std::vector<char> raw(64 * (size_t)k + 64);
  ears_detok(c, ids.data(), k, raw.data(), (int)raw.size());
  // Qwen3-ASR's answer is "language <L><asr_text><text>"
  std::string r = raw.data(), lang, text = r;
  const size_t at = r.find("<asr_text>");
  if (at != std::string::npos) {
    text = r.substr(at + 10);
    if (!r.compare(0, 9, "language ")) lang = r.substr(9, at - 9);
  }
  sb o = {0};
  sb_add(&o, "{\"ok\":true,\"text\":");
  json_quote(&o, text.c_str());
  sb_add(&o, ",\"language\":");
  json_quote(&o, lang.c_str());
  sb_add(&o, ",\"raw\":");
  json_quote(&o, r.c_str());
  sb_add(&o, ",\"ids\":[");
  for (int i = 0; i < k; i++) sb_fmt(&o, "%s%d", i ? "," : "", ids[i]);
  sb_add(&o, "],\"top1\":[");
  for (int i = 0; i < k; i++) sb_fmt(&o, "%s\"%a\"", i ? "," : "", t1[i]);
  sb_add(&o, "],\"top2\":[");
  for (int i = 0; i < k; i++) sb_fmt(&o, "%s\"%a\"", i ? "," : "", t2[i]);
  sb_fmt(&o, "],\"rows_sha256\":\"%s\",\"n_windows\":%d,\"n_audio_tokens\":%d,\"audio_s\":%.4f", rsha, st.n_windows,
         st.n_audio_tokens, st.audio_s);
  sb_fmt(&o, ",\"timings\":{\"mel_ms\":%.3f,\"tower_ms\":%.3f,\"tower_gpu_ms\":%.3f,\"ctx_ms\":%.3f,\"prefill_ms\":%.3f,"
             "\"decode_ms\":%.3f,\"first_token_ms\":%.3f,\"total_ms\":%.3f}}",
         st.mel_ms, st.tower_ms, st.tower_gpu_ms, st.llm_load_ms, st.prefill_ms, st.decode_ms, st.first_token_ms, st.total_ms);
  printf("%s\n", o.p);
  fflush(stdout);
  free(o.p);
}

static int usage() {
  fprintf(stderr, "usage: llb-ear worker --llm LLM.gguf --mmproj MMPROJ.gguf [--llm-cpu]\n"
                  "       llb-ear mel WAV GOLDEN_DIR\n"
                  "       llb-ear encb MMPROJ [--gemm seg|streams|cat] [--repeat R] [--max-batch B] [--out DIR] WAV...\n");
  return 2;
}

int main(int argc, char **argv) {
  if (argc >= 4 && !strcmp(argv[1], "mel")) {
    int nw = 0;
    char err[512] = {0};
    const int eq = ears_mel_check(argv[2], argv[3], &nw, err);
    if (eq < 0) return fprintf(stderr, "%s\n", err), 1;
    printf("%d/%d byte-equal\n", eq, nw);
    return eq == nw ? 0 : 1;
  }
  if (argc >= 4 && !strcmp(argv[1], "encb")) {  // (CHAIN-2-1-3 Stage 2) the encoder-batch gate
    int seg = 1, rep = 1, mb = 0, i = 3;
    const char *out = NULL;
    for (; i < argc && !strncmp(argv[i], "--", 2); i++) {
      if (!strcmp(argv[i], "--gemm") && i + 1 < argc) i++, seg = !strcmp(argv[i], "cat") ? 0 : !strcmp(argv[i], "streams") ? 2 : 1;
      else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) rep = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--max-batch") && i + 1 < argc) mb = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
      else return usage();
    }
    if (i >= argc) return usage();
    return ears_encb_gate(argv[2], argc - i, (const char *const *)argv + i, seg, rep < 1 ? 1 : rep, mb, out);
  }
  if (argc < 2 || strcmp(argv[1], "worker")) return usage();
  const char *llm = NULL, *mmproj = NULL;
  int gpu = 1;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--llm") && i + 1 < argc) llm = argv[++i];
    else if (!strcmp(argv[i], "--mmproj") && i + 1 < argc) mmproj = argv[++i];
    else if (!strcmp(argv[i], "--llm-cpu")) gpu = 0;
    else return usage();
  }
  if (!llm || !mmproj) return usage();
  ears_stats load;
  char err[512] = {0};
  ears_ctx *c = ears_open(llm, mmproj, gpu, &load, err);
  if (!c) return fprintf(stderr, "ear: load failed: %s\n", err), 1;
  fprintf(stderr, "ear: ready (%s; tower %.0f ms, decoder %s %.0f ms, rows of %d)\n", ears_version(), load.tower_load_ms,
          gpu ? "GPU" : "CPU", load.llm_load_ms, ears_proj_dim(c));
  char *line = NULL;
  size_t cap = 0;
  while (getline(&line, &cap, stdin) > 0) {
    const char *perr = NULL;
    jv *rq = json_parse(line, &perr);
    const jv *op = rq ? json_get(rq, "op") : NULL;
    if (!op || op->type != J_STR) reply_err(400, perr ? perr : "a request needs a string op");
    else if (!strcmp(op->str, "transcribe")) transcribe(c, rq);
    else if (!strcmp(op->str, "ping")) printf("{\"ok\":true,\"version\":\"%s\"}\n", ears_version()), fflush(stdout);
    else reply_err(400, "unknown op (transcribe, ping)");
    json_free(rq);
  }
  free(line);
  ears_close(c);
  return 0;
}
