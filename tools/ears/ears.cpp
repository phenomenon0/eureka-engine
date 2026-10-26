// ears.cpp: see ears.h. The LLM side is EARS-G0 agold.cpp's step 3 (prompt, chunking, greedy loop) with the embd
// override always on and the rows coming from our tower instead of a file. ios-ears2: no mtmd. EARS-CANON: the resident
// form (ears_open/ears_run: tower + LLM loaded once, a fresh llama context per request) for the served ear; the
// one-shot ears_transcribe is open + run + close. The chunk layout
// mtmd_tokenize builds for qwen3a (text | <|audio_start|> | one audio chunk per mel window | <|audio_end|> | text) is
// rebuilt here from the LLM vocab and the tower's per-window token counts, so the mmproj is loaded once (the tower).
#include "ears.h"
#include "aud_mel.h"
#include "aut.h"
#include "llama.h"
#include "llama-cpp.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// = mtmd_tokenize_text_internal (add_special false, parse_special true)
static std::vector<llama_token> tokenize(const llama_vocab *voc, const std::string &txt) {
  std::vector<llama_token> r(txt.size());
  int n = llama_tokenize(voc, txt.data(), (int32_t)txt.size(), r.data(), (int32_t)r.size(), false, true);
  if (n < 0) r.resize(-n), n = llama_tokenize(voc, txt.data(), (int32_t)txt.size(), r.data(), (int32_t)r.size(), false, true);
  r.resize(n < 0 ? 0 : n);
  return r;
}

// one prompt chunk: text tokens, or n audio rows (one mel window) taken from the tower output
struct Chunk {
  bool audio;
  std::vector<llama_token> toks;
  int n;
};

// = mtmd_tokenizer::tokenize for one audio marker: parts tokenized separately, consecutive text merged, BOS/EOS per vocab
static std::vector<Chunk> qwen3a_chunks(const llama_vocab *voc, const std::string &pre, const std::string &post,
                                        const std::vector<int> &win_tokens) {
  std::vector<Chunk> c;
  auto add_text = [&](const std::vector<llama_token> &t) {
    if (t.empty()) return;
    if (!c.empty() && !c.back().audio) c.back().toks.insert(c.back().toks.end(), t.begin(), t.end());
    else c.push_back({false, t, 0});
  };
  add_text(tokenize(voc, pre));
  add_text(tokenize(voc, "<|audio_start|>"));
  for (int n : win_tokens) c.push_back({true, {}, n});
  add_text(tokenize(voc, "<|audio_end|>"));
  add_text(tokenize(voc, post));
  if (llama_vocab_get_add_bos(voc)) {
    if (!c.empty() && !c[0].audio) c[0].toks.insert(c[0].toks.begin(), llama_vocab_bos(voc));
    else c.insert(c.begin(), Chunk{false, {llama_vocab_bos(voc)}, 0});
  }
  if (llama_vocab_get_add_eos(voc)) add_text({llama_vocab_eos(voc)});
  return c;
}

// = mtmd_helper_eval_chunk_single (text) / mtmd_helper_decode_image_chunk (audio, causal, normal or 1-d M-RoPE positions)
static int eval_chunk(llama_context *lctx, const Chunk &c, const float *embd, int n_embd, bool mrope, int n_batch,
                      bool logits_last, llama_pos *n_past) {
  llama_batch_ext_ptr b(llama_batch_ext_init(lctx));
  const int n = c.audio ? c.n : (int)c.toks.size();
  for (int i0 = 0; i0 < n; i0 += n_batch) {
    const int nb = std::min(n_batch, n - i0);
    llama_batch_ext_clear(b.get());
    int32_t idx = -1;
    for (int i = i0; i < i0 + nb; i++) {
      llama_pos p[GGML_MROPE_SECTIONS] = {0, 0, 0, 0};
      if (c.audio) {
        idx = llama_batch_ext_add_embd(b.get(), 0, llama_embd{embd + (size_t)i * n_embd, 1, (size_t)n_embd});
        for (int j = 0; j < (mrope ? 4 : 1); j++) p[j] = *n_past + i;
      } else {
        idx = llama_batch_ext_add_token(b.get(), 0, c.toks[i]);
        p[0] = *n_past + i;
      }
      if (idx < 0) return -1;
      llama_batch_ext_set_pos(b.get(), idx, p);
    }
    if (!c.audio && logits_last && i0 + nb == n) llama_batch_ext_set_output_logits(b.get(), idx, true);
    if (int r = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, b.get())) return r;
  }
  *n_past += n;
  return 0;
}

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static bool read_wav(const char *path, std::vector<float> &out, char *err) {  // = cudaud_cli read_wav (16 kHz mono)
  FILE *f = fopen(path, "rb");
  if (!f) { snprintf(err, 512, "ears: cannot open %s", path); return false; }
  std::vector<unsigned char> b;
  unsigned char buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + n);
  fclose(f);
  if (b.size() < 12 || memcmp(b.data(), "RIFF", 4) || memcmp(b.data() + 8, "WAVE", 4)) { snprintf(err, 512, "ears: not a wav"); return false; }
  int fmt = 0, ch = 0, sr = 0, bits = 0;
  for (size_t p = 12; p + 8 <= b.size();) {
    uint32_t sz;
    memcpy(&sz, b.data() + p + 4, 4);
    const unsigned char *d = b.data() + p + 8;
    if (!memcmp(b.data() + p, "fmt ", 4)) {
      uint16_t t, c, bp; uint32_t s;
      memcpy(&t, d, 2), memcpy(&c, d + 2, 2), memcpy(&s, d + 4, 4), memcpy(&bp, d + 14, 2);
      fmt = t, ch = c, sr = (int)s, bits = bp;
      if (fmt == 0xFFFE && sz >= 26) { uint16_t sub; memcpy(&sub, d + 24, 2); fmt = sub; }
    } else if (!memcmp(b.data() + p, "data", 4)) {
      if (ch != 1 || sr != 16000) { snprintf(err, 512, "ears: need 16 kHz mono (got %d ch, %d Hz)", ch, sr); return false; }
      if (fmt == 3 && bits == 32) { out.resize(sz / 4); memcpy(out.data(), d, out.size() * 4); return true; }
      if (fmt == 1 && bits == 16) {
        out.resize(sz / 2);
        for (size_t i = 0; i < out.size(); i++) { int16_t s; memcpy(&s, d + 2 * i, 2); out[i] = s / 32768.0f; }
        return true;
      }
      snprintf(err, 512, "ears: unsupported wav format %d/%d bits", fmt, bits);
      return false;
    }
    p += 8 + sz + (sz & 1);
  }
  snprintf(err, 512, "ears: no data chunk");
  return false;
}

extern "C" const char *ears_version(void) { return "ears-canon aut + slim chunker + llama.cpp 2ca15f5"; }

extern "C" int ears_mel_check(const char *wav, const char *golden_dir, int *n_windows, char *err) {
  std::vector<float> pcm;
  if (!read_wav(wav, pcm, err)) return -1;
  std::vector<AudMelWindow> wins;
  if (aud_mel_qwen3a(pcm.data(), pcm.size(), AudMelParams(), wins)) { snprintf(err, 512, "ears: mel failed"); return -1; }
  *n_windows = (int)wins.size();
  int eq = 0;
  for (size_t w = 0; w < wins.size(); w++) {
    std::string p = std::string(golden_dir) + "/pre.w" + std::to_string(w) + ".f32";
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) continue;
    std::vector<float> g(wins[w].data.size() + 1);
    const size_t got = fread(g.data(), 4, g.size(), f);
    fclose(f);
    if (got == wins[w].data.size() && !memcmp(g.data(), wins[w].data.data(), got * 4)) eq++;
  }
  return eq;
}

struct ears_ctx {
  AutW *tw;
  llama_model *model;
  int PD, llm_gpu;
  bool mrope;
};

extern "C" ears_ctx *ears_open(const char *llm, const char *mmproj, int llm_gpu, ears_stats *load, char *err) {
  ears_stats dummy;
  if (!load) load = &dummy;
  memset(load, 0, sizeof *load);
  double t = now_ms();
  char aerr[256] = {0};
  AutW *tw = aut_load(mmproj, aerr);
  if (!tw) { snprintf(err, 512, "%s", aerr); return nullptr; }
  load->tower_load_ms = now_ms() - t;
  t = now_ms();
  llama_backend_init();
  ggml_backend_load_all();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = llm_gpu ? 99 : 0;
  llama_model *model = llama_model_load_from_file(llm, mp);
  if (!model) { snprintf(err, 512, "ears: LLM load failed (%s)", llm); aut_free(tw); return nullptr; }
  load->llm_load_ms = now_ms() - t;
  const llama_rope_type rt = llama_model_rope_type(model);  // = mtmd_context's pos_type
  const bool mrope = rt == LLAMA_ROPE_TYPE_MROPE || rt == LLAMA_ROPE_TYPE_IMROPE;
  const int PD = aut_proj_dim(tw);
  if (!mrope && rt != LLAMA_ROPE_TYPE_NONE && rt != LLAMA_ROPE_TYPE_NORM && rt != LLAMA_ROPE_TYPE_NEOX)
    snprintf(err, 512, "ears: unsupported decoder rope type %d", (int)rt);
  else if (llama_model_n_embd_inp(model) != PD)
    snprintf(err, 512, "ears: LLM wants rows of %d, tower gave %d", llama_model_n_embd_inp(model), PD);
  else return new ears_ctx{tw, model, PD, llm_gpu, mrope};
  llama_model_free(model);
  aut_free(tw);
  return nullptr;
}

extern "C" void ears_close(ears_ctx *c) {
  if (!c) return;
  llama_model_free(c->model);
  aut_free(c->tw);
  delete c;
}

extern "C" int ears_proj_dim(const ears_ctx *c) { return c->PD; }

extern "C" int ears_detok(ears_ctx *c, const int32_t *ids, int n, char *out, int cap) {  // = agold's transcript.txt
  const llama_vocab *voc = llama_model_get_vocab(c->model);
  int k = 0;
  for (int i = 0; i < n && !llama_vocab_is_eog(voc, ids[i]); i++) {
    char piece[256];
    int np = llama_token_to_piece(voc, ids[i], piece, sizeof piece - 1, 0, true);
    if (np <= 0) continue;
    if (k + np >= cap) break;
    memcpy(out + k, piece, (size_t)np), k += np;
  }
  if (cap > 0) out[k < cap ? k : cap - 1] = 0;
  return k;
}

// (CHAIN-2-1-3 Stage 2) LLB_EAR_BATCH (default off): a request's mel windows go through the tower in one batched pass
// (aut_encode_batch), rows byte-identical to the per-window loop. "1" or "seg": GEMMs per window at the single shapes;
// "streams": those GEMMs on side streams. Returns the gemm_seg mode, 0 = off.
static int ears_batch_on() {
  const char *e = getenv("LLB_EAR_BATCH");
  if (!e) return 0;
  if (!strcmp(e, "1") || !strcmp(e, "seg")) return 1;
  if (!strcmp(e, "streams")) return 2;
  return 0;
}

// llb-ear encb: every window of every WAV through the tower alone, then all windows (or batches of max_batch) in one
// pass; per window BYTE-IDENTICAL or DIFFER, plus serial vs batched tower time (median of rep) and RTF. 0 = all identical.
extern "C" int ears_encb_gate(const char *mmproj, int nwav, const char *const *wavs, int gemm_seg, int rep, int max_batch,
                              const char *outdir) {
  char err[512] = {0}, aerr[256] = {0};
  AutW *tw = aut_load(mmproj, aerr);
  if (!tw) return fprintf(stderr, "encb: %s\n", aerr), 2;
  const int PD = aut_proj_dim(tw);
  std::vector<AudMelWindow> wins;
  std::vector<std::string> tag;
  double audio_s = 0;
  for (int k = 0; k < nwav; k++) {
    std::vector<float> pcm;
    std::vector<AudMelWindow> w;
    if (!read_wav(wavs[k], pcm, err)) return fprintf(stderr, "encb: %s\n", err), 2;
    if (aud_mel_qwen3a(pcm.data(), pcm.size(), AudMelParams(), w)) return fprintf(stderr, "encb: mel failed %s\n", wavs[k]), 2;
    audio_s += pcm.size() / 16000.0;
    for (size_t j = 0; j < w.size(); j++) wins.push_back(w[j]), tag.push_back(std::string(wavs[k]) + "#" + std::to_string(j));
  }
  const int n = (int)wins.size();
  if (max_batch <= 0 || max_batch > n) max_batch = n;
  std::vector<std::vector<float>> es(n), eb(n);
  std::vector<const float *> mp(n);
  std::vector<int> nl(n);
  std::vector<float *> ep(n);
  for (int i = 0; i < n; i++) {
    es[i].resize((size_t)aut_n_tokens(wins[i].n_len) * PD), eb[i].resize(es[i].size());
    mp[i] = wins[i].data.data(), nl[i] = wins[i].n_len, ep[i] = eb[i].data();
  }
  std::vector<double> ts, tb;
  int same = 0;
  size_t fr = 0, tot = 0;
  for (int r = 0; r < rep; r++) {
    double a = now_ms();
    for (int i = 0; i < n; i++)
      if (aut_encode(tw, mp[i], 128, nl[i], nullptr, es[i].data(), aerr)) return fprintf(stderr, "encb: %s\n", aerr), 2;
    ts.push_back(now_ms() - a);
    a = now_ms();
    for (int k0 = 0; k0 < n; k0 += max_batch) {
      const int m = std::min(max_batch, n - k0);
      if (aut_encode_batch(tw, m, mp.data() + k0, nl.data() + k0, gemm_seg, ep.data() + k0, aerr)) return fprintf(stderr, "encb: %s\n", aerr), 2;
    }
    tb.push_back(now_ms() - a);
    if (r == 0) {
      aut_mem_info(&fr, &tot);
      for (int i = 0; i < n; i++) {
        const bool eq = !memcmp(es[i].data(), eb[i].data(), es[i].size() * 4);
        same += eq;
        if (outdir) {
          char p[1024];
          FILE *f;
          snprintf(p, sizeof p, "%s/single-%d.f32", outdir, i);
          if ((f = fopen(p, "wb"))) fwrite(es[i].data(), 4, es[i].size(), f), fclose(f);
          snprintf(p, sizeof p, "%s/batch-%d.f32", outdir, i);
          if ((f = fopen(p, "wb"))) fwrite(eb[i].data(), 4, eb[i].size(), f), fclose(f);
        }
        printf("win\t%d\t%s\tn_len %d\t%d tokens\t%s\n", i, tag[i].c_str(), nl[i], aut_n_tokens(nl[i]), eq ? "BYTE-IDENTICAL" : "DIFFER");
      }
    }
  }
  std::sort(ts.begin(), ts.end()), std::sort(tb.begin(), tb.end());
  const double ms = ts[(rep - 1) / 2], mb = tb[(rep - 1) / 2];
  printf("verdict\t%d/%d BYTE-IDENTICAL\tgemm %s\tmax-batch %d\n", same, n, gemm_seg == 2 ? "streams" : gemm_seg ? "seg" : "cat", max_batch);
  printf("time\tserial %.2f ms (tower RTF %.5f)\tbatch %.2f ms (tower RTF %.5f)\tspeedup %.3fx\t(median of %d, %.2f s audio)\n",
         ms, ms / 1e3 / audio_s, mb, mb / 1e3 / audio_s, ms / mb, rep, audio_s);
  printf("mem\tweights %.1f MiB\tdevice free %.1f MiB of %.1f (after the first batch)\n", aut_weight_bytes(tw) / 1048576.0,
         fr / 1048576.0, tot / 1048576.0);
  aut_free(tw);
  return same == n ? 0 : 3;
}

extern "C" int ears_run(ears_ctx *c, const char *wav, int n_gen, const char *embd_out, int32_t *ids, float *top1, float *top2,
                        int max_ids, ears_stats *st, char *err) {
  memset(st, 0, sizeof *st);
  const double t0 = now_ms();
  std::vector<float> pcm;
  if (!read_wav(wav, pcm, err)) return -1;
  st->audio_s = pcm.size() / 16000.0;
  // 1. host mel + tower (resident)
  double t = now_ms();
  std::vector<AudMelWindow> wins;
  if (aud_mel_qwen3a(pcm.data(), pcm.size(), AudMelParams(), wins)) { snprintf(err, 512, "ears: mel failed"); return -1; }
  st->mel_ms = now_ms() - t, st->n_windows = (int)wins.size();
  if (wins.empty()) { snprintf(err, 512, "ears: no audio windows (clip too short)"); return -1; }
  char aerr[256] = {0};
  const int PD = c->PD;
  std::vector<float> embd;
  t = now_ms();
  const int bmode = ears_batch_on();
  if (bmode && wins.size() > 1) {  // (CHAIN-2-1-3 Stage 2) LLB_EAR_BATCH: all windows in one tower pass
    std::vector<const float *> mp;
    std::vector<int> nl;
    size_t tot = 0;
    for (auto &w : wins) mp.push_back(w.data.data()), nl.push_back(w.n_len), tot += (size_t)aut_n_tokens(w.n_len) * PD;
    embd.resize(tot);
    std::vector<float *> ep;
    for (size_t i = 0, o = 0; i < wins.size(); o += (size_t)aut_n_tokens(wins[i].n_len) * PD, i++) ep.push_back(embd.data() + o);
    if (aut_encode_batch(c->tw, (int)wins.size(), mp.data(), nl.data(), bmode, ep.data(), aerr)) { snprintf(err, 512, "%s", aerr); return -1; }
    st->tower_gpu_ms = aut_gpu_ms(c->tw);
  } else
    for (auto &w : wins) {
      std::vector<float> e((size_t)aut_n_tokens(w.n_len) * PD);
      if (aut_encode(c->tw, w.data.data(), w.n_mel, w.n_len, nullptr, e.data(), aerr)) { snprintf(err, 512, "%s", aerr); return -1; }
      st->tower_gpu_ms += aut_gpu_ms(c->tw);
      embd.insert(embd.end(), e.begin(), e.end());
    }
  st->tower_ms = now_ms() - t;
  st->n_audio_tokens = (int)(embd.size() / PD);
  if (embd_out) {
    FILE *f = fopen(embd_out, "wb");
    if (f) fwrite(embd.data(), 4, embd.size(), f), fclose(f);
  }
  // 2. LLM + the slim chunker (agold step 3 without mtmd: the tower already gave the rows)
  t = now_ms();
  int rc = -1;
  llama_context *lctx = nullptr;
  do {
    std::vector<int> win_tokens;
    for (auto &w : wins) win_tokens.push_back(aut_n_tokens(w.n_len));
    const std::vector<Chunk> chunks = qwen3a_chunks(llama_model_get_vocab(c->model), "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n",
                                                    "<|im_end|>\n<|im_start|>assistant\n", win_tokens);
    llama_context_params lp = llama_context_default_params();
    lp.n_ctx = 4096, lp.n_batch = 2048, lp.n_threads = 4;
    lctx = llama_init_from_model(c->model, lp);
    if (!lctx) { snprintf(err, 512, "ears: llama context failed"); break; }
    st->llm_load_ms = now_ms() - t;
    t = now_ms();
    llama_pos n_past = 0;
    size_t off = 0;
    bool ok = true;
    for (size_t i = 0; i < chunks.size() && ok; i++) {
      if (eval_chunk(lctx, chunks[i], embd.data() + off, PD, c->mrope, lp.n_batch, i == chunks.size() - 1, &n_past) != 0) {
        snprintf(err, 512, "ears: chunk %zu eval failed", i);
        ok = false;
      }
      if (chunks[i].audio) off += (size_t)chunks[i].n * PD;
    }
    if (!ok) break;
    st->prefill_ms = now_ms() - t;
    t = now_ms();
    const llama_vocab *voc = llama_model_get_vocab(c->model);
    const int nv = llama_vocab_n_tokens(voc);
    int n = 0;
    for (int s = 0; s < n_gen && n < max_ids; s++) {
      const float *lg = llama_get_logits_ith(lctx, -1);
      int b1 = 0, b2 = -1;
      for (int v = 1; v < nv; v++) {
        if (lg[v] > lg[b1]) { b2 = b1; b1 = v; } else if (b2 < 0 || lg[v] > lg[b2]) b2 = v;
      }
      ids[n] = b1, top1[n] = lg[b1], top2[n] = lg[b2], n++;
      if (n == 1) st->first_token_ms = now_ms() - t0;
      if (llama_vocab_is_eog(voc, b1)) break;
      llama_token tk = b1;
      llama_batch bt = llama_batch_get_one(&tk, 1);
      if (llama_decode(lctx, bt) != 0) { snprintf(err, 512, "ears: decode failed"); ok = false; break; }
    }
    if (!ok) break;
    st->decode_ms = now_ms() - t, st->n_ids = n;
    rc = n;
  } while (0);
  if (lctx) llama_free(lctx);
  st->total_ms = now_ms() - t0;
  return rc;
}

extern "C" int ears_transcribe(const char *llm, const char *mmproj, const char *wav, int n_gen, int llm_gpu, const char *embd_out,
                               int32_t *ids, float *top1, float *top2, int max_ids, ears_stats *st, char *err) {
  ears_stats load;
  ears_ctx *c = ears_open(llm, mmproj, llm_gpu, &load, err);
  if (!c) { memset(st, 0, sizeof *st); return -1; }
  const int rc = ears_run(c, wav, n_gen, embd_out, ids, top1, top2, max_ids, st, err);
  st->tower_load_ms = load.tower_load_ms, st->llm_load_ms += load.llm_load_ms;
  ears_close(c);
  return rc;
}
