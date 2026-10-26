// ears.h: the ear in process: host log-mel (aud_mel) -> the AuT tower (aut.h: aut.cu on CUDA, aut_metal.mm on Metal) ->
// the Qwen3-ASR-0.6B LLM (llama.cpp 2ca15f5) fed the tower's rows at the audio chunks (the slim qwen3a chunker: mtmd's
// layout, no second mmproj load) -> greedy ids. One source for the CLI twins, the iOS app (EARS.xcframework) and the
// served ear (llb-ear, tools/ears/ear_worker.cpp, driven by llb-serve --ear).
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  int n_windows, n_audio_tokens, n_ids;
  double mel_ms, tower_load_ms, tower_ms, tower_gpu_ms, llm_load_ms, prefill_ms, decode_ms, total_ms;
  double first_token_ms;  // request start -> first greedy id (mel + tower + prefill + one argmax); 0 if none
  double audio_s;         // seconds of audio (samples / 16000)
} ears_stats;
// ids[i] / top1[i] / top2[i] (logits) for up to max_ids greedy steps (the EOG step included, as transcript.tsv).
// llm_gpu: 1 = GPU LLM, 0 = CPU. embd_out: optional path for the tower rows (raw f32). Returns n_ids, < 0 on error.
// One-shot: loads the tower and the LLM, runs, frees both (the IOS-EARS contract, unchanged).
int ears_transcribe(const char *llm, const char *mmproj, const char *wav, int n_gen, int llm_gpu, const char *embd_out,
                    int32_t *ids, float *top1, float *top2, int max_ids, ears_stats *st, char *err /*512*/);
// host mel of wav vs golden_dir/pre.wN.f32: returns windows byte-equal, *n_windows = windows computed; < 0 on error.
int ears_mel_check(const char *wav, const char *golden_dir, int *n_windows, char *err /*512*/);
const char *ears_version(void);

// Resident form (the served ear): the tower and the LLM are loaded ONCE by ears_open; each ears_run is one request on
// a fresh llama context, so its ids and logits are those of ears_transcribe on the same clip.
typedef struct ears_ctx ears_ctx;
ears_ctx *ears_open(const char *llm, const char *mmproj, int llm_gpu, ears_stats *load /*tower_load_ms, llm_load_ms*/,
                    char *err /*512*/);
int ears_run(ears_ctx *c, const char *wav, int n_gen, const char *embd_out, int32_t *ids, float *top1, float *top2,
             int max_ids, ears_stats *st, char *err /*512*/);
// the detokenized text of ids (special tokens rendered, as the goldens' transcript.txt). Returns bytes written.
int ears_detok(ears_ctx *c, const int32_t *ids, int n, char *out, int cap);
int ears_proj_dim(const ears_ctx *c);
// (CHAIN-2-1-3 Stage 2) the encoder-batch gate (llb-ear encb): singles vs one batched tower pass over all windows.
int ears_encb_gate(const char *mmproj, int nwav, const char *const *wavs, int gemm_seg, int rep, int max_batch,
                   const char *outdir);
void ears_close(ears_ctx *c);
#ifdef __cplusplus
}
#endif
