// ec_acestep.h — 100% Pure Native Eureka ACE-Step 1.5 Turbo Music Generation Engine
// Zero Python, zero cuBLAS, zero ggml, zero external ML frameworks!
// Runs native Qwen3-0.6B text encoder, AceStepConditionEncoder (text proj + 8L lyric encoder
// + 4L timbre encoder), 24-layer GQA Turbo DiT (8-step Shift=3.0 ODE with cached cross-attn),
// and full AutoencoderOobleck 1D stereo audio VAE -> 48 kHz Stereo WAV + 4 mixer stems.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ec_acestep ec_acestep_t;

typedef struct {
  char dit_path[512];            // acestep_v15_turbo_bf16.safetensors
  char vae_path[512];            // acestep_v15_vae.safetensors
  char text_encoder_path[512];   // acestep_qwen3_0.6b.safetensors
  char silence_latent_path[512]; // acestep_silence_latent.safetensors
  char tokenizer_path[512];      // qwen3_tokenizer.json
} ec_acestep_paths_t;

typedef struct {
  const char *prompt;        // musical caption / style description
  const char *lyrics;        // optional lyrics (NULL or "" -> "[Instrumental]")
  uint32_t    seed;
  int         bpm;           // 40..240 (default 120)
  float       duration_sec;  // 4.0..60.0 (default 10.0)
  int         steps;         // 1..8 (default 8 for Turbo Shift=3.0 ODE)
  float       shift;         // 1.0, 2.0, or 3.0 (default 3.0)
} ec_acestep_req_t;

typedef struct {
  int    ok;
  int    sample_rate;       // 48000
  int    channels;          // 2
  int    num_samples;       // per channel
  int    latent_frames;     // T @ 25 Hz
  int    steps_run;
  float  peak_amp;
  float  rms_db;
  double text_enc_ms;
  double cond_enc_ms;
  double dit_ms;
  double vae_ms;
  double total_ms;
  char   error[256];
} ec_acestep_res_t;

typedef int (*ec_acestep_progress_fn)(int step, int total_steps, const char *stage, void *ctx);

ec_acestep_t *ec_acestep_create(const ec_acestep_paths_t *paths);
void          ec_acestep_destroy(ec_acestep_t *as);
int           ec_acestep_is_ready(const ec_acestep_t *as);

// Releases resident GPU weights (keeps lightweight mmap & tokenizer open) to share VRAM across engines.
void ec_acestep_unload_vram(ec_acestep_t *as);

// Generates 48 kHz Stereo WAV to out_wav_path, plus optional 4 mixer stems (drums/bass/harmony/lead)
// if stem_paths != NULL (array of 4 path strings, any entry may be NULL/empty to skip).
// Also optionally writes stereo interleaved float samples into out_pcm_lr (caller frees) if non-NULL.
int ec_acestep_generate_wav(
    ec_acestep_t *as,
    const ec_acestep_req_t *req,
    const char *out_wav_path,
    const char *const stem_paths[4],
    float **out_pcm_lr,
    ec_acestep_progress_fn cb, void *cb_ctx,
    ec_acestep_res_t *res);

#ifdef __cplusplus
}
#endif
