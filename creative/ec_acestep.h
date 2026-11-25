// ec_acestep.h — 100% Pure Native Eureka ACE-Step 1.5 Turbo Music Generation Engine
// Zero Python, zero cuBLAS, zero ggml, zero external ML frameworks!
// Runs native Qwen3-0.6B text encoder, AceStepConditionEncoder (text proj + 8L lyric encoder
// + 4L timbre encoder), 24-layer GQA Turbo DiT (8-step Shift=3.0 ODE with cached cross-attn),
// and full AutoencoderOobleck 1D stereo audio VAE -> 48 kHz Stereo WAV + 4 mixer stems.

#pragma once

#include "ec_lora.h"
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
  const char           *prompt;           // musical caption / style description
  const char           *lyrics;           // optional lyrics (NULL or "" -> "[Instrumental]")
  uint32_t              seed;
  int                   bpm;              // 40..240 (default 120)
  float                 duration_sec;     // 4.0..120.0 (default 20.0)
  int                   steps;            // 1..64 (default 8 for Turbo, 32-48 for SFT)
  float                 shift;            // 0.5..5.0 (default 3.0)
  float                 cfg_scale;        // 1.0 default (Turbo); 5.0..7.5 for undistilled SFT/Base!
  const char           *checkpoint_override; // optional SFT/Base/Turbo checkpoint path!
  // v8 Native Audio-to-Audio, Repaint [t0..t1], Extend (up to 120s), Musical Controls, Flexible Stems & LoRA/LoKr
  const char           *key_scale;        // e.g. "D minor", "C major", "F# minor" (NULL -> "N/A")
  const char           *time_sig;         // e.g. "4", "3", "6" (NULL -> "4")
  const char           *vocal_mode;       // "auto"|"instrumental"|"female"|"male"|"duet"|"choir"|"rap"
  const char           *audio_mode;       // "generate"|"cover"|"repaint"|"extend"
  const float          *source_pcm_lr;    // optional 48kHz stereo interleaved F32 source audio
  int                   source_samples;   // per channel sample count of source_pcm_lr
  float                 denoise;          // 0.1..1.0 for Cover / Style Transfer (default 0.65)
  float                 repaint_start_sec;// Repaint window start (sec)
  float                 repaint_end_sec;  // Repaint window end (sec)
  int                   stem_mode;        // 0=full_mix, 2=vocals/inst, 4=4stem, 8=8stem (default 4)
  const ec_lora_spec_t *loras;
  int                   n_loras;
} ec_acestep_req_t;

typedef struct {
  int    ok;
  int    sample_rate;       // 48000
  int    channels;          // 2
  int    num_samples;       // per channel
  int    latent_frames;     // T @ 25 Hz
  int    steps_run;
  int    lora_tensors_fused;
  int    n_stems;
  char   stem_paths[8][256];
  char   stem_names[8][24];
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

// Generates 48 kHz Stereo WAV to out_wav_path, plus flexible stems (0 / 2 / 4 / 8 based on req->stem_mode
// or legacy stem_paths[4]). Optionally writes stereo interleaved float samples into out_pcm_lr.
int ec_acestep_generate_wav(
    ec_acestep_t *as,
    const ec_acestep_req_t *req,
    const char *out_wav_path,
    const char *const stem_paths[4],
    float **out_pcm_lr,
    ec_acestep_progress_fn cb, void *cb_ctx,
    ec_acestep_res_t *res);

// Native Oobleck 1D Stereo Audio VAE Encoder (48 kHz stereo PCM -> [T, 64] GPU latent, exposed for G-C7 verification!)
int ec_acestep_encode_pcm(
    ec_acestep_t *as, const float *pcm_lr, int n_samples,
    float **out_latent_h, int *out_T, double *out_ms);

#ifdef __cplusplus
}
#endif
