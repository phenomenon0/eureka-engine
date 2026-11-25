// ec_ltx2.h — 100% Native CUDA Lightricks LTX-2.3 (22B Distilled FP8) Joint Synchronized Audio+Video Engine
// Zero Python, zero cuBLAS, zero ggml, zero external ML frameworks.
// End-to-end native GPU pipeline:
//   1. Gemma-3-12B-IT FP8 (48L, hybrid local/global RoPE, zero-centered RMSNorm) + FeatureExtractorV2 + 8L Embeddings1DConnector
//   2. 8-Step Distilled Dual-Stream AVTransformer3DModel (48L, FP8 E4M3 -> BF16 TensorCore + Hybrid VRAM/PCIe Block Cache, cfg=1.0)
//   3. Non-Causal 3D Conv Video Decoder (up_blocks.0..8 + PixelNorm + D2S + 4x4 Unpatchify -> RGB24)
//   4. Causal 2D Audio Mel Decoder + Pure IEEE FP32 BigVGAN v2 AMP1 Stereo Vocoder + 16k->48k BWE -> 48kHz Studio Stereo WAV + NVENC MP4!

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ec_ltx2 ec_ltx2_t;

typedef struct {
  char checkpoint_path[512];   // ltx-2.3-22b-distilled-fp8.safetensors (all-in-one DiT + Video VAE + Audio VAE + Vocoder + BWE)
  char text_encoder_path[512]; // gemma_3_12B_it_fp8_scaled.safetensors
  char tokenizer_path[512];    // gemma3_tokenizer.json
} ec_ltx2_paths_t;

typedef struct {
  const char *prompt;
  uint32_t    seed;
  int         width;      // multiple of 32 (default 768)
  int         height;     // multiple of 32 (default 512)
  int         n_frames;   // 8*k + 1 (default 49 = ~2.04s @ 24fps, or 73 = ~3.04s)
  int         fps;        // default 24
  int         steps;      // default 8 (official LTX-2.3 distilled schedule)
  int         with_audio; // 1 = synchronized native 48 kHz stereo audio + video
} ec_ltx2_req_t;

typedef struct {
  int    width;
  int    height;
  int    n_frames;
  int    fps;
  int    steps_run;
  int    sample_rate;       // 48000
  int    audio_frames;      // number of 48kHz stereo samples
  double text_enc_ms;
  double dit_ms;
  double video_vae_ms;
  double audio_vae_ms;
  double total_ms;
  char   encoder_used[32];  // "h264_nvenc" or "libx264"
} ec_ltx2_stats_t;

typedef int (*ec_ltx2_progress_cb_t)(int step, int total_steps, const char *stage, void *ctx);

ec_ltx2_t *ec_ltx2_create(const ec_ltx2_paths_t *paths);
void       ec_ltx2_destroy(ec_ltx2_t *eng);
int        ec_ltx2_is_ready(const ec_ltx2_t *eng);
void       ec_ltx2_unload_vram(ec_ltx2_t *eng);

// Generates synchronized Audio+Video MP4 (NVENC BT.709/tv H.264 + 48kHz AAC),
// optional standalone 48kHz studio stereo WAV, plus first/last PNG keyframes.
int ec_ltx2_generate(
    ec_ltx2_t *eng,
    const ec_ltx2_req_t *req,
    const char *out_mp4_path,
    const char *out_wav_path,
    const char *out_keyframe0_png,
    const char *out_last_frame_png,
    ec_ltx2_progress_cb_t cb,
    void *cb_ctx,
    ec_ltx2_stats_t *stats);

#ifdef __cplusplus
}
#endif
