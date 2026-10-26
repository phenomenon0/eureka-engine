// aud_mel.h: EARS-G1 host log-mel (= llama.cpp 2ca15f5 mtmd_audio_preprocessor_qwen3a). See aud_mel.cpp.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

struct AudMelParams {
  int n_mel = 128, n_fft = 400, window_len = 400, hop = 160, sample_rate = 16000;
};
struct AudMelWindow {
  int n_mel, n_len, n_valid;  // n_len = n_valid rounded up to x100 (zero-padded tail)
  std::vector<float> data;    // [n_mel][n_len] (= clip inp_raw, ne0 = frames)
};
// 16 kHz mono f32 samples -> 800-frame windows. 0 ok.
int aud_mel_qwen3a(const float *samples, size_t n_samples, const AudMelParams &hp, std::vector<AudMelWindow> &out);
