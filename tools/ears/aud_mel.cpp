// aud_mel.cpp: EARS-G1 host log-mel for the AuT tower = llama.cpp 2ca15f5 mtmd_audio_preprocessor_qwen3a, transliterated
// op for op (mtmd-audio.cpp: fill_sin_cos_table, fill_hann_window(periodic), fill_mel_filterbank_matrix(slaney, area
// norm), the recursive radix-2 / DFT fft_impl, log_mel_spectrogram(no_padding, log10), the (max-8) clamp, then the
// 800-frame windows zero-padded to x100). Gated BYTE-EQUAL vs the G0 `pre.wN`, so the float expressions are kept in
// the same order and types; build with the same g++ and -O3 -ffp-contract=off (mtmd itself is built without -march).
// Deliberately mirrors the 2ca15f5 behaviour of #30353 (n/hop + 1 frames, max over all frames): parity, not a fix.
#define _USE_MATH_DEFINES
#include "aud_mel.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>
#include <cstring>
#include "mel_tables_glibc.h"

namespace {
// IOS-EARS: the goldens were made with glibc, whose sinf/cosf are correctly rounded; Apple's libm sinf/cosf are not
// (10 of the 400 FFT table entries differ by 1 ulp). Evaluate in double and round once to float = correctly rounded
// (double rounding is the only exception; the tables are checked bit-equal to the glibc ones, probe/meltab).
inline float cr_sinf(float x) { return (float)std::sin((double)x); }
inline float cr_cosf(float x) { return (float)std::cos((double)x); }
struct Cache {
  std::vector<float> sin_vals, cos_vals, hann, filters;
};

void fill_sin_cos(Cache &c, uint32_t n) {
  c.sin_vals.resize(n), c.cos_vals.resize(n);
  if (n == 400) {  // IOS-EARS: the reference (glibc) tables, see mel_tables_glibc.h
    memcpy(c.sin_vals.data(), kGlibcSin400, sizeof kGlibcSin400), memcpy(c.cos_vals.data(), kGlibcCos400, sizeof kGlibcCos400);
    return;
  }
  for (uint32_t i = 0; i < n; i++) {
    double theta = (2 * M_PI * i) / n;
    c.sin_vals[i] = cr_sinf(theta);
    c.cos_vals[i] = cr_cosf(theta);
  }
}

void fill_hann(Cache &c, uint32_t length) {  // periodic
  c.hann.resize(length);
  if (length == 400) { memcpy(c.hann.data(), kGlibcHann400, sizeof kGlibcHann400); return; }  // IOS-EARS, as above
  for (uint32_t i = 0; i < length; i++) c.hann[i] = 0.5 * (1.0 - cr_cosf((2.0 * M_PI * i) / (length + 0)));
}

void fill_mel(Cache &c, int64_t n_mel, int64_t n_fft, int sample_rate) {  // fmin 0, fmax sr/2, slaney + area norm
  float fmin = 0.0f, fmax = 0.5f * sample_rate, scale = 1.0f;
  const double min_log_hz = 1000.0, lin_slope = 3 / 200., min_log_mel = min_log_hz * lin_slope, log_step = log(6.4) / 27.0;
  auto hz_to_mel = [&](const double f_hz) -> double {
    return (f_hz < min_log_hz) ? f_hz * lin_slope : min_log_mel + log(f_hz / min_log_hz) / log_step;
  };
  auto mel_to_hz = [&](const double m) -> double {
    return (m < min_log_mel) ? m / lin_slope : min_log_hz * exp((m - min_log_mel) * log_step);
  };
  const double bin_hz_step = double(sample_rate) / double(n_fft);
  const double m_lo = hz_to_mel(fmin), m_hi = hz_to_mel(fmax);
  std::vector<double> mel_pts(n_mel + 2), hz_pts(n_mel + 2);
  for (int i = 0; i < n_mel + 2; ++i) mel_pts[i] = m_lo + (m_hi - m_lo) * (double(i) / (n_mel + 1));
  for (int i = 0; i < n_mel + 2; ++i) hz_pts[i] = mel_to_hz(mel_pts[i]);
  const int64_t n_fft_bins = n_fft / 2 + 1;
  c.filters.assign((size_t)n_mel * (size_t)n_fft_bins, 0);
  for (int64_t m = 0; m < n_mel; ++m) {
    const double f_left = hz_pts[m], f_center = hz_pts[m + 1], f_right = hz_pts[m + 2];
    const double denom_l = std::max(1e-30, f_center - f_left), denom_r = std::max(1e-30, f_right - f_center);
    const double enorm = 2.0 / std::max(1e-30, f_right - f_left);
    for (int k = 0; k < n_fft_bins; ++k) {
      const double f = k * bin_hz_step;
      double w = 0.0;
      if (f >= f_left && f <= f_center) w = (f - f_left) / denom_l;
      else if (f > f_center && f <= f_right) w = (f_right - f) / denom_r;
      c.filters[size_t(m) * size_t(n_fft_bins) + size_t(k)] = float(w * enorm * scale);
    }
  }
}

void dft_real(const Cache &c, const float *in, int N, float *out) {  // dft_impl<false, true>
  const int n_sc = c.sin_vals.size(), step = n_sc / N;
  const float sign = -1.0f;
  for (int k = 0; k < N; k++) {
    float re = 0, im = 0;
    for (int n = 0; n < N; n++) {
      int idx = (k * n * step) % n_sc;
      float cos_val = c.cos_vals[idx], sin_val = c.sin_vals[idx];
      float in_re = in[n];
      re += in_re * cos_val;
      im += sign * in_re * sin_val;
    }
    out[k * 2 + 0] = re * 1.0f;
    out[k * 2 + 1] = im * 1.0f;
  }
}

void fft_real(const Cache &c, float *in, int N, float *out) {  // fft_impl<false, true>
  const int n_sc = c.sin_vals.size();
  if (N == 1) { out[0] = in[0], out[1] = 0.0f; return; }
  const int half_N = N / 2;
  if (N - half_N * 2 == 1) { dft_real(c, in, N, out); return; }
  float *even = in + N;
  for (int i = 0; i < half_N; ++i) even[i] = in[2 * i];
  float *even_fft = out + 2 * N;
  fft_real(c, even, half_N, even_fft);
  float *odd = even;
  for (int i = 0; i < half_N; ++i) odd[i] = in[2 * i + 1];
  float *odd_fft = even_fft + N;
  fft_real(c, odd, half_N, odd_fft);
  const int step = n_sc / N;
  const float sign = -1.0f, scale = 1.0f;
  for (int k = 0; k < half_N; k++) {
    int idx = k * step;
    float re = c.cos_vals[idx], im = sign * c.sin_vals[idx];
    float re_odd = odd_fft[2 * k + 0], im_odd = odd_fft[2 * k + 1];
    out[2 * k + 0] = scale * (even_fft[2 * k + 0] + re * re_odd - im * im_odd);
    out[2 * k + 1] = scale * (even_fft[2 * k + 1] + re * im_odd + im * re_odd);
    out[2 * (k + half_N) + 0] = scale * (even_fft[2 * k + 0] - re * re_odd + im * im_odd);
    out[2 * (k + half_N) + 1] = scale * (even_fft[2 * k + 1] - re * im_odd - im * re_odd);
  }
}
}  // namespace

int aud_mel_qwen3a(const float *samples, size_t n_samples, const AudMelParams &hp, std::vector<AudMelWindow> &out) {
  if (n_samples == 0) return -1;
  Cache c;
  fill_sin_cos(c, hp.n_fft);
  fill_hann(c, hp.window_len);
  fill_mel(c, hp.n_mel, hp.n_fft, hp.sample_rate);
  // reflection pad n_fft/2 at each end (mtmd_audio_preprocessor_qwen3a::preprocess)
  const int pad = hp.n_fft / 2;
  std::vector<float> padded(n_samples + 2 * pad, 0.0f);
  for (int i = 0; i < pad; i++) { int src = pad - i; padded[i] = (src < (int)n_samples) ? samples[src] : 0.0f; }
  std::copy(samples, samples + n_samples, padded.begin() + pad);
  for (int i = 0; i < pad; i++) { int src = (int)n_samples - 2 - i; padded[n_samples + pad + i] = (src >= 0) ? samples[src] : 0.0f; }
  // log_mel_spectrogram, no_padding, log10, power, mel_floor max
  const int n_s = (int)padded.size();
  const int64_t n_fft_bins = 1 + hp.n_fft / 2;
  const int frame_size = (n_fft_bins - 1) * 2, frame_step = hp.hop;
  std::vector<float> hann_p;
  const float *hann = c.hann.data();
  if (hp.window_len < frame_size) {
    hann_p.assign(frame_size, 0.0f);
    std::copy(hann, hann + hp.window_len, &hann_p[(frame_size - hp.window_len) / 2]);
    hann = hann_p.data();
  }
  const int64_t n_mel = hp.n_mel, n_len = (n_s - frame_size) / frame_step + 1;
  if (n_len <= 0 || n_s < frame_size) return -1;
  std::vector<float> mel((size_t)n_mel * n_len);
  {
    std::vector<float> fft_in(frame_size * 2, 0.0), fft_out(frame_size * 2 * 2 * 2);
    const float mel_floor = 5.960464477539063e-08f;
    int64_t i = 0;
    for (; i < std::min((int64_t)(n_s / frame_step + 1), n_len); i++) {
      const int64_t offset = i * frame_step;
      const int valid_len = std::min(frame_size, std::max(0, n_s - (int)offset));
      for (int j = 0; j < valid_len; j++) fft_in[j] = hann[j] * padded[offset + j];
      if (valid_len < frame_size) std::fill(fft_in.begin() + valid_len, fft_in.end(), 0.0);
      fft_real(c, fft_in.data(), frame_size, fft_out.data());
      for (int j = 0; j < n_fft_bins; j++) {
        float power = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        fft_out[j] = power;
      }
      for (int64_t j = 0; j < n_mel; j++) {
        double sum = 0.0;
        int k = 0;
        for (k = 0; k < n_fft_bins - 3; k += 4) {
          size_t idx = size_t(j) * size_t(n_fft_bins) + size_t(k);
          sum += fft_out[k + 0] * c.filters[idx + 0] + fft_out[k + 1] * c.filters[idx + 1] +
                 fft_out[k + 2] * c.filters[idx + 2] + fft_out[k + 3] * c.filters[idx + 3];
        }
        for (; k < n_fft_bins; k++) sum += fft_out[k] * c.filters[(size_t)j * n_fft_bins + k];
        sum = std::max(sum, (double)mel_floor);
        sum = log10(sum);
        mel[(size_t)j * n_len + i] = sum;
      }
    }
    double fill = log10(1e-10);
    for (; i < n_len; i++)
      for (int64_t j = 0; j < n_mel; j++) mel[(size_t)j * n_len + i] = fill;
  }
  {  // whisper normalisation over ALL frames (2ca15f5)
    double mmax = -1e20;
    for (float v : mel) if (v > mmax) mmax = v;
    mmax -= 8.0;
    for (float &v : mel) v = (std::max((double)v, mmax) + 4.0) / 4.0;
  }
  const int64_t n_eff = std::min(n_len, (int64_t)(n_samples / hp.hop) + 1);  // the unpadded count, as in mtmd
  const int chunk = 100, window = 800;
  out.clear();
  for (int64_t off = 0; off < n_eff; off += window) {
    const int64_t win_eff = std::min((int64_t)window, n_eff - off);
    const int64_t n_padded = (win_eff + chunk - 1) / chunk * chunk;
    AudMelWindow w;
    w.n_mel = (int)n_mel, w.n_len = (int)n_padded, w.n_valid = (int)win_eff;
    w.data.assign((size_t)n_mel * n_padded, 0.0f);
    for (int64_t m = 0; m < n_mel; m++) {
      const int64_t copy_len = std::min(win_eff, n_len - off);
      if (copy_len > 0)
        std::copy(mel.begin() + (size_t)m * n_len + off, mel.begin() + (size_t)m * n_len + off + copy_len,
                  w.data.begin() + (size_t)m * n_padded);
    }
    out.push_back(std::move(w));
  }
  return 0;
}
