// ec_nvenc.h — Native Hardware-Accelerated NVENC (h264_nvenc / libx264 fallback) MP4 + Audio Muxer
// Streams raw RGB24 frames via zero-disk POSIX pipe straight into NVIDIA NVENC hardware encoder
// and optionally muxes stereo WAV audio into a real playable H.264 + AAC .mp4 file.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ec_mp4_writer ec_mp4_writer_t;

// Opens hardware NVENC (falling back to libx264 if NVENC unavailable) MP4 writer.
// audio_wav_path may be NULL for video-only MP4, or a valid .wav path for synced A/V MP4 (LTX-2.5!).
ec_mp4_writer_t *ec_mp4_open(
    const char *out_mp4_path, const char *audio_wav_path,
    uint32_t width, uint32_t height, uint32_t fps, int prefer_nvenc,
    char *encoder_used, size_t enc_cap);

int ec_mp4_write_rgb24(ec_mp4_writer_t *w, const uint8_t *rgb, size_t nbytes);
int ec_mp4_close(ec_mp4_writer_t *w);

#ifdef __cplusplus
}
#endif
