// ec_nvenc.c — Zero-Intermediate-File Hardware NVENC / libx264 MP4 + Stereo Audio Muxer

#define _POSIX_C_SOURCE 200809L
#include "ec_nvenc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

struct ec_mp4_writer {
  pid_t  pid;
  int    stdin_fd;
  size_t frame_bytes;
  int    frames_written;
};

static int probe_encoder_works(const char *enc) {
  char cmd[320];
  snprintf(cmd, sizeof cmd,
           "ffmpeg -hide_banner -loglevel error -f lavfi -i color=black:s=256x256:r=16 -frames:v 1 "
           "-c:v %s -pix_fmt yuv420p -f null - >/dev/null 2>&1", enc);
  FILE *p = popen(cmd, "r");
  if (!p) return 0;
  int st = pclose(p);
  return (WIFEXITED(st) && WEXITSTATUS(st) == 0) ? 1 : 0;
}

static int has_nvenc(void) {
  static int cached = -1;
  if (cached >= 0) return cached;
  cached = probe_encoder_works("h264_nvenc");
  return cached;
}

static const char *select_video_codec(int prefer_nvenc) {
  const char *want = getenv("EC_VIDEO_CODEC");
  if (want && want[0]) {
    if (!strcmp(want, "av1") || !strcmp(want, "av1_nvenc")) {
      static int av1_hw = -1;
      if (av1_hw < 0) av1_hw = probe_encoder_works("av1_nvenc");
      if (av1_hw) return "av1_nvenc";
      static int svt = -1;
      if (svt < 0) svt = probe_encoder_works("libsvtav1");
      if (svt) return "libsvtav1";
    } else if (!strcmp(want, "hevc") || !strcmp(want, "hevc_nvenc")) {
      static int hevc_hw = -1;
      if (hevc_hw < 0) hevc_hw = probe_encoder_works("hevc_nvenc");
      if (hevc_hw) return "hevc_nvenc";
    }
  }
  return (prefer_nvenc && has_nvenc()) ? "h264_nvenc" : "libx264";
}

static ec_mp4_writer_t *spawn_ffmpeg(
    const char *out_mp4, const char *audio_wav,
    uint32_t w, uint32_t h, uint32_t fps, const char *vcodec) {
  int pipefd[2];
  if (pipe(pipefd) != 0) return NULL;

  pid_t pid = fork();
  if (pid < 0) {
    close(pipefd[0]); close(pipefd[1]);
    return NULL;
  }
  if (pid == 0) {
    dup2(pipefd[0], STDIN_FILENO);
    close(pipefd[0]); close(pipefd[1]);
    // Redirect ffmpeg stdout/stderr to /dev/null to keep server logs clean
    FILE *dn = fopen("/dev/null", "wb");
    if (dn) { dup2(fileno(dn), STDOUT_FILENO); dup2(fileno(dn), STDERR_FILENO); fclose(dn); }

    char size_arg[32], fps_arg[16];
    snprintf(size_arg, sizeof size_arg, "%ux%u", w, h);
    snprintf(fps_arg, sizeof fps_arg, "%u", fps);

    const char *argv[64];
    int ac = 0;
    argv[ac++] = "ffmpeg";
    argv[ac++] = "-y";
    argv[ac++] = "-loglevel"; argv[ac++] = "error";
    argv[ac++] = "-f";        argv[ac++] = "rawvideo";
    argv[ac++] = "-pix_fmt";  argv[ac++] = "rgb24";
    argv[ac++] = "-s";        argv[ac++] = size_arg;
    argv[ac++] = "-r";        argv[ac++] = fps_arg;
    argv[ac++] = "-i";        argv[ac++] = "pipe:0";
    if (audio_wav && audio_wav[0]) {
      argv[ac++] = "-i"; argv[ac++] = audio_wav;
    }
    argv[ac++] = "-c:v"; argv[ac++] = vcodec;
    if (!strcmp(vcodec, "h264_nvenc") || !strcmp(vcodec, "hevc_nvenc") || !strcmp(vcodec, "av1_nvenc")) {
      // Ampere/Ada NVENC psycho-visual HQ tuning: Spatial + Temporal Adaptive Quantization + Lookahead!
      argv[ac++] = "-preset";      argv[ac++] = "p5";
      argv[ac++] = "-tune";        argv[ac++] = "hq";
      argv[ac++] = "-rc";          argv[ac++] = "vbr";
      argv[ac++] = "-cq";          argv[ac++] = "19";
      argv[ac++] = "-spatial-aq";  argv[ac++] = "1";
      argv[ac++] = "-aq-strength"; argv[ac++] = "8";
      argv[ac++] = "-temporal-aq"; argv[ac++] = "1";
      argv[ac++] = "-rc-lookahead";argv[ac++] = "12";
    } else if (!strcmp(vcodec, "libsvtav1")) {
      argv[ac++] = "-preset"; argv[ac++] = "8";
      argv[ac++] = "-crf";    argv[ac++] = "24";
    } else {
      argv[ac++] = "-preset"; argv[ac++] = "fast";
      argv[ac++] = "-crf";    argv[ac++] = "18";
    }
    argv[ac++] = "-pix_fmt";         argv[ac++] = "yuv420p";
    argv[ac++] = "-colorspace";      argv[ac++] = "bt709";
    argv[ac++] = "-color_primaries"; argv[ac++] = "bt709";
    argv[ac++] = "-color_trc";       argv[ac++] = "bt709";
    if (audio_wav && audio_wav[0]) {
      argv[ac++] = "-c:a"; argv[ac++] = "aac";
      argv[ac++] = "-b:a"; argv[ac++] = "256k";
      argv[ac++] = "-shortest";
    }
    argv[ac++] = "-movflags"; argv[ac++] = "+faststart";
    argv[ac++] = out_mp4;
    argv[ac] = NULL;

    execvp("ffmpeg", (char *const *)argv);
    _exit(127);
  }

  close(pipefd[0]);
  ec_mp4_writer_t *wr = (ec_mp4_writer_t *)calloc(1, sizeof *wr);
  if (!wr) { close(pipefd[1]); waitpid(pid, NULL, 0); return NULL; }
  wr->pid = pid;
  wr->stdin_fd = pipefd[1];
  wr->frame_bytes = (size_t)w * h * 3u;
  return wr;
}

ec_mp4_writer_t *ec_mp4_open(
    const char *out_mp4_path, const char *audio_wav_path,
    uint32_t width, uint32_t height, uint32_t fps, int prefer_nvenc,
    char *encoder_used, size_t enc_cap) {
  if (!out_mp4_path || width < 16 || height < 16 || (width & 1) || (height & 1)) return NULL;
  const char *vcodec = select_video_codec(prefer_nvenc);
  ec_mp4_writer_t *wr = spawn_ffmpeg(out_mp4_path, audio_wav_path, width, height, fps ? fps : 24, vcodec);
  if (wr && encoder_used && enc_cap > 0) snprintf(encoder_used, enc_cap, "%s", vcodec);
  return wr;
}

int ec_mp4_write_rgb24(ec_mp4_writer_t *w, const uint8_t *rgb, size_t nbytes) {
  if (!w || w->stdin_fd < 0 || !rgb || nbytes != w->frame_bytes) return -1;
  size_t off = 0;
  while (off < nbytes) {
    ssize_t n = write(w->stdin_fd, rgb + off, nbytes - off);
    if (n <= 0) return -1;
    off += (size_t)n;
  }
  w->frames_written++;
  return 0;
}

int ec_mp4_close(ec_mp4_writer_t *w) {
  if (!w) return -1;
  if (w->stdin_fd >= 0) close(w->stdin_fd);
  int status = 0;
  waitpid(w->pid, &status, 0);
  int ok = (WIFEXITED(status) && WEXITSTATUS(status) == 0 && w->frames_written > 0) ? 0 : -1;
  free(w);
  return ok;
}
