/* backends/windows/cblas_sgemm.c: the one CBLAS routine the engine links (XPLAT-WIN).
 *
 * tools/clef3/head.c, the CLEF decide head, calls cblas_sgemm; Linux links OpenBLAS
 * (-lopenblas) for it. No OpenBLAS is installed on the Windows build machine, so
 * this is a plain-C SGEMM with the CBLAS signature (backends/windows/compat/openblas/
 * cblas.h). It is used only by decide requests that carry a head; generation never
 * calls it. Each output is a left-to-right sum over k in float (mul, then add: built with
 * -ffp-contract=off), so a head's probabilities can differ from OpenBLAS's in the last bits.
 *
 * DECIDE-FAST (2026-11-16): the same sums, faster. Each output's sum still runs p = 0..k-1 in order, so every result is
 * bit-identical to the plain triple loop; what changed is how many sums run at once. When B's columns are contiguous
 * (sbj == 1) the inner loop runs across outputs (independent sums the compiler vectorizes); otherwise four outputs share
 * each loaded A element. The rows (or, when there are few, the columns) are split across the CPU's threads
 * (LLB_SGEMM_THREADS caps them). Measured on clef-flash 9B's head (RTX 3090 box): 3.5 s a decision before.
 */
#include "compat/openblas/cblas.h"
#include <stddef.h>
#include <stdlib.h>
// lean: the CPU engine builds this file against the POSIX shim (compat/), whose socket types winsock.h would redefine
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

typedef struct {
  const float *a, *b;
  float *c;
  size_t sai, sap, sbp, sbj, sci, scj;
  int k;
  float alpha, beta;
  int i0, i1, j0, j1;
} Job;

static void put(const Job *J, int i, int j, float s) {
  float *o = J->c + (size_t)i * J->sci + (size_t)j * J->scj;
  *o = J->beta == 0.0f ? J->alpha * s : J->alpha * s + J->beta * *o;
}

#define JB 256
static void run(const Job *J) {
  const float *a = J->a, *b = J->b;
  if (J->sbj == 1) {
    // B(p, j) contiguous in j: for each row i, sweep p and add x * B(p, j..j+JB) into JB independent sums
    float acc[JB];
    for (int i = J->i0; i < J->i1; i++)
      for (int j0 = J->j0; j0 < J->j1; j0 += JB) {
        const int nb = J->j1 - j0 < JB ? J->j1 - j0 : JB;
        for (int q = 0; q < nb; q++) acc[q] = 0.0f;
        for (int p = 0; p < J->k; p++) {
          const float x = a[(size_t)i * J->sai + (size_t)p * J->sap];
          const float *row = b + (size_t)p * J->sbp + j0;
          for (int q = 0; q < nb; q++) acc[q] += x * row[q];
        }
        for (int q = 0; q < nb; q++) put(J, i, j0 + q, acc[q]);
      }
    return;
  }
  // otherwise dot products, four outputs sharing each A element
  for (int i = J->i0; i < J->i1; i++) {
    const float *ai = a + (size_t)i * J->sai;
    int j = J->j0;
    for (; j + 4 <= J->j1; j += 4) {
      const float *b0 = b + (size_t)j * J->sbj, *b1 = b0 + J->sbj, *b2 = b1 + J->sbj, *b3 = b2 + J->sbj;
      float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
      for (int p = 0; p < J->k; p++) {
        const float x = ai[(size_t)p * J->sap];
        const size_t o = (size_t)p * J->sbp;
        s0 += x * b0[o], s1 += x * b1[o], s2 += x * b2[o], s3 += x * b3[o];
      }
      put(J, i, j, s0), put(J, i, j + 1, s1), put(J, i, j + 2, s2), put(J, i, j + 3, s3);
    }
    for (; j < J->j1; j++) {
      const float *bj = b + (size_t)j * J->sbj;
      float s = 0.0f;
      for (int p = 0; p < J->k; p++) s += ai[(size_t)p * J->sap] * bj[(size_t)p * J->sbp];
      put(J, i, j, s);
    }
  }
}

static DWORD WINAPI worker(LPVOID arg) {
  run((const Job *)arg);
  return 0;
}

void cblas_sgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE ta, enum CBLAS_TRANSPOSE tb, int m, int n, int k,
  float alpha, const float *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
  if (m <= 0 || n <= 0) return;
  const int rm = order == CblasRowMajor, at_ = ta != CblasNoTrans, bt = tb != CblasNoTrans;
  // element strides: A(i,p), B(p,j), C(i,j) (column-major is the row-major view of the transpose)
  Job J = {a, b, c, 0, 0, 0, 0, 0, 0, k, alpha, beta, 0, m, 0, n};
  if (rm) {
    J.sai = at_ ? 1 : (size_t)lda, J.sap = at_ ? (size_t)lda : 1;
    J.sbp = bt ? 1 : (size_t)ldb, J.sbj = bt ? (size_t)ldb : 1;
    J.sci = (size_t)ldc, J.scj = 1;
  } else {
    J.sai = at_ ? (size_t)lda : 1, J.sap = at_ ? 1 : (size_t)lda;
    J.sbp = bt ? (size_t)ldb : 1, J.sbj = bt ? 1 : (size_t)ldb;
    J.sci = 1, J.scj = (size_t)ldc;
  }
  static int threads;
  if (!threads) {
    const char *e = getenv("LLB_SGEMM_THREADS");
    int t = e && atoi(e) > 0 ? atoi(e) : (int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    threads = t < 1 ? 1 : t > 32 ? 32 : t;
  }
  const double work = (double)m * n * (k > 0 ? k : 1);
  int T = work < 2e6 ? 1 : threads;
  const int by_rows = m >= 2 * T;
  if (!by_rows && n < 2 * T) T = 1;
  if (T == 1) { run(&J); return; }
  Job jobs[32];
  HANDLE hs[32];
  int nt = 0;
  for (int t = 0; t < T; t++) {
    Job x = J;
    const int span = by_rows ? m : n, lo = (int)((long long)span * t / T), hi = (int)((long long)span * (t + 1) / T);
    if (lo >= hi) continue;
    if (by_rows) x.i0 = lo, x.i1 = hi; else x.j0 = lo, x.j1 = hi;
    jobs[nt] = x;
    hs[nt] = CreateThread(NULL, 0, worker, &jobs[nt], 0, NULL);
    if (!hs[nt]) run(&jobs[nt]); else nt++;
  }
  WaitForMultipleObjects((DWORD)nt, hs, TRUE, INFINITE);
  for (int t = 0; t < nt; t++) CloseHandle(hs[t]);
}
