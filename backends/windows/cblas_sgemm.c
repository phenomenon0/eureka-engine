/* backends/windows/cblas_sgemm.c: the one CBLAS routine the engine links (XPLAT-WIN).
 *
 * tools/clef3/head.c, the CLEF decide head, calls cblas_sgemm; Linux links OpenBLAS
 * (-lopenblas) for it. No OpenBLAS is installed on the Windows build machine, so
 * this is a plain-C SGEMM with the CBLAS signature (backends/windows/compat/openblas/
 * cblas.h). It is used only by decide requests that carry a head; generation never
 * calls it. Each output is a left-to-right sum over k in float, so a head's
 * probabilities can differ from OpenBLAS's in the last bits.
 */
#include "compat/openblas/cblas.h"
#include <stddef.h>

static float at(const float *m, int ld, int r, int c, int rowmajor) {
  return rowmajor ? m[(size_t)r * ld + c] : m[(size_t)c * ld + r];
}

void cblas_sgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE ta, enum CBLAS_TRANSPOSE tb, int m, int n, int k,
  float alpha, const float *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
  int rm = order == CblasRowMajor;
  int at_ = ta != CblasNoTrans, bt = tb != CblasNoTrans;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float s = 0.0f;
      for (int p = 0; p < k; p++) {
        float x = at_ ? at(a, lda, p, i, rm) : at(a, lda, i, p, rm);
        float y = bt ? at(b, ldb, j, p, rm) : at(b, ldb, p, j, rm);
        s += x * y;
      }
      float *o = rm ? &c[(size_t)i * ldc + j] : &c[(size_t)j * ldc + i];
      *o = beta == 0.0f ? alpha * s : alpha * s + beta * *o;
    }
  }
}
