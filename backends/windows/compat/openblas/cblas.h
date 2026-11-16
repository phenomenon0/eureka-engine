/* XPLAT-WIN: part of the Windows POSIX shim (backends/windows/compat/README.md). */
/* XPLAT-WIN: OpenBLAS is not on this machine. This declares the one CBLAS routine
 * tools/clef3/head.c calls (row- and column-major SGEMM), implemented in plain C in
 * backends/windows/cblas_sgemm.c. It feeds only the CLEF decide head, never
 * generation. Its sums are not ordered as OpenBLAS orders them, so head outputs can
 * differ from Linux in the last bits. */
#pragma once
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };
typedef enum CBLAS_ORDER CBLAS_LAYOUT;
void cblas_sgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE ta, enum CBLAS_TRANSPOSE tb, int m, int n, int k,
  float alpha, const float *a, int lda, const float *b, int ldb, float beta, float *c, int ldc);
