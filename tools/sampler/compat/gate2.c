// gate 2: distribution test + seed determinism.
// usage: gate2 CASEDIR REFDIR case_a case_b case_c
//
// THRESHOLDS (fixed before the first run, 2026-10-02):
//   N = 200000 draws per parameter set, history/bias fixed (no accept between draws).
//   Analytic truth = llama.cpp's own dist-stage probabilities (ref_chain, stage 6).
//   PASS per set iff ALL of:
//     (1) zero draws outside the survivor set;
//     (2) every survivor: |obs - N*p| <= 5*sqrt(N*p*(1-p))        (5 sigma, per category);
//     (3) chi-square over survivors with expected >= 5 (rest pooled as one bin)
//         <= df + 5*sqrt(2*df),  df = bins-1.
//   Determinism: seed A stream of 1000 tokens (accepting each token, penalties on) is
//   byte-identical on a second run; seeds B and C each differ from A in >= 1 token.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "case_load.h"

static int stream(const Q35Params *Pin, const float *logits, int V, int len, uint64_t seed, int *out) {
    Q35Params P = *Pin; P.seed = seed;
    Q35S *S = q35s_new(&P, V);
    float *sp = malloc(4 * (size_t)V); int *si = malloc(4 * (size_t)V);
    for (int i = 0; i < len; i++) { out[i] = q35s_sample(S, logits, sp, si); q35s_accept(S, out[i]); }
    q35s_free(S); free(sp); free(si); return 0;
}

int main(int argc, char **argv) {
    const int N = 200000;
    int bad = 0;
    for (int a = 3; a < argc; a++) {
        int ci = atoi(argv[a]); char path[512], line[1 << 16];
        snprintf(path, sizeof path, "%s/%03d.args", argv[1], ci);
        FILE *fa = fopen(path, "r"); if (!fgets(line, sizeof line, fa)) return 2; fclose(fa);
        CaseP *CP = malloc(sizeof *CP); parse_case_args(line, CP);
        snprintf(path, sizeof path, "%s/%03d.f32", argv[1], ci);
        int V; float *logits = load_logits(path, &V);
        // llama's final dist probabilities
        snprintf(path, sizeof path, "%s/%03d.ref", argv[2], ci);
        FILE *fr = fopen(path, "rb"); int32_t hdr[2]; int *rid = malloc(4 * (size_t)V); float *rp = malloc(4 * (size_t)V), *tmp = malloc(4 * (size_t)V);
        int rn = 0;
        while (fread(hdr, 4, 2, fr) == 2) {
            int n = hdr[1];
            if (fread(rid, 4, (size_t)n, fr) != (size_t)n || fread(tmp, 4, (size_t)n, fr) != (size_t)n || fread(rp, 4, (size_t)n, fr) != (size_t)n) return 2;
            rn = hdr[0] == 6 ? n : rn;
        }
        fclose(fr);
        double *pv = calloc((size_t)V, sizeof(double)); char *surv = calloc((size_t)V, 1);
        for (int j = 0; j < rn; j++) { pv[rid[j]] = rp[j]; surv[rid[j]] = 1; }
        Q35S *S = q35s_new(&CP->P, V);
        for (int i = 0; i < CP->nh; i++) q35s_accept(S, CP->hist[i]);
        float *sp = malloc(4 * (size_t)V); int *si = malloc(4 * (size_t)V);
        long *obs = calloc((size_t)V, sizeof(long)); long outside = 0;
        for (int i = 0; i < N; i++) { int t = q35s_sample(S, logits, sp, si); if (surv[t]) obs[t]++; else outside++; }
        int fail = outside != 0; double chi = 0, pooled_e = 0, pooled_o = 0; int bins = 0; double worst_sig = 0;
        for (int j = 0; j < rn; j++) {
            int t = rid[j]; double e = N * pv[t], sd = sqrt(N * pv[t] * (1 - pv[t]));
            double dev = fabs(obs[t] - e); if (sd > 0 && dev / sd > worst_sig) worst_sig = dev / sd;
            if (dev > 5 * sd + 1e-9) fail = 1;
            if (e >= 5) { chi += dev * dev / e; bins++; } else { pooled_e += e; pooled_o += obs[t]; }
        }
        if (pooled_e >= 5) { chi += (pooled_o - pooled_e) * (pooled_o - pooled_e) / pooled_e; bins++; }
        int df = bins - 1; double lim = df + 5 * sqrt(2.0 * (df > 0 ? df : 1));
        if (df > 0 && chi > lim) fail = 1;
        printf("dist case %03d: survivors %d, outside %ld, worst %.2f sigma, chi2 %.1f (df %d, limit %.1f) -> %s\n",
               ci, rn, outside, worst_sig, chi, df, lim, fail ? "FAIL" : "PASS");
        bad |= fail;
        if (a == 3) {                      // determinism on the first set
            int *A = malloc(4000), *A2 = malloc(4000), *B = malloc(4000), *C = malloc(4000);
            stream(&CP->P, logits, V, 1000, 111, A); stream(&CP->P, logits, V, 1000, 111, A2);
            stream(&CP->P, logits, V, 1000, 222, B); stream(&CP->P, logits, V, 1000, 333, C);
            int same = !memcmp(A, A2, 4000), dB = memcmp(A, B, 4000) != 0, dC = memcmp(A, C, 4000) != 0;
            printf("determinism: same-seed identical=%d, seed B differs=%d, seed C differs=%d -> %s\n", same, dB, dC, same && dB && dC ? "PASS" : "FAIL");
            bad |= !(same && dB && dC);
        }
        q35s_free(S);
    }
    printf("gate2: %s\n", bad ? "FAIL" : "PASS");
    return bad;
}
