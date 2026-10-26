// gate 1: our C chain vs llama.cpp's own samplers (ref_chain output), stage by stage.
// usage: gate1 CASEDIR REFDIR   (NNN.f32/.args/.name in CASEDIR, NNN.ref in REFDIR)
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "case_load.h"

static const char *SN[7] = {"logit_bias", "penalties", "top_k", "top_p", "min_p", "temp", "dist"};
static int cmpf_desc(const void *a, const void *b) { float x = *(const float *)a, y = *(const float *)b; return (x < y) - (x > y); }

static double relerr(double a, double b) {
    if (a == b) return 0;                       // includes +-inf equal
    if (isinf(a) || isinf(b)) return 1e30;
    double d = fabs(a - b), m = fabs(b);
    return m > 1e-30 ? d / m : d;
}

int main(int argc, char **argv) {
    (void)argc;
    const char *cd = argv[1], *rd = argv[2];
    int ncase = 0, nfail = 0;
    int cases[7] = {0}, setbad[7] = {0}, tiediff[7] = {0};
    double wlog[7] = {0}, wp[7] = {0};
    int greedy_n = 0, greedy_bad = 0;
    for (int ci = 0;; ci++) {
        char path[512], line[1 << 16], name[256];
        snprintf(path, sizeof path, "%s/%03d.args", cd, ci);
        FILE *fa = fopen(path, "r"); if (!fa) break;
        if (!fgets(line, sizeof line, fa)) break;
        fclose(fa);
        snprintf(path, sizeof path, "%s/%03d.name", cd, ci);
        FILE *fn = fopen(path, "r"); if (!fgets(name, sizeof name, fn)) break; fclose(fn);
        name[strcspn(name, "\n")] = 0;
        int ties = strstr(name, "ties=1") != NULL;
        snprintf(path, sizeof path, "%s/%03d.f32", cd, ci);
        int V; float *logits = load_logits(path, &V);

        CaseP *CP = (CaseP *)malloc(sizeof *CP); parse_case_args(line, CP);
        Q35Params P = CP->P; int *hist = CP->hist, nh = CP->nh;
        Q35S *S = q35s_new(&P, V);
        for (int i = 0; i < nh; i++) q35s_accept(S, hist[i]);

        snprintf(path, sizeof path, "%s/%03d.ref", rd, ci);
        FILE *fr = fopen(path, "rb"); if (!fr) { printf("case %d: missing ref\n", ci); nfail++; continue; }
        int *ids = malloc(sizeof(int) * (size_t)V), *rid = malloc(sizeof(int) * (size_t)V), *mark = calloc((size_t)V, sizeof(int));
        float *lg = malloc(4 * (size_t)V), *rlg = malloc(4 * (size_t)V), *rp = malloc(4 * (size_t)V), *pd = malloc(4 * (size_t)V);
        float *mlg_sorted = malloc(4 * (size_t)V), *rlg_sorted = malloc(4 * (size_t)V);
        int bad = 0; int ref_surv = -1; int tainted = 0;
        int32_t hdr[2];
        while (fread(hdr, 4, 2, fr) == 2) {
            int st = hdr[0], rn = hdr[1];
            if (fread(rid, 4, (size_t)rn, fr) != (size_t)rn || fread(rlg, 4, (size_t)rn, fr) != (size_t)rn || fread(rp, 4, (size_t)rn, fr) != (size_t)rn) return 2;
            int n = q35s_trace(S, logits, st, ids, lg, pd);
            cases[st]++;
            int why = 0;
            if (st == 5 && P.temp <= 0)        // llama's greedy-by-temp survivor = the one non -inf logit
                for (int j = 0; j < rn; j++) if (rlg[j] > -INFINITY) { ref_surv = rid[j]; break; }
            if (n != rn) why = 1;
            else {
                for (int j = 0; j < rn; j++) mark[rid[j]] = 1;
                int diff = 0; for (int j = 0; j < n; j++) if (!mark[ids[j]]) diff++;
                for (int j = 0; j < rn; j++) mark[rid[j]] = 0;
                if (diff) {
                    if (ties && (tainted || (st >= 2 && st <= 4))) { tiediff[st]++; tainted = 1; }   // ambiguous boundary: compare logit multiset below
                    else why = 2;
                }
                if (!why) {
                    int idset_equal = !diff;
                    if (idset_equal) {      // per-id logit + normalised-probability comparison
                        // map ref logit by id
                        for (int j = 0; j < rn; j++) mark[rid[j]] = j + 1;
                        double mx_m = -INFINITY, mx_r = -INFINITY;
                        for (int j = 0; j < n; j++) { if (lg[j] > mx_m) mx_m = lg[j]; }
                        for (int j = 0; j < rn; j++) { if (rlg[j] > mx_r) mx_r = rlg[j]; }
                        double sm = 0, sr = 0;
                        for (int j = 0; j < n; j++) sm += exp((double)lg[j] - mx_m);
                        for (int j = 0; j < rn; j++) sr += exp((double)rlg[j] - mx_r);
                        for (int j = 0; j < n; j++) {
                            int rj = mark[ids[j]] - 1;
                            double e = relerr(lg[j], rlg[rj]); if (e > wlog[st]) wlog[st] = e;
                            if (e > 0 && st != 5) why = 3;   // logits must be bit-identical except temp's division
                            double pm = exp((double)lg[j] - mx_m) / sm, pr = exp((double)rlg[rj] - mx_r) / sr;
                            double ep = fabs(pm - pr) / (pr > 1e-12 ? pr : 1.0);
                            if (ep > wp[st]) wp[st] = ep;
                            if (ep > 1e-6) why = 4;
                            if (st == 6) {                  // dist: our p vs llama's own p
                                double e2 = fabs(pd[j] - rp[rj]) / (rp[rj] > 1e-12 ? rp[rj] : 1.0);
                                if (e2 > wp[6]) wp[6] = e2;
                                if (e2 > 1e-6) why = 5;
                            }
                        }
                        for (int j = 0; j < rn; j++) mark[rid[j]] = 0;
                    } else {                // tie-ambiguous boundary: compare sorted logit multisets
                        memcpy(mlg_sorted, lg, 4 * (size_t)n); memcpy(rlg_sorted, rlg, 4 * (size_t)rn);
                        qsort(mlg_sorted, (size_t)n, 4, cmpf_desc); qsort(rlg_sorted, (size_t)rn, 4, cmpf_desc);
                        if (memcmp(mlg_sorted, rlg_sorted, 4 * (size_t)n)) why = 6;
                    }
                }
            }
            if (why) { setbad[st]++; bad = 1; printf("  FAIL case %03d [%s] stage %s why=%d (n mine %d ref %d)\n", ci, name, SN[st], why, n, rn); }
        }
        fclose(fr);
        if (P.temp <= 0 && ref_surv >= 0) {
            greedy_n++;
            int g = q35s_sample(S, logits, lg, ids);
            if (g != ref_surv) { greedy_bad++; bad = 1; printf("  FAIL case %03d greedy token %d vs llama %d\n", ci, g, ref_surv); }
        }
        ncase++; if (bad) nfail++;
        q35s_free(S); free(logits); free(CP); free(ids); free(rid); free(mark); free(lg); free(rlg); free(rp); free(pd); free(mlg_sorted); free(rlg_sorted);
    }
    printf("gate1: %d cases, %d failing\n", ncase, nfail);
    printf("%-11s %6s %7s %9s %12s %12s\n", "stage", "cmps", "setbad", "tieamb", "worst_logit", "worst_p");
    for (int s = 0; s < 7; s++) printf("%-11s %6d %7d %9d %12.3g %12.3g\n", SN[s], cases[s], setbad[s], tiediff[s], wlog[s], wp[s]);
    printf("greedy token checks: %d, bad %d\n", greedy_n, greedy_bad);
    return nfail != 0;
}
