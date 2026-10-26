// case_load.h — parse a NNN.args line into Q35Params + history (shared by gate1/gate2).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "qwen35_sampler.h"
typedef struct { Q35Params P; int hist[4096], nh, bid[64]; float bv[64]; } CaseP;
static void parse_case_args(char *line, CaseP *C) {
    Q35Params *Pp = &C->P; q35_params_default(Pp, 0);
    Pp->temp = 1.0f; Pp->repeat_last_n = 64; Pp->seed = 12345;
    C->nh = 0; int nb = 0;
    for (char *tok = strtok(line, " \n"); tok; tok = strtok(NULL, " \n")) {
        char *eq = strchr(tok, '='); *eq = 0; const char *k = tok, *v = eq + 1;
        if (!strcmp(k, "top_k")) Pp->top_k = atoi(v);
        else if (!strcmp(k, "top_p")) Pp->top_p = (float)atof(v);
        else if (!strcmp(k, "min_p")) Pp->min_p = (float)atof(v);
        else if (!strcmp(k, "temp")) Pp->temp = (float)atof(v);
        else if (!strcmp(k, "min_keep")) Pp->min_keep = atoi(v);
        else if (!strcmp(k, "pen")) { sscanf(v, "%d,%f,%f,%f", &Pp->repeat_last_n, &Pp->repeat_penalty, &Pp->frequency, &Pp->presence); }
        else if (!strcmp(k, "hist")) { for (char *s = (char *)v; *s;) { C->hist[C->nh++] = (int)strtol(s, &s, 10); if (*s == ',') s++; } }
        else if (!strcmp(k, "bias")) { for (char *s = (char *)v; *s;) { C->bid[nb] = (int)strtol(s, &s, 10); s++; C->bv[nb++] = (float)strtod(s, &s); if (*s == ',') s++; } }
    }
    Pp->n_bias = nb; Pp->bias_ids = C->bid; Pp->bias_vals = C->bv;
}
static float *load_logits(const char *path, int *V) {
    FILE *fl = fopen(path, "rb"); fseek(fl, 0, SEEK_END); *V = (int)(ftell(fl) / 4); fseek(fl, 0, SEEK_SET);
    float *x = (float *)malloc(4 * (size_t)*V); if (fread(x, 4, (size_t)*V, fl) != (size_t)*V) exit(2); fclose(fl); return x;
}
