// dump_misc: `dump_misc presets` prints q35_params_default for modes 0..2; `dump_misc lp F32 N` prints
// top-N logprobs of a logits file.  Checked against presets.json / numpy by check_misc.py.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "case_load.h"
int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "presets")) {
        for (int m = 0; m < 3; m++) { Q35Params p; q35_params_default(&p, m);
            printf("%d %.9g %.9g %d %.9g %.9g %.9g\n", m, p.temp, p.top_p, p.top_k, p.min_p, p.presence, p.repeat_penalty); }
        return 0;
    }
    int V; float *x = load_logits(argv[2], &V); int n = atoi(argv[3]);
    int ids[64]; float lp[64]; q35s_topn_logprobs(x, V, n, ids, lp);
    for (int i = 0; i < n; i++) printf("%d %.9g\n", ids[i], lp[i]);
    return 0;
}
