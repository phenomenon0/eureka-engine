// greedy2.c - llama.cpp reference for the Metal MoE lane (CPU only): greedy decoding from prompt ids, or teacher-forced
// scoring of a given continuation. Derived from the any-shape lane's greedy.c.
// Usage: greedy2 MODEL.gguf IDS_FILE N_GEN [--force FORCED_IDS_FILE] [--threads T] [--ctx C] [--gpu]
//   --gpu: llama.cpp on Metal (every layer offloaded): the field's own GPU path, the floor for the CPU reference
//   IDS_FILE / FORCED_IDS_FILE: whitespace-separated token ids. Prints prompt_ids, gen_ids (argmax at each step, or the
//   forced ids when --force) and per step "top K argmax logit second logit" (strict '>': ties to the lowest id).
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void quiet_log(enum ggml_log_level level, const char * text, void * ud) {
    (void) ud;
    if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

static llama_token * read_ids(const char * path, int * n) {
    FILE * f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    int cap = 1024, k = 0; long v;
    llama_token * t = malloc(sizeof(llama_token) * cap);
    while (fscanf(f, "%ld", &v) == 1) {
        if (k == cap) t = realloc(t, sizeof(llama_token) * (cap *= 2));
        t[k++] = (llama_token) v;
    }
    fclose(f);
    *n = k;
    return t;
}

int main(int argc, char ** argv) {
    llama_log_set(quiet_log, NULL);
    if (argc < 4) { fprintf(stderr, "usage: %s MODEL IDS N_GEN [--force F] [--threads T] [--ctx C]\n", argv[0]); return 1; }
    int n_gen = atoi(argv[3]), n_threads = 16, n_ctx_arg = 0, nf = 0, gpu = 0;
    llama_token * forced = NULL;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--force") && i + 1 < argc) forced = read_ids(argv[++i], &nf);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) n_ctx_arg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gpu")) gpu = 1;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (forced && nf < n_gen) n_gen = nf;
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = gpu ? 999 : 0;
    mp.use_extra_bufts = true;
    struct llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    int n_prompt; llama_token * prompt = read_ids(argv[2], &n_prompt);
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx_arg > 0 ? (uint32_t) n_ctx_arg : (uint32_t) (n_prompt + n_gen + 16);
    cp.n_batch = 2048; cp.n_ubatch = 2048;
    cp.n_threads = n_threads; cp.n_threads_batch = n_threads;
    cp.offload_kqv = gpu; cp.op_offload = gpu; cp.no_perf = true;
    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }
    fprintf(stderr, "n_prompt=%d n_ctx=%u n_vocab=%d threads=%d forced=%d\n", n_prompt, llama_n_ctx(ctx), n_vocab, n_threads, nf);
    for (int i = 0; i < n_prompt; i += 2048) {
        int n = n_prompt - i < 2048 ? n_prompt - i : 2048;
        if (llama_decode(ctx, llama_batch_get_one(prompt + i, n))) { fprintf(stderr, "decode failed at %d\n", i); return 1; }
    }
    llama_token * gen = malloc(sizeof(llama_token) * (n_gen + 1));
    int * t1 = malloc(sizeof(int) * (n_gen + 1)), * t2 = malloc(sizeof(int) * (n_gen + 1));
    float * l1 = malloc(sizeof(float) * (n_gen + 1)), * l2 = malloc(sizeof(float) * (n_gen + 1));
    for (int k = 0; k < n_gen; k++) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        int b1 = -1, b2 = -1; float v1 = 0, v2 = 0;
        for (int t = 0; t < n_vocab; t++) {
            float v = lg[t];
            if (b1 < 0 || v > v1) { b2 = b1; v2 = v1; b1 = t; v1 = v; }
            else if (b2 < 0 || v > v2) { b2 = t; v2 = v; }
        }
        t1[k] = b1; t2[k] = b2; l1[k] = v1; l2[k] = v2;
        gen[k] = forced ? forced[k] : b1;
        if (k + 1 < n_gen && llama_decode(ctx, llama_batch_get_one(&gen[k], 1))) { fprintf(stderr, "decode failed at step %d\n", k); return 1; }
    }
    printf("prompt_ids %d", n_prompt); for (int i = 0; i < n_prompt && i < 64; i++) printf(" %d", prompt[i]); printf("\n");
    printf("gen_ids %d", n_gen); for (int i = 0; i < n_gen; i++) printf(" %d", gen[i]); printf("\n");
    for (int k = 0; k < n_gen; k++) printf("top %d %d %.6f %d %.6f\n", k, t1[k], l1[k], t2[k], l2[k]);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
