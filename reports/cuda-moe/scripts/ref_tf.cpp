// ref_tf MODEL IDS(comma) NGPU : llama.cpp teacher-forced top-5 at every position of IDS (logits for all tokens),
// CPU (NGPU 0) or offloaded (NGPU 99). Prints JSON {"top":[[[id,logit] x5] per position]}.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage\n"); return 2; }
    std::vector<llama_token> ids; { std::string s = argv[2]; size_t p = 0; while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); ids.push_back(atoi(s.substr(p, q - p).c_str())); p = q + 1; } }
    int ngpu = atoi(argv[3]);
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = ngpu;
    llama_model * model = llama_model_load_from_file(argv[1], mp); if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    auto cp = llama_context_default_params(); if (ngpu == 0) cp.op_offload = false; cp.n_ctx = 1024; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_threads = 16; cp.n_threads_batch = 16;
    llama_context * ctx = llama_init_from_model(model, cp); if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch bt = llama_batch_init((int)ids.size(), 0, 1);
    for (size_t i = 0; i < ids.size(); i++) { bt.token[i] = ids[i]; bt.pos[i] = (int)i; bt.n_seq_id[i] = 1; bt.seq_id[i][0] = 0; bt.logits[i] = 1; }
    bt.n_tokens = (int)ids.size();
    if (llama_decode(ctx, bt)) { fprintf(stderr, "decode failed\n"); return 1; }
    std::string top = "[";
    for (size_t i = 0; i < ids.size(); i++) {
        const float * L = llama_get_logits_ith(ctx, (int)i); std::vector<int> o(nv); for (int k = 0; k < nv; k++) o[k] = k;
        std::partial_sort(o.begin(), o.begin() + 5, o.end(), [&](int a, int b) { return L[a] > L[b] || (L[a] == L[b] && a < b); });
        char b[256]; std::string t = "["; for (int k = 0; k < 5; k++) { snprintf(b, sizeof b, "%s[%d,%.5f]", k ? "," : "", o[k], L[o[k]]); t += b; } t += "]";
        top += (i ? "," : "") + t;
    }
    printf("{\"top\":%s]}\n", top.c_str());
    llama_batch_free(bt); llama_free(ctx); llama_model_free(model); return 0;
}
