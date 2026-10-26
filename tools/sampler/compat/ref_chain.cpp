// tools/sampler2/ref_chain.cpp — the ORACLE for the sampler2 gates: builds llama.cpp's
// own samplers (libllama) and applies them stage by stage on a llama_token_data_array.
// No model.  usage: ref_chain LOGITS.f32 OUT.bin key=val...
//   keys: bias=id:b,id:b  hist=id,id,..  pen=last_n,rep,freq,pres  top_k= top_p= min_p=
//         temp= min_keep= seed=
// OUT.bin = records per applied stage:
//   int32 stage, int32 n, int32 ids[n], float logit[n], float p_llama[n]
// stage ids: 0 bias 1 penalties 2 top_k 3 top_p 4 min_p 5 temp 6 dist.  Disabled stages are
// skipped (llama's "empty" samplers).  A real llama_sampler_chain with the same stages is
// also run; its post-chain candidates must equal the staged result (checked here, exit 3).
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<std::string> split(const std::string &s, char c) {
    std::vector<std::string> r; size_t i = 0;
    while (i <= s.size()) { size_t j = s.find(c, i); if (j == std::string::npos) j = s.size(); if (j > i) r.push_back(s.substr(i, j - i)); i = j + 1; }
    return r;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage\n"); return 2; }
    FILE *f = fopen(argv[1], "rb"); if (!f) { perror("logits"); return 2; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    int V = (int)(sz / 4);
    std::vector<float> lg(V); if (fread(lg.data(), 4, V, f) != (size_t)V) return 2; fclose(f);

    std::vector<llama_logit_bias> bias; std::vector<llama_token> hist;
    int last_n = 64; float rep = 1, fr = 0, pr = 0, top_p = 1, min_p = 0, temp = 1; int top_k = 0, min_keep = 0;
    uint32_t seed = 1;
    for (int a = 3; a < argc; a++) {
        std::string kv = argv[a]; size_t e = kv.find('='); std::string k = kv.substr(0, e), v = kv.substr(e + 1);
        if (k == "bias") for (auto &t : split(v, ',')) { auto p = split(t, ':'); bias.push_back({(llama_token)atoi(p[0].c_str()), (float)atof(p[1].c_str())}); }
        else if (k == "hist") for (auto &t : split(v, ',')) hist.push_back(atoi(t.c_str()));
        else if (k == "pen") { auto p = split(v, ','); last_n = atoi(p[0].c_str()); rep = atof(p[1].c_str()); fr = atof(p[2].c_str()); pr = atof(p[3].c_str()); }
        else if (k == "top_k") top_k = atoi(v.c_str());
        else if (k == "top_p") top_p = atof(v.c_str());
        else if (k == "min_p") min_p = atof(v.c_str());
        else if (k == "temp") temp = atof(v.c_str());
        else if (k == "min_keep") min_keep = atoi(v.c_str());
        else if (k == "seed") seed = (uint32_t)atoll(v.c_str());
        else { fprintf(stderr, "bad key %s\n", k.c_str()); return 2; }
    }

    std::vector<llama_sampler *> st; std::vector<int> sid;
    auto add = [&](int id, llama_sampler *s) { st.push_back(s); sid.push_back(id); };
    if (!bias.empty()) add(0, llama_sampler_init_logit_bias(V, bias.size(), bias.data()));
    llama_sampler *pen = llama_sampler_init_penalties(V, last_n, rep, fr, pr);
    add(1, pen);
    add(2, llama_sampler_init_top_k(top_k));
    add(3, llama_sampler_init_top_p(top_p, min_keep));
    add(4, llama_sampler_init_min_p(min_p, min_keep));
    add(5, llama_sampler_init_temp_ext(temp, 0.0f, 1.0f));
    add(6, llama_sampler_init_dist(seed));
    // same stages in a real chain (separate instances)
    llama_sampler *chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler *cpen = llama_sampler_init_penalties(V, last_n, rep, fr, pr);
    if (!bias.empty()) llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(V, bias.size(), bias.data()));
    llama_sampler_chain_add(chain, cpen);
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, min_keep));
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(min_p, min_keep));
    llama_sampler_chain_add(chain, llama_sampler_init_temp_ext(temp, 0.0f, 1.0f));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    for (auto t : hist) { llama_sampler_accept(pen, t); llama_sampler_accept(chain, t); }

    auto mk = [&](std::vector<llama_token_data> &d) { d.resize(V); for (int i = 0; i < V; i++) d[i] = {i, lg[i], 0.0f}; };
    std::vector<llama_token_data> d, d2; mk(d); mk(d2);
    llama_token_data_array arr = {d.data(), (size_t)V, -1, false}, arr2 = {d2.data(), (size_t)V, -1, false};

    FILE *o = fopen(argv[2], "wb");
    for (size_t i = 0; i < st.size(); i++) {
        llama_sampler_apply(st[i], &arr);
        int32_t hdr[2] = {sid[i], (int32_t)arr.size};
        fwrite(hdr, 4, 2, o);
        for (size_t j = 0; j < arr.size; j++) fwrite(&arr.data[j].id, 4, 1, o);
        for (size_t j = 0; j < arr.size; j++) fwrite(&arr.data[j].logit, 4, 1, o);
        for (size_t j = 0; j < arr.size; j++) fwrite(&arr.data[j].p, 4, 1, o);
    }
    fclose(o);
    llama_sampler_apply(chain, &arr2);
    // chain vs staged: same candidate count and same ids/logits at every position
    int rc = 0;
    if (arr2.size != arr.size) rc = 3;
    else for (size_t j = 0; j < arr.size; j++)
        if (arr2.data[j].id != arr.data[j].id || !(arr2.data[j].logit == arr.data[j].logit || (std::isinf(arr.data[j].logit) && std::isinf(arr2.data[j].logit)))) { rc = 3; break; }
    if (rc) fprintf(stderr, "staged != chain\n");
    return rc;
}
