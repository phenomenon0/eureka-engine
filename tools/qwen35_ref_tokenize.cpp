// tools/qwen35_ref_tokenize.cpp — llama.cpp-side oracle for the tokenizer gate.
// Same interface as qwen35_toktest, but uses the GGUF's own tokenizer via
// libllama (add_special=false, parse_special=true; remove_special=false,
// unparse_special=true for decode).
// usage: qwen35_ref_tokenize MODEL.gguf encode|decode|string|decode_ids ...
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_ids(const std::vector<llama_token> &v) {
    for (size_t i = 0; i < v.size(); i++) printf("%d%c", v[i], i + 1 == v.size() ? '\n' : ' ');
    if (v.empty()) printf("\n");
}

static std::vector<llama_token> enc(const llama_vocab *vocab, const std::string &text) {
    std::vector<llama_token> toks(text.size() + 256);
    int n = llama_tokenize(vocab, text.data(), (int)text.size(), toks.data(), (int)toks.size(), false, true);
    if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, text.data(), (int)text.size(), toks.data(), (int)toks.size(), false, true); }
    toks.resize(n);
    return toks;
}

static std::string dec(const llama_vocab *vocab, const std::vector<llama_token> &ids) {
    std::vector<char> buf(ids.size() * 16 + 256);
    int n = llama_detokenize(vocab, ids.data(), (int)ids.size(), buf.data(), (int)buf.size(), false, true);
    if (n < 0) { buf.resize(-n + 1); n = llama_detokenize(vocab, ids.data(), (int)ids.size(), buf.data(), (int)buf.size(), false, true); }
    return std::string(buf.data(), n);
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s MODEL.gguf encode|decode|string|decode_ids ...\n", argv[0]); return 2; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.vocab_only = true;
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const char *mode = argv[2];

    if (!strcmp(mode, "string")) {
        print_ids(enc(vocab, argv[3]));
    } else if (!strcmp(mode, "encode") || !strcmp(mode, "decode")) {
        FILE *f = fopen(argv[3], "rb");
        if (!f) { perror("in.bin"); return 1; }
        while (1) {
            uint32_t n;
            if (fread(&n, 4, 1, f) != 1) break;
            if (!strcmp(mode, "encode")) {
                std::string s(n, 0);
                if (fread(&s[0], 1, n, f) != n) break;
                print_ids(enc(vocab, s));
            } else {
                std::vector<llama_token> ids(n);
                if (fread(ids.data(), 4, n, f) != n) break;
                std::string s = dec(vocab, ids);
                uint32_t un = (uint32_t)s.size();
                fwrite(&un, 4, 1, stdout);
                fwrite(s.data(), 1, s.size(), stdout);
            }
        }
        fclose(f);
    } else if (!strcmp(mode, "decode_ids")) {
        std::vector<llama_token> ids;
        for (int i = 3; i < argc; i++) ids.push_back(atoi(argv[i]));
        std::string s = dec(vocab, ids);
        fwrite(s.data(), 1, s.size(), stdout);
    } else { fprintf(stderr, "unknown mode\n"); return 2; }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
