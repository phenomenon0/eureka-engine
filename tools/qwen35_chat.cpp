// tools/qwen35_chat.cpp — chat template + detokenizer for Gate (c), using the
// system llama.cpp libllama (the GGUF's own tokenizer/template).
// usage:
//   qwen35_chat template MODEL.gguf "user text"      -> prints token ids
//   qwen35_chat detok    MODEL.gguf ID ID ...        -> prints the text
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s template|detok MODEL ...\n", argv[0]); return 2; }
  const char *mode = argv[1], *mpath = argv[2];
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  llama_model *model = llama_model_load_from_file(mpath, mp);
  if (!model) { fprintf(stderr, "load failed\n"); return 1; }
  const llama_vocab *vocab = llama_model_get_vocab(model);

  if (!strcmp(mode, "template")) {
    std::string text = argv[3];
    llama_chat_message msg[1] = {{"user", text.c_str()}};
    std::vector<char> buf(2 * (strlen(text.c_str()) + 256) + 1024);
    int n = llama_chat_apply_template("chatml", msg, 1, true, buf.data(), (int)buf.size());
    if (n < 0) { fprintf(stderr, "template failed\n"); return 1; }
    std::vector<llama_token> toks(text.size() + 256);
    int nt = llama_tokenize(vocab, buf.data(), n, toks.data(), (int)toks.size(), false, true);
    if (nt < 0) { toks.resize(-nt); nt = llama_tokenize(vocab, buf.data(), n, toks.data(), (int)toks.size(), false, true); }
    fprintf(stderr, "prompt_text %s\n", buf.data());
    for (int i = 0; i < nt; i++) printf("%d\n", toks[i]);
  } else if (!strcmp(mode, "detok")) {
    std::vector<llama_token> toks;
    for (int i = 3; i < argc; i++) toks.push_back(atoi(argv[i]));
    std::vector<char> buf(toks.size() * 32 + 256);
    int n = llama_detokenize(vocab, toks.data(), (int)toks.size(), buf.data(), (int)buf.size(), false, true);
    if (n < 0) { buf.resize(-n + 1); n = llama_detokenize(vocab, toks.data(), (int)toks.size(), buf.data(), (int)buf.size(), false, true); }
    fwrite(buf.data(), 1, n, stdout);
    printf("\n");
  } else { fprintf(stderr, "unknown mode %s\n", mode); return 2; }
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
