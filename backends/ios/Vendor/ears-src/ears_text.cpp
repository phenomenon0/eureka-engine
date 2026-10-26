// ears_text.cpp (G38, EARS v3): the transcript's text for the app. ears_transcribe is untouched (its ids, logits and
// tower rows stay byte-identical to ios-ears / ios-ears-carry); this loads the LLM's vocab only (no weights, no context)
// and renders the greedy ids as llama.cpp's detokenizer does, the EOG step dropped, special tokens kept so the caller
// can cut Qwen3-ASR's "language English<asr_text>" preamble itself. The vocab-only model is kept after the first call
// (keyed by path; a few MB) — reloading it cost ~110 ms a turn on the M3U. Callers serialize (EarsASR's lock).
#include "ears.h"
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>

extern "C" int ears_detokenize(const char *llm, const int32_t *ids, int n_ids, char *out, int cap, char *err) {
  if (cap <= 0) return -1;
  out[0] = 0;
  static std::string cached_path;
  static llama_model *model = nullptr;
  if (!model || cached_path != llm) {
    if (model) llama_model_free(model), model = nullptr;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    mp.n_gpu_layers = 0;
    model = llama_model_load_from_file(llm, mp);
    if (!model) { snprintf(err, 512, "ears: vocab load failed (%s)", llm); return -1; }
    cached_path = llm;
  }
  const llama_vocab *voc = llama_model_get_vocab(model);
  int n = n_ids;
  while (n > 0 && llama_vocab_is_eog(voc, ids[n - 1])) n--;
  std::string text(64 + (size_t)n * 16, '\0');
  int k = llama_detokenize(voc, ids, n, text.data(), (int32_t)text.size(), false, false);
  if (k < 0) { text.resize((size_t)-k); k = llama_detokenize(voc, ids, n, text.data(), (int32_t)text.size(), false, false); }
  if (k < 0) { snprintf(err, 512, "ears: detokenize failed (%d)", k); return -1; }
  text.resize((size_t)k);
  if ((int)text.size() >= cap) { snprintf(err, 512, "ears: text needs %zu bytes, have %d", text.size() + 1, cap); return -1; }
  memcpy(out, text.data(), text.size() + 1);
  return (int)text.size();
}
