// ec_unigram.h — 100% Pure C11 HuggingFace Unigram / SentencePiece Tokenizer
// Zero Python, zero SentencePiece C++, zero Rust tokenizers!
// Loads umt5_tokenizer.json (256,300 Unigram pieces) and runs exact Viterbi DP
// segmentation with Metaspace (U+2581) + trailing </s> (id=1) for UMT5-XXL (Wan 2.2).

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ec_unigram ec_unigram_t;

ec_unigram_t *ec_unigram_load(const char *tokenizer_json_path);
void          ec_unigram_free(ec_unigram_t *ug);

// Encodes UTF-8 prompt using exact SentencePiece Metaspace + Viterbi Unigram DP + trailing </s> (id=1).
// Returns number of tokens written (<= max_out), or <0 on error.
int ec_unigram_encode(const ec_unigram_t *ug, const char *text, int32_t *out_ids, int max_out);

#ifdef __cplusplus
}
#endif
