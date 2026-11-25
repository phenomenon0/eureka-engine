// ec_bpe.h — 100% Pure C11 HuggingFace tokenizer.json Byte-Level BPE Tokenizer
// Zero Python, zero Rust/tokenizers crate. Powers Qwen3-4B (FLUX.2-klein-4B)
// and Qwen3-0.6B (ACE-Step 1.5) natively inside eureka-engine/creative/.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ec_bpe ec_bpe_t;

ec_bpe_t *ec_bpe_load(const char *tokenizer_json_path);
void      ec_bpe_free(ec_bpe_t *tok);

// Encode UTF-8 text into token IDs (splits special tokens first, then GPT-2 byte-level BPE).
// Returns number of tokens written (<= max_out), or <0 on error.
int ec_bpe_encode(const ec_bpe_t *tok, const char *text, int32_t *out_ids, int max_out);

// Encode prompt with the exact FLUX.2-klein-4B Qwen3 chat template (enable_thinking=False),
// right-padded with <|endoftext|> (151643) to fixed_len (512).
// Writes real token count into *out_real_len and attention mask (1=real, 0=pad) into out_mask.
int ec_bpe_encode_flux2_qwen3(
    const ec_bpe_t *tok, const char *prompt,
    int32_t *out_ids, uint8_t *out_mask, int fixed_len, int *out_real_len);

// Encode raw prompt for Gemma-3 (LTX-2.3): Replace ' ' -> '▁' (U+2581), SentencePiece BPE with
// <0xXX> byte fallback, and prepend <bos> (id=2). Returns token count (<= max_out), or <0 on error.
int ec_bpe_encode_gemma3(
    const ec_bpe_t *tok, const char *prompt,
    int32_t *out_ids, int max_out);

#ifdef __cplusplus
}
#endif
