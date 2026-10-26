// tools/qwen35_tokenizer.h — the Qwen3.5 (GPT-2 byte-level) BPE tokenizer.
//
// Vocabulary comes from the GGUF metadata, extracted into <packdir>/tokenizer.bin
// by tools/vocab_extract_qwen35.py. The algorithm is a faithful port of
// llama.cpp's BPE path: the Qwen3.5 pre-tokenizer regex (custom splitter),
// GPT-2 byte encoding, rank-ordered merges (min-heap by rank then left index),
// the special-token partition and byte fallback.
#ifndef QWEN35_TOKENIZER_H
#define QWEN35_TOKENIZER_H

#include <stdint.h>

typedef struct {
    unsigned char *buf;      // token text blob
    uint32_t *off;           // token -> offset in buf
    uint16_t *len;           // token -> byte length
    uint8_t  *type;          // token -> GGUF token_type (1 normal,3 control,4 user,5 unused)
    int32_t  *text_to_id;    // open-addressing hash: index -> id, -1 empty
    uint32_t tok_mask;

    unsigned char *mbuf;     // merge text blob
    uint32_t *ml_off; uint16_t *ml_len;
    uint32_t *mr_off; uint16_t *mr_len;
    int32_t  *rank;          // hash: index -> merge rank, -1 empty
    uint32_t hash_mask;

    int32_t *special;        // ids with control/user_defined/unknown attrs, len desc
    int      n_special;

    int n_vocab;
    int n_merges;
    int max_token_len;
    int bos, eos, pad, add_bos;
    int mask_id;             // -1
} Q35Tokenizer;

int  q35_tok_load(Q35Tokenizer *t, const char *dir);
void q35_tok_free(Q35Tokenizer *t);

// Encode `text` (NUL-terminated UTF-8). `parse_special` splits out control /
// user-defined / unknown tokens before BPE (llama.cpp semantics). Returns the
// number of ids written, or -needed if out_cap is too small.
int  q35_tok_encode(Q35Tokenizer *t, const char *text, int parse_special, int *out, int out_cap);

// Decode one token to bytes (special=true semantics); returns byte count written.
int  q35_tok_piece(Q35Tokenizer *t, int id, char *out, int out_cap);
const char *q35_tok_text(Q35Tokenizer *t, int id, int *len);

// Decode a whole sequence (special=true) into `out`; returns bytes (may exceed
// out_cap, in which case only out_cap bytes are written).
int  q35_tok_decode(Q35Tokenizer *t, const int *ids, int n, char *out, int out_cap);

#endif
