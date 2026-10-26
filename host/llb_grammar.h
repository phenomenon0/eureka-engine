// host/llb_grammar.h — grammar-constrained decoding (G1): a GBNF grammar (llama.cpp syntax)
// or a JSON-schema subset compiled to one, and a token-mask applier for the sampler hook.
//
// Grammar: llama.cpp's GBNF (grammars/README.md): rules `name ::= alt | alt`, "literals"
// with \n \r \t \\ \" \[ \] \xHH \uHHHH \UHHHHHHHH escapes, [classes] with ranges and ^,
// `.`, ( groups ), * + ? {m} {m,} {m,n}, # comments; a top-level rule ends at a newline,
// nested ones may span lines. Left recursion is rejected at compile time (as llama.cpp).
// Not supported: token references <[id]> (rejected).
//
// Matching is byte-level: every char class is compiled into the UTF-8 byte-range sequences
// of its code points (Russ Cox / regex-syntax Utf8Sequences), so a token that ends inside a
// multi-byte character needs no special case. The parse state is a set of pushdown stacks
// (llama.cpp's representation); stacks and stack-sets are interned and the byte transition
// set x byte -> set is memoised, so a full vocab mask is one pass over a DFS-ordered byte
// trie of the vocabulary that prunes whole subtrees at the first dead byte. Masks are cached
// per stack-set (string interiors repeat the same set every token).
//
// Token semantics follow llama.cpp's llama_grammar_apply: a token's text is its piece with
// special = true (control / user-defined tokens match as their literal text), an empty piece
// or one starting with NUL is never allowed, and the end-of-generation ids are allowed only
// when the grammar can end here (accepting state); accepting one ends the match.
//
// JSON schema (llb_gram_schema_to_gbnf): the subset type object / array / string / number /
// integer / boolean / null (also a list of types), properties + required (properties in
// declared order, the non-required ones optional; additionalProperties only when given as
// true or a schema, as llama.cpp's json_schema_to_grammar, but additional keys are not kept
// apart from the declared ones), items, minItems / maxItems,
// minLength / maxLength, enum, const, anyOf / oneOf, local $ref (#/$defs/.., #/definitions/..),
// and {} / true = any JSON value. Whitespace placement and the string / number terminals
// are llama.cpp's current converter's (space ::= | " " | "\n"{1,2} [ \t]{0,20} after "{" "["
// "," and keys and before "}" "]"; none after a value). Other validation keywords (pattern,
// format, minimum, ...) are an error, never a silent widening; annotations are ignored.
#ifndef LLB_GRAMMAR_H
#define LLB_GRAMMAR_H

#include <stdint.h>

typedef struct LlbGram LlbGram;
// Compile GBNF text; root = the start rule ("root" when NULL). NULL on error (message in err).
LlbGram* llb_gram_from_gbnf(const char* src, const char* root, char* err, int errlen);
void llb_gram_free(LlbGram* g);
// JSON schema (text) -> malloc'd GBNF text with start rule "root"; NULL on error.
char* llb_gram_schema_to_gbnf(const char* schema_json, char* err, int errlen);

// The vocabulary as byte strings (what the mask walks).
typedef struct LlbGVocab LlbGVocab;
// from the pack's tokenizer.bin (tools/vocab_extract_qwen35.py layout, Q35TOK01)
LlbGVocab* llb_gv_load(const char* tokenizer_bin, char* err, int errlen);
// from caller-provided pieces (tests, other tokenizers)
LlbGVocab* llb_gv_new(int n_vocab, const unsigned char* const* bytes, const int* lens);
// end-of-generation ids (Qwen: <|im_end|> 248046, <|endoftext|> 248044; llb_gv_load sets
// these two when the vocabulary has them)
void llb_gv_set_eog(LlbGVocab* v, const int* ids, int n);
int llb_gv_n_vocab(const LlbGVocab* v);
int llb_gv_piece(const LlbGVocab* v, int id, const unsigned char** bytes);   // byte length
void llb_gv_free(LlbGVocab* v);

// A match in progress (one per request).
typedef struct LlbGMatch LlbGMatch;
LlbGMatch* llb_gm_new(const LlbGram* g, const LlbGVocab* v);
void llb_gm_free(LlbGMatch* m);
void llb_gm_reset(LlbGMatch* m);
// Allowed-token bitmask over n words = (n_logits + 31) / 32 (bit t of word t/32); ids at or
// past the vocabulary are never allowed. Returns the number of allowed ids (0 = dead end).
int llb_gm_mask(LlbGMatch* m, uint32_t* bits, int n_logits);
// logits[t] = -INFINITY where bit t is clear (the applier; a bias stage ahead of the chain)
void llb_gm_apply(const uint32_t* bits, float* logits, int n_logits);
// 1 iff tok is allowed now (one token's walk, no mask): the greedy fast path
int llb_gm_allows(LlbGMatch* m, int tok);
// advance by tok: 0, or -1 if tok is not allowed (state unchanged)
int llb_gm_accept(LlbGMatch* m, int tok);
// advance by raw bytes (tests, prefilled assistant text): 0 or -1 (state unchanged)
int llb_gm_accept_bytes(LlbGMatch* m, const char* bytes, int len);
int llb_gm_can_end(const LlbGMatch* m);   // the grammar may end here
int llb_gm_done(const LlbGMatch* m);      // an end-of-generation token was accepted

#endif
