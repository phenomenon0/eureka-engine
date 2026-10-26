// kevhead.h: the Kev-4B decide head for dualm.c (KEV-HEAD-METAL). The contract is tools/kevserve/kev.c
// (llama-bend-nullseat-gpucache): render (assemble.py), the cls.output readout (kevhead.c) and the answer.
// Only the request side lives here; the forward is the engine's exact pass with the hidden hook on (one pass per question).
#pragma once
#include "clef3/json.h"
#include "qwen35_tokenizer.h"
#include <stddef.h>

// what the GGUF carries: 1 = Kev head (<arch>.decision.type "kev" + cls.output.{weight,bias}), 0 = none.
// why (if non-NULL) gets a one-line reason when the answer is 0 but a Kev key or tensor is present.
int kev_head_probe(const char *gguf, char *why, size_t why_n);
// load cls.output + the three temperatures + the five render markers; exits on a malformed head (clef_fail).
void kev_head_open(const char *gguf, Q35Tokenizer *tok);
// one decide request -> the question queue. NULL = queued (kev_count() passes), else the refusal text.
const char *kev_request(const jv *req, unsigned ctx);
int kev_count(void);
const int *kev_ids(int qi, int *n);
// read question qi's hidden rows (the hook's file) and append "qid":{answer} to ans (comma-separated after the first)
void kev_answer(int qi, const char *hidden_path, sb *ans);
// the stderr record: every question's "raw <key> s=.. p=.." line, as a JSON object (qid -> line)
const char *kev_raw(void);
int kev_tokens(void);
void kev_drop(void);
