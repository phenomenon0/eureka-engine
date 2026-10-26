// serve_structured.h — G16: structured output for llb-serve (--structured). The request
// side: response_format / json_schema / grammar / tools -> the prompt and ONE GBNF grammar
// sent to the engine as dual.c's "grammar" field (G1's mask, host/llb_grammar.h). The
// response side (EOG stop, reasoning + tool-call parsing) lives in serve.c on top of G1's
// streaming parser (host/llb_toolcall.h). Docs: docs/guides/structured-output.md.
#pragma once
#include "../clef3/json.h"

// Qwen3.5 end-of-generation ids (<|endoftext|>, <|im_end|>): the ids G1's llb_gv_load marks
// EOG, i.e. the only ones a finished grammar allows. The server stops the response at the
// first one (the engine itself still runs max_tokens steps).
#define ST_EOG_A 248044
#define ST_EOG_B 248046

typedef struct {
    int on;        // a structured request: EOG stop + (chat) the reasoning/tool-call parser
    int reasoning; // the generation prompt ends inside <think> (parse reasoning_content)
    int tools;     // tool calls are parsed into tool_calls
    int forced;    // tool_choice required / named: the grammar forces >= 1 call
    char *gbnf;    // the constraint (malloc'd), NULL = unconstrained
    char *tools_json; // the tools array (malloc'd JSON) for the parser's value typing
    char *prompt;  // the rendered chat prompt when the tool template was used (else NULL)
} StReq;

// Classify a request; with the server flag off every structured field is a 400 (tools keep
// W1-SERVER's message). Returns NULL, or an error message (a 400). chat: the chat endpoint.
// thinking / agp: the chat template's enable_thinking / add_generation_prompt.
const char *st_request(const jv *body, int chat, int thinking, int agp, int flag_on, StReq *s);
void st_free(StReq *s);

// building blocks (exported for tests/st_test.c)
char *st_gbnf_prefix(const char *g, const char *pfx); // every rule name -> pfx+name
void st_gbnf_until(sb *o, const char *name, const char *marker); // text without marker, then marker
void st_gbnf_value(sb *o, const char *name); // a raw tool-parameter value as the parser delimits it
char *st_think_wrap(const char *gbnf);                // reasoning, "</think>", then the grammar
char *st_tools_gbnf(const jv *tools, const char *only, int multi, const char **err);
