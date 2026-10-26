// host/llb_toolcall.h — Qwen3.5 tool calling (G1): the chat-template renderer (prompt side)
// and a streaming tool-call / reasoning parser (response side). Plain C11, built on the
// JSON DOM in llb_json.h.
//
// Prompt side (llb_tc_render): a C port of the HF chat template
// (qwen3.5-9b-hf/chat_template.jinja) for an OpenAI-style request
//   {"messages": [...], "tools": [...]}
// rendered byte-identical to transformers' apply_chat_template (jinja2 sandbox, trim_blocks,
// lstrip_blocks, tojson = json.dumps(ensure_ascii=False)). Supported: system/user/assistant/
// tool roles; content as a string, null, or a list of {"type":"text","text":..} items;
// assistant reasoning_content and the in-content <think>...</think> split; the
// multi_step_tool / last_query_index rule (think blocks only after the last real user
// query); assistant tool_calls with or without the "function" wrapper, arguments as an
// object or as a JSON string (OpenAI wire form; parsed first — "" and whitespace-only mean
// {}); the tools system block; `args_value | string` (strings raw, Python str() for
// scalars: True/False/None, ints as digits, floats as Python repr) and tojson for
// containers; |trim = Python str.strip() (Unicode whitespace). Numbers are re-formatted
// the Python way (json.loads then repr: 1.0e2 -> 100.0, -0 -> 0) and duplicate object keys
// collapse like a Python dict (first position, last value), so this module does NOT use
// llb_json_dump's lexeme-faithful numbers. Every raise_exception of the template returns
// NULL with the template's message in err. Unsupported (NULL + err): image/video content
// items, non-object content items, container-valued "text", non-string tool-call names.
//
// Response side (LlbTcParser): feed detokenised bytes in any chunking; pull events.
//   CONTENT    text outside tool calls. A suffix that could start "<tool_call>" (or an
//              incomplete UTF-8 sequence) is held back until disambiguated, so deltas never
//              split a tag or a code point. Raw bytes otherwise (the "\n\n" the template
//              puts before the first call is passed through, as vLLM does).
//   REASONING  text before "</think>" when the parser starts in reasoning mode
//              (llb_tcp_set_reasoning: the Qwen3.5 thinking generation prompt ends with
//              "<think>\n", so the stream begins inside the reasoning). The "\n" run right
//              after "</think>" is dropped (the template writes "\n</think>\n\n"); the
//              reasoning itself is raw (trailing "\n" included). Tool tags inside reasoning
//              are reasoning text. Without set_reasoning, "<think>" is plain content.
//   TOOL_CALL  one completed call: name + arguments as a JSON object string (Python
//              json.dumps default separators ", " ": ", ensure_ascii=False — what vLLM puts
//              in tool_calls[].function.arguments). One event per <function=...> block
//              (a <tool_call> holding two functions yields two calls, as vLLM).
//   ERROR      malformed or unterminated call; the raw bytes of that call are then
//              re-emitted as CONTENT (no byte is ever lost).
// Call syntax (vLLM qwen3coder parser, lenient the same way): inside <tool_call>...
// </tool_call>, one or more "<function=NAME>" blocks; "</function>" may be missing; each
// "<parameter=ARG>" value ends at the first "</parameter>", "<parameter=" or end of the
// function; exactly one leading and one trailing "\n" are stripped from the value.
// Unlike vLLM, non-whitespace text between blocks inside a call is an ERROR (not dropped).
// After a call, whitespace up to the next call / end of stream is dropped (template glue);
// any other text there is CONTENT (vLLM drops it; we keep bytes).
// Value typing by the tool's JSON schema (tools[].function.parameters.properties[ARG].type;
// a type list uses its first non-"null" entry, and if it lists "null" then "null"/"None"
// -> null even for ["string", "null"]; no "type" (anyOf, $ref...) -> string), vLLM semantics:
//   string/str/text/varchar/char/enum  -> JSON string, verbatim (also "null")
//   int*/uint*/long*/short*/unsigned*  -> integer if [+-]?digits (canonicalised), else string
//   num*/float*                         -> number if it parses as a decimal float: integral
//                                          -> integer, else Python float repr; else string
//   boolean/bool/binary                 -> true/false for any-case "true"/"false", else string
//                                          (vLLM yields false here; we keep the text)
//   object/array/arr/dict*/list*        -> parsed JSON, else string (vLLM also tries
//                                          ast.literal_eval; we do not)
//   any other type                      -> parsed JSON, True/False/None, else string
//   non-string types: "null"/"None"    -> null (vLLM maps "null" for every param, strings
//                                          included; we keep strings verbatim)
//   param not in the schema, unknown tool, no tools -> JSON string (vLLM behaviour)
//
// Ownership: llb_tc_render's result is malloc'd (caller frees). Event strings belong to the
// parser and stay valid until the next llb_tcp_next / llb_tcp_free.
#ifndef LLB_TOOLCALL_H
#define LLB_TOOLCALL_H

#include <stddef.h>

// NULL on error (message in err). enable_thinking = 0 renders the empty think block.
char* llb_tc_render(const char* request_json, int add_generation_prompt, int enable_thinking, char* err, int errlen);

typedef enum { LLB_TCE_NONE = 0, LLB_TCE_CONTENT, LLB_TCE_REASONING, LLB_TCE_TOOL_CALL, LLB_TCE_ERROR } LlbTcEvType;

typedef struct {
  LlbTcEvType type;
  const char* text;  // CONTENT / REASONING delta bytes, ERROR message (NUL-terminated too)
  size_t len;        // byte length of text
  const char* name;  // TOOL_CALL function name
  const char* args;  // TOOL_CALL arguments JSON object
  int index;         // TOOL_CALL ordinal in the response (0, 1, ...)
} LlbTcEvent;

typedef struct LlbTcParser LlbTcParser;

// tools_json: the tools array, a request object holding "tools", or NULL. NULL on bad JSON.
LlbTcParser* llb_tcp_new(const char* tools_json);
// start inside a reasoning block (generation prompt ended with "<think>\n"); before feeding
void llb_tcp_set_reasoning(LlbTcParser* p, int in_reasoning);
// returns the number of events waiting in the queue (or -1 after finish)
int llb_tcp_feed(LlbTcParser* p, const char* bytes, int len);
// end of stream: flush held-back text, flag an unterminated call; returns queued events
int llb_tcp_finish(LlbTcParser* p);
// pop the next event: 1 and *ev filled, 0 when the queue is empty
int llb_tcp_next(LlbTcParser* p, LlbTcEvent* ev);
void llb_tcp_free(LlbTcParser* p);

#endif
