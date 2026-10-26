// host/llb_structured.c: G1 structured output as ONE standalone object for a server or test
// (the grammar mask + JSON-schema converter, the Qwen tool-call renderer/parser and the JSON
// DOM they share). Link this instead of llb_grammar.c + llb_toolcall.c together: those each
// carry llb_json_impl.h, whose functions are external, so linking both duplicates them. The
// engine inlines the _impl.h headers instead.
#include "llb_grammar_impl.h"
#include "llb_toolcall_impl.h"
