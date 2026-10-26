# Use structured output in llb-serve

This guide shows you how to turn on constrained decoding in the
OpenAI-compatible server (`tools/sys1dual/serve.c`). With it, you can
constrain output to a JSON schema or a GBNF grammar, and you can make Qwen3.5
tool calls (G16).

The feature is built on two existing parts:

- The grammar mask at the sampler hook (W1-SAMPLER), from the grammar lane
  (G1): `host/llb_grammar.h`.
- The G1 tool-call template port and streaming parser (`host/llb_toolcall.h`).

The feature is off by default. Without `--structured`, the server answers
exactly as the base server (W1-SERVER) does. With `--structured`, a plain request still
takes the base server's path unchanged: the engine gets the same request line
and the client gets the same response. A plain request is one with no
`response_format`, `json_schema`, `grammar`, `tools`, or tool history.

## Before you begin

- Build `llb-serve` and `llb-engine` with `tools/sys1dual/build-serve.sh`.
- Use an engine that contains the grammar mask (the `grammar` field in
  `dual.c`). That is an engine emitted from `g1-structured` or later.
- Have the engine's own `tokenizer.bin` from its sidecar directory.

## Start the server

1. Build the server and the engine:

   ```sh
   OUT=/tmp/x LLB_ENGINE_C=... LLB_ENGINE_GPU=... tools/sys1dual/build-serve.sh
   ```

2. Start the server with `--structured`:

   ```sh
   llb-serve --structured $SYS1_SIDECAR/tokenizer.bin --port 8080 --alias clef \
             --engine /tmp/x/llb-engine -- --threads 16 -- $SYS1_SIDECAR 0 --steps 4096
   ```

   `SYS1_SIDECAR` is the engine's sidecar directory.

3. Check that the log line says `structured=on`.

`--structured TOKENIZER.BIN` turns the feature on and gives the engine its
mask vocabulary. The server exports the path as `LLB_TOKENIZER` to the engine
that it starts. The file must be the engine's own `tokenizer.bin`.

## Send a structured request

1. Request JSON that matches a schema:

   ```sh
   curl -s localhost:8080/v1/chat/completions -H content-type:application/json -d '{
     "messages":[{"role":"user","content":"Profile of Ada Lovelace as JSON."}],
     "chat_template_kwargs":{"enable_thinking":false}, "max_tokens":128,
     "response_format":{"type":"json_schema","json_schema":{"name":"p","schema":{
       "type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},
       "required":["name","age"]}}}}'
   ```

2. Constrain a completion with a GBNF grammar:

   ```sh
   curl -s localhost:8080/v1/completions -H content-type:application/json -d '{
     "prompt":"Is Paris the capital of France? Answer:","max_tokens":8,
     "grammar":"root ::= \" \" (\"yes\" | \"no\") \".\""}'
   ```

3. Force a tool call:

   ```sh
   curl -s localhost:8080/v1/chat/completions -H content-type:application/json -d '{
     "messages":[{"role":"user","content":"Weather in Paris?"}], "tool_choice":"required",
     "tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object",
       "properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},
       "required":["city"]}}}]}'
   ```

4. Send the tool's result back. Add the assistant message with its
   `tool_calls`, then add `{"role":"tool","tool_call_id":...,"content":...}`.
   The same template port renders the tool history. The rendering is
   byte-identical to the Hugging Face Jinja template.

## Request fields

| Field | Endpoint | Meaning |
|---|---|---|
| `response_format: {"type":"json_schema","json_schema":{"schema":S}}` | chat, completions | The output is an instance of S. |
| `response_format: {"type":"json_object"}` | chat, completions | The output is a JSON object (`{"type":"object"}`). |
| `response_format: {"type":"text"}` | both | No constraint. The request is a plain request. |
| `json_schema: S` | both | llama.cpp-server shorthand for the `json_schema` response format. |
| `grammar: "GBNF"` | both | llama.cpp GBNF, passed through verbatim. `root` is the start rule. |
| `tools: [...]` | chat | Puts the Qwen3.5 tool template in the prompt. The server parses calls into `tool_calls`. |
| `tool_choice` | chat | `auto` (default): unconstrained, parsed. `none`: tools are left out of the prompt. `required`: the grammar forces one or more calls. `{"type":"function","function":{"name":N}}`: forces exactly one call of N. |
| `parallel_tool_calls: false` | chat | With `required`, forces exactly one call. |
| `chat_template_kwargs.enable_thinking` | chat | Default true. Reasoning is free text up to `</think>` and is returned as `reasoning_content`. The constraint applies after it. |

Rules for combining fields:

- Send at most one of `response_format`, `json_schema`, and `grammar`.
- Do not send any of them with `tools`.
- `tools: []` means no tools. OpenAI clients send it.
- `functions` (deprecated) returns a 400 error.

### Supported JSON schema keywords

The schema converter is `llb_gram_schema_to_gbnf`. It supports these
keywords:

- Types: object, array, string, number, integer, boolean, null, and type
  lists.
- `properties`, `required`, `enum`, `const`, `anyOf`, `oneOf`, local `$ref`.
- `minItems`, `maxItems`, `minLength`, `maxLength`, `additionalProperties`.
- `{}`, which means any value.

A bare `{"type":"object"}` means any object. Before G16, only `{}` meant any
object.

The converter ignores the annotations `title`, `description`, and `$schema`.

Any other validation keyword, such as `pattern`, `format`, or `minimum`,
returns a 400 error. The server never widens the schema silently.

There is one exception. A tool parameter schema is used only for
`tool_choice` `required` or a named function. In that schema, a parameter
with an unsupported keyword is constrained to its bare type, and the server
logs a line. The tool still works, but the server does not enforce the
keyword.

## Response format

| Item | Behavior |
|---|---|
| Stop at end of generation | For a structured request, the server ends the response at the first `<\|im_end\|>` or `<\|endoftext\|>` id. The engine still runs `max_tokens` steps, and the server discards the extra steps. `usage.completion_tokens` counts up to and including the end-of-generation id. |
| `finish_reason` | `tool_calls` if the server parsed any call. Otherwise `stop` at end of generation. Otherwise `length`. A `length` finish under a grammar means the instance may be incomplete. Set `max_tokens` high enough. |
| Chat message | `content` is null when there are only calls. `reasoning_content` is present when thinking is on. `tool_calls` is `[{"id","type":"function","function":{"name","arguments": JSON string}}]`. |
| Argument types | The tool's parameter schema sets the argument types, with vLLM qwen3coder semantics. `"7"` becomes `7` for an integer parameter. Arrays and objects are parsed. Strings are kept raw. |
| Streaming (`stream: true`) | The server sends `content` and `reasoning_content` deltas as text arrives. Each tool call is one delta, `{"tool_calls":[{"index":i,"id",...,"function":{"name","arguments"}}]}`, sent whole once its `</function>` is seen. There are no partial-argument deltas. The server holds back partial tags and split UTF-8, so the deltas concatenate to exactly the non-stream message. This is tested at every split. |
| Malformed call | The server logs it and keeps its raw text as content. |
| Errors | 400 `invalid_request_error` with the converter's or compiler's message, for example `json_schema: unsupported keyword 'pattern'` or `grammar: ...`. The server checks the request before the engine sees it. |
| Errors without `--structured` | The structured fields return a 400 error that names the flag. `tools` returns the base server's own 400 error. |

## How it works

The engine sees one GBNF text in the existing `grammar` field of `dual.c`.
The engine is unchanged from G1.

The request side is `tools/sys1dual/serve_structured.c`. It does these
things:

- Converts schemas with the G1 converter, passes GBNF through, composes the
  result, and compiles it once with the G1 compiler, so that errors are a
  clean 400.
- With thinking on, it builds `root ::= st-think-0 "\n"{0,2} u-root`.
  `st-think` is a KMP automaton for "any text, then `</think>`". The user's
  rules are renamed `u-*`.
- For `tool_choice` `required` or a named function, it builds a grammar of the
  template's exact call form:
  `<tool_call>\n<function=N>\n(<parameter=P>\nVALUE\n</parameter>\n)*</function>\n</tool_call>`.
  - Parameters appear in declared order. Parameters that are not required are
    optional.
  - String values are raw text that cannot contain the parser's delimiters
    (`st-str`).
  - String enums are literals.
  - Other types use the G1 JSON grammar for the parameter schema.
  - Several calls are joined by `\n`.
- With tools or tool history, it renders the prompt with `llb_tc_render`, the
  chat-template port, instead of the base server's plain renderer.

The response side is `tools/sys1dual/serve.c`. It handles the
end-of-generation stop, runs `LlbTcParser` for reasoning and calls, and builds
the OpenAI message and chunk shapes.

## Tests

| Suite | What it tests | Command |
|---|---|---|
| `tools/sys1dual/st_test.c` | Request side: grammar composition against the G1 engine on a byte vocabulary, the forced-call grammar matching the template's rendering, parse-back, and every 400 error. | `OUT=/tmp/b tools/sys1dual/build-structured.sh && /tmp/b/st_test unit` |
| `tools/sys1dual/structured_tests.py` | End to end over HTTP against `fake_engine.c`, which implements the `dual.c` stream protocol with the real tokenizer and the real G1 mask, and no model. Covers flag-on equal to flag-off goldens (responses and engine lines), schemas, thinking, tools in auto, required, named, and history modes, malformed calls, seeded off-script fuzz, stream equal to non-stream for every case, and the 400 surface. Needs `jsonschema`. | `python3 tools/sys1dual/structured_tests.py --bin /tmp/b --tokenizer $SYS1_SIDECAR/tokenizer.bin --work /tmp/w` |
| `tools/toolcall/run_tests.py` and the G1 grammar suite | The G1 gates, unchanged. | See `tools/toolcall/` and [the grammar README](../../tools/grammar/README.md). |

To build the same tests with ASan and UBSan, set `SAN=1` on
`build-structured.sh`.

## Limitations

- Grammars do not work with speculative decoding (`LLB_SPEC`). The engine
  refuses the combination (G1).
- The cold mask inside long free strings costs about 7 ms once per new stack
  state (G1, `host/llb_grammar.h`).
- Undeclared keys under `additionalProperties` are not kept apart from
  declared names.
- Tool-call arguments do not stream incrementally. Each call is one delta.
- Prompt text is not NFC-normalized (G1 tokenizer audit). Decomposed input
  tokenizes differently from Hugging Face.
- `tool_choice: auto` is unconstrained. The server returns a malformed call
  from the model as content.
- The forced-call grammar guarantees the form of a call: the right function,
  the required parameters, valid typed values, and a parseable result. It
  does not guarantee sensible string contents. String values are free text
  minus the parser's delimiters, so `</x>`-style markup stays legal. HTML and
  code arguments need this. On Clef-flash, which is a decide model and not a
  tool caller, `tool_choice: required` gave `{"city": "Paris</单位>celsius"}`.
  The remedy is a tool-tuned model, not a narrower grammar.
- The template's inner `\n` characters are optional in the forced grammar,
  because the G1 parser strips them. Requiring them forced junk output on a
  model that writes `Paris</parameter>` (first GPU run).
