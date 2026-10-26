# Structured output and tool calls

`tools/grammar/` holds the tests for the structured output modules: grammar
and JSON-schema constraints, and Qwen tool calls (G1). The modules are in
`host/`. See also [Structured output](../../docs/guides/structured-output.md).

## Files

The modules are plain C11 in the house style of `host/llb_sampler*`. `X.h` is
the API, `X_impl.h` holds the definitions that the engine inlines, and `X.c`
is the standalone object.

| Module | Description |
|---|---|
| `host/llb_grammar.h` | GBNF compiler (llama.cpp syntax), byte-level pushdown matcher, vocabulary-trie token mask, and JSON schema to GBNF conversion. |
| `host/llb_toolcall.h` | Qwen3.5 chat template with tools (prompt side, byte-identical to the HF jinja template), and a streaming `<tool_call>` / `<think>` parser. |
| `host/llb_json.h` | The JSON DOM that both modules use. |
| `host/llb_structured.c` | All three as one object. Link this file. Do not link `llb_grammar.c` and `llb_toolcall.c` together. |
| `gram_test.c` | Test driver. The modes are listed in its header. |

## How the engine applies a grammar

The hook is in `host/llb_host.c`, at the sampler's `smp_pick` (W1-SAMPLER).

- From the first generated position on, the engine masks the vocabulary row
  that the sampler already copies from device to host. It works on a copy and
  sets `-inf` on disallowed ids.
- The mask runs before the sampler chain. This is the grammar-first order of
  llama.cpp: a bias stage ahead of
  `bias → penalties → top_k → top_p → min_p → temp → dist`.
- The match advances by the picked token.
- Logprobs and the logit written back stay those of the raw model.
- Greedy fast path: if the raw argmax is allowed, it is the masked argmax.
  This takes one token walk and no mask.
- The sampler module itself is unchanged. It is reused, not forked.

## Constrain output

1. Write a GBNF grammar file or a JSON schema file.
2. Set the constraint, the first constrained position, and the tokenizer.

   ```sh
   LLB_GRAMMAR=GRAMMAR.gbnf LLB_GRAMMAR_AT=PROMPT_TOKENS LLB_TOKENIZER=TOKENIZER_BIN ENGINE_COMMAND
   ```

   Use `LLB_JSON_SCHEMA=SCHEMA.json` instead of `LLB_GRAMMAR` for a JSON
   schema. `PROMPT_TOKENS` is the prompt's token count. `TOKENIZER_BIN` is the
   `tokenizer.bin` of the pack or of the Clef sidecar. `ENGINE_COMMAND` is the
   engine command line.

For per-request use, call `llb_grammar_configure` from C, or send the SYS1
request fields listed in the next section.

## Options

| Option | Description |
|---|---|
| `LLB_GRAMMAR=file.gbnf` / `LLB_JSON_SCHEMA=file.json` | The constraint (environment, CLI runs). |
| `LLB_GRAMMAR_AT=n` | The prompt's token count, which is the first constrained position. Required with the two variables above, because the host never sees the prompt length. |
| `LLB_TOKENIZER=…/tokenizer.bin` | The `tokenizer.bin` of the pack or of the Clef sidecar. Gives the token pieces for the mask. |
| `int llb_grammar_configure(kind, text, at)` | Per-request C ABI. kind is `"gbnf"` or `"json_schema"`. NULL turns the constraint off. |
| SYS1 request fields | `"json_schema":TEXT` or `"grammar":GBNF`, plus `"sample":SPEC` (sent as the sampler's `--sample` word). See [`tools/sys1dual/dual.c`](../sys1dual/dual.c). |

The constraint is off by default. With none of these set, `gr_on()` is false,
`smp_on()` is exactly the sampler's expression, and no host code path
changes.

## Token rules

- A token's text is its piece with special = true. Control and user-defined
  tokens match as their literal text, as in the llama.cpp grammar.
- Empty pieces, pieces that start with NUL, and unused padding rows are never
  allowed. For Qwen3.5, the padding rows are ids 248077–248319.
- `<|im_end|>` and `<|endoftext|>` are allowed only when the grammar can end.
  Accepting one ends the match. After that, only EOG ids are allowed.

## Tests

| Test | What it checks |
|---|---|
| `gram_test.c` | Modes are listed in its header. |
| `runs/gap-chase/G1/scripts/grammar_suite.py` | Differential tests against the llama.cpp `test-gbnf-validator` and `json_schema_to_grammar.py`, and jsonschema soundness. |
| `gram_test maskcheck` | The trie mask equals a per-token brute force. |
| [`tools/toolcall/run_tests.py`](../toolcall/run_tests.py) | Tool calls: render parity against jinja2, streaming round trips at every split, and adversarial streams. |

The receipts are in `runs/gap-chase/G1/receipts/`.

## Limitations

- A grammar is refused with `LLB_SPEC`. A spec cycle verifies T rows at once,
  and the mask is per accepted token.
- A grammar is refused on the unfolded head op, as `LLB_SAMPLE` is.
- `LLB_GRAMMAR_AT` is required for CLI runs, because the host never sees the
  prompt length.
