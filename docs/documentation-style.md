# Documentation style

This page defines how documentation in this repository is named and written.
It follows the conventions of the Google developer documentation style guide.

## File and directory names

- Use lowercase words separated by hyphens: `load-gguf-directly.md`.
- Name a guide for the task it describes, in imperative form:
  `build-moe-pack.md`, `load-gguf-directly.md`.
- Name a reference page for the thing it describes, as a noun phrase:
  `http-api.md`, `module-contract.md`.
- Name the main document in a directory `README.md`.
- Do not use lane names, milestone codes, or dates in file names of guides or
  reference pages. Records may carry a milestone code, because the code is how
  receipts and reports refer to them.

## Directory layout

| Directory | Contents |
|---|---|
| `docs/guides/` | Task-oriented procedures. |
| `docs/reference/` | Interfaces, formats, configuration, and limits. |
| `reports/` | Measurement reports that back numbers in the docs. Do not edit. |
| `tests/` | Tests and fixtures. |


## Branch names

Use `<type>/<component>-<subject>`, in lowercase with hyphens. See
[Branches](branches.md) for the types and the map of existing branches.

## Writing

- Write for a reader who knows C, CUDA, and LLM inference, but who has not
  worked in this repository.
- Use second person and present tense: "Run `./build.sh`." Not "We then run".
- Use short declarative sentences. Put one fact in each sentence.
- Use sentence case for headings: "Load a GGUF file directly".
- Start a guide with one or two sentences that state what the guide does,
  then a "Before you begin" section that lists prerequisites.
- Write procedures as numbered steps. Put one action in each step. Put the
  command in a code block after the step.
- Put lists of options, variables, and fields in tables.
- State limits and conditions plainly in a "Limitations" or "Known issues"
  section.
- Do not use metaphors, slogans, rhetorical questions, exclamation points, or
  emphasis for effect. Use bold only for a term on first definition or for a
  warning label such as **Caution:**.
- Do not use internal lane jargon in running text. If a code such as `G16` or
  `W1-SERVER` is the only identifier of a record, put it in parentheses after
  a plain description, once.
- Write "the engine", "the server", "the CLI". Do not personify components.

## Numbers and evidence

- Keep every measured number exactly as recorded. Do not round or restate it.
- After a measured number, cite the report or receipt that holds it, as a
  relative link or path: `(reports/unify1-gates.md)`.
- Give the conditions that a number depends on: model, quantization, GPU,
  prompt length, and the number of rounds.
- Label untested behavior as untested. Do not describe it as working.

## Code blocks

- Use `sh` for shell commands. Do not include a prompt character.
- Use placeholder names in uppercase, such as `MODEL_PATH`, and define each
  one before or after the block.
- Show output in a separate block labeled `text`, if the output matters.
