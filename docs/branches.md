# Branches

This page defines the branch naming convention for this repository.

## Naming convention

Use the form `<type>/<component>-<subject>`:

- Use lowercase letters, digits, and hyphens only.
- Use one `/`, after the type.
- Keep the subject to three words or fewer. Describe the change.
- Do not put dates in the subject.

### Types

| Type | Use |
|---|---|
| `feature` | New behavior. |
| `fix` | A correction to existing behavior. |
| `perf` | A performance change that keeps output within its registered tolerance. |
| `docs` | Documentation only. |
| `release` | A release candidate or packaging change. |
| `research` | Measurement, probes, and plans. Not intended to merge. |
| `pins` | Content records that the build reads. Never rename. |

### Components

Use one of these components where it applies: `build`, `cuda`, `metal`,
`decode`, `prefill`, `spec`, `quant`, `moe`, `gguf`, `pack`, `tok`, `sampler`,
`serve`, `batch`, `kv`, `clef`, `vision`, `video`, `tts`, `ear`, `cli`.

## Fixed branches

| Branch | Role |
|---|---|
| `main` | Default branch. |
| `pins/bend-240ecef4` | Content record of the pinned Bend compiler. `build.sh` reads this name. |
