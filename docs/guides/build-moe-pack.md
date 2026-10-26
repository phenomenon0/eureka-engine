# Build a Qwen 35B-A3B pack with gguf2pack

This guide shows you how to convert the Qwen 35B-A3B MoE GGUF into a pack with
the C converter `gguf2pack`. The conversion needs no Python and no prebuilt
pack.

Tested on Linux, 2026-10-06. The converter builds the registered
`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (`qwen35moe`) into the same 44 files as the
moe2b reference pack.

## Before you begin

- Get `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`. The input is 22,360,456,160 bytes.
- Keep the original GGUF after conversion. The engine loads the routed
  experts from it.
- Use an absolute path to the GGUF, and keep that file accessible.
- Have at least 3 GB free in the output filesystem for one build. Keep the
  input next to it.
- Use an output directory that does not contain an existing pack.
- Have a C11 compiler (`cc`), or use the release converter.

## Build the pack

1. From the source checkout, create the build directory.

   ```sh
   mkdir -p build
   ```

2. Compile the converter.

   ```sh
   cc -std=c11 -O2 tools/gguf2pack.c -o build/gguf2pack
   ```

3. Run the converter.

   ```sh
   ./build/gguf2pack /absolute/path/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf /absolute/path/pack-35b
   ```

   With the release converter, run it directly:

   ```sh
   ./gguf2pack MODEL.gguf OUT_DIR
   ```

   Replace `MODEL.gguf` with the absolute GGUF path and `OUT_DIR` with the
   output directory.

4. Optional: inspect the GGUF header without converting.

   ```sh
   ./gguf2pack MODEL.gguf --probe
   ```

## Options

| Flag | MoE behavior |
|---|---|
| `--format quant` | Optional. The converter detects `qwen35moe` without it. |
| `--probe` | Inspects the GGUF header. |
| `--sidecar-only` | Rejected for MoE. Dense only. |
| `--sha` | Rejected for MoE. Dense only. |
| `--head-only` | Rejected for MoE. Dense only. |
| `--verify-against` | Rejected for MoE. Dense only. |
| `--format q8` | Rejected for MoE. Dense only. |

The MoE conversion rejects the dense-only flags on purpose. Existing dense
commands are unchanged.

## Output files

The generated files total 2,678,327,538 bytes (2.678 GB decimal).

| File | Bytes | Contents |
|---|---:|---|
| `pack.bin` | 1,080,696,832 | Output head, output norm, and token embedding, in that order. |
| `layers/blk.N.bin` (30 layers) | 41,771,776 each | Linear-attention trunk, shared expert, router. |
| `layers/blk.N.bin` (N = 3,7,...,39) | 34,433,024 each | Full-attention trunk, shared expert, router. |
| `table.bin` | 5,120 | 40 rows of 32 little-endian u32 words. The router and gate are at words 21 and 22. |
| `experts.idx` | 2,644 for the registered source path | GGUF path, per-layer expert offsets, format IDs, and per-expert bytes. |
| `manifest.json` | 139,422 for the registered source path | Config, tensor layout and hashes, head hash, and virtual layer offsets. |

The converter writes the files as follows:

- It rearranges Q8_0 rows to int8 values followed by f16 scales.
- It copies F32 bytes verbatim.
- It renames shared-expert tensors to the dense `ffn_gate`, `ffn_up`, and
  `ffn_down` aliases.
- Layer bytes exist only in their layer files. The manifest's layer offsets
  are virtual positions after the head region.
- The MoE format has no `index.tsv`.
- The converter streams weights with bounded buffers. It never copies the
  routed-expert pool.

The source path appears in line 1 of `experts.idx` and in
`manifest.source.file`. If you change the source path, those two values
change, and so do the sizes and hashes of the two files. The spelling of the
path does not affect any other output byte.

## Measured results

- Two Linux builds took 28.80 s and 25.51 s, including tracing, under
  `env -i PATH=/usr/bin:/bin`. These are observations on one machine. They are
  not a Mac estimate or a promised time. They exclude hashing and comparison
  time.
- G1 and G2: both full builds match all 44 reference files by SHA-256 and
  `cmp`.
- G3: the 27B and 9B Q4_K_M sidecars match their stored references exactly.
- G4: the release-candidate binary has no Python invocation strings and no
  process launch imports. It ran with only `PATH` in its environment.
- G5: the write receipts and limitations are in the final report.

See [`reports/pack35-final.md`](../../reports/pack35-final.md).

## Limitations

- Other 35B quantizations are not tested. The stored heads, trunk, and shared
  weights must be Q8_0 or F32. The routed experts use the registered
  256-element blocks.
- The engine-level load is not tested. The unified Linux build has no MoE
  support yet. Equality to the reference pack carries over the evidence from
  the moe2b gates.
- This work claims no new inference timing, no Linux MoE capability, and no
  Mac release validation.
- Not tested: whether the Mac `run.sh` overlay rewrites `experts.idx` to name
  the user's GGUF path.
