# Load a GGUF file directly

This guide shows you how to make the GPU engine read its weights from a GGUF
file instead of from a pack. You set one environment variable and point the
engine at a small sidecar directory.

The final report is [`reports/gguf2-final.md`](../../reports/gguf2-final.md).

## Before you begin

- Build `llb-engine` with `build.sh`.
- Build the `gguf2pack` converter.
- Use a model shape that `mf.validate` accepts: the 9B or the 27B.
- Keep the GGUF file on a readable disk. The engine reads it at every load.

## Load a GGUF file

1. Write the sidecar directory from the GGUF header. This takes about 1 s and
   reads 11.5 MB.

   ```sh
   gguf2pack MODEL.gguf --sidecar-only DIR
   ```

   Add `--sha` to also write the sha256 fields:

   ```sh
   gguf2pack MODEL.gguf --sidecar-only DIR --sha
   ```

   Replace `MODEL.gguf` with the path to the GGUF file and `DIR` with the
   output directory. The command writes `manifest.json` and `table.bin`.

2. Set `LLB_GGUF` to the GGUF path and run the engine with `DIR` as the
   positional pack directory.

   ```sh
   LLB_GGUF=MODEL.gguf llb-engine ... -- DIR ...
   ```

3. Optional: check every loaded tensor against a pack manifest. This is a
   debug mode.

   ```sh
   LLB_GGUF=MODEL.gguf LLB_GGUF_VERIFY=PACK/manifest.json llb-engine ... -- DIR ...
   ```

   Replace `PACK` with a pack directory for the same model. The engine exits
   with status 1 on a mismatch.

## How the loader works

- `engine_qwen35_gpu.bend` changes by +17/-8 lines. It adds one `IO.get_env`
  in `mf.model`. A Bool is threaded into `ws.layers`, `ws.layer`, and the two
  head loads.
- When `LLB_GGUF` is unset, the engine makes the same `ws.load` calls as
  before. With the flag off, the token streams are identical to the base
  receipts (9B Q4_K_M, 9B Q8_0, 27B mode 9).
- `gguf.bend` and `host/llb_gguf.c` implement a Bend user effect,
  `Gguf.read(which, off, n)`. `which` is a layer number N, or `0xFFFFFFFF` for
  a virtual `pack.bin` range.
- The change needs no compiler patch. It does not edit `llb_host.c` or
  `build.sh`.
- The C code opens the GGUF once and parses only the header. It derives the
  pack offset of every tensor (64-aligned, in GGUF order).
- The C code fills the region with 8 pthreads (`LLB_GGUF_THREADS`). It uses
  plain `pread` for tensors that are not Q8_0. For Q8_0 tensors, it re-lays
  whole rows to the pack's Q8_0 row split (`[cols int8][cols/32 f16]`).
- The block goes into the Corpus through the runtime's `io_put`.
- If the region size differs from the GGUF's own layout, for example because
  the sidecar is stale, the load fails with an error.
- `table.bin` in the sidecar is byte-identical to the pack's. `manifest.json`
  is identical except for `source.file` and the sha256 fields. Without
  `--sha`, the sha256 fields are empty. With `--sha`, the file is
  byte-identical (checked on Kev-4B).
- `LLB_GGUF_VERIFY` reads every loaded tensor back from the device
  (`cuMemcpyDtoH`) and checks its sha256. It is off by default. It is not on
  any timed path.

## Options

| Variable or flag | Default | Description |
|---|---|---|
| `LLB_GGUF=PATH` | Unset | Load weights from the GGUF at `PATH`. Unset means the pack path. |
| `LLB_GGUF_THREADS` | 8 | Number of pthreads that fill a region. |
| `LLB_GGUF_VERIFY=PACK/manifest.json` | Off | Read every loaded tensor back from the device, check its sha256, and exit 1 on a mismatch. |
| `gguf2pack --sidecar-only DIR` | Not applicable | Write `manifest.json` and `table.bin` from the GGUF header. |
| `gguf2pack --sha` | Off | Fill the sha256 fields of the sidecar manifest. |

## Tools

| File | Description |
|---|---|
| [`tools/gguf2/unit.c`](../../tools/gguf2/unit.c) | CPU loader test with 3 mutations. Build with `-DGG_CUDA` for a device round trip. |
| [`tools/gguf2/sidecar_cmp.sh`](../../tools/gguf2/sidecar_cmp.sh) | Compares a sidecar with a pack. |
| [`tools/gguf2/cache.py`](../../tools/gguf2/cache.py) | Evicts or warms the page cache. |

## Measured results

The verdicts are in [`reports/gguf2/s4cmp.txt`](../../reports/gguf2/s4cmp.txt).

All gates pass:

- Device bytes equal the pack manifest: 9B Q4_K_M 427/427, 27B 851/851, and
  Kev-4B 428/428 through `unit-dev`.
- Token streams and logits are identical: 9B Q8_0 s8/s64 with stream sha
  `5f07a4ec32f1`; 9B Q4_K_M 12/68/83/523 lines; 27B mode 9 68 lines, equal to
  q27c `1-fz`.

The load time is the engine's "resident head ... in N ms" line.

| Load | GGUF | Pack path |
|---|---|---|
| 9B Q4_K_M cold | 5.1 s | 248 s (head on an HDD) |
| 9B Q4_K_M warm | 1.2 s | 2.9 s ; 20.5 / 39.6 s fresh, because the HDD's FUSE `pack.bin` does not stay cached |
| 9B Q8_0 cold | 8.5 s (NVMe2, ntfs-3g) | 5.7 s (`/home`, btrfs NVMe0). The GGUF is slower cold, on a different disk. |
| 9B Q8_0 warm | 2.5 s | 4.5 s |
| 27B Q4_K_M cold | 15.4 s | 317 s (head on an HDD); 77.8 s fresh with its HDD head pre-read |

Host VmHWM of the engine process is 0.5-1.3 GB for the GGUF path and
0.9-1.2 GB for the pack path. The loader's transient buffer is one region
(<= 1.04 GB, the 27B `output.weight`) plus 8 x 16 MiB scratch. The plan
predicted < 200 MB.

## Limitations

- The Mac and Metal follow-up is not done. The Metal engine still loads packs.
  Nothing here was built or run on Metal.
- The pack path stays the default and the oracle. The CPU twin
  (`tools/qwen35_engine.c`), the KAT and bench tools, and the Python verify
  tools still read `pack.bin` and `layers/`.
- Only the shapes that `mf.validate` accepts (9B, 27B) work. Kev-4B is
  checked at the loader level only. The engine refuses its shapes.
- There is no `--gguf` CLI flag. The switch is the environment variable
  (`IO.get_env`), so the CLI parser is unchanged.
- A warm 27B GGUF load was not measured. The file is 19 GB under a 12 GB unit
  on a loaded machine. Its second run (18.7 s) is another cold read.

## Planned work

1. Remove the sidecar. Parse `config` and the tensor table in Bend, or have
   the effect return them, so that `LLB_GGUF` needs no directory. The sidecar
   takes about 1 s and two small files, but it is still a step.
2. Port to Metal. Point the Metal engine's dense loads at the same
   `llb_gguf.c`. The file uses only libc and pthreads; only the readback in
   `gguf_verify` is CUDA. Gate it against its packs the same way (G1 device
   bytes, G2 streams).
3. Stream into the Corpus. Write each 16 MiB unit straight into the device
   block, instead of one region-sized malloc plus `io_put`. This removes the
   transient host buffer of up to 1 GB and overlaps the read with the upload.
   It needs the Corpus write off the loop thread (risk 1 in the plan).
