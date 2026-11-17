# Hybrid expert placement (design)

This page specifies a fourth residency for the routed experts of a MoE model:
the experts that are used most stay in VRAM, and the rest are computed on the
CPU from RAM while the GPU works on the cached ones. It is a design. The code
does not implement it yet. Each section names the function it changes.

The idea comes from Strata's Qwen3.6-35B-A3B engine
([docs/QWEN36.md](https://github.com/Niko1221/Strata/blob/main/docs/QWEN36.md)),
which reports 64 tokens/s on an RTX 3060 12 GB with 37 of 256 experts per layer
in VRAM, against 41 for llama.cpp's best CPU/GPU split.

## Why

Today `llb_moe_load` (`host/llb_host.c`) picks one of three residencies for all
experts of a model ([Limits](../limits.md)):

| Mode | Where the experts are | Who computes them | Measured |
|---|---|---|---|
| 0, device | VRAM | GPU | 35B-A3B Q4_K_S, RTX 3090: 137 tokens/s |
| 1, host | Pinned RAM, mapped | GPU, over PCIe | Synthetic 35B-A3B-like: 305 tokens/s, bound by about 60 MB of reads a token |
| 2, stream | The mmapped GGUF | GPU, after a copy of the k chosen experts | 35B-A3B Q4_K_S, RTX 3090: about 1 token/s |

A model whose experts miss the free VRAM by any amount drops to mode 1 or 2.
On a 24 GB card with the Windows desktop open, a 35B-A3B Q4_K_S misses by
0.1 GiB. On 8, 12 and 16 GB cards every 35B-A3B misses by far. Mode 1 needs
the experts plus 4 GiB of available RAM in pinned memory. Mode 2 copies
about 60 MB a token over PCIe and is bound by it.

At decode, a token reads k of the ne experts of a layer (8 of 256 for the
35B-A3B). Routing is skewed, so a cache of the most-used experts serves most
of the reads. A missed expert is one row-block GEMV per matrix (gate, up and
down) on one activation vector. A CPU does that from RAM at RAM bandwidth,
with no PCIe copy.

## Mode 3, hybrid

### Placement at load

`llb_moe_load` gets a mode 3, chosen when mode 0 does not fit and the GGUF can
be memory-mapped:

1. Keep every expert stack mapped from the GGUF (as mode 2 does: `moe_map`).
2. Size a VRAM cache: `C = (free VRAM - rv - KV at full context) / (layers x expert bytes)`
   slots a layer, at most `ne`.
3. Fill each layer's slots with experts 0..C-1, and keep a slot table per
   layer: `slot[layer][expert]`, -1 when the expert is not resident.
4. Start a CPU worker pool (`LLB_MOE_CPU_THREADS`, default physical cores
   less one) with a Q4_K / Q5_K / Q6_K / Q8_0 GEMV of its own. The CPU
   engine's kernels (`backends/windows/engine_cpu.c`, `tools/qwen35_engine.c`)
   are the starting point.

### A token's MoE layer

`llb_moe_run` changes after the top-k launch (`gpu_moetk`):

1. Read back the k chosen ids and weights (`llb_read` of `idp`, `wtp`: 2 x k
   words) and the normed activation `h2b` (`d_model` floats). This is one
   synchronization a layer, about 10 us on PCIe 4.
2. Split the k experts into hits (resident) and misses.
3. Launch the GPU part on the hits only: `gpu_moegu` and `gpu_moedn` take the
   slot ids, with the misses' weights set to 0.
4. At the same time, the CPU pool computes each miss: gate, up, SiLU and down
   from the mapped GGUF, scaled by its router weight and summed into one
   `d_model` vector.
5. Copy that vector to the device (`cuMemcpyHtoDAsync`) and add it in the
   combine step (`mx_fcomb` / the `gpu_moedn` epilogue takes one more addend).

The CPU's misses and the GPU's hits overlap. A layer costs about
`max(GPU hits, CPU misses) + sync`, not their sum.

### Keeping the cache warm

Count the hits of each expert in a decay-weighted histogram per layer. Every N
tokens (`LLB_MOE_REBALANCE`, default 256) swap the coldest resident expert
for the hottest missed one. The swap is a `cuMemcpyHtoDAsync` of one expert's
three stacks on a second stream, so it overlaps decode.

### Prefill

A prompt chunk reads most experts once per chunk, so a copy over PCIe
amortizes over the chunk's tokens. At prefill, send the missed experts that
the chunk uses to the GPU through the mode-2 staging slots, grouped by expert,
and let the CPU take only the experts the chunk uses for one or two tokens.
Strata measures 2,022 prompt tokens/s this way on an RTX 3060 at chunk 2048.

### The adaptive split

Measure the CPU's GEMV throughput and the PCIe bandwidth at load (one expert
each way). At decode, when the CPU's queue for a layer would outlast the GPU's
hits, send the cheapest misses over PCIe instead (the mode-2 path for those
experts only).

## Exactness

The CPU GEMV rounds differently from the GPU kernels. A token can differ from
mode 0's after a near tie. Keep the existing modes the default where they
fit, and make mode 3's numbers comparable:

- `LLB_MOE_EXPERTS=hybrid` forces mode 3; `auto` takes it only where mode 0
  does not fit.
- `LLB_MOE_HYBRID_GPU_ONLY=1` sends every miss over PCIe instead of to the
  CPU. That gives mode 0's bits and isolates the cost of placement from the
  cost of CPU rounding.
- Compare against mode 0 with the KL and top-1 agreement method of
  [reports/cuda-moe](../../reports/cuda-moe/REPORT.md).

## Backends

| Backend | What changes |
|---|---|
| CUDA, Linux and Windows | `llb_moe_load` and `llb_moe_run` in `host/llb_host.c`, as above. |
| Metal | Nothing needed: unified memory holds every expert where the GPU reads it. |
| WebGPU | The same split with the browser's worker threads (WASM SIMD GEMV) for misses and a `GPUBuffer` cache. The storage-buffer limit (often 2 to 4 GB) makes the cache the only way to run a 35B-A3B there. |
| CPU | Not applicable: every expert is already on the CPU. |

## Measurements to take

Run each with the same GGUF, prompt, context and greedy decoding, on 8, 12, 16
and 24 GB cards:

1. Mode 0 where it fits, as the reference.
2. Mode 3 with misses on the CPU.
3. Mode 3 with misses over PCIe (`LLB_MOE_HYBRID_GPU_ONLY=1`).
4. Mode 3 adaptive.
5. llama.cpp with its best `-ncmoe` split.

Report decode tokens/s, prompt tokens/s at 512 and 2048, the cache hit rate,
and the KL and top-1 agreement against mode 0. This separates the gain from
placement, from concurrent execution, and from speculative decoding, which is
a separate change.
