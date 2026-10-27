# WebGPU trunk: any Qwen3.5 model, format, and context length

The browser trunk engine (`backends/webgpu/trunk/`) used to take Qwen3.5
shapes with head size 256, delta-net state 128, conv 4, dense FFN only, five
weight formats, and a context fixed by `TRUNK_CAPACITIES` (at most 4,096)
with a 16,384-token prompt cap. It now takes:

- every dense size and the MoE models (`qwen35moe`: softmax top-k routing,
  renormalized, plus the sigmoid-gated shared expert; separate or fused
  `gate_up` experts);
- head size, state size, and conv width read from the file;
- 21 GGUF formats (F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K..Q6_K,
  IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS);
- any context whose f16 KV buffer for one full-attention layer fits the GPU's
  binding limit, with split-KV attention over 256-token chunks.

Machines: RTX 3090 with Chrome (Vulkan/ANGLE), and an M3 Ultra with Chrome
155 (Metal). Reference: llama.cpp `9d57ce45`, CPU, greedy.

## Results

| Receipt | What it shows |
|---|---|
| `qcheck.txt` | `trunk/qcheck.html` vs gguf-py's dequantizers (`tools/qref.py`), 16 rows a tensor, 30 tensors over 21 formats: every dequantized value equal; GEMV relative error ≤ 2.6e-6. |
| `vs-llamacpp.txt` | Greedy ids vs llama.cpp, 4 prompts x 64 tokens on 0.8B/2B/4B/9B Q4_K_M and the `sky` prompt on 0.8B in every format. At each first divergence, llama.cpp's own top-2 gap and where the trunk's pick ranks in llama.cpp's top 5. |
| `synthetic.txt` | Random-weight files: head 128 / state 64 / conv 3 / 2 KV heads, and MoE 16 experts / 4 used with a shared expert (separate and fused): 24/24 greedy ids equal to llama.cpp. IQ1_S, IQ1_M, IQ2_XXS against llama.cpp on the same weights dequantized to F32: equal (32/32, 1/1, 32/32). |
| `old-vs-new-ids.txt` | The engine before this change vs after, on the four sizes it already ran: same ids 16/16. |
| `ab-decode.txt` | Decode speed, old vs new, interleaved, 3 runs each: 0.8B 105.5 vs 104.1 tok/s, 2B 54.5 vs 54.0 tok/s (medians). `ab-decode-before-direct-dispatch.txt` is the same A/B before attention chunks were dispatched directly at small capacities (an 11% loss on the 0.8B). |
| `ctx-scaling-3090-0.8b.txt` | `[position, ms/token in prefill, ms for one decode step]` every 8,192 tokens of a 160k-capacity 0.8B: 6.1 ms/token at 8k, 9.7 at 74k. |
| `long-context.txt` | Needle prompts on the 0.8B, RTX 3090: found at 20,973 and 153,352 tokens. |
| `mac-chrome.txt` | Chrome on the M3 Ultra (Metal): 0.8B 168.7 tok/s, 2B 106.7, 4B 56.4, clef-flash 9B 36.1; IQ1_M, IQ2_XXS, IQ3_S, Q5_1, BF16; the synthetic dense and MoE files. Needles found at 20,973 tokens on the 2B (95.5 tok/s decode after) and 153,352 tokens on the 0.8B (115.9 tok/s prefill, 59 tok/s decode after). |
| `ctx-scaling-m3-2b.txt` | The same scaling run on the M3 Ultra, 2B: 7.4 ms/token at 4k, 8.1 at 33k. |

## Reading the llama.cpp comparison

llama.cpp's CPU path quantizes activations to 8 bits for quantized matmuls;
the trunk keeps them in f32. Where the two split, llama.cpp's top two logits
were within 0.007 to 0.17 of each other and the trunk chose llama.cpp's #2 or
#3. On the lowest-bit 0.8B files (IQ1_S, IQ1_M, IQ2_XXS) the split comes at
the first token; with the same weights dequantized to F32 (exact f32 in
llama.cpp too) the outputs are equal, so the difference is activation
rounding, not the kernels.

## Not covered

- No real MoE checkpoint was run (the smallest, 35B-A3B, is about 20 GB);
  the MoE graph is checked on random-weight files only.
- The 27B was not run in the browser.
- Safari: WebDriver automation is off on the test Mac. Chrome on Metal was run.
- Prefill is one decode-graph pass a token (in 64-token submissions), so long
  prompts take minutes: 153k tokens on the 0.8B took 26 minutes on the RTX 3090
  and 22 minutes on the M3 Ultra.

## SUMMARY

The WebGPU trunk runs any Qwen3.5 shape (dense or MoE), 21 GGUF formats, and
contexts bounded by the GPU's binding limit. The kernels equal the reference
dequantizers, real models agree with llama.cpp up to near ties, the old shapes
produce the same ids at the same speed, and a 153k-token needle was found on
both the RTX 3090 and the M3 Ultra.
