#!/usr/bin/env bash
# run27b.sh [--gguf FILE] [--prefill fast|exact] [--spec K] ENGINE-ARGS... - launcher for the Qwen3.8-27B Q4_K_M pack with the gated fast tier ON by default.
# LLB_Q4_DP4A=9 = two-plane dp4a + fused quantizers (q27c final: 25.690 ms/step vs llama.cpp 26.035; gates K1-K5 PASS, 27B K2 20/20).
# Fast prefill is ON by default (unify1: R2 gate PASS, 1,297 / 1,301 tok/s at N = 1009 / 3011); --prefill exact turns it off.
# --spec K = LLB_SPEC=K + LLB_MTP=$LLB_MTP (NextN speculation; unify1: 65.4 tok/s at K=2; default MTP file below). Opt-in: with fast prefill it peaks at 21.4 GB, 25-65 MiB under the 22.5 GB line.
# --gguf FILE = LLB_GGUF=FILE (gguf2: weights straight from the GGUF; the positional dir is then gguf2pack --sidecar-only's).
# Both together: fast prefill, then speculative decode (integ1: the same tokens as plain greedy after the same prefill).
# Scope: 27B ONLY. Mode 9 was never gated on the 9B (only plain dp4a, 18/20); do not use this launcher for other packs.
# Override: LLB_Q4_DP4A=0 ./run27b.sh ... gives the exact tier (31.355 ms/step, byte-fixed-tree).
# Pack: build/gguf2pack MODEL.gguf PACK_DIR (plain C, from ./build.sh; no Python).
# ENG overrides the engine binary (default build/llb-engine, from ./build.sh; the q27c record build was c5-eng).
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
export LLB_Q4_DP4A="${LLB_Q4_DP4A-9}" LLB_PREFILL="${LLB_PREFILL-fast}"
while [ $# -gt 0 ]; do
  case $1 in
    --gguf) export LLB_GGUF=$2; shift 2 ;;
    --prefill) export LLB_PREFILL=$2; shift 2 ;;
    --spec) export LLB_SPEC=$2 LLB_MTP="${LLB_MTP:?--spec needs LLB_MTP, the drafter pack (tools/mtp27/make_drafter.sh)}"; shift 2 ;;
    *) break ;;
  esac
done
exec "${ENG:-${LLB_OUT:-$here/build}/llb-engine}" "$@"
