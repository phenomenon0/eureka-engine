#!/bin/bash
# mk_synth.sh: the synthetic MoE shapes (gen_moe.py, F32) quantized (llama-quantize); the F32 file deleted after
set -e
cd ~/scratch/metal-catchup; . ./paths.sh; Q=$HOME/llama.cpp/build/bin/llama-quantize
gen() { PYTHONPATH=$HOME/llama.cpp/gguf-py:$PWD/pystub python3 gen_moe.py $M08 models/$1-F32.gguf "$2"
  for q in $3; do $Q models/$1-F32.gguf models/$1-$q.gguf $q > runs/q-$1-$q.log 2>&1; done; rm -f models/$1-F32.gguf; }
gen s128x8b d=1024,L=4,nh=8,nkv=2,hk=8,hv=16,nexp=128,kexp=8,ffe=768,ffs=1024,seed=11 Q4_K_M
gen s60x6 d=1024,L=8,nh=8,nkv=2,hk=8,hv=16,nexp=60,kexp=6,ffe=768,ffs=512,seed=12 Q4_K_M
gen s40x5f d=1536,L=4,nh=12,nkv=4,hk=8,hv=16,nexp=40,kexp=5,ffe=512,ffs=1536,fused=1,seed=13 "Q4_K_M Q8_0"
