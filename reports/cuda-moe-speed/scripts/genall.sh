#!/usr/bin/env bash
# genall.sh : regenerate the gate files (seeded, the commands of reports/cuda-moe) + S35x8, on $BIG (a second disk); sha256 each
S=$HOME/worktrees/.scratch-cuda-moe-speed; M=${BIG:?a scratch dir for the big files}; L=$S/receipts/gen
mkdir -p $L; export TMPDIR=$S/tmp PYTHONPATH=$HOME/llama.cpp/gguf-py
cap() { systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=0 "$@"; }
src=${SRC08B:-$HOME/worktrees/.scratch-web/models/q08b.gguf}  # any Qwen3.5-0.8B GGUF: only its tokenizer is read
gen() { t=$1; shift; [ -f $M/$t-q8.gguf ] || cap python3 $S/scripts/old/gen_moe.py $src $M/$t-q8.gguf "$@" > $L/gen-$t.txt 2>&1; sha256sum $M/$t-q8.gguf >> $L/sha256.txt; }
qz() { t=$1 o=$2 ty=$3; shift 3; [ -f $M/$t-$o.gguf ] || cap $HOME/llama.cpp/build/bin/llama-quantize --allow-requantize "$@" $M/$t-q8.gguf $M/$t-$o.gguf $ty > $L/quant-$t-$o.log 2>&1; sha256sum $M/$t-$o.gguf >> $L/sha256.txt; }
gen s60 d=1024 L=4 ne=60 k=6 ef=768 sf=512 nh=8 nkv=2 lkh=8 lvh=16 V=0; qz s60 q4km Q4_K_M
gen s128f d=1024 L=4 ne=128 k=8 ef=768 sf=768 nh=8 nkv=2 lkh=8 lvh=16 V=8192 fused=1 mtp=1; qz s128f q4km Q4_K_M
gen s512 d=512 L=4 ne=512 k=10 ef=256 sf=256 nh=4 nkv=1 lkh=4 lvh=8 V=8192; qz s512 q5km Q5_K_M --token-embedding-type q6_k
gen s35x8 d=2048 L=8 ne=256 k=8 ef=512 sf=512 nh=16 nkv=2 lkh=16 lvh=32 V=8192; qz s35x8 q4km Q4_K_M
echo done >> $L/sha256.txt
