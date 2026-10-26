#!/usr/bin/env bash
# build vision/build/cudavis-cli (FUS-A: the 9B tower, D 72): host C -ffp-contract=off, kernels -fmad=false, sm_86.
# The loader is this tree's own host/llb_gguf.c (mmproj section, V1); vmm.c includes it.
set -euo pipefail; cd "$(dirname "$0")"; mkdir -p build
CF="-std=c11 -D_GNU_SOURCE -O3 -ffp-contract=off -Wall -Wno-unused-function -I. -I../host"
gcc $CF -c vpre.c -o build/vpre.o -w
gcc $CF -c vmm.c -o build/vmm.o
gcc $CF -c cudavis_cli.c -o build/cli.o
nvcc -O3 -fmad=false -arch=sm_86 -std=c++17 -Xcompiler -ffp-contract=off -I. -I../host -c vitc.cu -o build/vitc.o
nvcc -arch=sm_86 build/cli.o build/vitc.o build/vpre.o build/vmm.o -lcublas -lm -lpthread -o build/cudavis-cli
ls -la build/cudavis-cli
