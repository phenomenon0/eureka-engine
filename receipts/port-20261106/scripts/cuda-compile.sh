#!/usr/bin/env bash
# compile-only check of an emitted CUDA engine C (tools/remap_cc.sh's flags, CUDA 12 headers from NVIDIA's pip wheels;
# no link: there is no libcuda here). $1 = the emitted C, $2 = the repo (for tools/pkg2)
set -uo pipefail
I=(-I/opt/nv/nvidia/cuda_runtime/include -I/opt/nv/nvidia/cuda_nvrtc/include)
clang-19 --version | head -1
clang-19 -include "$2/tools/pkg2/glibc234.h" -include "$2/tools/pkg2/cu12.h" -DBEND_CUDA=1 "${I[@]}" -std=c11 -O3 -ffp-contract=off \
  -c "$1" -o "/tmp/$(basename "$1" .c).o"; rc=$?
echo "compile $(basename "$1") rc=$rc"; [ $rc = 0 ] && nm "/tmp/$(basename "$1" .c).o" | grep -c ' [Tt] ' | sed 's/^/text symbols: /'
nm "/tmp/$(basename "$1" .c).o" 2>/dev/null | grep -E 'llb_sk8_(launch|fits|enabled)' ; exit $rc
