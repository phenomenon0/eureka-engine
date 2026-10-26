#!/usr/bin/env bash
# remap_cc.sh <in.c> <out>: the engine's host build (the headprobe recipe). GLIBC <= 2.34 (tools/pkg2/glibc234.h),
# DT_RPATH $ORIGIN (a bundled libnvrtc next to the binary wins; RUNPATH would not reach libnvrtc.so.13's own dlopen of
# libnvrtc-builtins, as that library has no RUNPATH); CUDA 12 headers add tools/pkg2/cu12.h.
# LLB_CC_EXTRA: extra objects/libs to link (build.sh sets the request objects; tools/sys1dual/build.sh its own).
set -euo pipefail
C=${CUDA_HOME:-/usr/local/cuda} t=$(cd "$(dirname "$0")" && pwd)/pkg2
inc=(-include "$t/glibc234.h")
[ "$(awk '/#define CUDA_VERSION/ { print $3; exit }' "$C/include/cuda.h")" -ge 13000 ] || inc+=(-include "$t/cu12.h")
${CC:-clang} "${inc[@]}" -DBEND_CUDA=1 -I"$C/include" -L"$C/lib64" -std=c11 -O3 -ffp-contract=off \
  "$1" ${LLB_CC_EXTRA:-} -lpthread -lm -lcuda -lnvrtc -Wl,--disable-new-dtags,-rpath,'$ORIGIN' -o "$2"
