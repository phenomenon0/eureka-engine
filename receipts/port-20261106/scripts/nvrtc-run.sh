#!/usr/bin/env bash
set -uo pipefail
NV=/opt/nv/nvidia/cuda_nvrtc
clang-19 -O1 -I$NV/include /s/nvrtc-check.c -L$NV/lib -l:libnvrtc.so.12 -Wl,-rpath,$NV/lib -o /tmp/nvrtc-check || exit 9
ls $NV/lib
/tmp/nvrtc-check "$1" 7 sm_86
