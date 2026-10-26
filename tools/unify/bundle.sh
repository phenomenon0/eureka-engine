#!/usr/bin/env bash
# bundle.sh VERSION BUILD_DIR: release/llb-linux-x86_64-cuda-VERSION/ + .tar.gz + SHA256SUMS (dist PACKAGING.md naming).
# Ships: llb (the front end, tools/llb/llb) + defaults.conf + models.conf, llb-tok (its C helper, tools/llb/llb-tok.c:
# GGUF info, tokenizer.bin extract, enc/dec, ChatML, run wrapper), llb-engine (binary; its NVRTC kernel source is
# llb-engine.c, embedded by #embed __FILE__), llb-engine.c + build.sh (source path), gguf2pack (+ .c), llb-toktest
# (tools/qwen35_toktest.c, the ids-workflow tokenizer tool; was named llb-tok in 0.1.0), run.sh (the raw ids launcher),
# README.md (release/README-cuda.md). No Python. The .gpu kernel cache is NOT shipped (built per GPU on first run).
# pkg2: + llb-probe (tools/llb/llb-probe.c) and NVRTC from CUDA_HOME (libnvrtc.so.N + its builtins, found through
# llb-engine's DT_RPATH $ORIGIN); C helpers built with tools/pkg2/glibc234.h (GLIBC <= 2.34). CUDA_HOME with CUDA 12
# headers (BUILD_DIR's engine built against them, tools/remap_cc.sh) -> llb-linux-x86_64-cuda12-VERSION, probe floor 12.2.
set -euo pipefail
cd "$(dirname "$0")/../.."
C=${CUDA_HOME:-/usr/local/cuda} I="-include tools/pkg2/glibc234.h" fl="-DLLB_FLOOR=13000 -DLLB_FLOOR_R=580" k=cuda
[ "$(awk '/#define CUDA_VERSION/ { print $3; exit }' "$C/include/cuda.h")" -ge 13000 ] || { fl="-DLLB_FLOOR=12020 -DLLB_FLOOR_R=535" k=cuda12; }
v=$1 b=$2 n=llb-linux-x86_64-$k-$1; d=release/$n
rm -rf $d; mkdir -p $d/tools/llb
cp $b/llb-engine $b/llb-engine.c $b/gguf2pack tools/gguf2pack.c $d/
for l in "$C"/lib64/libnvrtc.so.[0-9]* "$C"/lib64/libnvrtc-builtins.so.[0-9]*; do  # each soname once, as a plain file
  [ -L "$l" ] && continue; so=$(readelf -d "$l" | sed -n 's/.*SONAME.*\[\(.*\)\]/\1/p'); cp "$l" $d/$so; chmod 755 $d/$so
done
nv=$(readelf -d $d/llb-engine | sed -n 's/.*NEEDED.*\[\(libnvrtc[^]]*\)\]/\1/p')
[ -f $d/$nv ] || { echo "bundle.sh: $b/llb-engine needs $nv; CUDA_HOME=$C has another NVRTC" >&2; exit 1; }
cc $I $fl -O2 -Wall -o $d/llb-probe tools/llb/llb-probe.c -ldl
cp tools/llb/llb-probe.c $d/tools/llb/; cp tools/pkg2/glibc234.h tools/pkg2/cu12.h $d/tools/
sed "s/^LLB_VERSION=.*/LLB_VERSION=$v/" tools/llb/llb > $d/llb; chmod +x $d/llb
cp tools/llb/defaults.conf tools/llb/models.conf $d/
gcc $I -O2 -ffp-contract=off -fno-fast-math -I tools tools/qwen35_toktest.c -o $d/llb-toktest -lm
cc $I -O2 -o $d/llb-tok tools/llb/llb-tok.c -lm
cp tools/qwen35_tokenizer.c tools/qwen35_tokenizer.h tools/qwen35_unicode_data.inc tools/qwen35_toktest.c $d/tools/
cp tools/llb/llb-tok.c $d/tools/llb/
sed "s/@@NAME@@/$n/g; s/@@VERSION@@/$v/g; s/@@COMMIT@@/$(git rev-parse --short=8 HEAD)/g" release/README-cuda.md > $d/README.md
cat > $d/build.sh <<'B'
#!/bin/sh
# build.sh: rebuild the binaries from the shipped source (clang >= 19 for #embed; CUDA headers + libnvrtc).
# tools/glibc234.h keeps GLIBC <= 2.34; CUDA 12 headers add tools/cu12.h; DT_RPATH $ORIGIN finds the libnvrtc (and, for its dlopen, the builtins) shipped here.
set -eu
cd "$(dirname "$0")"; C=${CUDA_HOME:-/usr/local/cuda}; I="-include tools/glibc234.h"
x=; [ "$(awk '/#define CUDA_VERSION/ { print $3; exit }' "$C/include/cuda.h")" -ge 13000 ] || x="-include tools/cu12.h"
${CC:-clang} $I $x -DBEND_CUDA=1 -I"$C/include" -L"$C/lib64" -std=c11 -O3 -ffp-contract=off llb-engine.c -lpthread -lm -lcuda -lnvrtc -Wl,--disable-new-dtags,-rpath,'$ORIGIN' -o llb-engine
${CC:-clang} -std=c11 -O2 -Wall -o gguf2pack gguf2pack.c
gcc $I -O2 -ffp-contract=off -fno-fast-math -I tools tools/qwen35_toktest.c -o llb-toktest -lm
${CC:-cc} $I -O2 -o llb-tok tools/llb/llb-tok.c -lm
fl="-DLLB_FLOOR=13000 -DLLB_FLOOR_R=580"; [ -z "$x" ] || fl="-DLLB_FLOOR=12020 -DLLB_FLOOR_R=535"
${CC:-cc} $I $fl -O2 -o llb-probe tools/llb/llb-probe.c -ldl
sha256sum llb-engine gguf2pack llb-tok llb-toktest llb-probe
B
cat > $d/run.sh <<'R'
#!/bin/sh
# run.sh [--gguf FILE] [--prefill fast|exact] [--spec K] ENGINE-ARGS...: the 27B Q4_K_M launcher; defaults LLB_Q4_DP4A=9 + LLB_PREFILL=fast.
#   --gguf FILE  read weights straight from the GGUF (LLB_GGUF); the positional dir is then the gguf2pack --sidecar-only dir
#   --prefill exact: token-by-token prefill; --spec K (needs LLB_MTP=drafter file): opt-in, see README
set -eu
here=$(cd "$(dirname "$0")" && pwd)
export LLB_Q4_DP4A="${LLB_Q4_DP4A-9}" LLB_PREFILL="${LLB_PREFILL-fast}"
while [ $# -gt 0 ]; do
  case $1 in
    --gguf) export LLB_GGUF=$2; shift 2 ;;
    --prefill) export LLB_PREFILL=$2; shift 2 ;;
    --spec) export LLB_SPEC=$2; [ -n "${LLB_MTP-}" ] || { echo "run.sh: --spec needs LLB_MTP=<drafter file> (README)" >&2; exit 2; }; shift 2 ;;
    *) break ;;
  esac
done
exec "$here/llb-engine" "$@"
R
chmod +x $d/build.sh $d/run.sh
tools/llb/check-isa.sh "$d"
(cd $d && find . -type f ! -name SHA256SUMS.files | sed 's|^\./||' | LC_ALL=C sort | xargs sha256sum > SHA256SUMS.files)
tar -C release -czf release/$n.tar.gz $n
(cd release && sha256sum llb-*.tar.gz > SHA256SUMS && cat SHA256SUMS && ls -l $n.tar.gz)  # every tarball in release/
