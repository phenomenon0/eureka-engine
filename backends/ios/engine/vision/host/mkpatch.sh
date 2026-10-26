#!/usr/bin/env bash
# regenerate patches/engine-mmproj.patch: vision/host/mmproj_section.c inserted before the effect glue
# (#ifdef CID_GGUF_READ at line start) of the Mac tree's host/llb_gguf.c (post engine-reopen.patch)
set -euo pipefail; cd "$(dirname "$0")/../.."; V=work/vpatch; rm -rf $V; mkdir -p $V/a/host $V/b/host
cp work/mac/tree/engine/host/llb_gguf.c $V/a/host/
python3 -c "
src=open('$V/a/host/llb_gguf.c').read(); sec=open('vision/host/mmproj_section.c').read()
i=src.index('\n#ifdef CID_GGUF_READ\n')+1; open('$V/b/host/llb_gguf.c','w').write(src[:i]+sec+'\n'+src[i:])"
(cd $V && diff -u a/host/llb_gguf.c b/host/llb_gguf.c | sed '1,2s/\t.*//') > patches/engine-mmproj.patch || true
shasum -a 256 patches/engine-mmproj.patch
