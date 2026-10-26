#!/bin/sh
# install.sh DEST: install llb into DEST (llb, its two .conf files, llb-tok and gguf2pack built from source, the engine,
# and llb-probe: the build's own next to the engine, else built from source with the CUDA 13 floor; llb doctor needs it).
# Engine source: $LLB_ENGINE, else $LLB_OUT/llb-engine, else build/llb-engine (with its .gpu kernel cache when present).
# DEST must be writable at run time too: the engine builds its kernel cache next to itself on first use.
set -eu
[ $# = 1 ] || { echo "usage: tools/llb/install.sh DEST" >&2; exit 2; }
src=$(cd "$(dirname "$0")" && pwd)
dest=$1
mkdir -p "$dest"
cp "$src/llb" "$src/defaults.conf" "$src/models.conf" "$dest/"
${CC:-cc} -O2 -o "$dest/llb-tok" "$src/llb-tok.c" -lm
${CC:-cc} -std=c11 -O2 -o "$dest/gguf2pack" "$src/../gguf2pack.c"
eng=
for e in "${LLB_ENGINE-}" "${LLB_OUT:+$LLB_OUT/llb-engine}" "$src/../../build/llb-engine"; do
  [ -n "$e" ] && [ -x "$e" ] && { eng=$e; break; }
done
if [ -n "$eng" ]; then
  cp "$eng" "$dest/llb-engine"
  [ -f "$eng.gpu" ] && cp "$eng.gpu" "$dest/llb-engine.gpu"
  [ -x "$(dirname "$eng")/llb-probe" ] && cp "$(dirname "$eng")/llb-probe" "$dest/llb-probe"
  echo "llb: installed to $dest (engine from $eng)"
else
  echo "llb: installed to $dest WITHOUT an engine: build it (./build.sh) and rerun, or set LLB_ENGINE" >&2
fi
[ -x "$dest/llb-probe" ] || ${CC:-cc} -O2 -o "$dest/llb-probe" "$src/llb-probe.c" -ldl
echo "llb: run $dest/llb doctor; put $dest on PATH (or ln -s $dest/llb ~/.local/bin/llb)"
