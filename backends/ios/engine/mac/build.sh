#!/usr/bin/env bash
# mac/build.sh (ENGINE-EARLYSTOP copy of engine-repeat's, re-rooted to ~/engine-earlystop, + engine-stop.patch; ENGINE-REPEAT copy of engine-phone2's, re-rooted to ~/engine-repeat, + the reopen pair; ENGINE-PHONE2 copy of engine-decidechunk's, re-rooted to ~/engine-phone2; ENGINE-DECIDECHUNK copy of engine-prefill's, + the dchunk pair; ENGINE-PREFILL copy of engine-one's; ENGINE-ONE copy of engine-phone's, itself engine-4b's): the ENGINE-4B Mac engine = engine-emcee's sys1mac recipe (re-rooted to ~/engine-one) + the 4B pair:
#   1. base_tree.sh: fresh41.sh (M36 v4 + M38 + M41) + metal-/engine-sys1mac.patch (patch --fuzz=0)
#   2. patches/metal-4b.patch into cc/, patches/engine-4b.patch into engine/ (--fuzz=0, hashes in patches/SHA256SUMS)
#      (NO4B=1 skips step 2: the base build, byte-identical to engine-emcee's sys1mac)
#      (ENGINE-ONE) + patches/metal-kev.patch into cc/ (Kev-4B's Q4_K embedding at d 2560); NOKEV=1 skips it
#      (ENGINE-PREFILL) + patches/{metal,engine}-4bpf.patch (the fast-prefill chunk path at d 2560); NOPF=1 skips them
#   3. tools/sys1dual/sys1.{bend,c,js} copied unchanged; bun -> C (M36's emit); the dual + CLEF-3 objects;
#      clang -O3 -ffp-contract=off link + Homebrew OpenBLAS. The metallib (.gpu) is made by the first --gpu run.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); top=$here/..; root=${ISILDUR_S1D:-$top/S1D}
W=${W:-${ISILDUR_OUT:-$top}/work/mac}  # (ISILDUR) re-rooted to this repo (PHONECAM, KEVPHONE before)
export HOME=$W/home TMPDIR=$W/tmp CLANG_MODULE_CACHE_PATH=$W/modules BUN_INSTALL_CACHE_DIR=$W/tmp/bun
mkdir -p "$HOME" "$TMPDIR" "$CLANG_MODULE_CACHE_PATH" "$W/inc"
S=$W/tree O=${O:-$W/build}
bash "$here/base_tree.sh" "$S"
if [ -z "${NO4B:-}" ]; then
  (cd "$top/patches" && shasum -a 256 -c --quiet SHA256SUMS) || { echo "4B patch hashes differ"; exit 3; }
  (cd "$S/cc" && patch -s --fuzz=0 -p1 < "$top/patches/metal-4b.patch")
  (cd "$S/engine" && patch -s --fuzz=0 -p1 < "$top/patches/engine-4b.patch")
  # (ENGINE-ONE) Kev-4B's Q4_K embedding at d 2560 in the fused step (NOKEV=1: engine-phone's build exactly)
  [ -n "${NOKEV:-}" ] || (cd "$S/cc" && patch -s --fuzz=0 -p1 < "$top/patches/metal-kev.patch")
  # (ENGINE-PREFILL) the fast-prefill chunk path at d 2560 (NOPF=1: engine-one's build exactly)
  if [ -z "${NOPF:-}" ]; then
    (cd "$S/cc" && patch -s --fuzz=0 -p1 < "$top/patches/metal-4bpf.patch")
    (cd "$S/engine" && patch -s --fuzz=0 -p1 < "$top/patches/engine-4bpf.patch")
  fi
  # (ENGINE-DECIDECHUNK) the fast decide: the srv loop's --fastpf and the chunk's hook rows
  # (NODCH=1: engine-prefill's build exactly); needs the 4bpf pair (the chunk at d 2560)
  if [ -z "${NODCH:-}" ] && [ -z "${NOPF:-}" ]; then
    (cd "$S/cc" && patch -s --fuzz=0 -p1 < "$top/patches/metal-dchunk.patch")
    (cd "$S/engine" && patch -s --fuzz=0 -p1 < "$top/patches/engine-dchunk.patch")
  fi
  # (ENGINE-PREFILL) the default-tier flip for the 4B (m.dfk: fast at nt >= 13). (ENGINE-PHONE2) ON by default,
  # matching llb_decide's fast default; NODEF4B=1 skips it (engine-decidechunk's a36cb56e / 9566cd54 builds)
  [ -n "${NODEF4B:-}${NOPF:-}" ] || (cd "$S/engine" && patch -s --fuzz=0 -p1 < "$top/patches/engine-4bdef.patch")
  # (ENGINE-REPEAT) N runs per process: bend_close (pool + io helpers joined, every static back to its initializer,
  # the span and the loop's stack unmapped) and gguf_close; NOREOPEN=1 skips (engine-phone2's 515c9fee exactly)
  if [ -z "${NOREOPEN:-}" ]; then
    (cd "$S/cc" && patch -s --fuzz=0 -p1 < "$top/patches/metal-reopen.patch")
    (cd "$S/engine" && patch -s --fuzz=0 -p1 < "$top/patches/engine-reopen.patch")
  fi
  # (ENGINE-EARLYSTOP) --stop ID: the generate loop ends after it prints a stop id (n new ids is a maximum); no
  # --stop = n new ids exactly, as before. NOSTOP=1 skips (engine-repeat's 382c871a exactly)
  [ -n "${NOSTOP:-}" ] || (cd "$S/engine" && patch -s --fuzz=0 -p1 < "$top/patches/engine-stop.patch")
  # (KEVPHONE) the srv loop passes a command's --stop ids to its generate (the resident session's chat); decide's
  # commands carry none, as before. NORESIDENT=1 skips (engine-earlystop's c13b9e28 exactly)
  [ -n "${NORESIDENT:-}${NOSTOP:-}" ] || (cd "$S/engine" && patch -s --fuzz=0 -p1 < "$top/patches/engine-resident.patch")
  # (PHONECAM) ENGINE-IPHONEVIS's vision stack on top of the resident session (same order as engine-iphonevis's
  # build.sh: mmproj, then imrope + vision). OPT-IN: VISION=1; unset = KEVPHONE's bccd7cea exactly
  if [ -n "${VISION:-}" ]; then
    (cd "$S/engine" && patch -s --fuzz=0 -p1 < "$top/patches/engine-mmproj.patch")
    (cd "$S/cc" && patch -s --fuzz=0 -p1 < "$top/patches/metal-imrope.patch")
    (cd "$S/cc" && patch -s --fuzz=0 -p1 < "$top/patches/metal-vision.patch")
    # (PHONECAM) metal-camera.patch: the session image area (one image per resident job) + the Yes/No readout (two
    # named logits of the head at a prompt pos, in process); VISION=1 only, NORO=1 skips
    [ -n "${NORO:-}" ] || (cd "$S/cc" && patch -s --fuzz=0 -p1 < "$top/patches/metal-camera.patch")
  fi
  # (ENGINE-PHONE2) no gate instrument here: gate-logits*.patch are not carried (never in a product build)
  find "$S" -name '*.orig' -delete
fi
mkdir -p "$S/engine/tools/sys1dual"
cp "$root"/tools/sys1dual/sys1.{bend,c,js} "$S/engine/tools/sys1dual/"
mkdir -p "$O/obj"
(cd "$S/engine" && bun ../cc/bend2/main.ts engine_qwen35_gpu.bend -o "$O/sys1mac.c" 2>&1 | grep -v '^- \|bend update' | tail -3)
ln -sfn /opt/homebrew/opt/openblas/include "$W/inc/openblas"   # head.c's <openblas/cblas.h>
for f in sys1dual/dual.c clef3/head.c clef3/weights.c clef3/spans.c clef3/json.c qwen35_tokenizer.c; do
  clang -D_POSIX_C_SOURCE=200809L -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Wno-unused-function -I"$W/inc" \
    -c "$root/tools/$f" -o "$O/obj/$(basename "$f" .c).o"
done
# (PHONECAM, from engine-iphonevis's build.sh) VISION=1: the tower objects (vit.c = the CPU reference, OpenBLAS;
# vitm.m = the Metal tower, vitm.metal embedded) in obj-vis/, linked after obj/*.o
VOBJ=()
if [ -n "${VISION:-}" ]; then
  mkdir -p "$O/obj-vis"; rm -f "$O"/obj-vis/*.o
  for f in llbv vpre vit vmm; do
    clang -std=c11 -D_POSIX_C_SOURCE=200809L -D_DARWIN_C_SOURCE -O3 -ffp-contract=off -fblocks -w \
      -I"$top/vision/host" -I"$S/engine/host" -I/opt/homebrew/opt/openblas/include \
      -c "$top/vision/host/$f.c" -o "$O/obj-vis/$f.o"
    VOBJ+=("obj-vis/$f.o")
  done
  (cd "$top/vision/host" && xxd -i vitm.metal) > "$O/obj-vis/vitm_msl.h"
  clang -std=c11 -D_POSIX_C_SOURCE=200809L -D_DARWIN_C_SOURCE -O3 -ffp-contract=off -fobjc-arc -x objective-c -w \
    -I"$top/vision/host" -I"$O/obj-vis" -c "$top/vision/host/vitm.m" -o "$O/obj-vis/vitm.o"
  VOBJ+=("obj-vis/vitm.o" -framework Metal -framework Foundation)
fi
(cd "$O" && clang -DBEND_METAL=1 -x objective-c -fobjc-arc -fmodules -fmodules-cache-path="$CLANG_MODULE_CACHE_PATH" \
  -std=c11 -O3 -ffp-contract=off sys1mac.c -x none obj/*.o ${VOBJ[@]+"${VOBJ[@]}"} -L/opt/homebrew/opt/openblas/lib -lopenblas -lpthread -lm \
  -o sys1mac)
shasum -a 256 "$S/cc/bend2/comp.ts" "$S/engine/engine_qwen35_gpu.bend" "$O/sys1mac.c" "$O/sys1mac" "$O"/obj/*.o $([ -z "${VISION:-}" ] || ls "$O"/obj-vis/*.o) | sed "s|$W/||"
