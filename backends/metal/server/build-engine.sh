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
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); top=${EUREKA_ENGINE_SRC:?set EUREKA_ENGINE_SRC to the Mac engine tree}; here=$top/mac; root=$top/S1D; MY=$HERE
W=${W:-$HERE/work}
export HOME=$W/home TMPDIR=$W/tmp CLANG_MODULE_CACHE_PATH=$W/modules BUN_INSTALL_CACHE_DIR=$W/tmp/bun
mkdir -p "$HOME" "$TMPDIR" "$CLANG_MODULE_CACHE_PATH" "$W/inc"
S=$W/tree O=${O:-$W/build}
# (FUS-M-V46) the m38 patch directory (fresh41.sh's patches-m38) was deleted by a cleanup, so base_tree.sh + the default
# patch stack cannot be replayed. base/tree-default = ENGINE-VISION3's last work/mac/tree with the three opt-in
# vision patches reversed (metal-vision, metal-imrope, engine-mmproj; patch -R --fuzz=0). It is proven by the emitted-C
# hashes: default = c13b9e28 and VISION=1 = e15c60e0, as in ENGINE-VISION3 (the build log prints them). Only the
# opt-in patches below are applied on it. FROMSCRATCH=1 = the old path (needs the m38 patch directory)
if [ -z "${FROMSCRATCH:-}" ]; then
  rm -rf "$S"; mkdir -p "$S"; cp -R "$top/base/tree-default/." "$S/"
  [ -z "${MMPROJ:-}${VISION:-}" ] || (cd "$S/engine" && patch -s --fuzz=0 -p1 < "$top/patches/engine-mmproj.patch")
  if [ -n "${VISION:-}" ]; then
    (cd "$S/cc" && patch -s --fuzz=0 -p1 < "$top/patches/metal-imrope.patch")
    (cd "$S/cc" && patch -s --fuzz=0 -p1 < "$top/patches/metal-vision.patch")
    # (FUS-M-V46 M-V6X) image rows at the exact step too (the CUDA V6's tier); NOEXACTVIS=1 = ENGINE-VISION3's e15c60e0
    [ -n "${NOEXACTVIS:-}" ] || [ ! -s "$top/patches/metal-visexact.patch" ] || (cd "$S/cc" && patch -s --fuzz=0 -p1 < "$top/patches/metal-visexact.patch")
  fi
  find "$S" -name '*.orig' -delete
else
bash "$here/base_tree.sh" "$S"
fi
if [ -n "${FROMSCRATCH:-}" ] && [ -z "${NO4B:-}" ]; then
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
  # (ENGINE-VISION V1) the mmproj loader section in host/llb_gguf.c (gg_mm_*: clip.* keys, v.*/mm.* tensors, the
  # qwen3vl_merger list check, F16/F32 upload). OPT-IN: MMPROJ=1 applies it; unset = earlystop's c13b9e28 exactly
  # (the section is static C the Bend glue does not call yet, but it is spliced into the emitted C, so it moves the hash)
  [ -z "${MMPROJ:-}${VISION:-}" ] || (cd "$S/engine" && patch -s --fuzz=0 -p1 < "$top/patches/engine-mmproj.patch")
  # (ENGINE-VISION2 V4-V6) OPT-IN: VISION=1 (implies MMPROJ=1) adds metal-imrope.patch (V4: the rope kernels' angle
  # from the image map LLB_MRP, the KV slot unchanged; text-only = the same expressions) and metal-vision.patch (V5:
  # <|image_pad|> batch rows := image rows in the fast chunk, the map set at the first pad), and links
  # vision/host/{llbv,vpre,vit,vmm}.c (the R1 tower, CPU). Unset = earlystop's c13b9e28 exactly
  if [ -n "${VISION:-}" ]; then
    (cd "$S/cc" && patch -s --fuzz=0 -p1 < "$top/patches/metal-imrope.patch")
    (cd "$S/cc" && patch -s --fuzz=0 -p1 < "$top/patches/metal-vision.patch")
  fi
  # (ENGINE-PHONE2) no gate instrument here: gate-logits*.patch are not carried (never in a product build)
  find "$S" -name '*.orig' -delete
fi
# (ENGFOLD) VISION=1: the exact-prefill keeper set (PREFILL-VERDICT §5.1) is ON by default. engine-xpf.patch routes the
# 9B's --exactpf to the chunk bang at the default tier's K; after the emit, mac/xpf/chain.sh (the prefill-speed /
# prefill-speed2 patchers, vendored) and mac/xpf/apply_xdef.py (LLB_XCHUNK 2 under --exactpf, LLB_XB 0x1467BFF) make the
# host serve it with the exact chunk. NOXDEF=1: the chain only (emitted C = prefill-speed2's q8f 664b0ba1, env opt-in as
# in the lanes); NOXPF=1: neither (fus-m-v46's 2f328a0b exactly)
XPF=; XDEF=
if [ -n "${VISION:-}" ] && [ -z "${NOXPF:-}" ]; then
  XPF=1; [ -n "${NOXDEF:-}" ] || XDEF=1
  (cd "$top/mac/xpf" && shasum -a 256 -c --quiet SHA256SUMS) || { echo "xpf hashes differ"; exit 3; }
  [ -z "$XDEF" ] || (cd "$S/engine" && patch -s --fuzz=0 -p1 < "$top/patches/engine-xpf.patch")
  find "$S" -name '*.orig' -delete
fi
(cd "$S/engine" && patch -s --fuzz=0 -p1 < "$MY/patches/engine-srvstop.patch")
mkdir -p "$S/engine/tools/sys1dual"
cp "$root"/tools/sys1dual/sys1.{bend,c,js} "$S/engine/tools/sys1dual/"
mkdir -p "$O/obj"
(cd "$S/engine" && bun ../cc/bend2/main.ts engine_qwen35_gpu.bend -o "$O/sys1mac.c" 2>&1 | grep -v '^- \|bend update' | tail -3)
if [ -n "$XPF" ]; then
  mv "$O/sys1mac.c" "$O/sys1mac.emit.c"
  if [ -n "$XDEF" ]; then
    bash "$top/mac/xpf/chain.sh" "$O/sys1mac.emit.c" "$O/sys1mac.chain.c"
    python3 -I "$top/mac/xpf/apply_xdef.py" "$O/sys1mac.chain.c" "$O/sys1mac.c"
  else
    bash "$top/mac/xpf/chain.sh" "$O/sys1mac.emit.c" "$O/sys1mac.c"
  fi
fi
# (E-METAL-LEAD) the fast decode K-quant GEMV tier, linked beside the bit-exact fixed-tree bodies. OFF at run time unless
# LLB_FASTGEMV=1: metal-lead 581933da's "fast by default" (serve sets LLB_FASTGEMV=1, --canon) is HELD for the owner (METAL-PORT).
mv "$O/sys1mac.c" "$O/sys1mac.pre-fg.c"
python3 -I "$MY/patches/apply_fastgemv.py" "$O/sys1mac.pre-fg.c" "$O/sys1mac.fg.c" "$MY/patches/kern_fastgemv.metal"
# (CHUNKF-F32) the fast chunk's GEMM gets an F32 arm (llb_f27_mm32) and the CHUNKF gate takes clef 9B's F32 ssm_beta / ssm_alpha
# (slots 7, 8), as the exact PASS arm does: clef 9B's default --fastpf tier exited on its first chunk (bang fid 295, "warp hybrid:
# unsupported operands"). NOF32MM=1 skips it (= metal-decide-chat c665a1ac's emitted C exactly)
if [ -n "${NOF32MM:-}" ]; then cp "$O/sys1mac.fg.c" "$O/sys1mac.f32.c"
else python3 -I "$MY/patches/apply_f27mm32.py" "$O/sys1mac.fg.c" "$O/sys1mac.f32.c"; fi
# (DECIDE-CHAT-FIX) the chunk area reserved whatever the first pass was (an exact first pass, e.g. a decide, then a fast-tier
# generate exited with "fast chunk: no chunk area"). NOCHUNKAREA=1 skips it (= KEV-HEAD-METAL's emitted C exactly)
if [ -n "${NOCHUNKAREA:-}" ]; then mv "$O/sys1mac.f32.c" "$O/sys1mac.c"
else python3 -I "$MY/patches/apply_chunkarea.py" "$O/sys1mac.f32.c" "$O/sys1mac.c"; fi
ln -sfn /opt/homebrew/opt/openblas/include "$W/inc/openblas"   # head.c's <openblas/cblas.h>
clang -D_POSIX_C_SOURCE=200809L -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Wno-unused-function -I"$W/inc" -I"$root/tools" -I"$MY/src" -c "$MY/src/dualm.c" -o "$O/obj/dual.o"
# (KEV-HEAD-METAL) the Kev decide head beside the clef objects (dualm.c serves decide through it for a Kev GGUF)
clang -D_POSIX_C_SOURCE=200809L -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Wno-unused-function -I"$root/tools" -I"$MY/src" -c "$MY/src/kevhead.c" -o "$O/obj/kevhead.o"
for f in clef3/head.c clef3/weights.c clef3/spans.c clef3/json.c qwen35_tokenizer.c; do
  clang -D_POSIX_C_SOURCE=200809L -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Wno-unused-function -I"$W/inc" \
    -c "$root/tools/$f" -o "$O/obj/$(basename "$f" .c).o"
done
# (ENGINE-VISION2) VISION=1: the tower objects (vmm.c includes the tree's MMPROJ-patched host/llb_gguf.c; vit.c +
# libdispatch blocks + OpenBLAS, -ffp-contract=off as R1's vit_run) in obj-vis/, linked after obj/*.o
# (ENGINE-VISION3 V7) + vitm.m, the Metal tower (the default tower; vit.c stays linked as the CPU reference,
# LLB_VIS_TOWER=cpu), its kernels vitm.metal embedded as obj-vis/vitm_msl.h (xxd -i)
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
shasum -a 256 "$S/cc/bend2/comp.ts" "$S/engine/engine_qwen35_gpu.bend" $([ -z "$XPF" ] || ls "$O"/sys1mac.emit.c "$O"/sys1mac.chain.c 2>/dev/null) "$O/sys1mac.c" "$O/sys1mac" "$O"/obj/*.o $([ -z "${VISION:-}" ] || ls "$O"/obj-vis/*.o) | sed "s|$W/||"
