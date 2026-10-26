#!/usr/bin/env bash
# build.sh: fresh clone -> engine binary. CPU work only; never touches the GPU.
#   1. export the pinned Bend compiler commit from a Bend git checkout ($LLB_BEND)
#   2. apply the compiler patch stack (tools/build_gpu.sh owns the list/order, --fuzz=0)
#   3. emit C from engine_qwen35_gpu.bend (9B and 27B Qwen3.5-family engine)
#   4. compile it with the CUDA host (clang + libcuda + libnvrtc) and the request objects it calls
#      (dual_next / dual_done: tools/sys1dual/dual.c + the CLEF-3 sources, -lopenblas)
#   5. compile tools/gguf2pack.c (GGUF -> pack, plain C) to $LLB_OUT/gguf2pack, tools/llb/llb-probe.c to $LLB_OUT/llb-probe
# Env: LLB_ROOT (this repo), LLB_BEND (Bend checkout, default $LLB_ROOT/../bend),
#      LLB_OUT (default $LLB_ROOT/build), CUDA_HOME (default /usr/local/cuda), CC (default clang),
#      LLB_CC_EXTRA (set = link exactly these objects/libs instead of step 4's, tools/sys1dual/build.sh),
#      LLB_MTP_PACK=1 (also build the LLB_MTP drafter pack, tools/mtp27/README.md).
#      LLB_BACKEND (cuda, the default | metal: backends/metal/build.sh, see backends/metal/README.md).
set -euo pipefail
LLB_ROOT=${LLB_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
# LLB_BACKEND=metal: the Mac Metal engine (backends/metal/build.sh, opt-in). Unset or cuda: the CUDA build below, unchanged.
case "${LLB_BACKEND:-cuda}" in cuda) ;; metal) export LLB_ROOT; exec bash "$LLB_ROOT/backends/metal/build.sh" ;; *) echo "build.sh: LLB_BACKEND=$LLB_BACKEND (cuda|metal)" >&2; exit 1 ;; esac
LLB_BEND=${LLB_BEND:-$LLB_ROOT/../bend}
LLB_OUT=${LLB_OUT:-$LLB_ROOT/build}
BEND_COMMIT=240ecef4b52a1efcc2151df689cca9a2a769aac6
# the 240ecef4 commit object is lost; its bend2/ survives as the content record pins/bend-240ecef4 (PIN.md there),
# accepted by this git tree id when LLB_BEND has no 240ecef4
BEND_TREE=aada4cbf6e04d0ade6a7e421ed9fa945a9504177
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} CC=${CC:-clang}

die() { echo "build.sh: $*" >&2; exit 1; }
for t in bun git patch "$CC" sha256sum; do command -v "$t" >/dev/null || die "missing tool: $t"; done
[ -f "$CUDA_HOME/include/cuda.h" ] || die "no CUDA headers at CUDA_HOME=$CUDA_HOME"
bend_git=$LLB_BEND bend_rev=$BEND_COMMIT
if ! git -C "$LLB_BEND" cat-file -e "$BEND_COMMIT^{commit}" 2>/dev/null; then
  bend_rev=
  for r in "$LLB_BEND" "$LLB_ROOT"; do for ref in pins/bend-240ecef4 origin/pins/bend-240ecef4; do
    [ "$(git -C "$r" rev-parse -q --verify "$ref:bend2" 2>/dev/null)" = "$BEND_TREE" ] && bend_git=$r bend_rev=$ref && break 2
  done; done
  [ -n "$bend_rev" ] || die "LLB_BEND=$LLB_BEND has no $BEND_COMMIT, and no pins/bend-240ecef4 with bend2 tree $BEND_TREE here or there (git fetch origin pins/bend-240ecef4:pins/bend-240ecef4)"
fi

mkdir -p "$LLB_OUT"
"$CC" -std=c11 -O2 -Wall -o "$LLB_OUT/gguf2pack" "$LLB_ROOT/tools/gguf2pack.c"
# llb-probe: tools/llb/llb reads it from $LLB_OUT for the capacity table's GPU check (tools/unify/bundle.sh's recipe)
fl="-DLLB_FLOOR=13000 -DLLB_FLOOR_R=580"
[ "$(awk '/#define CUDA_VERSION/ { print $3; exit }' "$CUDA_HOME/include/cuda.h")" -ge 13000 ] || fl="-DLLB_FLOOR=12020 -DLLB_FLOOR_R=535"
"$CC" -include "$LLB_ROOT/tools/pkg2/glibc234.h" $fl -O2 -Wall -o "$LLB_OUT/llb-probe" "$LLB_ROOT/tools/llb/llb-probe.c" -ldl
src=$LLB_OUT/bend-pinned
if [ ! -f "$src/.commit" ] || [ "$(cat "$src/.commit")" != "$BEND_COMMIT" ]; then
  rm -rf "$src" "$LLB_OUT/bend-joint"
  mkdir -p "$src"
  git -C "$bend_git" archive "$bend_rev" bend2 | tar -xm -C "$src"
  echo "$BEND_COMMIT" > "$src/.commit"
fi

BEND_ROOT=$src BEND_BUILD=$LLB_OUT/bend-joint \
  bash "$LLB_ROOT/tools/build_gpu.sh" "$LLB_ROOT/engine_qwen35_gpu.bend" -o "$LLB_OUT/llb-engine.c"
# the emitted C calls dual_next / dual_done (tools/sys1dual/sys1.c) unconditionally: unless the caller set
# LLB_CC_EXTRA, link the request objects tools/sys1dual/build.sh links (same sources, flags and order)
if [ -z "${LLB_CC_EXTRA+set}" ]; then
  o=$LLB_OUT/obj
  rm -rf "$o"; mkdir -p "$o"
  for f in sys1dual/dual.c metrics/metrics.c metrics/tokstream.c glue/glue_out.c clef3/head.c clef3/weights.c clef3/spans.c clef3/json.c qwen35_tokenizer.c; do
    "$CC" -D_POSIX_C_SOURCE=200809L -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Wno-unused-function \
      -c "$LLB_ROOT/tools/$f" -o "$o/$(basename "$f" .c).o"
  done
  export LLB_CC_EXTRA="$(echo "$o"/*.o) -lopenblas"
fi
bash "$LLB_ROOT/tools/remap_cc.sh" "$LLB_OUT/llb-engine.c" "$LLB_OUT/llb-engine"

# optional: the LLB_MTP drafter pack (tools/mtp27/README.md); needs the 27B GGUFs + python gguf
if [ "${LLB_MTP_PACK:-0}" = 1 ]; then
  bash "$LLB_ROOT/tools/mtp27/make_drafter.sh" "$LLB_OUT/mtp-drafter.mtp1"
fi
echo "== built (sha256) =="
echo "bend commit   $BEND_COMMIT"
echo "patch stack   $(grep -o 'tools/[a-z0-9_-]*\.patch' "$LLB_ROOT/tools/build_gpu.sh" | LC_ALL=C sort -u | sed "s|^|$LLB_ROOT/|" | xargs cat | sha256sum | cut -d' ' -f1) ($(grep -o 'tools/[a-z0-9_-]*\.patch' "$LLB_ROOT/tools/build_gpu.sh" | LC_ALL=C sort -u | wc -l) patches)"
echo "compiler      $(cd "$LLB_OUT/bend-joint/bend2" && find . -type f | LC_ALL=C sort | xargs cat | sha256sum | cut -d' ' -f1)"
sha256sum "$LLB_OUT/llb-engine.c" "$LLB_OUT/llb-engine" "$LLB_OUT/gguf2pack" "$LLB_OUT/llb-probe"
