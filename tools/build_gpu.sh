#!/usr/bin/env bash
# Bend CLI through an isolated compiler with scalar borrowing, CUDA word loads
# and device-chosen iterations queued in batches.
# Usage: tools/build_gpu.sh engine_gpu.bend -o build/engine_gpu
# A .c/.js output only emits source; a native GPU build initializes CUDA.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
bend=${BEND_ROOT:-"$root/../bend"}
compiler=${BEND_BUILD:-"$root/build/bend-joint"}
patches=("$root/tools/astra2-borrow.patch" "$root/tools/astra2-aligned.patch"
  "$root/tools/gemv-tile.patch" "$root/tools/grow-batch.patch" "$root/tools/remap.patch"
  "$root/tools/payload-attn.patch"
  "$root/tools/payload-rms.patch" "$root/tools/payload-hand.patch"
  "$root/tools/payload-add.patch" "$root/tools/merge3-q8.patch"
  "$root/tools/interiors-add.patch" "$root/tools/interiors-rope.patch" "$root/tools/interiors-ffn.patch"
  "$root/tools/merge3-f16.patch" "$root/tools/quant-fix.patch"
  "$root/tools/qknorm.patch" "$root/tools/unblock.patch"
  "$root/tools/payload-hybrid.patch" "$root/tools/payload-rmsnorm.patch" "$root/tools/payload-vadd.patch" "$root/tools/payload-og.patch"
  "$root/tools/payload-head.patch" "$root/tools/payload-attn4.patch"
  "$root/tools/payload-silumul.patch" "$root/tools/payload-grid.patch"
  "$root/tools/payload-fbn.patch" "$root/tools/payload-linb.patch"
  "$root/tools/quant-doors.patch" "$root/tools/payload-drain.patch"
  "$root/tools/payload-aob.patch" "$root/tools/payload-linf.patch"
  "$root/tools/payload-qkr.patch" "$root/tools/payload-slot.patch"
  "$root/tools/payload-sqg.patch" "$root/tools/payload-aqg.patch"
  "$root/tools/gemv-q8-rw.patch" "$root/tools/payload-fulb.patch"
  "$root/tools/payload-pass.patch" "$root/tools/payload-attn4b.patch" "$root/tools/gemv-nv.patch" "$root/tools/payload-fold.patch" "$root/tools/payload-fuse.patch" "$root/tools/payload-seam.patch" "$root/tools/payload-attn4c.patch" "$root/tools/payload-cat.patch" "$root/tools/payload-down.patch" "$root/tools/payload-host.patch" "$root/tools/payload-shape.patch" "$root/tools/payload-table.patch" "$root/tools/payload-kpair.patch" "$root/tools/payload-q4glue.patch" "$root/tools/payload-27b.patch" "$root/tools/payload-dp4a.patch" "$root/tools/payload-dp4b.patch" "$root/tools/payload-pf.patch" "$root/tools/payload-spec.patch" "$root/tools/payload-spec2.patch" "$root/tools/payload-pfchunk.patch" "$root/tools/payload-batch.patch" "$root/tools/payload-passdrop.patch" "$root/tools/payload-imrope.patch" "$root/tools/payload-w1024.patch")
# --forward forbids automatic direction reversal, including ignoring --reverse.

if [[ ! -e "$compiler" ]]; then
  mkdir -p "$compiler/bend2"
  cp -r "$bend/bend2/"{bend.ts,comp.ts,main.ts,hand.ts,export.ts,base.bend,effs} "$compiler/bend2/"
  for patch_file in "${patches[@]}"; do
    patch --batch --silent --fuzz=0 --forward -d "$compiler" -p1 < "$patch_file"
  done
fi
# Never overwrite a compiler another build may be using. Refuse stale or
# incomplete copies; choose a fresh BEND_BUILD directory to recreate one.
[[ -f "$compiler/bend2/main.ts" ]] || { printf 'Incomplete compiler copy: %s\n' "$compiler" >&2; exit 1; }
# Patches stack (a later one may edit an earlier one's lines): unwind them on
# a scratch copy, last first.
check=$(mktemp -d)
trap 'rm -rf "$check"' EXIT
cp -r "$compiler/bend2" "$check/"
for ((i = ${#patches[@]} - 1; i >= 0; i--)); do
  if ! patch --batch --silent --fuzz=0 --forward --reverse -d "$check" -p1 < "${patches[i]}"; then
    printf 'Compiler copy does not match %s: %s\nUse a fresh BEND_BUILD directory.\n' "${patches[i]}" "$compiler" >&2
    exit 1
  fi
done
rm -rf "$check"
trap - EXIT
export LLB_HOST_DIR="${LLB_HOST_DIR:-$root/host}"
exec bun "$compiler/bend2/main.ts" "$@"
