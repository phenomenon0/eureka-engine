#!/usr/bin/env bash
# tools/vision/f1.sh IMAGE "PROMPT IDS WITH ONE <|image_pad|> MARK @IMG@" [engine args...] (FUS-V46): image -> the V3 CUDA
# tower (vision/build/cudavis-cli) -> rows.f32 -> the engine with LLB_VIS_EMBD (llb_host.c llb_img_step). @IMG@ expands
# to nx * ny ids 248056. Env: LLB_GGUF (the 9B GGUF), SIDE (its --sidecar-only dir), MMPROJ, ENG (default build/llb-engine).
# Example ("Describe this image." in mtmd's template, = FUS-A V0 prompt.txt):
#   tools/vision/f1.sh img.png "248045 846 198 248053 @IMG@ 248054 72240 411 2099 13 248046 198 248045 74455 198 248068 271 248069 271" --steps 32
set -euo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); img=$1 p=$2; shift 2
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
"$R/vision/build/cudavis-cli" "$img" "${MMPROJ:?}" --out "$t/rows.f32" > "$t/tower.log"
read -r px py < <(sed -n 's/.*: \([0-9]*\)x\([0-9]*\) -> .*/\1 \2/p' "$t/tower.log" | head -1)
nx=$((px / 32)) ny=$((py / 32)); pads=$(yes 248056 | head -n $((nx * ny)) | paste -sd' ')
LLB_VIS_EMBD="$t/rows.f32:$nx:$ny" "${ENG:-$R/build/llb-engine}" "${SIDE:?}" ${p/@IMG@/$pads} "$@"
