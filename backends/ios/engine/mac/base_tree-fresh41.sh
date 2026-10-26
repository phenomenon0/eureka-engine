#!/usr/bin/env bash
# mac/base_tree.sh <dir>: the sys1mac Mac tree (engine-emcee's recipe, re-rooted) WITHOUT the 4B patches:
# fresh41.sh (M36 v4 + M38 + M41), then metal-/engine-sys1mac.patch, each patch --fuzz=0. The 4B patches
# (patches/*.patch) are made against this tree and applied on top by build.sh.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); S=$1
bash "$here/fresh41.sh" "$S" >/dev/null
(cd "$S/cc" && patch -s --fuzz=0 -p1 < "$here/metal-sys1mac.patch")
(cd "$S/engine" && patch -s --fuzz=0 -p1 < "$here/engine-sys1mac.patch")
find "$S" -name '*.orig' -delete
