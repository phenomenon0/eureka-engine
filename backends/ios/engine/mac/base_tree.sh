#!/usr/bin/env bash
# mac/base_tree.sh <dir> (ISILDUR): the sys1mac Mac tree WITHOUT the 4B patches, copied from this repo's base/
# (hash-checked against base/SHA256SUMS). base/ is that tree, frozen: the output of the old recipe
# (mac/base_tree-fresh41.sh = fresh41.sh (M36 v4 + M38 + M41) + metal-/engine-sys1mac.patch), recovered by
# reverse-applying patches/ from PHONECAM's patched tree and checked file-for-file (442/442) against the sys1mac
# lane's inventory (base/PROVENANCE). The old recipe needs the lane trees m36, m38, m41 and emitted-unlock; m38 is gone, so
# BASE=fresh41 is kept only as provenance.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); S=$1
if [ "${BASE:-}" = fresh41 ]; then exec bash "$here/base_tree-fresh41.sh" "$S"; fi
# (Eureka import) base/ is not vendored here: ISILDUR_BASE names it (default $here/../base); it is checked against
# ../base.SHA256SUMS (a copy of isildur 8ff026e's base/SHA256SUMS) and, if present, its own SHA256SUMS
B=${ISILDUR_BASE:-$here/../base}
(cd "$B" && shasum -a 256 -c --quiet "$here/../base.SHA256SUMS") || { echo "$B differs from base.SHA256SUMS"; exit 3; }
[ ! -f "$B/SHA256SUMS" ] || (cd "$B" && shasum -a 256 -c --quiet SHA256SUMS) || { echo "base/ differs from base/SHA256SUMS"; exit 3; }
rm -rf "$S"; mkdir -p "$S"
rsync -a --exclude SHA256SUMS --exclude PROVENANCE "$B/" "$S/"
