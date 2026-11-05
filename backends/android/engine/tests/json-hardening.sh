#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp -r "$here/vendor/." "$work/"
for patchfile in "$here"/patches/*.patch; do patch --batch --silent --fuzz=0 -d "$work" -p1 < "$patchfile"; done
cc -D_POSIX_C_SOURCE=200809L -std=c11 -I "$work/tools/clef3" "$here/tests/json-hardening.c" "$work/tools/clef3/json.c" -lm -o "$work/test"
"$work/test"
