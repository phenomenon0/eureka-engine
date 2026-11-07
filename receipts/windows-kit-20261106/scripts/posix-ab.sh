#!/usr/bin/env bash
# posix-ab.sh GGUF WORKDIR [BASE]: build gguf2pack + qwen35-cli at BASE (a git rev, default dd824b9, before
# tools/llb_platform.h) and at the working tree, pack GGUF with both, run both CLIs (mmap and --preload), and
# compare: pack directory bytes, verify-against, token ids, logit bits. Prints PASS/FAIL per check.
# BASE may also be a directory holding that rev's tools/ and host/ (for a container without git).
set -u
G=$1 W=$2 BASE=${3:-dd824b9}
R=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
CC=${CC:-cc}
rm -rf "$W"; mkdir -p "$W/base-src" "$W/base" "$W/new"
if [ -d "$BASE" ]; then cp -r "$BASE/tools" "$BASE/host" "$W/base-src/"; else git -C "$R" archive "$BASE" tools host | tar -x -C "$W/base-src"; fi
bad=0; ok() { if [ "$1" = 0 ]; then echo "PASS $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
for side in base new; do
  src=$R; [ $side = base ] && src=$W/base-src
  echo "\$ $CC -std=c11 -O2 -Wall -o $side/gguf2pack tools/gguf2pack.c   # $side"
  $CC -std=c11 -O2 -Wall -o "$W/$side/gguf2pack" "$src/tools/gguf2pack.c" || bad=$((bad + 1))
  echo "\$ $CC -O2 -ffp-contract=off -fno-fast-math tools/qwen35_engine.c -o $side/qwen35-cli -lm   # $side"
  $CC -O2 -ffp-contract=off -fno-fast-math "$src/tools/qwen35_engine.c" -o "$W/$side/qwen35-cli" -lm || bad=$((bad + 1))
  "$W/$side/gguf2pack" "$G" "$W/$side/pack" --format quant 2>&1 | tail -1 | sed "s|$W/||"
  "$W/$side/gguf2pack" "$G" "$W/$side/ho" --head-only > /dev/null 2>&1
  "$W/$side/qwen35-cli" "$W/$side/pack" 760 6511 314 9338 369 --steps 8 > "$W/$side/run.out" 2> "$W/$side/run.err"
  echo "$side cli exit=$?"
  "$W/$side/qwen35-cli" "$W/$side/pack" 760 6511 314 9338 369 --steps 8 --preload > "$W/$side/pre.out" 2> "$W/$side/pre.err"
  echo "$side cli --preload exit=$?"
done
diff -r "$W/base/pack" "$W/new/pack" > /dev/null; ok $? "pack dir byte-identical (diff -r)"
diff -r "$W/base/ho" "$W/new/ho" > /dev/null; ok $? "--head-only pack byte-identical (ftruncate path)"
"$W/new/gguf2pack" "$G" --verify-against "$W/base/pack" --format quant 2>&1 | tail -1
ids() { grep -E '^[0-9]+$' "$1" | tr '\n' ' '; }
for k in run pre; do
  diff <(ids "$W/base/$k.out") <(ids "$W/new/$k.out") > /dev/null; ok $? "$k: token ids identical: $(ids "$W/new/$k.out")"
  diff <(grep ^logits "$W/base/$k.err") <(grep ^logits "$W/new/$k.err") > /dev/null; ok $? "$k: logit bits identical ($(grep -c ^logits "$W/new/$k.err") lines)"
done
diff <(grep ^logits "$W/new/run.err") <(grep ^logits "$W/new/pre.err") > /dev/null; ok $? "new: mmap == --preload logit bits"
mkdir -p "$W/nopack"; cp "$W/new/pack/index.tsv" "$W/nopack/"
diff <("$W/base/qwen35-cli" "$W/nopack" 1 2>&1 | tail -1) <("$W/new/qwen35-cli" "$W/nopack" 1 2>&1 | tail -1) > /dev/null; ok $? "missing pack.bin: same error text ($("$W/new/qwen35-cli" "$W/nopack" 1 2>&1 | tail -1))"
echo "posix-ab $((9 - bad))/9 PASS"
