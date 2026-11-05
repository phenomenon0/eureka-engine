#!/usr/bin/env bash
# isildur.sh (ISILDUR): the one entrypoint from Bend source to the library the apps link. Apps depend on this repo;
# this repo never reads an app tree (verify/vendor only write or read the Vendor dir you name).
#
#   ./isildur.sh emit             base/ + patches/ (VISION stack) -> work/mac/b-cam/sys1mac.c, checked = patches/CAM_SRC_SHA
#   ./isildur.sh lib              emit, then build-fast.sh: + patches/host-prefix.patch -> build/cam-{macos,ios,sim}/
#                                 libllbengine.a, build/cam/LLBEngine.xcframework (ZERO_AR_DATE=1: archives reproducible)
#   ./isildur.sh dist             lib, then dist/ = the app's Vendor layout: LLBEngine.xcframework/, macos/libllbengine.a,
#                                 SHA256SUMS (same format the app's scripts check) + ENGINE.pin (what the app pins)
#   ./isildur.sh verify <Vendor>  an app's ios/Vendor vs dist/: every archive member's object, the archive modulo ar
#                                 header dates, the header, Info.plist; and the app's LLBEngine.programTag if found
#   ./isildur.sh vendor <Vendor>  dist, then copy it into <Vendor> (LLBEngine.xcframework, macos/, SHA256SUMS, ENGINE.pin).
#                                 The app's lane reviews and commits that; nothing here commits in an app repo.
#
# Needs: Xcode (xcrun clang/libtool, xcodebuild), bun (the Bend compiler is TypeScript), Homebrew OpenBLAS (only for
# emit's Mac CLI link). No models, no network.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# (Eureka import) inputs and outputs are env-configurable; defaults are upstream's in-repo layout:
#   ISILDUR_BASE  pre-patch Bend tree (base/, checked against base.SHA256SUMS here)   default $here/base
#   ISILDUR_S1D   the S1D subset (checked against S1D.SHA256SUMS here)                 default $here/S1D
#   ISILDUR_OUT   where work/, build/ and dist/ are written (use a scratch dir)        default $here
OUTR=${ISILDUR_OUT:-$here}; export ISILDUR_OUT=$OUTR
W=$OUTR/work/mac; export W
export ZERO_AR_DATE=${ZERO_AR_DATE:-1}
CAM_SHA=$(cat "$here/patches/CAM_SRC_SHA")
PROGRAM_TAG=${CAM_SHA:0:8}   # replaced by the final emitted+host source hash after lib()

emit() {
  # the VISION tree must be the last one mac/build.sh wrote to work/mac/tree: build-fast.sh reads its llb_gguf.c
  mkdir -p "$W"
  VISION=1 O="$W/b-cam" bash "$here/mac/build.sh" > "$W/emit-vision.log" 2>&1 \
    || { tail -20 "$W/emit-vision.log"; echo "emit failed (log: $W/emit-vision.log)"; exit 3; }
  local got; got=$(shasum -a 256 "$W/b-cam/sys1mac.c" | cut -d' ' -f1)
  [ "$got" = "$CAM_SHA" ] || { echo "emitted C $got != patches/CAM_SRC_SHA $CAM_SHA"; exit 3; }
  echo "== emit: work/mac/b-cam/sys1mac.c $got"
}

lib() { emit; bash "$here/build-fast.sh"; }

dist() {
  lib
  PROGRAM_TAG=$(shasum -a 256 "$OUTR/build/cam-gen/src/sys1mac.c" | cut -c1-8)
  local D=$OUTR/dist B=$OUTR/build; rm -rf "$D"; mkdir -p "$D/macos"
  cp -R "$B/cam/LLBEngine.xcframework" "$D/"
  # xcodebuild lists the slices in no fixed order; sorted, Info.plist depends only on its content
  python3 "$here/scripts/xcf-plist-normalize.py" "$D/LLBEngine.xcframework/Info.plist"
  cp "$B/cam-macos/libllbengine.a" "$D/macos/"
  (cd "$D" && find ./LLBEngine.xcframework ./macos -type f | LC_ALL=C sort | xargs shasum -a 256) > "$D/SHA256SUMS"
  local commit dirty=""; commit=$(git -C "$here" rev-parse HEAD)
  git -C "$here" diff --quiet HEAD -- . || dirty=" (DIRTY: uncommitted changes in isildur; do not pin this)"
  {
    echo "# ENGINE.pin: the isildur build an app's ios/Vendor holds. Written by isildur.sh dist; check: isildur.sh verify"
    echo "isildur_commit=$commit$dirty"
    echo "emitted_c_device=$CAM_SHA"
    echo "emitted_c=$(shasum -a 256 "$B/cam-gen/src/sys1mac.c" | cut -d' ' -f1)"
    echo "program_tag=$PROGRAM_TAG"
    echo "header=$(shasum -a 256 "$B/cam/include/llb_engine.h" | cut -d' ' -f1)"
    for s in ios sim macos; do
      echo "objects_$s=$(cd "$B/cam-$s/obj" && shasum -a 256 *.o | awk '{printf "%s:%s ", $2, substr($1,1,16)}' | sed 's/ $//')"
    done
  } > "$D/ENGINE.pin"
  echo "== dist: $D"; cat "$D/ENGINE.pin"
}

verify() {  # verify <Vendor dir>
  local V=${1:?usage: isildur.sh verify <app>/ios/Vendor} D=$OUTR/dist fail=0
  V=$(cd "$V" && pwd)
  [ -f "$D/SHA256SUMS" ] || { echo "no dist/ yet: run ./isildur.sh dist"; exit 2; }
  local T; T=$(mktemp -d); trap 'rm -rf "$T"' RETURN
  for s in "ios LLBEngine.xcframework/ios-arm64/libllbengine.a" "sim LLBEngine.xcframework/ios-arm64-simulator/libllbengine.a" \
           "macos macos/libllbengine.a"; do
    set -- $s
    mkdir -p "$T/$1/v" "$T/$1/d"; (cd "$T/$1/v" && ar x "$V/$2"); (cd "$T/$1/d" && ar x "$D/$2")
    local n=0 bad=0 o
    for o in $(cd "$T/$1/d" && ls *.o); do
      n=$((n + 1)); cmp -s "$T/$1/v/$o" "$T/$1/d/$o" || { bad=$((bad + 1)); echo "   $1 $o DIFFERS"; }
    done
    [ "$(cd "$T/$1/v" && ls *.o | wc -l)" -eq "$n" ] || { echo "   $1: member count differs"; bad=$((bad + 1)); }
    local arc; arc=$(python3 "$here/scripts/ar-normalize.py" "$V/$2" "$D/$2") || bad=$((bad + 1))
    echo "== $1: $((n - bad))/$n objects identical; archive: $arc"
    [ "$bad" -eq 0 ] || fail=1
  done
  for f in LLBEngine.xcframework/Info.plist LLBEngine.xcframework/ios-arm64/Headers/llb_engine.h \
           LLBEngine.xcframework/ios-arm64-simulator/Headers/llb_engine.h; do
    cmp -s "$V/$f" "$D/$f" && echo "== $f identical" || { echo "== $f DIFFERS"; fail=1; }
  done
  (cd "$V" && shasum -a 256 -c --quiet SHA256SUMS) && echo "== $V/SHA256SUMS: the Vendor files match it" \
    || { echo "== $V/SHA256SUMS does not match its files"; fail=1; }
  local sw=$V/../Bend/Engines/LLB/LLBEngine.swift
  if [ -f "$sw" ]; then
    grep -q "programTag = \"$PROGRAM_TAG\"" "$sw" && echo "== LLBEngine.programTag = $PROGRAM_TAG" \
      || { echo "== LLBEngine.programTag != $PROGRAM_TAG ($sw)"; fail=1; }
  fi
  [ "$fail" -eq 0 ] && echo "VERIFY PASS: $V is isildur $(git -C "$here" rev-parse --short HEAD)'s engine" \
    || { echo "VERIFY FAIL"; exit 1; }
}

vendor() {  # vendor <Vendor dir>
  local V=${1:?usage: isildur.sh vendor <app>/ios/Vendor}
  [ -d "$V" ] || { echo "no such dir: $V"; exit 2; }
  dist
  rm -rf "$V/LLBEngine.xcframework" "$V/macos/libllbengine.a"; mkdir -p "$V/macos"
  cp -R "$OUTR/dist/LLBEngine.xcframework" "$V/"; cp "$OUTR/dist/macos/libllbengine.a" "$V/macos/"
  cp "$OUTR/dist/SHA256SUMS" "$OUTR/dist/ENGINE.pin" "$V/"
  echo "== vendored into $V. In the app: set LLBEngine.programTag = \"$PROGRAM_TAG\", review, commit."
}

case "${1:-}" in
  emit) emit ;; lib) lib ;; dist) dist ;; verify) shift; verify "$@" ;; vendor) shift; vendor "$@" ;;
  *) sed -n '2,17p' "$0"; exit 2 ;;
esac
