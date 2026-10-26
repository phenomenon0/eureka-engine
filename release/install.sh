#!/bin/sh
# llb installer.
#   curl -fsSL https://github.com/phenomenon0/eureka-engine/releases/latest/download/install.sh | sh
#   sh install.sh [--variant cuda|cuda12|cpu] [--dry-run]
# LLB_MANIFEST overrides the manifest URL (default: the latest GitHub release).
# Detects OS/arch, downloads the tarball the manifest lists, verifies sha256 BEFORE unpacking,
# unpacks under ~/.local/share/llb and links ~/.local/bin/llb. Read it before you run it.
set -eu

dry=0
variant=${LLB_VARIANT-cuda}
variant_explicit=0
[ "${LLB_VARIANT+x}" != x ] || variant_explicit=1
usage() { echo "usage: sh install.sh [--variant cuda|cuda12|cpu] [--dry-run] (variants select Linux x86_64 bundles; LLB_VARIANT is also accepted)"; }
while [ "$#" -gt 0 ]; do
  case $1 in
    --dry-run) dry=1; shift ;;
    --variant) [ "$#" -ge 2 ] || { usage >&2; exit 2; }; variant=$2; variant_explicit=1; shift 2 ;;
    --variant=*) variant=${1#--variant=}; variant_explicit=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
case $variant in cuda|cuda12|cpu) ;; *) echo "install.sh: unknown Linux variant '$variant' (cuda, cuda12, cpu)" >&2; exit 2 ;; esac
die() { echo "install.sh: $*" >&2; exit 1; }
say() { echo "$*"; }

os=${LLB_UNAME_S:-$(uname -s)}   # LLB_UNAME_* exist only so the detection can be tested
arch=${LLB_UNAME_M:-$(uname -m)}
case "$os/$arch" in
  Linux/x86_64) plat=linux-x86_64-$variant
    case $variant in
      cuda) need="native Linux, glibc 2.34+, NVIDIA Ampere or newer with concurrent managed memory access; NVRTC 13.1 is bundled, no CUDA toolkit needed; driver R580+ (CUDA 13.0); tested on driver 580.119.02 and an RTX 3090" ;;
      cuda12) need="native Linux, glibc 2.34+, NVIDIA Ampere or newer with concurrent managed memory access; NVRTC 12.9 is bundled, no CUDA toolkit needed; driver R535+ (CUDA 12.2); tested on driver 580.119.02 and an RTX 3090" ;;
      cpu) need="Linux x86_64, glibc 2.34+; experimental CPU-only build: read the CPU findings and memory limits; successful installation does not establish successful inference or usability" ;;
    esac;;
  Darwin/arm64) [ "$variant_explicit" = 0 ] || die "--variant / LLB_VARIANT selects Linux x86_64 bundles; omit it on macOS."
    plat=macos-arm64
    need="an Apple silicon Mac (tested on M3 Ultra only; unified memory needed depends on the model)";;
  Darwin/*) die "unsupported: $os/$arch. Only Apple silicon (arm64) is built; the Metal path is not built or tested for Intel Macs.";;
  Linux/*) die "unsupported: $os/$arch. Only x86_64 Linux variants are built; no $arch build exists.";;
  MINGW*|MSYS*|CYGWIN*|Windows*) die "unsupported: $os. There is no Windows release yet. WSL2 is not supported: the CUDA engine needs concurrent managed memory access, which NVIDIA does not provide on Windows or WSL2.";;
  *) die "unsupported: $os/$arch. Built for Linux x86_64 and macOS arm64 only.";;
esac

manifest=${LLB_MANIFEST:-https://github.com/phenomenon0/eureka-engine/releases/latest/download/manifest.json}
command -v curl >/dev/null 2>&1 || die "curl is required"
if command -v sha256sum >/dev/null 2>&1; then shacmd="sha256sum"; elif command -v shasum >/dev/null 2>&1; then shacmd="shasum -a 256"; else die "need sha256sum or shasum"; fi

mtmp=$(mktemp); trap 'rm -f "$mtmp"' EXIT
curl -fsSL "$manifest" -o "$mtmp" || die "cannot fetch manifest $manifest"
# manifest.json is pretty-printed, one field per line, 4-space platform keys (tools/publish-artifact.sh keeps it so)
field() { awk -v p="$plat" -v k="$1" '
  $0 ~ "^    \"" p "\": \\{" { inb=1; next }
  inb && /^    \}/ { exit }
  inb && $0 ~ "^      \"" k "\":" { v=$0; sub(/^[^:]*: */,"",v); sub(/,$/,"",v); gsub(/^"|"$/,"",v); print v; exit }' "$mtmp"; }

status=$(field status)
say "platform  : $plat ($os/$arch)"
say "manifest  : $manifest"
say "status    : ${status:-missing}"
if [ "$status" != "available" ]; then
  say "no download is available for $plat: $(field missing)"
  [ "$dry" = 1 ] && say "(dry run: nothing would be installed)"
  exit 1
fi
url=$(field url); sum=$(field sha256); ver=$(field version)
case "$url" in */*://*|*://*) ;; *) url=${manifest%/*}/$url;; esac   # relative url -> next to the manifest
dest=${LLB_PREFIX:-$HOME/.local/share/llb}
bin=${LLB_BIN:-$HOME/.local/bin}
say "version   : $ver"
say "download  : $url"
say "sha256    : $sum (checked before unpacking)"
say "unpack to : $dest"
say "link      : $bin/llb -> <unpacked dir>/llb"
say "you need  : $need"
if [ "$dry" = 1 ]; then say "(dry run: nothing downloaded, unpacked or linked)"; exit 0; fi

work=$(mktemp -d); trap 'rm -rf "$work" "$mtmp"' EXIT
curl -fsSL "$url" -o "$work/pkg.tar.gz" || die "download failed: $url"
got=$($shacmd "$work/pkg.tar.gz" | cut -d' ' -f1)
[ "$got" = "$sum" ] || die "sha256 mismatch: manifest says $sum, downloaded file is $got. Nothing was unpacked."
say "sha256 ok"
top=$(tar -tzf "$work/pkg.tar.gz" | head -1 | cut -d/ -f1)
case "$top" in ""|.|..|/*) die "unexpected tarball layout";; esac
mkdir -p "$dest" "$bin"
tar -xzf "$work/pkg.tar.gz" -C "$dest"
[ -f "$dest/$top/llb" ] || die "tarball has no $top/llb; unpacked to $dest/$top, no link made"
ln -sf "$dest/$top/llb" "$bin/llb"
say "unpacked  : $dest/$top"
say "linked    : $bin/llb"
case ":$PATH:" in *":$bin:"*) ;; *) say "note: $bin is not on your PATH";; esac
say "next: llb doctor    (checks this machine)"
if [ "$plat" = linux-x86_64-cpu ]; then
  say "then: read $dest/$top/CPU-STATUS.txt and README.md; this CPU variant is experimental"
  say "command: llb run /path/to/model.gguf \"your text\" --threads 16 -n 8"
else
  say "then: llb run 27b \"your text\"    (needs the Qwen3.8-27B Q4_K_M GGUF; see $dest/$top/README.md)"
fi
