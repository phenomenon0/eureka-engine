#!/bin/bash
# Builds the Bend iOS shell for the simulator from the current web app.
#
#   scripts/build-ios.sh              npm run build → sync dist/ into ios/web → xcodegen → simulator build
#   scripts/build-ios.sh --launch     …then install + launch in the booted simulator
#   scripts/build-ios.sh --shot FILE  …then also screenshot the simulator into FILE (implies --launch)
#   scripts/build-ios.sh --sync-only  stop after xcodegen (scripts/deploy-phone.sh uses this)
#   scripts/build-ios.sh --skip-web   reuse the existing dist/ instead of running npm run build
#
# Env: SIM_NAME (default "iPhone 18 Pro"), CONFIG (default Debug),
#      BEND_ENGINE: llb (default: OUR engine, ios/Vendor/LLBEngine.xcframework, scheme Bend) or llama (the R7
#        fallback: llama.cpp + llama-server, scheme BendLlama; see docs/ENGINE-NATIVE.md),
#      LLAMA_XCFRAMEWORK (BEND_ENGINE=llama only; default ~/bend-phone-prep/ios/llama.xcframework).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IOS="$ROOT/ios"
SIM_NAME="${SIM_NAME:-iPhone 18 Pro}"
CONFIG="${CONFIG:-Debug}"
DERIVED="$IOS/build/sim"
BUNDLE_ID="com.example.bend"

SYNC_ONLY=0 SKIP_WEB=0 LAUNCH=0 SHOT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --sync-only) SYNC_ONLY=1 ;;
    --skip-web) SKIP_WEB=1 ;;
    --launch) LAUNCH=1 ;;
    --shot) SHOT="${2:?--shot needs a file path}"; LAUNCH=1; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "build-ios: unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

log() { printf '\n== %s\n' "$*"; }
die() { printf '\nbuild-ios: %s\n' "$*" >&2; exit 1; }

command -v xcodegen >/dev/null || die "xcodegen not found. Install it with: brew install xcodegen"
command -v xcodebuild >/dev/null || die "xcodebuild not found. Install Xcode and run: sudo xcode-select -s /Applications/Xcode.app"

if [ "$SKIP_WEB" = 0 ]; then
  log "web: npm run build"
  (cd "$ROOT" && npm run build --silent)
fi
[ -f "$ROOT/dist/index.html" ] || die "dist/index.html missing. Run npm run build (or drop --skip-web)."

log "web: sync dist/ → ios/web/"
mkdir -p "$IOS/web"
rsync -a --delete "$ROOT/dist/" "$IOS/web/"
# Absolute asset URLs would resolve to the filesystem root under file://; vite.config.ts sets base './'.
if grep -Eq '(src|href)="/assets/' "$IOS/web/index.html"; then
  die "dist/index.html uses absolute /assets/ URLs; keep base: './' in vite.config.ts."
fi

# VOICEREAL: onnxruntime + the neural voice's files (pinned; fetched once, verified every build).
log "voice: scripts/fetch-voice.sh (onnxruntime + Supertonic 2 F1 → ios/Vendor)"
"$ROOT/scripts/fetch-voice.sh"

BEND_ENGINE="${BEND_ENGINE:-llb}"
case "$BEND_ENGINE" in
  llb)
    SCHEME=Bend
    # OUR engine: engine-earlystop's LLBEngine.xcframework, checked in under ios/Vendor (not rebuilt here).
    log "engine: LLBEngine (ios/Vendor, emitted C bccd7cea…, resident session)"
    (cd "$IOS/Vendor" && shasum -a 256 -c SHA256SUMS --quiet) || die "ios/Vendor does not match ios/Vendor/SHA256SUMS"
    ;;
  llama)
    SCHEME=BendLlama
    # The R7 fallback: llama.cpp (+ the in-process llama-server), built by bend-phone-prep/ios/build-llama-ios.sh.
    LLAMA_XCFRAMEWORK="${LLAMA_XCFRAMEWORK:-$HOME/bend-phone-prep/ios/llama.xcframework}"
    [ -f "$LLAMA_XCFRAMEWORK/Info.plist" ] || die "llama.xcframework not found at $LLAMA_XCFRAMEWORK. Set LLAMA_XCFRAMEWORK, or build it with bend-phone-prep/ios/build-llama-ios.sh."
    mkdir -p "$IOS/Frameworks"
    ln -sfn "$LLAMA_XCFRAMEWORK" "$IOS/Frameworks/llama.xcframework"
    log "engine: llama.cpp fallback ($LLAMA_XCFRAMEWORK)"
    ;;
  *) die "BEND_ENGINE must be llb or llama (got $BEND_ENGINE)" ;;
esac

log "xcodegen generate"
(cd "$IOS" && xcodegen generate --quiet)

[ "$SYNC_ONLY" = 1 ] && { log "synced; project at ios/Bend.xcodeproj"; exit 0; }

log "xcodebuild: $SCHEME for '$SIM_NAME' ($CONFIG)"
xcodebuild -project "$IOS/Bend.xcodeproj" -scheme "$SCHEME" -configuration "$CONFIG" \
  -sdk iphonesimulator -destination "platform=iOS Simulator,name=$SIM_NAME" \
  -derivedDataPath "$DERIVED" build -quiet
APP="$DERIVED/Build/Products/$CONFIG-iphonesimulator/Eureka.app"
[ -d "$APP" ] || die "build finished but $APP is missing"
echo "built $APP"

[ "$LAUNCH" = 1 ] || exit 0

log "simulator: install + launch on '$SIM_NAME'"
UDID="$(xcrun simctl list devices available | grep -F "    $SIM_NAME (" | grep -m1 -oE '[0-9A-F-]{36}' || true)"
[ -n "$UDID" ] || die "no available simulator named '$SIM_NAME' (xcrun simctl list devices available)"
xcrun simctl bootstatus "$UDID" -b >/dev/null
xcrun simctl install "$UDID" "$APP"
xcrun simctl terminate "$UDID" "$BUNDLE_ID" >/dev/null 2>&1 || true
xcrun simctl launch "$UDID" "$BUNDLE_ID"

if [ -n "$SHOT" ]; then
  # The demo engines connect after ~0.4s and seed the thread; give the page time to settle.
  sleep "${SHOT_DELAY:-4}"
  mkdir -p "$(dirname "$SHOT")"
  xcrun simctl io "$UDID" screenshot "$SHOT" >/dev/null
  echo "screenshot $SHOT"
fi
