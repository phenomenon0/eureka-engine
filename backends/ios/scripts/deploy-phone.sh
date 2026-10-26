#!/bin/bash
# Builds Bend for the operator's iPhone, installs it, launches it and captures a screenshot.
# Mirrors the vizvim device flow (xcodebuild → devicectl install → launch).
#
#   scripts/deploy-phone.sh                 deploy to the default phone
#   UDID=<udid> scripts/deploy-phone.sh     deploy to another device
#   scripts/deploy-phone.sh --skip-web      reuse the current dist/ (no npm run build)
#   scripts/deploy-phone.sh --dry-run       preflight + print the plan; no build, no device calls
#
# Env: UDID, WAIT_SECS (device wait, default 180), BUILD_SECS (xcodebuild limit, default 1200),
#      CONFIG (default Debug), SHOT (screenshot path, default shots/device-<time>.png),
#      BEND_ENGINE (llb = OUR engine, the default; llama = the R7 llama.cpp fallback; docs/ENGINE-NATIVE.md).
# Exit codes: 2 usage/preflight, 3 device never connected, 4 signing/provisioning,
#             5 build, 6 install, 7 launch, 8 screenshot.
#
# Run it from a logged-in GUI session (Terminal on the Mac, or a LaunchAgent like vizvim's
# viz-run-device.sh). Over plain ssh the login keychain is locked and codesign fails.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UDID="${UDID:?set UDID to the device UDID (xcrun devicectl list devices)}"
BUNDLE_ID="com.example.bend"
TEAM="${BEND_TEAM_ID:?set BEND_TEAM_ID to your Apple team ID}"
CONFIG="${CONFIG:-Debug}"
BEND_ENGINE="${BEND_ENGINE:-llb}"
case "$BEND_ENGINE" in llb) SCHEME=Bend ;; llama) SCHEME=BendLlama ;; *) echo "deploy-phone: BEND_ENGINE must be llb or llama" >&2; exit 2 ;; esac
export BEND_ENGINE
WAIT_SECS="${WAIT_SECS:-180}"
BUILD_SECS="${BUILD_SECS:-1200}"
STAMP="$(date +%Y%m%d-%H%M%S)"
DERIVED="$ROOT/ios/build/device"
LOG_DIR="$ROOT/ios/build/logs"
SHOT="${SHOT:-$ROOT/shots/device-$STAMP.png}"

SKIP_WEB=0 DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --skip-web) SKIP_WEB=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "deploy-phone: unknown option $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/deploy-$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

log() { printf '\n[%s] == %s\n' "$(date +%H:%M:%S)" "$*"; }
info() { printf '[%s]    %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { local code="$1"; shift; printf '\n[%s] FAILED (exit %s): %s\n' "$(date +%H:%M:%S)" "$code" "$*"; printf 'Full log: %s\n' "$LOG"; exit "$code"; }

# Runs a command with a wall-clock limit (macOS ships no `timeout`). Returns 124 on timeout.
with_limit() {
  local secs="$1"; shift
  local flag; flag="$(mktemp -t bend-limit)"; rm -f "$flag"
  "$@" & local pid=$!
  ( sleep "$secs"; touch "$flag"; kill -TERM "$pid" 2>/dev/null; sleep 5; kill -KILL "$pid" 2>/dev/null ) >/dev/null 2>&1 & local watchdog=$!
  wait "$pid"; local status=$?
  pkill -P "$watchdog" 2>/dev/null; kill "$watchdog" 2>/dev/null; wait "$watchdog" 2>/dev/null
  if [ -e "$flag" ]; then rm -f "$flag"; status=124; fi
  return "$status"
}

# Prints the device's tunnel state ("connected", "disconnected", "unavailable", or "missing").
tunnel_state() {
  local json="$LOG_DIR/devices.json"
  xcrun devicectl list devices -q -t 20 -j "$json" >/dev/null 2>&1 || { echo "unknown"; return; }
  python3 - "$json" "$UDID" <<'PY'
import json, sys
devices = json.load(open(sys.argv[1])).get("result", {}).get("devices", [])
for d in devices:
    if d.get("hardwareProperties", {}).get("udid") == sys.argv[2]:
        print(d.get("connectionProperties", {}).get("tunnelState", "unknown")); break
else:
    print("missing")
PY
}

log "Bend ($SCHEME, engine $BEND_ENGINE) → device $UDID ($CONFIG)"
info "log: $LOG"

# ---------------------------------------------------------------- preflight
log "preflight"
for tool in xcodegen xcodebuild xcrun python3 npm rsync; do
  command -v "$tool" >/dev/null || fail 2 "$tool not found on PATH"
done
info "$(xcodebuild -version | tr '\n' ' ')"
[ -f "$ROOT/ios/project.yml" ] || fail 2 "ios/project.yml missing"
if [ "$SKIP_WEB" = 1 ] && [ ! -f "$ROOT/dist/index.html" ]; then fail 2 "--skip-web but dist/ is missing; run npm run build"; fi

if [ "$DRY_RUN" = 1 ]; then
  log "dry run: plan"
  info "1. scripts/build-ios.sh --sync-only$([ "$SKIP_WEB" = 1 ] && echo ' --skip-web')   (web build, sync ios/web, xcodegen)"
  info "2. wait ≤${WAIT_SECS}s for tunnelState=connected (devicectl list devices + device info details)"
  info "3. xcodebuild -project ios/Bend.xcodeproj -scheme $SCHEME -configuration $CONFIG -destination 'platform=iOS,id=$UDID' -allowProvisioningUpdates -derivedDataPath ios/build/device build   (≤${BUILD_SECS}s)"
  info "4. xcrun devicectl device install app --device $UDID ios/build/device/Build/Products/$CONFIG-iphoneos/Eureka.app"
  info "5. xcrun devicectl device process launch --device $UDID --terminate-existing $BUNDLE_ID"
  info "6. xcrun devicectl device capture screenshot --device $UDID --destination $SHOT"
  log "dry run: no build, no device calls"
  exit 0
fi

# ---------------------------------------------------------------- web + project
log "web bundle + xcodegen"
SYNC_ARGS=(--sync-only); [ "$SKIP_WEB" = 1 ] && SYNC_ARGS+=(--skip-web)
"$ROOT/scripts/build-ios.sh" "${SYNC_ARGS[@]}" || fail 5 "web build / sync / xcodegen failed (see above)"

# ---------------------------------------------------------------- device
log "waiting for the device (≤${WAIT_SECS}s): unlock it, keep it on the cable or the same Wi-Fi"
deadline=$(( $(date +%s) + WAIT_SECS ))
state="$(tunnel_state)"
while [ "$state" != "connected" ]; do
  [ "$state" = "missing" ] && fail 3 "device $UDID is not paired with this Mac (xcrun devicectl list devices). Pair it in Xcode ▸ Devices and Simulators."
  [ "$(date +%s)" -ge "$deadline" ] && fail 3 "device $UDID still '$state' after ${WAIT_SECS}s. Unlock the phone, check the cable / Wi-Fi, and that Developer Mode is on; then rerun."
  info "state: $state; asking CoreDevice to connect"
  # A details request brings up the tunnel for a paired device that is reachable.
  xcrun devicectl device info details --device "$UDID" -q -t 20 >/dev/null 2>&1 || true
  sleep 5
  state="$(tunnel_state)"
done
info "state: connected"
xcrun devicectl device info details --device "$UDID" -t 20 2>/dev/null | grep -iE '^\s*(name|marketingName|osVersionNumber|developerModeStatus|ddiServicesAvailable):' | sed 's/^ */    /' || true

# ---------------------------------------------------------------- build
log "xcodebuild for the device (≤${BUILD_SECS}s)"
BUILD_LOG="$LOG_DIR/xcodebuild-$STAMP.log"
with_limit "$BUILD_SECS" xcodebuild -project "$ROOT/ios/Bend.xcodeproj" -scheme "$SCHEME" -configuration "$CONFIG" \
  -destination "platform=iOS,id=$UDID" -allowProvisioningUpdates \
  -derivedDataPath "$DERIVED" build > "$BUILD_LOG" 2>&1
status=$?
tail -n 25 "$BUILD_LOG"

if [ "$status" -ne 0 ]; then
  [ "$status" = 124 ] && fail 5 "xcodebuild did not finish in ${BUILD_SECS}s (log: $BUILD_LOG)"
  if grep -qE 'errSecInternalComponent|User interaction is not allowed|No Accounts' "$BUILD_LOG"; then
    # The keychain's ACL refuses codesign outside the GUI session. Don't fight it: a .command file opened in
    # Terminal runs inside the session, where codesign is allowed (open(1) hands it to Terminal.app).
    SIGN=/tmp/bend-sign.command
    cat > "$SIGN" <<CMD
#!/bin/bash
# Bend: build + sign + install + launch from a Terminal window (codesign needs the GUI session's keychain).
cd "$ROOT" && BEND_ENGINE=$BEND_ENGINE UDID=$UDID CONFIG=$CONFIG scripts/deploy-phone.sh --skip-web
echo; read -r -p "done; press return to close " _
CMD
    chmod +x "$SIGN"
    fail 4 "this session cannot sign (errSecInternalComponent or \"No Accounts\": the keychain / Xcode account is only reachable from the GUI session).
    One click: open $SIGN   (or double-click it in Finder). It reruns this deploy in a Terminal window.
    Or unlock first: security unlock-keychain ~/Library/Keychains/login.keychain-db"
  fi
  if grep -qiE 'No profiles for|requires a provisioning profile|provisioning profile|No Accounts|No signing certificate|Signing for "Bend"|not registered|doesn.t include the currently selected device|Automatic signing' "$BUILD_LOG"; then
    fail 4 "code signing / provisioning failed for $BUNDLE_ID (team $TEAM). Nothing was installed.
    Remedy (one time, operator):
      1. Xcode ▸ Settings ▸ Accounts: signed in to the account for team $TEAM.
      2. open ios/Bend.xcodeproj ▸ target $SCHEME ▸ Signing & Capabilities: 'Automatically manage
         signing' on, Team $TEAM. Pick the phone as the run destination so Xcode registers it and
         creates the 'iOS Team Provisioning Profile: $BUNDLE_ID'. (Or create an App ID +
         development profile for $BUNDLE_ID that includes device $UDID in the developer portal.)
      3. Rerun scripts/deploy-phone.sh.
    Signing errors: $(grep -iE 'error:' "$BUILD_LOG" | sort -u | head -5 | tr '\n' ' ')"
  fi
  fail 5 "xcodebuild failed (log: $BUILD_LOG)"
fi

APP="$DERIVED/Build/Products/$CONFIG-iphoneos/Eureka.app"
[ -d "$APP" ] || APP="$(find "$DERIVED/Build/Products" -maxdepth 2 -name Eureka.app -path '*iphoneos*' | head -1)"
[ -n "$APP" ] && [ -d "$APP" ] || fail 5 "build succeeded but Eureka.app was not found under $DERIVED/Build/Products"
[ -f "$APP/web/index.html" ] || fail 5 "$APP has no web/index.html; the web bundle was not copied in"
info "app: $APP"

# ---------------------------------------------------------------- install, launch, screenshot
log "install"
xcrun devicectl device install app --device "$UDID" -t 300 "$APP" || fail 6 "devicectl install failed. If it mentions trust or Developer Mode: on the phone, Settings ▸ Privacy & Security ▸ Developer Mode, and Settings ▸ General ▸ VPN & Device Management ▸ trust the developer."

log "launch"
xcrun devicectl device process launch --device "$UDID" -t 60 --terminate-existing "$BUNDLE_ID" || fail 7 "devicectl launch failed. A locked phone refuses launches; unlock it and rerun with --skip-web."

log "screenshot"
sleep "${SHOT_DELAY:-5}" # demo engines connect and seed the thread
mkdir -p "$(dirname "$SHOT")"
xcrun devicectl device capture screenshot --device "$UDID" -t 60 --destination "$SHOT" || fail 8 "app is installed and running, but the screenshot failed"
[ -s "$SHOT" ] || fail 8 "screenshot file is empty: $SHOT"

log "done"
info "installed + launched $BUNDLE_ID on $UDID"
info "screenshot: $SHOT"
info "log: $LOG"
