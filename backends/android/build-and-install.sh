#!/usr/bin/env bash
# One command: engine -> APK -> install -> (first time) model files -> launch. Every adb op holds /tmp/pixel6a.lock.
#   ./build-and-install.sh            build + install + push models if absent + launch
#   AUTORUN=1 ./build-and-install.sh  launch with the web app's autorun (one chat turn + decisions)
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); cd "$here"
SERIAL=${SERIAL:?set SERIAL to the device serial from adb devices}
ADB="${ADB:-adb} -s $SERIAL"
PACK=${PACK:?set PACK to the Kev-4B Q4_K_M pack dir}            # Kev-4B Q4_K_M pack (pack.bin, tokenizer.bin, manifest.json, index.tsv, chat.flags)
HEAD=${HEAD:?set HEAD to kev-head-q8.gguf}  # kev head GGUF (tools/feddecide mkhead.py)
export JAVA_HOME=${JAVA_HOME:-$HOME/.gradle/jdks/eclipse_adoptium-17-amd64-linux.2}
D=/sdcard/Android/data/dev.bend.app/files
lock() { flock /tmp/pixel6a.lock "$@"; }

engine/build.sh android
cp build/engine/kev-engine-android app/src/main/jniLibs/arm64-v8a/libbendengine.so
[ -f local.properties ] || echo "sdk.dir=${ANDROID_HOME:?set ANDROID_HOME}" > local.properties
./gradlew --offline -q assembleDebug
lock $ADB shell dumpsys battery | grep level
lock $ADB install -r app/build/outputs/apk/debug/app-debug.apk
if ! lock $ADB shell "test -f $D/pack/pack.bin && test -f $D/kev-head-q8.gguf"; then
  # The app must create its own external dir first (a dir made by adb shell is not the app's).
  lock $ADB shell am start -W -n dev.bend.app/.MainActivity; sleep 4; lock $ADB shell am force-stop dev.bend.app
  lock $ADB shell mkdir -p $D/pack
  lock $ADB push "$PACK"/{manifest.json,index.tsv,chat.flags,tokenizer.bin,pack.bin} $D/pack/
  lock $ADB push "$HEAD" $D/
fi
lock $ADB shell am force-stop dev.bend.app
lock $ADB shell am start -n dev.bend.app/.MainActivity ${AUTORUN:+--ez autorun true}
