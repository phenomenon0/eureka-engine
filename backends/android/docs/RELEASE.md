# Release build (APKSIGN)

A debug APK is signed with the shared Android debug key and marked debuggable. Installers and scanners treat that as a test build.
The release build is signed with our own key, minified, and not debuggable.

```
cd android
JAVA_HOME=$HOME/.gradle/jdks/eclipse_adoptium-17-amd64-linux.2 ./gradlew --offline assembleRelease
# -> app/build/outputs/apk/release/app-release.apk  (signed v2 + v3, zipaligned by AGP)
```

## The key (outside every repo)

| | |
|---|---|
| Keystore | `~/.dsh/keys/bend-release.p12` (PKCS12, alias `bend`, RSA 4096, SHA384withRSA, valid until 2054-02-23) |
| Cert SHA-256 | `5e8cc5432e5e20f41547815ac70609cb70336967d83f4e3d25659a098b2620b8` |
| Gradle props | `~/.dsh/keys/bend-release.properties`: `storeFile`, `keyAlias`, `passwordFile`. Override the path with `-PbendSigning=<file>`. |
| Password | Lives only in `~/.config/bend-release.pass` (0600), the same pattern as the other `~/.config/*.pass` files. It is never in git or chat. Store and key share the password, as PKCS12 requires. |

If the props file is missing, `assembleRelease` still succeeds but leaves `app-release-unsigned.apk`, which does not install.

**Back up the keystore and the password file together, now, to offline media.** Every update must be signed with this key. If the key
is lost, installed copies can never be updated: users would have to uninstall, which deletes their model files. Example:
`tar czf - ~/.dsh/keys/bend-release.* ~/.config/bend-release.pass | gpg -c > /media/<usb>/bend-release-key.tgz.gpg`.

## Verify

```
BT=~/Android/Sdk/build-tools/36.1.0
$BT/zipalign -c -P 16 -v 4 app-release.apk | tail -1          # Verification successful
$BT/apksigner verify --print-certs -v app-release.apk         # cert SHA-256 above
$BT/apksigner verify -v --min-sdk-version 24 app-release.apk  # v2: true, v3: true
```

At minSdk 34, apksigner stops after v3 and reports "v2: false". Pass `--min-sdk-version 24` to make it check the v2 block too.

## What changed for release

- `signingConfigs.release` uses v2 + v3. v3 is set explicitly because AGP leaves it off by default.
- `isMinifyEnabled` and `isShrinkResources` are on. `proguard-rules.pro` keeps `ai.onnxruntime.**`, because ORT's JNI looks its Java classes up by name and the AAR ships no rules.
- WebView devtools are on only in debug builds (`BuildConfig.DEBUG`). On a **userdebug** ROM (the LineageOS Pixel), WebView turns devtools on for every app anyway, so `voice/test-voices.py` still drives the release build there. A stock `user` ROM does not do this.
- There is now a launcher icon (adaptive, `res/mipmap-anydpi-v26/ic_launcher.xml`). Before this, the installer showed the generic Android robot.
- `versionCode` is now 2 and `versionName` is 0.2. Bump both on every release.

## Switching a phone from debug to release

The debug and release builds are signed with different keys. `adb install -r` of one over the other fails with `INSTALL_FAILED_UPDATE_INCOMPATIBLE`.
Uninstalling deletes `Android/data/dev.bend.app/files`, which holds the 2.7 GB pack. Move it aside first:

```
adb shell 'mkdir /sdcard/bend-stash && mv /sdcard/Android/data/dev.bend.app/files/* /sdcard/bend-stash/'
adb uninstall dev.bend.app && adb install bend-release.apk
adb shell am start -W -n dev.bend.app/.MainActivity; adb shell am force-stop dev.bend.app   # the app creates its own dir
adb shell 'mv /sdcard/bend-stash/* /sdcard/Android/data/dev.bend.app/files/ && rmdir /sdcard/bend-stash'
```

## Before giving it to anyone

- **Licence line:** `libespeak.so` and `espeak-ng-data` are espeak-ng, GPL-3.0. They are the fallback TTS for phones with no TTS engine. `voice/VOICES.md` says
  every shipped piece is permissive, but that is true only of the Kokoro set. Either ship the GPL notice plus a source offer, or drop espeak from the release build.
- **Model host:** `Voices.CATALOG_URL` is still `https://models.bend.invalid/…`, and the 2.7 GB chat pack is still pushed by adb. A user who installs the APK alone gets
  "no models". Decide where the pack and voice catalog are hosted before distributing.
- **Cleartext:** `usesCleartextTraffic="true"` is required by the in-app `127.0.0.1` server and the LAN shop test. A
  `network_security_config` limited to localhost would be tighter. It is not done yet.
- **Play:** targetSdk 35. Google Play raises its minimum target each August (35 since 2025-08-31). Expect 36 to be required now, but check the Play Console; not verified here. Bump `compileSdk` and `targetSdk` to 36 before
  any Play upload; `android-36` is already installed in the SDK.
