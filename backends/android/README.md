# Android

Android phone, CPU. Tested on a Pixel 6a.

## Source

Branch `ship/android/voice/ship-wip/app` on this repository's remote. The
engine is `engine.bend`, built into `libbendengine.so`.

## Build

```sh
git fetch origin ship/android/voice/ship-wip/app
git worktree add ../llb-android origin/ship/android/voice/ship-wip/app
cd ../llb-android/android
./gradlew assembleDebug
```

## Result

Kev-4B Q4_K_M: 3,176.4 ms for a 19-token prompt and a 2-token reply.

## Gaps

Runs on the CPU. There is no Vulkan build yet.
