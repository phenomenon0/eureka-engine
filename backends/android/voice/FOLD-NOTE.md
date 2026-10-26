# FOLD-NOTE: kokorodroid → Android trunk

**Status (droidfold, 2026-10-09): folded.** Branch `droidfold` fast-forwarded 221c5b15 → 84254449 with no conflicts. The build gives rc 0, an APK of
113,531,826 bytes, sha256 `2c223095…ab9582`, and every entry matches kokorodroid's APK by CRC. On the Pixel, `test-voices.py --packs --host` passed
26/26. To adopt, move `android-vis-r1` to the `droidfold` head. The kokorodroid row is closed. The fold queue below is unchanged: apkfetch, appdev,
andvis and capvision are still off trunk. Test gap: the shop half of `test-voices.py` resets the source but leaves the test catalog in
`cache/voice-catalog.json`. The offline shop then lists dead `test_*` rows. droidfold cleared it on the phone with `run-as … rm`; the fix should be
to clear the cache in the suite's `finally`.

**Trunk.** The trunk is `android-vis-r1` at 221c5b15, checked out in the worktree `llama-bend-androidapp`. It is the newest lineage that carries the
`dev.bend.app` app. Lane branch: `kokorodroid` in the worktree `~/Documents/Project/kokorodroid/app`, one commit on 221c5b15. It is local and unpushed.
Fold it with a fast-forward or cherry-pick. The diff is confined to `android/`.

**What lands.**

New:
- `Kokoro.kt`: the ORT runtime and the G2P port.
- `Voices.kt`: packs, choice, shop.
- `assets/web/voices.html`.
- `voice/`: bundled.json, catalog.json, fetch-voice.sh, page.py, test-voices.py, the g2p golden, VOICES.md.
- `tools/serve-models.py`.

Edited:
- `Voice.kt`: Kokoro first, then platform TTS, then espeak; preview.
- `Server.kt`: `/voices*` routes and the voices.html injection and hook.
- `MainActivity.kt`: Voices wiring; 127.0.0.1 navigation kept in-app; Back.
- `build.gradle.kts`: ORT AAR, `noCompress onnx`.
- `.gitignore`.

**Not in git (fetch before building).** Run `android/voice/fetch-voice.sh`. It fetches the assets (about 100 MB) and the ORT AAR, all sha-pinned. The AAR is required to
compile. Without the assets the app still builds and runs: `Voices.available` is false, and speech falls back to the platform TTS (by design; not exercised in this lane).

**Merge points to watch.**
- `Server.kt` and `Voice.kt` are also touched by the vis lane (image turns). This branch sits on top of vis R1, so there is no conflict today.
- `MainActivity` now sets a `WebViewClient`. Before this, every navigation left the app, so this is a behavior change for any future in-app link
  to 127.0.0.1. Off-host links still open outside the app.
- `Voices.download` repeats the apkfetch `Models.kt` resume and sha logic. Once apkfetch folds, factor the shared fetch out of the
  two. `tools/serve-models.py` is a verbatim copy of apkfetch's.

**Isolated Android lineages (the fold queue).** None of these are on trunk. Each is ahead by the count shown, measured against `android-vis-r1`.

| Branch | Head | Ahead | Base | Touches |
|---|---|---|---|---|
| apkfetch | 3db394cd | 1 | 98ff68fd (on trunk) | 8 `android/` files: Models.kt downloader, setup.html, serve-models.py |
| appdev | 14cd6833 | 1 | 98ff68fd | 15 `android/` files |
| andvis | 72520c72 | 1 | 98ff68fd | 15 `android/` files |
| capvision | 7cbb909f | 1 | 98ff68fd | 5 `android/` files |
| andbuild | e99534e7 | 491 | 48b9e39a | receipts, tools, reports, docs, kernels; no `android/` files |
| andbench / andpow / devphone / vvoice | b66cf65b / 0409eed3 / 515db61d / 3281a7f2 | 1–2 | 48b9e39a | receipts and reports only (devphone also `tools/devphone`) |

The first four carry app code and need a real fold; apkfetch is the one this lane depends on. The rest are evidence and tooling,
and can fold without touching the app.
