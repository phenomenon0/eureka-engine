# iOS engine (4B / Kev phone engine)

This directory builds `LLBEngine.xcframework`, the static library that the
iPhone app links (`../Vendor/`), plus a macOS slice of the same library. The
engine runs Qwen3.5-4B, including the Kev-4B fine-tune with its Q4_K
embedding, on Metal, inside the app process. It provides:

- `llb_load`, `llb_generate`, and `llb_decide` (the Jev decision head).
- The resident session, `llb_open` / `llb_close`, with early stop.
- 4B vision: `llb_classify_rgb`, `llb_generate_rgb`, `llb_vision_keep`, and
  prefix snapshots.

The C API is [`engine/llb_engine.h`](engine/llb_engine.h).

## Source

Imported from `github.com/phenomenon0/isildur`, branch `main`, commit
`8ff026e` (2026-10-09, "fresh-clone receipt"). That repository is private.

| Path | Contents |
|---|---|
| `isildur.sh` | The entry point: `emit`, `lib`, `dist`, `verify`, `vendor`. |
| `build-fast.sh` | Builds the three library slices and the xcframework from the emitted C (`isildur.sh lib`). |
| `mac/` | `build.sh`: base tree, then patches, then `bun` emit, then the Mac CLI. `base_tree.sh` copies and checks `base/`. `metal-sys1mac.patch` and `engine-sys1mac.patch` are the sys1mac deltas (already folded into `base/`). `fresh36.sh`, `fresh41.sh` and `base_tree-fresh41.sh` are the legacy recipe, kept for provenance; their lane trees are gone. |
| `patches/` | The 4B patch stack, `SHA256SUMS`, `CAM_SRC_SHA`, `host-prefix.patch`, and the lane hash notes. |
| `engine/` | The native wrapper `llb_engine.{h,m}` and the Mac harnesses (`llb_host.c`, `llb_repeat.c`, `llb_resident.c`, `llb_guard_test.c`). |
| `head/` | The Jev decision head, from jev-head `8736c62`, hash-checked (`head/SHA256SUMS`). |
| `vision/host/`, `vision/lib/`, `vision/third_party/` | The 4B Metal vision tower (`vitm.m`, `vitm.metal`), preprocessing (`vpre`), the mmproj loader (`vmm`), the resident tower (`llbv`), and stb_image 2.30. |
| `ios/engine-host/` | EngineHost, a standalone probe app for a device (XcodeGen). |
| `tools/cam_host.c`, `scripts/*.py` | The macOS camera harness that `lib` links, and the dist normalizers (`Info.plist` order, `ar` dates). |
| `base.SHA256SUMS`, `base.PROVENANCE`, `S1D.SHA256SUMS`, `S1D.commit` | Pins for the two inputs that are not vendored. See [Inputs](#inputs). |

Left out: `receipts/`, `results/`, `runs/`, `dist/`, `prompts/`, `docs/lanes/`,
the lane harness scripts, the legacy library recipes (`build-ios.sh`,
`build-cam.sh`, `build-img.sh`; they need older wrapper tags), `pc.sh`,
`fetch_s1d.sh`, binaries, models and xcframeworks.

### Changes from the source

- Paths are relative or set by environment variables. `isildur.sh`,
  `build-fast.sh` and `mac/build.sh` take `ISILDUR_BASE`, `ISILDUR_S1D` and
  `ISILDUR_OUT`. `mac/fresh36.sh` and `mac/fresh41.sh` take `M36`, `M38`,
  `M41`, `INTEGRATE` and `FRESH_ROOT`. The defaults keep upstream's in-repo
  layout.
- `base_tree.sh` checks the base tree against `base.SHA256SUMS` in this
  directory. Upstream kept that file inside `base/`.
- `isildur.sh dist` marks the build dirty only when this directory has
  uncommitted changes, not when any part of the repository does.
- `ios/engine-host/project.yml`: the bundle prefix is `com.example`, and the
  team is `${BEND_TEAM_ID}`.
- Developer machine paths were removed from comments, from
  `base.PROVENANCE`, and from `patches/PHONECAM-HASHES.txt`. One of those
  comments is in `engine/llb_engine.h`, so the header hash changes. See
  [Pins](#pins).

No patch, kernel, wrapper or tower source changed. (Later, METAL-ANYSHAPE changed the wrapper's
`llb_open` ctx cap and `llb_span` default, and one comment in
`engine/llb_engine.h`; see [`../README.md`](../README.md#any-shape-and-long-context).) The 23 patches in
`patches/` and the 2 in `mac/` are byte-identical to `8ff026e`, and
`patches/SHA256SUMS` still verifies.

## Inputs

Two source trees are not in this repository. The build reads them from the
environment.

| Variable | Default | What it is | Check |
|---|---|---|---|
| `ISILDUR_BASE` | `./base` | The pre-patch Bend tree: the compiler `cc/bend2` and the Qwen3.5 engine `engine/` (442 files) | `base.SHA256SUMS`; `mac/base_tree.sh` refuses any other tree |
| `ISILDUR_S1D` | `./S1D` | The compiled subset of llama-bend-sys1dual at `7e61bda` (`S1D.commit`): `tools/sys1dual`, `tools/clef3`, the tokenizer, `host/llb_gguf.c` | `S1D.SHA256SUMS` |

Both trees are in isildur at `8ff026e` (`base/` and `S1D/`). This
repository's root tree and `../../metal/base/` are different revisions:
96 of the 113 `base/` files that exist at the root match, and 6 of the 20
S1D files that exist here match. They cannot be used in their place.

## Build

You need macOS with Xcode 27 (`xcrun`, `xcodebuild`), `bun` (the Bend
compiler is TypeScript), and Homebrew OpenBLAS. OpenBLAS is used only for the
Mac CLI link in `emit`; the library does not use BLAS. No model or network is
needed. Keep the outputs out of the source tree:

```sh
export ISILDUR_BASE=/path/to/isildur/base
export ISILDUR_S1D=/path/to/isildur/S1D
export ISILDUR_OUT=/path/to/scratch
./isildur.sh emit     # base + patches -> $ISILDUR_OUT/work/mac/b-cam/sys1mac.c, checked = patches/CAM_SRC_SHA
./isildur.sh lib      # + host-prefix.patch -> $ISILDUR_OUT/build/cam-{ios,sim,macos}/libllbengine.a + LLBEngine.xcframework
./isildur.sh dist     # $ISILDUR_OUT/dist: LLBEngine.xcframework, macos/libllbengine.a, SHA256SUMS, ENGINE.pin
./isildur.sh verify ../Vendor   # is that Vendor directory exactly this build?
./isildur.sh vendor ../Vendor   # dist, then copy it in
```

The steps, in order:

1. **emit** (`mac/build.sh` with `VISION=1`). It copies `base/`, applies the
   patch stack with `patch --fuzz=0`, copies `sys1.{bend,c,js}` from S1D,
   and emits C with `bun cc/bend2/main.ts engine_qwen35_gpu.bend`. It then
   links a Mac CLI against OpenBLAS as a smoke check. The emitted C must
   hash to `patches/CAM_SRC_SHA`.
2. **lib** (`build-fast.sh`). It applies `patches/host-prefix.patch`, which
   adds the host-only prefix snapshots and does not change the device
   program. It then compiles the wrapper, the head, the tokenizer and the
   tower (`-DLLBV_NO_CPU`, no OpenBLAS) for three targets:
   `arm64-apple-ios27.0`, `arm64-apple-ios27.0-simulator`, and
   `arm64-apple-macos27.0`. Each slice is `libtool -static`. The script
   checks for duplicate symbols and for any BLAS reference, and links a probe
   with `-all_load`.
   `xcodebuild -create-xcframework` wraps the iOS and simulator slices. The
   macOS slice ships as a separate `macos/libllbengine.a`, next to the
   xcframework.
3. **dist**. It sorts `Info.plist` and writes `SHA256SUMS` and `ENGINE.pin`
   (emitted C, program tag, header and per-object hashes). `ZERO_AR_DATE=1`
   makes the archives reproducible.

`dist` takes about 20 s on an M3 Ultra. Here are the switches of
`mac/build.sh`. Each one reproduces an earlier lane's build:

| Switch | Effect | Emitted C |
|---|---|---|
| `VISION=1` (what `isildur.sh` uses) | The full stack, plus vision and the camera | `b0ffd9af…` (PHONECAM) |
| (none) | The resident engine, without vision | `bccd7cea…` (KEVPHONE) |
| `NORESIDENT=1` | Without `engine-resident` | `c13b9e28…` (ENGINE-EARLYSTOP) |
| `NOSTOP=1` | Without `engine-stop` | `382c871a…` (ENGINE-REPEAT) |
| `NOREOPEN=1` | Without the reopen pair | `515c9fee…` (ENGINE-PHONE2) |
| `NORO=1`, `NODEF4B=1`, `NODCH=1`, `NOPF=1`, `NOKEV=1`, `NO4B=1` | Each drops one more layer | See the header of `mac/build.sh` |

## Pins

| What | Hash |
|---|---|
| Emitted C, device program (`patches/CAM_SRC_SHA`, program tag `b0ffd9af`) | `b0ffd9afe033cdb1a9a15a5012551301717f528062b8ab49c7e89f08ee3b32b5` |
| Emitted C after `host-prefix.patch` (what `lib` compiles) | `88fd9bc8cf8ba4d2813adfeec964758f9f7c2691587390de56bb5f8315b908dc` |
| `bccd7cea…` (default), `c13b9e28…` (`NORESIDENT=1`) | Reproduced byte for byte by isildur |
| `comp.ts`, `engine_qwen35_gpu.bend`, `host/llb_gguf.c` in `base/` | `70473751…`, `e0eaf6df…`, `ec6ba09c…` (all files: `base.SHA256SUMS`) |
| `engine/llb_engine.h` at isildur `8ff026e` | `314c2654…` |
| `engine/llb_engine.h` here (one comment scrubbed) | `fc745df1…` |

isildur's own receipts (not imported) show that `emit` reproduces
`b0ffd9af`, `bccd7cea` and `c13b9e28` from a fresh clone. They also show
that `verify` against the app's `foldin-r8` Vendor directory found 30/30
objects identical, plus identical headers and `Info.plist`.

Because of the scrubbed comment, a `dist` built from here differs from
isildur's in the header file, and so in `ENGINE.pin`'s `header=` line. The
objects are unaffected, because comments do not reach the compiled code.
`isildur.sh verify` compares headers, so it reports that header as
different against a Vendor directory built by isildur.

The earlier app Vendor contained later G3 APIs absent from the original isildur wrapper. The build now applies the G3 transformations from `g18-memlevers-eng` commit `030443f` and preserves those APIs. The app Vendor manifest records the rebuilt slices.

## Relation to `backends/metal`

[`../../metal`](../../metal/README.md) builds `sys1mac`, the Mac product
engine for the 9B model. It comes from the same lineage: integrate, then the
M36 release, M38, M41, sys1mac, and then the 4B stack. It ships the patches
in collapsed form.

| Layer | `backends/metal` | This directory |
|---|---|---|
| Base | The pinned Bend compiler and this repository's engine, through `tools/gemv-q8-rw.patch` (fork point) | `base/`, a frozen copy of the sys1mac tree (M36 + M38 + M41 + `metal-/engine-sys1mac.patch`) |
| Compiler stack | `cc/metal-cc.patch`: integrate metal-port, 9b and batch; M36 cc 01–14; M38; M41; sys1mac; and `metal-4b`, `metal-kev`, `metal-4bpf`, `metal-dchunk`, `metal-reopen` | The same 4B patches as separate files in `patches/`, applied to `base/` |
| Engine stack | `engine/metal-engine.patch`: M36 engine 01–11, M41, `engine-sys1mac`, and `engine-4b` through `engine-stop` | `engine-4b`, `4bpf`, `dchunk`, `4bdef`, `reopen`, `stop`, plus `engine-resident` |
| Vision (`VISION=1`) | `vision/engine-mmproj.patch`, `metal-imrope.patch`, `metal-vision.patch` | The same three files, byte-identical (`31b9d6c8…`, `2efb9ffa…`, `5dc8838d…`), plus `metal-camera.patch` |
| Default emit | `c13b9e28…` | `NORESIDENT=1` gives the same `c13b9e28…` |
| Vision emit | `fbdefec7…` (no resident session and no camera) | `b0ffd9af…` |
| Host and wrapper | `link/dual.c` and the Mac CLI | `host-prefix.patch`, the `llb_engine.m` wrapper (the `wrapper-*.patch` files record how it grew), the Jev head, and the 4B tower |

The overlap is the 4B compiler patches, the engine patches through
`engine-stop`, and the three vision patches. Only this directory has
`engine-resident`, `metal-camera`, `host-prefix`, the library wrapper, and
the vision tower sources. The vision tower speed patch in
[`../../metal/vision/`](../../metal/vision/README.md) is for the 9B tower
revision and does not apply to `vision/host/` here.

## Gaps

- `base/` (4.2 MB) and `S1D/` are not vendored. See [Inputs](#inputs).
- This copy has been rebuilt for macOS, iOS and simulator from the pinned inputs. A Mac hardware test exercised cancellation and subsequent reuse; an iOS simulator build compiled and linked. Real iOS device cancellation has not been tested.
- `mac/fresh36.sh` and `mac/fresh41.sh` cannot run, because their lane trees
  no longer exist.
- The EngineHost project expects `../../build/LLBEngine.xcframework`
  relative to `ios/engine-host`, as in upstream's layout. Copy or link the
  xcframework there, or edit `project.yml`.

### Cooperative cancellation

The iOS-specific `engine-cancel.patch` checks the host cancellation flag between
Bend decode steps and between prefill chunks or exact-token steps. `llb_cancel()` sets that atomic flag from another thread;
`llb_cancel_reset()` must be called before a new generation. The poll cannot interrupt a GPU command already submitted. A
caller must associate cancellation with the active request and clear the flag
before starting the next one. `scripts/cancel-test.sh MODEL_DIR CACHE_DIR` checks
that a cancelled resident generation stops and a subsequent generation works.

The emitted-C input is pinned by `patches/CAM_SRC_SHA`. The G3 source transformations are from `g18-memlevers-eng` commit `030443f`;
they preserve `llb_generate_px`, `llb_state_save`, `llb_state_load`, and
`llb_packmap_report`. The host source tag changes with the emitted program, so
snapshots from a previous build are safely rejected.

The current Bend-emitted C SHA256 is `9884e4c50b95e4e58c5bfd326ec40a8278a49ce99d3b7a9a0130bf8e887aadfb`; after the G3 host transformations it is `da60adc346edcb76a598ac4ccffb238f9e1478fbc4daad28ae515602aa959ba7`. The app cache tag is `da60adc3`. Base and S1D inputs are from isildur `8ff026e4d723803385bcd089f61191bddd22dd57` and are checked against the repository manifests.
