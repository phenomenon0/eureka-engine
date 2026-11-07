# Receipts: release 0.2.0

The release gate for `llb-linux-x86_64-cuda-0.2.0.tar.gz`
(sha256 `95c9f22e3369890ce80ae2dae3bece869c42065679ce6855a01ca5784dfb6bfa`).
It ran on the RTX 3090 machine described in
[`../cuda-proof-20261106/`](../cuda-proof-20261106/README.md), from commit `f9169e2`.

| File | What it shows |
|---|---|
| `commit.txt` | The source commit. |
| `build.txt` | `./build.sh`, exit 0. |
| `bundle.txt` | `tools/unify/bundle.sh 0.2.0 build`, exit 0. The ISA check passes (`RESULT=PASS`). |
| `manifest.json` | The manifest published with the release. Only `linux-x86_64-cuda` is available. |
| `install.txt` | `release/install.sh` against that manifest: sha256 checked before unpacking, `llb` linked, exit 0. |
| `install-macos-dryrun.txt` | The same installer on macOS arm64. It says that no bundle exists and points at the source build. |
| `doctor.txt` | `llb version` and `llb doctor` from the installed bundle, exit 0. |
| `llb-run-4b.txt` | `llb run` on the Qwen3.5-4B Q8_0 from the installed bundle: " Paris.", exit 0. |
