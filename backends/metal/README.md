# Metal

macOS, Apple silicon. Tested on a Mac Studio (M3 Ultra, 96 GB). The same
emitted C (`c13b9e28`) is the engine in the [iOS app](../ios/README.md).

## Source

- Engine: this directory. The Metal compiler and engine deltas are
  `cc/metal-cc.patch` and `engine/metal-engine.patch`.
- Server provider (`llb-serve` + `sys1mac`): `desktop/engine/metal/` in
  `github.com/phenomenon0/bend-app`.

## Build

On a Mac:

```sh
LLB_BACKEND=metal ./build.sh
```

To build the server provider, run `desktop/engine/metal/build.sh` in
`bend-app`.

## Result

clef-flash Q4_K_S (9B): 70.2 tok/s decode.

## Gaps

- The engine is a vendored fork. It is not rebased onto the CUDA stack.
- `see` runs on the ONNX Runtime CPU provider, not on Metal.

How the build reproduces the Mac product: [build-internals.md](build-internals.md).
