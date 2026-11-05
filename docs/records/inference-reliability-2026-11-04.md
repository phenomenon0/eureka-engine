# Inference reliability fixes, 2026-11-04

These changes carry the Eureka app review into the maintained engine source.
They repair request parsing, cancellation, worker ownership, and terminal
responses. They do not change model weights or sampling math.

| Backend | Change at its owning source | Verification |
| --- | --- | --- |
| Metal server | Strict finite JSON numbers, bounded nesting, decision request framing, canceled queue removal, and replacement of only the affected worker | Parser sanitizers; one/two-slot protocol regressions; real Kev-4B generation canceled over HTTP, followed by a successful reply in 1.433 s |
| CUDA server | Same parser/framing fixes; cancellation and recovery for generation, decision, native batches, and continuous batches | Parser sanitizers and protocol regressions; real RTX 3090 generation canceled over HTTP, followed by a successful reply in 2.815 s |
| iOS resident engine | Bend prefill/decode polling through a host atomic cancellation API; reproducible G3 wrapper transformations retain prefix snapshots, state persistence, and pack mapping | Device/simulator/Mac library slices built; real Qwen 4B Mac tests cancel a 600-token prefill before output, cancel decoding, and reuse the resident engine; prefix save/load round trip checked |
| Android CPU engine | Strict JSON parser and explicit stop/length finish reasons in the reproducible patch build | Native parser regression and NDK cross-build; artifact hash below |
| WebGPU worker | Serialized opens/free, load identity, settlement of queued and active chat/decision requests, and cancellation checks between GPU steps | Six simulated worker regressions, including cancellation during batched prefill and Kev decision steps |

The real Metal server test used this repository's HTTP server with the existing
production-pinned `sys1mac` model worker. The CUDA run used an isolated server on
the Tailscale NVIDIA host. Both tests retained source/artifact hashes and used
the machines' GPU locks; neither changed a running service or installed an app.

The Android app now authenticates before allocating request bodies and rejects
invalid framing. Its socket disconnect monitor registers cancellation only while
the request owns the process or upstream connection, so a canceled queued request
cannot kill another request. Its rebuilt native engine is
`82c8a5dbce523e79a06096e90c0205b4445ce48d44c1c6a599b905f524bb1091`.
The build records source, compiler, flags, and artifact hashes in
`backends/android/engine/receipts/android-20261104.txt`.

The iOS app uses the rebuilt library, watches socket closure during prefill, and
resets cancellation under its engine lock before every job. The wrapper preserves
the later G3 APIs used by the app; replacing it with the older camera wrapper
would remove functionality. See `backends/ios/engine/README.md` for the pinned
inputs and build recipe. Cache identity is derived from the final emitted host C.
The actual Swift HTTP host also passed a Mac GPU test: disconnecting a 312-token
prompt produced zero generated IDs, and the following chat returned four IDs.
The final iOS simulator app built with these libraries and the current web bundle.

## Re-run checks

From this repository:

```sh
bash tests/inference/run-host.sh
```

This compiles both HTTP fronts, runs parser sanitizers and test-worker cancellation
regressions, checks Android JSON parsing, and runs the WebGPU worker tests. It
uses no model or GPU. Test-only macOS shims allow compiling the Linux CUDA HTTP
front; this is not a CUDA compute build.

The hardware harnesses are `tests/inference/metal_real_cancellation.py`,
`tools/sys1dual/tests/cancellation_gpu_smoke.py`, and
`backends/ios/engine/scripts/cancel-test.sh`. They require compatible models,
engine inputs, and the host's GPU lock. Local review receipts are retained in
`/Users/dog/scratch/eureka-engine-evidence-20261104/`.

## Limits of this verification

- Physical iOS and Android device inference was not run.
- WebGPU lifecycle tests simulate the session; no browser/device/model matrix was run.
- CUDA native-batch and continuous-batch recovery was tested with a protocol worker.
  Available NVIDIA model workers did not expose the required batch protocol, so
  real GPU batch cancellation remains unverified.
- Cooperative cancellation observes prefill/chunk/token boundaries; it cannot
  interrupt an already submitted GPU command. Shared model warmup is not canceled
  merely because one request disconnects.

Eureka's supervisor, Swift/Kotlin lifecycle integration, stream parsing, and UI
terminal handling belong to the app repository. Their matching fixes are kept
there; future vendoring must preserve both layers.
