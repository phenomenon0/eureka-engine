# Inference source restoration, 2026-11-05

The local scratch checkout that held `e36049349a710523ae0e7c4cea0ccd32386211a5`
was no longer present when delivery resumed. The app integration at `31cdbfd7`
and its mobile libraries remained intact. This checkout restores the engine
changes onto the same base, `b0514e16e4c80ef8ad70cdfe816e3b1537557627`.
It uses the app's committed Metal patch, the retained CUDA and Android source
on the NVIDIA test host, and the recorded iOS and WebGPU changes.

## Current verification

- Metal and CUDA HTTP server sources and the WebGPU worker match the hashes
  recorded when the original source commit was completed. The Metal JSON parser
  matches the app's committed parser.
- All Android build inputs match `backends/android/engine/receipts/android-20261104.txt`.
- The restored iOS recipe regenerates the original Bend C and final host C hashes.
  All three rebuilt static libraries (Mac, iOS device, iOS simulator) are identical
  to the app's checked-in Vendor libraries after normalizing archive metadata.
  The generated public header is identical too. Exact hashes and results are in
  [the restoration receipt](inference-restoration-2026-11-05.json).
- `bash tests/inference/run-host.sh` passes parser sanitizer tests, Metal one/two-slot
  and CUDA one/two-bank protocol tests, native and continuous batch cancellation
  with test workers, Android parsing checks, and all six WebGPU worker tests.

This establishes that the recovered iOS source builds the libraries already
integrated in the app. It does not add physical-phone or live WebGPU/CUDA batch
claims to the [original verification record](inference-reliability-2026-11-04.md).
The source is now kept in `/Users/dog/Projects/eureka-engine`; delivery receipts
are in `/Users/dog/macops/workroom/LANE-INTEGRATION/evidence/inference-finish-20261104`.
