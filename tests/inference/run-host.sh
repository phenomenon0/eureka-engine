#!/usr/bin/env bash
# Protocol, parser, and worker lifecycle regressions. Uses test workers, no model or GPU.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
work=$(mktemp -d "${TMPDIR:-/tmp}/bend-inference-tests.XXXXXX")
trap 'rm -rf "$work"' EXIT
cc=${CC:-clang}
metal=backends/metal/server/src
for parser in "$metal" tools/clef3; do
    "$cc" -fsanitize=address,undefined -g -I"$parser" tests/inference/native_json_regression.c "$parser/json.c" -lm -o "$work/json-test"
    "$work/json-test"
done
"$cc" -D_GNU_SOURCE -std=c11 -O2 -pthread "$metal/serve.c" "$metal/json.c" "$metal/speak_route.c" -lm -o "$work/metal-serve"
EUREKA_TEST_SERVE="$work/metal-serve" python3 tests/inference/native_server_regression.py
EUREKA_TEST_SERVE="$work/metal-serve" EUREKA_TEST_SLOTS=2 python3 tests/inference/native_server_regression.py
compat=()
if [ "$(uname -s)" = Darwin ]; then compat=(-include tests/inference/cuda-host-compat.h); fi
"$cc" "${compat[@]}" -D_GNU_SOURCE -std=c11 -O2 -pthread tools/sys1dual/serve.c \
    tools/sys1dual/serve_structured.c host/llb_structured.c tools/sys1dual/mm.c tools/clef3/json.c -lm -o "$work/cuda-serve"
EUREKA_TEST_SERVE="$work/cuda-serve" EUREKA_TEST_CUDA=1 python3 tests/inference/native_server_regression.py
EUREKA_TEST_SERVE="$work/cuda-serve" EUREKA_TEST_CUDA=1 EUREKA_TEST_BANKS=2 python3 tests/inference/native_server_regression.py
python3 tools/sys1dual/tests/cancellation_batch_regression.py "$work/cuda-serve"
bash backends/android/engine/tests/json-hardening.sh
node --test backends/webgpu/tests/worker-lifecycle.test.mjs
