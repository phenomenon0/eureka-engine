// tjs.js: transformers.js v4 pinned to this tab: models from ../models/hf/ or the HF hub (hosts.js), ORT wasm from our vendor dir, errors-only logs.
import * as T from "./vendor/tjs/transformers.min.js";
import { HF } from "./hosts.js";
// WEBGPU-E2E-P2: hosts.js picks the weights' home: ../models/hf/ next to the page, or the Hugging Face hub itself (the box)
T.env.allowRemoteModels = HF; T.env.allowLocalModels = !HF; T.env.localModelPath = new URL("../models/hf/", import.meta.url).pathname;
T.env.backends.onnx.wasm.wasmPaths = new URL("./vendor/ort/", import.meta.url).href;
if (T.LogLevel) T.env.logLevel = T.LogLevel.ERROR;
if (self.crossOriginIsolated) T.env.backends.onnx.wasm.numThreads = Math.min(8, navigator.hardwareConcurrency);
export default T;
