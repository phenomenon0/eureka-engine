#if !VISION_STANDALONE
import CoreGraphics
import Foundation
import onnxruntime

/// G38: a pocket detector from an ONNX file on onnxruntime (Vendor/onnxruntime.xcframework 1.24, static) with its CoreML
/// execution provider: ORT hands the graph's supported partitions to Core ML as one MLProgram (fp16 on the ANE/GPU) and
/// runs any rest on its CPU EP. Chosen when the staged detector is `<-pocketDet>.onnx` (e.g. `-pocketDet dfine-n` with
/// Documents/ml/dfine-n.onnx: D-FINE-N, mac-reflex-coreml's fp32 export, logits [1,300,80] + boxes [1,300,4] cxcywh).
///   -pocketEP coreml|cpu   coreml (default) = the CoreML EP, compute units from -pocketUnits (ane = CPUAndNeuralEngine,
///                          gpu = CPUAndGPU, cpu = CPUOnly, all = ALL), compiled model cached in Caches/ort-coreml;
///                          cpu = ORT's own CPU EP (the reference the CoreML EP is judged against)
///   -pocketOrtLog          ORT's INFO log (the CoreML EP's partition count: which nodes Core ML took)
/// Input: one RGB image, stretched to the model's side (640), 0…1, NCHW — the export's own preprocessing (no mean/std).
final class OrtDetector: @unchecked Sendable {
    let name: String, ep: String, side: Int
    private let api: OrtApi
    private var env: OpaquePointer?, options: OpaquePointer?, session: OpaquePointer?, memory: OpaquePointer?
    private let inName: String, inNames: [String], outNames: [String]
    /// True for the 3-head COCO export (`labels`/`boxes`/`scores`): boxes in target pixels, scores already sigmoid'd.
    let coco: Bool

    struct Failure: Error, LocalizedError { let errorDescription: String?; init(_ s: String) { errorDescription = s } }

    private static func check(_ api: OrtApi, _ status: OpaquePointer?) throws {
        guard let status else { return }
        let message = String(cString: api.GetErrorMessage(status)!)
        api.ReleaseStatus(status)
        throw Failure("onnxruntime: \(message)")
    }
    private func check(_ status: OpaquePointer?) throws { try Self.check(api, status) }

    init(url: URL, ep: String, units: String, cache: URL, verbose: Bool) throws {
        guard let base = OrtGetApiBase(), let a = base.pointee.GetApi(UInt32(ORT_API_VERSION)) else { throw Failure("onnxruntime: no API v\(ORT_API_VERSION)") }
        let api = a.pointee
        func check(_ status: OpaquePointer?) throws { try Self.check(api, status) }
        var env: OpaquePointer?, options: OpaquePointer?, session: OpaquePointer?, memory: OpaquePointer?
        try check(api.CreateEnv(verbose ? ORT_LOGGING_LEVEL_INFO : ORT_LOGGING_LEVEL_WARNING, "bend-pocket", &env))
        try check(api.CreateSessionOptions(&options))
        try check(api.SetIntraOpNumThreads(options, 2))
        try check(api.SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL))
        if verbose { try check(api.SetSessionLogSeverityLevel(options, 1)) }
        var how = "cpu"
        if ep == "coreml" {
            let mlUnits = ["ane": "CPUAndNeuralEngine", "gpu": "CPUAndGPU", "cpu": "CPUOnly", "all": "ALL"][units] ?? "CPUAndNeuralEngine"
            try? FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
            let kv: [(String, String)] = [("ModelFormat", "MLProgram"), ("MLComputeUnits", mlUnits), ("RequireStaticInputShapes", "1"),
                                          ("ModelCacheDirectory", cache.path)]
            let keys = kv.map { strdup($0.0) }, vals = kv.map { strdup($0.1) }
            defer { (keys + vals).forEach { free($0) } }
            let kp = keys.map { UnsafePointer<CChar>($0) }, vp = vals.map { UnsafePointer<CChar>($0) }
            let st = kp.withUnsafeBufferPointer { k in vp.withUnsafeBufferPointer { v in
                api.SessionOptionsAppendExecutionProvider(options, "CoreML", k.baseAddress, v.baseAddress, kv.count)
            } }
            try check(st)  // ORT 1.24's generic entry point; the module map does not export coreml_provider_factory.h
            how = "coreml(MLProgram,\(mlUnits))"
        }
        try check(api.CreateSession(env, url.path, options, &session))
        try check(api.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory))
        var alloc: UnsafeMutablePointer<OrtAllocator>?
        try check(api.GetAllocatorWithDefaultOptions(&alloc))
        func names(_ count: (OpaquePointer?, UnsafeMutablePointer<Int>?) -> OpaquePointer?,
                   _ get: (OpaquePointer?, Int, UnsafeMutablePointer<OrtAllocator>?, UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) -> OpaquePointer?) throws -> [String] {
            var n = 0
            try check(count(session, &n))
            return try (0..<n).map { i in
                var p: UnsafeMutablePointer<CChar>?
                try check(get(session, i, alloc, &p))
                defer { _ = api.AllocatorFree(alloc, p) }
                return String(cString: p!)
            }
        }
        let ins = try names(api.SessionGetInputCount, api.SessionGetInputName)
        let outs = try names(api.SessionGetOutputCount, api.SessionGetOutputName)
        guard let first = ins.first, outs.count >= 2 else { throw Failure("onnx: want inputs + logits/boxes outputs, got \(ins) → \(outs)") }
        // The input's side from its static shape [1, 3, S, S] (640 when dynamic).
        var ti: OpaquePointer?
        try check(api.SessionGetInputTypeInfo(session, 0, &ti))
        defer { api.ReleaseTypeInfo(ti) }
        var tsi: OpaquePointer?
        try check(api.CastTypeInfoToTensorInfo(ti, &tsi))
        var nd = 0
        try check(api.GetDimensionsCount(tsi, &nd))
        var dims = [Int64](repeating: -1, count: nd)
        try check(api.GetDimensions(tsi, &dims, nd))
        self.api = api
        self.env = env; self.options = options; self.session = session; self.memory = memory
        name = url.deletingPathExtension().lastPathComponent
        self.ep = how
        inName = first; inNames = ins; outNames = outs
        coco = outs.contains { $0.lowercased().contains("label") } && outs.contains { $0.lowercased().contains("score") }
        side = nd == 4 && dims[3] > 0 ? Int(dims[3]) : 640
    }

    deinit {
        api.ReleaseSession(session); api.ReleaseMemoryInfo(memory); api.ReleaseSessionOptions(options); api.ReleaseEnv(env)
    }

    /// The photo stretched to side² RGB, 0…1, planar (CHW).
    func pixels(_ image: CGImage) throws -> [Float] {
        let s = side
        guard let ctx = CGContext(data: nil, width: s, height: s, bitsPerComponent: 8, bytesPerRow: s * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { throw Failure("pre: no context") }
        ctx.interpolationQuality = .medium
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: s, height: s))
        guard let data = ctx.data else { throw Failure("pre: no pixels") }
        let px = data.assumingMemoryBound(to: UInt8.self), plane = s * s
        var out = [Float](repeating: 0, count: 3 * plane)
        out.withUnsafeMutableBufferPointer { o in
            for i in 0..<plane {
                o[i] = Float(px[i * 4]) / 255; o[plane + i] = Float(px[i * 4 + 1]) / 255; o[2 * plane + i] = Float(px[i * 4 + 2]) / 255
            }
        }
        return out
    }

    /// logits [1,Q,C] + boxes [1,Q,4] (cxcywh normalized), as floats, with Q and C.
    /// A D-FINE export carries a second input, `orig_target_sizes` [1,2] int64 (h,w): the postprocessor multiplies the
    /// normalized cxcywh by it. Feed [1,1] so the head's boxes stay normalized — the decode below says normalized.
    func run(_ input: [Float]) throws -> (logits: [Float], boxes: [Float], q: Int, c: Int) {
        var value: OpaquePointer?, sizes: OpaquePointer?
        var result: (logits: [Float], boxes: [Float], q: Int, c: Int)?
        let shape: [Int64] = [1, 3, Int64(side), Int64(side)]
        var target: [Int64] = [1, 1]
        let targetShape: [Int64] = [1, 2]
        var data = input
        try data.withUnsafeMutableBytes { raw in
            try check(shape.withUnsafeBufferPointer { s in
                api.CreateTensorWithDataAsOrtValue(memory, raw.baseAddress, raw.count, s.baseAddress, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value)
            })
            defer { api.ReleaseValue(value) }
            var namePtrs = inNames.map { strdup($0) }
            defer { namePtrs.forEach { free($0) } }
            var ins: [UnsafePointer<CChar>?] = namePtrs.map { UnsafePointer($0) }
            var inV: [OpaquePointer?] = [value]
            if inNames.count >= 2 {
                try target.withUnsafeMutableBytes { traw in
                    try check(targetShape.withUnsafeBufferPointer { ts in
                        api.CreateTensorWithDataAsOrtValue(memory, traw.baseAddress, traw.count, ts.baseAddress, 2,
                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &sizes)
                    })
                }
                inV.append(sizes)
            }
            defer { api.ReleaseValue(sizes) }
            let wantL = outNames.first { $0.contains("logit") } ?? outNames[0], wantB = outNames.first { $0.contains("box") } ?? outNames[1]
            let lC = strdup(wantL), bC = strdup(wantB)
            defer { free(lC); free(bC) }
            var outs: [UnsafePointer<CChar>?] = [UnsafePointer(lC), UnsafePointer(bC)]
            var outV: [OpaquePointer?] = [nil, nil]
            try check(api.Run(session, nil, &ins, &inV, ins.count, &outs, 2, &outV))
            defer { outV.forEach { api.ReleaseValue($0) } }
            func floats(_ v: OpaquePointer?) throws -> ([Float], [Int64]) {
                var info: OpaquePointer?
                try check(api.GetTensorTypeAndShape(v, &info))
                defer { api.ReleaseTensorTypeAndShapeInfo(info) }
                var nd = 0, n = 0
                try check(api.GetDimensionsCount(info, &nd))
                var dims = [Int64](repeating: 0, count: nd)
                try check(api.GetDimensions(info, &dims, nd))
                try check(api.GetTensorShapeElementCount(info, &n))
                var p: UnsafeMutableRawPointer?
                try check(api.GetTensorMutableData(v, &p))
                return (Array(UnsafeBufferPointer(start: p!.assumingMemoryBound(to: Float.self), count: n)), dims)
            }
            let (l, ld) = try floats(outV[0]), (b, _) = try floats(outV[1])
            result = (l, b, Int(ld[1]), Int(ld[2]))
        }
        return result!
    }

    /// The 3-head COCO graph (`labels` [1,Q], `boxes` [1,Q,4] xyxy in `orig_target_sizes` pixels, `scores` [1,Q],
    /// already sigmoid'd). `orig_target_sizes` is (w, h) — D-FINE's convention; the boxes arrive in photo pixels,
    /// VisionPocket normalizes.
    func runCoco(_ input: [Float], width: Int, height: Int) throws -> (labels: [Int64], boxes: [Float], scores: [Float]) {
        var value: OpaquePointer?, sizes: OpaquePointer?
        var out: (labels: [Int64], boxes: [Float], scores: [Float])?
        let shape: [Int64] = [1, 3, Int64(side), Int64(side)]
        var target: [Int64] = [Int64(width), Int64(height)]
        let targetShape: [Int64] = [1, 2]
        var data = input
        try data.withUnsafeMutableBytes { raw in
            try check(shape.withUnsafeBufferPointer { s in
                api.CreateTensorWithDataAsOrtValue(memory, raw.baseAddress, raw.count, s.baseAddress, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value)
            })
            defer { api.ReleaseValue(value) }
            var namePtrs = inNames.map { strdup($0) }
            defer { namePtrs.forEach { free($0) } }
            var ins: [UnsafePointer<CChar>?] = namePtrs.map { UnsafePointer($0) }
            var inV: [OpaquePointer?] = [value]
            if inNames.count >= 2 {
                try target.withUnsafeMutableBytes { traw in
                    try check(targetShape.withUnsafeBufferPointer { ts in
                        api.CreateTensorWithDataAsOrtValue(memory, traw.baseAddress, traw.count, ts.baseAddress, 2,
                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &sizes)
                    })
                }
                inV.append(sizes)
            }
            defer { if sizes != nil { api.ReleaseValue(sizes) } }
            var outPtrs = outNames.map { strdup($0) }
            defer { outPtrs.forEach { free($0) } }
            var outs: [UnsafePointer<CChar>?] = outPtrs.map { UnsafePointer($0) }
            var outV = [OpaquePointer?](repeating: nil, count: outNames.count)
            try check(api.Run(session, nil, &ins, &inV, ins.count, &outs, outV.count, &outV))
            defer { outV.forEach { api.ReleaseValue($0) } }
            func vals<T>(_ v: OpaquePointer?, _ t: T.Type) throws -> [T] {
                var info: OpaquePointer?
                try check(api.GetTensorTypeAndShape(v, &info))
                defer { api.ReleaseTensorTypeAndShapeInfo(info) }
                var n = 0
                try check(api.GetTensorShapeElementCount(info, &n))
                var p: UnsafeMutableRawPointer?
                try check(api.GetTensorMutableData(v, &p))
                return Array(UnsafeBufferPointer(start: p!.assumingMemoryBound(to: T.self), count: n))
            }
            func isInt32(_ v: OpaquePointer?) throws -> Bool {
                var info: OpaquePointer?
                try check(api.GetTensorTypeAndShape(v, &info))
                defer { api.ReleaseTensorTypeAndShapeInfo(info) }
                var t = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                try check(api.GetTensorElementType(info, &t))
                return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
            }
            let li = outNames.firstIndex { $0.lowercased().contains("label") } ?? 0
            let bi = outNames.firstIndex { $0.lowercased().contains("box") } ?? 1
            let si = outNames.firstIndex { $0.lowercased().contains("score") } ?? 2
            let labels: [Int64] = try isInt32(outV[li])
                ? vals(outV[li], Int32.self).map { Int64($0) }
                : vals(outV[li], Int64.self)
            out = (labels, try vals(outV[bi], Float.self), try vals(outV[si], Float.self))
        }
        return out!
    }
}
#endif
