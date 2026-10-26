import Foundation
import onnxruntime

/// VOICEREAL: the phone's own neural voice. Supertonic 2 (Supertone; code MIT, weights OpenRAIL-M), voice F1, run on the
/// CPU by onnxruntime (MIT, Vendor/onnxruntime.xcframework, static). No phonemizer (Supertonic reads characters), so no
/// espeak-ng / GPL code in the app. The pipeline follows Supertone's own Swift example (swift/Sources/Helper.swift, MIT):
/// text → duration predictor → text encoder → 5 flow-matching steps of the vector estimator → vocoder → 44.1 kHz mono.
/// One chunk at a time (the web's chunker already cuts the reply into clauses/sentences): ~0.2 s for the first clause and
/// RTF ≈ 0.05 on two threads (simulator on the Mac; docs/VOICE.md). Not thread-safe: the Speaker calls it from one queue.
///
/// Files (`dir`): duration_predictor.onnx, text_encoder.onnx, vector_estimator.int8.onnx, vocoder.int8.onnx, tts.json,
/// unicode_indexer.json, and the voice style (F1.json). scripts/fetch-voice.sh puts them in ios/Vendor/voices/supertonic2
/// (pinned SHA-256s); the app bundles that folder as /voices/supertonic2.
final class NeuralVoice: @unchecked Sendable {
    let name: String
    let sampleRate: Int
    private let ort: ORT
    private let dp, textEnc, vectorEst, vocoder: ORT.Session
    private let indexer: [Int64]
    private let styleTTL, styleDP: (data: [Float], shape: [Int64])
    private let baseChunkSize, chunkCompress, latentDim: Int
    /// Supertone's defaults: 5 denoising steps, speed 1.05.
    var steps = 5
    var speed: Float = 1.05

    struct Config: Decodable {
        struct AE: Decodable { let sample_rate: Int; let base_chunk_size: Int }
        struct TTL: Decodable { let chunk_compress_factor: Int; let latent_dim: Int }
        let ae: AE
        let ttl: TTL
    }
    private struct Style: Decodable {
        struct Part: Decodable { let data: [[[Float]]]; let dims: [Int] }
        let style_ttl: Part
        let style_dp: Part
    }

    static let fileNames = ["duration_predictor.onnx", "text_encoder.onnx", "vector_estimator.int8.onnx", "vocoder.int8.onnx",
                            "tts.json", "unicode_indexer.json"]

    /// Where the voice lives: Documents/models/voices/supertonic2 (a newer copy dropped in, as models arrive) wins over
    /// the bundled one; BEND_VOICE_DIR overrides both (the Mac harness).
    static func directory() -> URL? {
        var candidates: [URL] = []
        if let env = ProcessInfo.processInfo.environment["BEND_VOICE_DIR"] { candidates.append(URL(fileURLWithPath: env)) }
        if let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            candidates.append(docs.appendingPathComponent("models/voices/supertonic2"))
        }
        if let bundled = Bundle.main.url(forResource: "supertonic2", withExtension: nil, subdirectory: "voices") { candidates.append(bundled) }
        return candidates.first { dir in fileNames.allSatisfy { FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path) } }
    }

    init(dir: URL, voice: String = "F1", threads: Int32 = 2) throws {
        name = "Supertonic \(voice)"
        let cfg = try JSONDecoder().decode(Config.self, from: Data(contentsOf: dir.appendingPathComponent("tts.json")))
        sampleRate = cfg.ae.sample_rate
        baseChunkSize = cfg.ae.base_chunk_size
        chunkCompress = cfg.ttl.chunk_compress_factor
        latentDim = cfg.ttl.latent_dim
        indexer = try JSONDecoder().decode([Int64].self, from: Data(contentsOf: dir.appendingPathComponent("unicode_indexer.json")))
        let style = try JSONDecoder().decode(Style.self, from: Data(contentsOf: dir.appendingPathComponent("\(voice).json")))
        styleTTL = (style.style_ttl.data.flatMap { $0.flatMap { $0 } }, style.style_ttl.dims.map(Int64.init))
        styleDP = (style.style_dp.data.flatMap { $0.flatMap { $0 } }, style.style_dp.dims.map(Int64.init))
        ort = try ORT(threads: threads)
        dp = try ort.session(dir.appendingPathComponent("duration_predictor.onnx"))
        textEnc = try ort.session(dir.appendingPathComponent("text_encoder.onnx"))
        vectorEst = try ort.session(dir.appendingPathComponent("vector_estimator.int8.onnx"))
        vocoder = try ort.session(dir.appendingPathComponent("vocoder.int8.onnx"))
    }

    /// One chunk of text → mono Float32 samples at `sampleRate`, trimmed to the predicted duration.
    func synthesize(_ text: String) throws -> [Float] {
        let ids = Self.preprocess(text).unicodeScalars.map { s -> Int64 in Int(s.value) < indexer.count ? indexer[Int(s.value)] : -1 }
        guard !ids.isEmpty else { return [] }
        let n = Int64(ids.count)
        let textIds = try ort.tensor(ids, [1, n])
        let textMask = try ort.tensor([Float](repeating: 1, count: ids.count), [1, 1, n])
        let ttl = try ort.tensor(styleTTL.data, styleTTL.shape)
        let dpStyle = try ort.tensor(styleDP.data, styleDP.shape)

        let durationOut = try dp.run(["text_ids": textIds, "style_dp": dpStyle, "text_mask": textMask], ["duration"])[0]
        let duration = max(try durationOut.floats()[0] / speed, 0.05)
        let textEmb = try textEnc.run(["text_ids": textIds, "style_ttl": ttl, "text_mask": textMask], ["text_emb"])[0]

        let wavLen = Int(duration * Float(sampleRate))
        let chunk = baseChunkSize * chunkCompress
        let latentLen = (wavLen + chunk - 1) / chunk
        let dim = latentDim * chunkCompress
        var xt = (0..<(dim * latentLen)).map { _ in Self.gaussian() }
        let latentMask = try ort.tensor([Float](repeating: 1, count: latentLen), [1, 1, Int64(latentLen)])
        let total = try ort.tensor([Float(steps)], [1])
        for step in 0..<steps {
            let out = try vectorEst.run([
                "noisy_latent": try ort.tensor(xt, [1, Int64(dim), Int64(latentLen)]), "text_emb": textEmb, "style_ttl": ttl,
                "latent_mask": latentMask, "text_mask": textMask, "current_step": try ort.tensor([Float(step)], [1]), "total_step": total,
            ], ["denoised_latent"])[0]
            xt = try out.floats()
        }
        let wav = try vocoder.run(["latent": try ort.tensor(xt, [1, Int64(dim), Int64(latentLen)])], ["wav_tts"])[0].floats()
        return Array(wav.prefix(wavLen))
    }

    private static func gaussian() -> Float {
        let u1 = Float.random(in: 0.0001...1), u2 = Float.random(in: 0...1)
        return (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
    }

    /// Supertone's preprocessText (en): NFKD, emoji out, typographic marks folded, a closing period, the language tags.
    static func preprocess(_ input: String, lang: String = "en") -> String {
        var text = String(String.UnicodeScalarView(input.decomposedStringWithCompatibilityMapping.unicodeScalars.filter { s in
            let v = s.value
            return !((0x1F300...0x1FAFF).contains(v) || (0x2600...0x27BF).contains(v) || (0x1F1E6...0x1F1FF).contains(v))
        }))
        for (old, new) in [("–", "-"), ("‑", "-"), ("—", "-"), ("_", " "), ("\u{201C}", "\""), ("\u{201D}", "\""), ("\u{2018}", "'"),
                           ("\u{2019}", "'"), ("´", "'"), ("`", "'"), ("[", " "), ("]", " "), ("|", " "), ("/", " "), ("#", " "),
                           ("→", " "), ("←", " "), ("♥", ""), ("☆", ""), ("♡", ""), ("©", ""), ("\\", ""), ("@", " at "),
                           ("e.g.,", "for example, "), ("i.e.,", "that is, "), (" ,", ","), (" .", "."), (" !", "!"), (" ?", "?"),
                           (" ;", ";"), (" :", ":"), (" '", "'")] {
            text = text.replacingOccurrences(of: old, with: new)
        }
        while text.contains("\"\"") { text = text.replacingOccurrences(of: "\"\"", with: "\"") }
        while text.contains("''") { text = text.replacingOccurrences(of: "''", with: "'") }
        text = text.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression).trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty, text.range(of: "[.!?;:,'\"\u{201C}\u{201D}\u{2018}\u{2019})\\]}…]$", options: .regularExpression) == nil { text += "." }
        return "<\(lang)>\(text)</\(lang)>"
    }
}

// MARK: - onnxruntime, through its C API

/// The few onnxruntime C calls the voice needs: an env, sessions, CPU tensors, run. Errors become NeuralVoiceError.
final class ORT: @unchecked Sendable {
    let api: OrtApi
    private var env: OpaquePointer?
    private var options: OpaquePointer?
    private var memory: OpaquePointer?

    init(threads: Int32) throws {
        guard let base = OrtGetApiBase(), let api = base.pointee.GetApi(UInt32(ORT_API_VERSION)) else { throw NeuralVoiceError("onnxruntime: no API v\(ORT_API_VERSION)") }
        self.api = api.pointee
        try check(self.api.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "bend-voice", &env))
        try check(self.api.CreateSessionOptions(&options))
        try check(self.api.SetIntraOpNumThreads(options, threads))
        try check(self.api.SetInterOpNumThreads(options, 1))
        try check(self.api.SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL))
        try check(self.api.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory))
    }

    deinit {
        api.ReleaseMemoryInfo(memory)
        api.ReleaseSessionOptions(options)
        api.ReleaseEnv(env)
    }

    func check(_ status: OpaquePointer?) throws {
        guard let status else { return }
        let message = String(cString: api.GetErrorMessage(status)!)
        api.ReleaseStatus(status)
        throw NeuralVoiceError("onnxruntime: \(message)")
    }

    func session(_ url: URL) throws -> Session {
        var s: OpaquePointer?
        try check(api.CreateSession(env, url.path, options, &s))
        return Session(ort: self, handle: s!)
    }

    /// A CPU tensor that owns a copy of its data for its lifetime.
    final class Value {
        let ort: ORT
        let handle: OpaquePointer
        private let storage: UnsafeMutableRawPointer?
        init(ort: ORT, handle: OpaquePointer, storage: UnsafeMutableRawPointer?) { self.ort = ort; self.handle = handle; self.storage = storage }
        deinit { ort.api.ReleaseValue(handle); storage?.deallocate() }

        func floats() throws -> [Float] {
            var info: OpaquePointer?
            try ort.check(ort.api.GetTensorTypeAndShape(handle, &info))
            defer { ort.api.ReleaseTensorTypeAndShapeInfo(info) }
            var count = 0
            try ort.check(ort.api.GetTensorShapeElementCount(info, &count))
            var raw: UnsafeMutableRawPointer?
            try ort.check(ort.api.GetTensorMutableData(handle, &raw))
            guard let raw else { return [] }
            return Array(UnsafeBufferPointer(start: raw.assumingMemoryBound(to: Float.self), count: count))
        }
    }

    func tensor<T>(_ data: [T], _ shape: [Int64]) throws -> Value {
        let type: ONNXTensorElementDataType = T.self == Int64.self ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        let bytes = data.count * MemoryLayout<T>.stride
        let storage = UnsafeMutableRawPointer.allocate(byteCount: max(bytes, 1), alignment: 16)
        data.withUnsafeBytes { storage.copyMemory(from: $0.baseAddress!, byteCount: bytes) }
        var value: OpaquePointer?
        let status = shape.withUnsafeBufferPointer { s in
            api.CreateTensorWithDataAsOrtValue(memory, storage, bytes, s.baseAddress, shape.count, type, &value)
        }
        do { try check(status) } catch { storage.deallocate(); throw error }
        return Value(ort: self, handle: value!, storage: storage)
    }

    final class Session {
        let ort: ORT
        let handle: OpaquePointer
        init(ort: ORT, handle: OpaquePointer) { self.ort = ort; self.handle = handle }
        deinit { ort.api.ReleaseSession(handle) }

        func run(_ inputs: [String: Value], _ outputs: [String]) throws -> [Value] {
            let inNames = inputs.keys.map { strdup($0) }, outNames = outputs.map { strdup($0) }
            defer { (inNames + outNames).forEach { free($0) } }
            var inValues: [OpaquePointer?] = inputs.values.map { $0.handle }
            var outValues = [OpaquePointer?](repeating: nil, count: outputs.count)
            let inPtrs = inNames.map { UnsafePointer<CChar>($0) }, outPtrs = outNames.map { UnsafePointer<CChar>($0) }
            try ort.check(inPtrs.withUnsafeBufferPointer { ip in
                outPtrs.withUnsafeBufferPointer { op in
                    ort.api.Run(handle, nil, ip.baseAddress, &inValues, inputs.count, op.baseAddress, outputs.count, &outValues)
                }
            })
            return outValues.map { Value(ort: ort, handle: $0!, storage: nil) }
        }
    }
}

struct NeuralVoiceError: LocalizedError {
    let errorDescription: String?
    init(_ message: String) { errorDescription = message }
}
