import CoreML
import Foundation

/// VOICESHOP: Kokoro-82M v1.0 on the Apple Neural Engine: a drop-in for `KokoroRuntime.run` (same ids, same style
/// table, same 24 kHz mono output) over the 7-stage Core ML chain FluidInference publishes.
///
/// Models: huggingface.co/FluidInference/kokoro-82m-coreml, `ANE/` (Apache-2.0; converted from hexgrad/Kokoro-82M,
/// Apache-2.0, via laishere/kokoro-coreml). The glue below (stage order, fp16/fp32 boundaries, duration rounding,
/// compute-unit routing, zeroed tail slack on inputs) is re-implemented in Swift after FluidAudio's
/// `KokoroAneSynthesizer` / `KokoroAneArrays` / `KokoroAneModelStore`
/// (github.com/FluidInference/FluidAudio, Copyright FluidInference, Apache License 2.0); no code dependency on it.
///
/// Chain (T = ids + 2 pads ≤ 512, T_a = acoustic frames ≤ 2000, 600 samples per frame):
///   Albert(input_ids, attention_mask) → bert_dur
///   PostAlbert(bert_dur, input_ids, style_s, speed, attention_mask) → duration, d, t_en
///   durations: round, clamp ≥ 1 → pred_dur (int32)
///   Alignment(pred_dur, d, t_en) → en, asr          (the duration → alignment matrix is inside the model)
///   Prosody(en, style_s) → F0, N
///   Noise(F0 fp32, style_timbre fp32) → x_source_0, x_source_1
///   Vocoder(asr, F0, N, x_source_0, x_source_1, style_timbre; fp16) → x_pre (anchor discarded)
///   Tail(x_pre fp32: iSTFT) → audio
/// Style row r = min(ids.count, 509) (as KokoroRuntime); cols 0..<128 = style_timbre (decoder), 128..<256 = style_s
/// (predictor). Every input is copied to the dtype the model declares, into a buffer with 16 KB of zeroed slack
/// (macOS/iOS 27 BNNS kernels read past the end of page-sized inputs: FluidAudio's fix).
///
/// Routing: `units` for Albert, PostAlbert, Alignment, Prosody and Vocoder; Noise and Tail are fp32 graphs the ANE
/// cannot take, they run on `tailUnits` (default .cpuOnly: no Metal, which aborts under Core ML on the 27 line and
/// is not allowed in the background on iOS anyway).
/// Limits: at most `maxIds` (510) phoneme ids per call → `KokoroANEError.tooManyIds`; more than 2000 predicted frames
/// (~50 s of audio) → `.tooManyFrames`. The caller splits or falls back to the CPU runtime.
/// Thread-safe (one call at a time, serialised by a lock).
///
/// G28: `bucket`. The models are range-flexible, and Core ML only puts a stage on the Neural Engine when its input has
/// the model's default shape (T = 48 tokens, 120 acoustic frames; any other length runs on the CPU, bit-identical to
/// `.cpuOnly`: voiceshop2 work/ane/RESULT.md). With `bucket`, a call under 120 predicted frames (3 s: a 4-word first
/// chunk) runs the decoder (Prosody, Noise, Vocoder, Tail: ~85% of the time) at 120 frames: the closing pad token's
/// duration is stretched to fill them, and the audio is cut back to the real frames (600 samples each). The tokens are
/// not padded: padding ids to 48 under a zero mask changed PostAlbert's durations (voiced audio 5–11% shorter, G28).
final class KokoroANE: @unchecked Sendable {
    enum Stage: String, CaseIterable, Sendable {
        case albert = "KokoroAlbert", postAlbert = "KokoroPostAlbert", alignment = "KokoroAlignment"
        case prosody = "KokoroProsody_v2", noise = "KokoroNoise_v2", vocoder = "KokoroVocoder", tail = "KokoroTail_v2"
        var file: String { rawValue + ".mlmodelc" }
    }

    /// What must be bundled (as voices/kokoro-ane/…). The ids are the same as KokoroRuntime's vocab.json (identical
    /// 114-entry table), so no vocab file is needed here.
    static let fileNames: [String] = Stage.allCases.map(\.file)
    static let sampleRate = 24_000
    static let maxIds = 510, maxFrames = 2000
    static let styleRows = 510, styleDim = 256
    /// G28: the decoder's default (ANE-eligible) frame count, and the samples per acoustic frame.
    static let bucketFrames = 120, samplesPerFrame = 600

    /// BEND_KOKORO_ANE_DIR, else the bundled voices/kokoro-ane; nil unless every model is there.
    static func directory() -> URL? {
        var candidates: [URL] = []
        if let env = ProcessInfo.processInfo.environment["BEND_KOKORO_ANE_DIR"] { candidates.append(URL(fileURLWithPath: env)) }
        if let bundled = Bundle.main.url(forResource: "kokoro-ane", withExtension: nil, subdirectory: "voices") { candidates.append(bundled) }
        return candidates.first { dir in fileNames.allSatisfy { FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path) } }
    }

    let dir: URL
    let units: MLComputeUnits
    let bucket: Bool
    private let models: [Stage: MLModel]
    private let inputTypes: [Stage: [String: MLMultiArrayDataType]]
    private let lock = NSLock()
    /// Per-stage wall ms and frame count of the last run (for measurements).
    private(set) var lastStageMs: [Stage: Double] = [:]
    private(set) var lastFrames = 0
    /// G28: whether the last run's decoder ran in the 120-frame bucket.
    private(set) var lastBucketed = false

    init(dir: URL, units: MLComputeUnits = .cpuAndNeuralEngine, tailUnits: MLComputeUnits = .cpuOnly, bucket: Bool = false) throws {
        self.dir = dir
        self.units = units
        self.bucket = bucket
        var models: [Stage: MLModel] = [:], types: [Stage: [String: MLMultiArrayDataType]] = [:]
        for stage in Stage.allCases {
            let url = dir.appendingPathComponent(stage.file)
            guard FileManager.default.fileExists(atPath: url.path) else { throw KokoroANEError.missing(stage.file) }
            let cfg = MLModelConfiguration()
            cfg.computeUnits = (stage == .noise || stage == .tail) ? tailUnits : units
            cfg.allowLowPrecisionAccumulationOnGPU = true
            let model = try MLModel(contentsOf: url, configuration: cfg)
            models[stage] = model
            types[stage] = model.modelDescription.inputDescriptionsByName.compactMapValues { $0.multiArrayConstraint?.dataType }
        }
        self.models = models
        inputTypes = types
    }

    /// Same contract as KokoroRuntime.run: ids without the two 0 pads, the whole 510 × 256 style table, → 24 kHz mono.
    func run(_ ids: [Int64], style: [Float], speed: Float) throws -> [Float] {
        guard !ids.isEmpty else { return [] }
        guard ids.count <= Self.maxIds else { throw KokoroANEError.tooManyIds(ids.count, max: Self.maxIds) }
        guard style.count == Self.styleRows * Self.styleDim else { throw KokoroANEError.badStyle(style.count) }
        lock.lock()
        defer { lock.unlock() }
        var ms: [Stage: Double] = [:]

        let row = min(ids.count, Self.styleRows - 1) * Self.styleDim
        let timbre = Array(style[row ..< row + 128]), styleS = Array(style[row + 128 ..< row + Self.styleDim])
        let tokens = [Int32(0)] + ids.map { Int32(truncatingIfNeeded: $0) } + [0]
        let t = tokens.count
        let idsArr = try Self.int32([1, t], tokens)
        let mask = try Self.int32([1, t], [Int32](repeating: 1, count: t))
        let styleSArr = try Self.float32([1, 128], styleS)
        let timbreArr = try Self.float32([1, 128], timbre)
        let speedArr = try Self.float32([1], [speed])

        let albert = try predict(.albert, ["input_ids": idsArr, "attention_mask": mask], &ms)
        let post = try predict(.postAlbert, ["bert_dur": try out(albert, "bert_dur", .albert), "input_ids": idsArr,
                                             "style_s": styleSArr, "speed": speedArr, "attention_mask": mask], &ms)
        let durations = Self.floats(try out(post, "duration", .postAlbert))
        guard durations.allSatisfy(\.isFinite) else { throw KokoroANEError.nonFinite("PostAlbert.duration") }
        var predDur = durations.map { Int32(min(max($0.rounded(), 1), Float(Self.maxFrames))) }
        let frames = predDur.reduce(0) { $0 + Int($1) }
        guard frames <= Self.maxFrames else { throw KokoroANEError.tooManyFrames(frames, max: Self.maxFrames) }
        // G28: under 120 real frames the closing pad token holds the rest, so the decoder runs at its ANE shape (cut below).
        let frameBucket = bucket && frames < Self.bucketFrames
        if frameBucket { predDur[predDur.count - 1] += Int32(Self.bucketFrames - frames) }

        let align = try predict(.alignment, ["pred_dur": try Self.int32([1, predDur.count], predDur),
                                             "d": try out(post, "d", .postAlbert), "t_en": try out(post, "t_en", .postAlbert)], &ms)
        let prosody = try predict(.prosody, ["en": try out(align, "en", .alignment), "style_s": styleSArr], &ms)
        let f0 = try out(prosody, "F0", .prosody), n = try out(prosody, "N", .prosody)
        let noise = try predict(.noise, ["F0_curve": f0, "style_timbre": timbreArr], &ms)
        let voc = try predict(.vocoder, ["asr": try out(align, "asr", .alignment), "F0_curve": f0, "N_pred": n,
                                         "x_source_0": try out(noise, "x_source_0", .noise),
                                         "x_source_1": try out(noise, "x_source_1", .noise), "style_timbre": timbreArr], &ms)
        let tail = try predict(.tail, ["x_pre": try out(voc, "x_pre", .vocoder)], &ms)
        var audio = Self.floats(try out(tail, "audio", .tail))
        if frameBucket {  // back to the real frames, with a 20 ms fade: the vocoder hears the stretched pad up to the cut
            audio = Array(audio.prefix(frames * Self.samplesPerFrame))
            let fade = min(audio.count, Self.sampleRate / 50)
            for k in 0 ..< fade { audio[audio.count - fade + k] *= Float(fade - k) / Float(fade) }
        }
        lastStageMs = ms
        lastFrames = frames
        lastBucketed = frameBucket
        return audio
    }

    /// One short throwaway run (a few ids, row 0 style): lets Core ML specialise the chain before the first real
    /// reply. Each new input length can still cost a little (the models have flexible shapes).
    func prewarm() throws {
        _ = try run([Int64](repeating: 16, count: 12), style: [Float](repeating: 0, count: Self.styleRows * Self.styleDim), speed: 1)
    }

    // MARK: stages

    private func predict(_ stage: Stage, _ inputs: [String: MLMultiArray], _ ms: inout [Stage: Double]) throws -> MLFeatureProvider {
        let types = inputTypes[stage] ?? [:]
        var feed: [String: MLFeatureValue] = [:]
        for (name, arr) in inputs {
            feed[name] = MLFeatureValue(multiArray: try Self.copy(arr, to: types[name] ?? arr.dataType))
        }
        let t0 = DispatchTime.now().uptimeNanoseconds
        do {
            let result = try models[stage]!.prediction(from: try MLDictionaryFeatureProvider(dictionary: feed))
            ms[stage] = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
            return result
        } catch {
            throw KokoroANEError.prediction(stage.rawValue, error.localizedDescription)
        }
    }

    private func out(_ p: MLFeatureProvider, _ name: String, _ stage: Stage) throws -> MLMultiArray {
        guard let a = p.featureValue(for: name)?.multiArrayValue else { throw KokoroANEError.prediction(stage.rawValue, "no output \(name)") }
        return a
    }

    // MARK: arrays (Foundation + CoreML only: Swift's Float16 for the conversions)

    /// Zeroed bytes after every input buffer (FluidAudio's `tailSlackBytes`: BNNS overreads page-sized inputs on OS 27).
    private static let slack = 16_384

    private static func size(_ t: MLMultiArrayDataType) -> Int {
        switch t {
        case .float16: return 2
        case .double: return 8
        default: return 4
        }
    }

    private static func make(_ shape: [Int], _ type: MLMultiArrayDataType) throws -> MLMultiArray {
        let count = shape.reduce(1, *), bytes = count * size(type) + slack
        let buf = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 64)
        buf.initializeMemory(as: UInt8.self, repeating: 0, count: bytes)
        var strides = [Int](repeating: 1, count: shape.count)
        if shape.count > 1 { for i in stride(from: shape.count - 2, through: 0, by: -1) { strides[i] = strides[i + 1] * shape[i + 1] } }
        return try MLMultiArray(dataPointer: buf, shape: shape.map { NSNumber(value: $0) }, dataType: type,
                                strides: strides.map { NSNumber(value: $0) }, deallocator: { $0.deallocate() })
    }

    private static func int32(_ shape: [Int], _ v: [Int32]) throws -> MLMultiArray {
        let a = try make(shape, .int32)
        v.withUnsafeBufferPointer { a.dataPointer.bindMemory(to: Int32.self, capacity: v.count).update(from: $0.baseAddress!, count: v.count) }
        return a
    }

    private static func float32(_ shape: [Int], _ v: [Float]) throws -> MLMultiArray {
        let a = try make(shape, .float32)
        v.withUnsafeBufferPointer { a.dataPointer.bindMemory(to: Float.self, capacity: v.count).update(from: $0.baseAddress!, count: v.count) }
        return a
    }

    private static func contiguous(_ a: MLMultiArray) -> Bool {
        var expect = 1
        for (dim, st) in zip(a.shape.map(\.intValue), a.strides.map(\.intValue)).reversed() {
            if dim > 1, st != expect { return false }
            expect *= max(dim, 1)
        }
        return true
    }

    /// Element offsets (in elements) of a strided array, in row-major logical order.
    private static func offsets(_ a: MLMultiArray) -> [Int] {
        let shape = a.shape.map(\.intValue), strides = a.strides.map(\.intValue)
        var out: [Int] = []
        out.reserveCapacity(a.count)
        var idx = [Int](repeating: 0, count: shape.count)
        for _ in 0 ..< a.count {
            var o = 0
            for d in 0 ..< shape.count { o += idx[d] * strides[d] }
            out.append(o)
            var d = shape.count - 1
            while d >= 0 { idx[d] += 1; if idx[d] < shape[d] { break }; idx[d] = 0; d -= 1 }
        }
        return out
    }

    /// A compact copy of `src` as `type` (with slack): fp16 ↔ fp32 ↔ int32.
    private static func copy(_ src: MLMultiArray, to type: MLMultiArrayDataType) throws -> MLMultiArray {
        let shape = src.shape.map(\.intValue), n = src.count
        let dst = try make(shape, type)
        let s = src.dataPointer, d = dst.dataPointer
        if src.dataType == type, contiguous(src) { d.copyMemory(from: s, byteCount: n * size(type)); return dst }
        let idx: [Int]? = contiguous(src) ? nil : offsets(src)
        func read(_ body: (Int, Float) -> Void) {
            switch src.dataType {
            case .float16: let p = s.assumingMemoryBound(to: Float16.self); for i in 0 ..< n { body(i, Float(p[idx?[i] ?? i])) }
            case .int32: let p = s.assumingMemoryBound(to: Int32.self); for i in 0 ..< n { body(i, Float(p[idx?[i] ?? i])) }
            case .double: let p = s.assumingMemoryBound(to: Double.self); for i in 0 ..< n { body(i, Float(p[idx?[i] ?? i])) }
            default: let p = s.assumingMemoryBound(to: Float.self); for i in 0 ..< n { body(i, p[idx?[i] ?? i]) }
            }
        }
        if src.dataType == .float16, type == .float32, idx == nil {  // the hot conversions, without the closure
            let p = s.assumingMemoryBound(to: Float16.self), q = d.assumingMemoryBound(to: Float.self)
            for i in 0 ..< n { q[i] = Float(p[i]) }
            return dst
        }
        if src.dataType == .float32, type == .float16, idx == nil {
            let p = s.assumingMemoryBound(to: Float.self), q = d.assumingMemoryBound(to: Float16.self)
            for i in 0 ..< n { q[i] = Float16(p[i]) }
            return dst
        }
        switch type {
        case .float16: let q = d.assumingMemoryBound(to: Float16.self); read { q[$0] = Float16($1) }
        case .int32: let q = d.assumingMemoryBound(to: Int32.self); read { q[$0] = Int32($1) }
        case .double: let q = d.assumingMemoryBound(to: Double.self); read { q[$0] = Double($1) }
        default: let q = d.assumingMemoryBound(to: Float.self); read { q[$0] = $1 }
        }
        return dst
    }

    private static func floats(_ a: MLMultiArray) -> [Float] {
        let c = a.dataType == .float32 && contiguous(a) ? a : (try? copy(a, to: .float32)) ?? a
        return Array(UnsafeBufferPointer(start: c.dataPointer.assumingMemoryBound(to: Float.self), count: c.count))
    }
}

enum KokoroANEError: LocalizedError, Equatable {
    /// More phoneme ids than the chain's Albert window (510 + the two pads): split, or run it on the CPU runtime.
    case tooManyIds(Int, max: Int)
    /// More predicted acoustic frames than the models' 2000 (~50 s at 25 ms a frame): split the input.
    case tooManyFrames(Int, max: Int)
    case badStyle(Int)
    case missing(String)
    case nonFinite(String)
    case prediction(String, String)

    var errorDescription: String? {
        switch self {
        case .tooManyIds(let n, let m): return "Kokoro ANE: \(n) phoneme ids, at most \(m) per run"
        case .tooManyFrames(let n, let m): return "Kokoro ANE: \(n) acoustic frames predicted, at most \(m) per run"
        case .badStyle(let n): return "Kokoro ANE: style table has \(n) floats, expected 510 × 256"
        case .missing(let f): return "Kokoro ANE: \(f) missing"
        case .nonFinite(let w): return "Kokoro ANE: non-finite \(w) (Core ML mis-executed the model on this OS)"
        case .prediction(let s, let e): return "Kokoro ANE: stage \(s) failed: \(e)"
        }
    }
}
