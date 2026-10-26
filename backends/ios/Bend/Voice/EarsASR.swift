#if BEND_EARS
import AVFoundation
import Foundation

/// G38: the ASR tower in the voice path, behind a flag. Hearing is Apple's SpeechAnalyzer (or the on-device
/// SFSpeechRecognizer) unless asked:
///   -voiceASR ears | env BEND_VOICE_ASR=ears | UserDefaults bend.voice.asr = "ears"   (`-engineCheck ears` implies it)
/// Then every turn is heard by EARS (Vendor/EARS.xcframework v3: ios-ears' Metal Qwen3 AuT tower + Qwen3-ASR-0.6B Q8_0 on
/// llama.cpp 2ca15f5, the slim chunker of ios-ears2/ios-ears-carry): the turn's audio is kept (16 kHz mono float), and at
/// release it goes wav → mel → tower → LLM → greedy ids → text in one call. No partials (the text lands at the end), no
/// server, no network. The models are not bundled: Documents/ears (or -earsDir / BEND_EARS_DIR) holds
///   Qwen3-ASR-0.6B-Q8_0.gguf (805 MB) + mmproj-Qwen3-ASR-0.6B-bf16.gguf (379 MB)   (sha: ios-ears models/SHAS.txt)
/// and with them missing the flag is ignored (SpeechAnalyzer hears, as before; the log says why).
///   -earsDump <dir>    keep each turn's wav + the tower's rows (raw f32) there (receipts: compare with ears-cli)
///   -earsLLM gpu|cpu   where the LLM runs (default gpu; cpu in the simulator, ios-ears' finding: its Metal LLM there is
///                      not the phone's). The tower is always Metal.
/// Loaded per turn and freed after (tower first, then the LLM: the vision lane's order); ~1.1 GB peak in the simulator
/// (ios-ears-carry C4), so never resident beside the chat model. One turn at a time.
enum EarsASR {
    static let llmFile = "Qwen3-ASR-0.6B-Q8_0.gguf"
    static let towerFile = "mmproj-Qwen3-ASR-0.6B-bf16.gguf"
    static let label = "ears(Qwen3-ASR-0.6B)"

    private static func value(_ flag: String) -> String? {
        let a = ProcessInfo.processInfo.arguments
        guard let i = a.firstIndex(of: flag), i + 1 < a.count else { return nil }
        return a[i + 1]
    }

    static let asked: Bool = {
        let a = ProcessInfo.processInfo.arguments
        if value("-voiceASR") == "ears" || ProcessInfo.processInfo.environment["BEND_VOICE_ASR"] == "ears" { return true }
        if let i = a.firstIndex(of: "-engineCheck"), i + 1 < a.count, a[i + 1] == "ears" { return true }
        return UserDefaults.standard.string(forKey: "bend.voice.asr") == "ears"
    }()

    static var dir: URL {
        let p = value("-earsDir") ?? ProcessInfo.processInfo.environment["BEND_EARS_DIR"] ?? "ears"
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return URL(fileURLWithPath: p, relativeTo: docs).standardizedFileURL
    }

    static var llmOnGPU: Bool {
        if let v = value("-earsLLM") { return v != "cpu" }
        #if targetEnvironment(simulator)
        return false
        #else
        return true
        #endif
    }

    /// Asked for and the two files are there.
    static var enabled: Bool {
        guard asked else { return false }
        let fm = FileManager.default
        let ok = fm.fileExists(atPath: dir.appendingPathComponent(llmFile).path) && fm.fileExists(atPath: dir.appendingPathComponent(towerFile).path)
        if !ok { warnOnce("ears: asked for, but \(dir.path) lacks \(llmFile) + \(towerFile): SpeechAnalyzer hears") }
        return ok
    }

    nonisolated(unsafe) private static var warned = false
    private static func warnOnce(_ s: String) { if !warned { warned = true; Voice.log(s) } }

    struct Heard: Sendable {
        var raw: String, text: String, ids: [Int32]
        var windows: Int, audioTokens: Int, seconds: Double
        var melMs: Double, towerLoadMs: Double, towerMs: Double, towerGpuMs: Double, llmLoadMs: Double, prefillMs: Double
        var decodeMs: Double, totalMs: Double, detokMs: Double
        var footprintBefore: Int, footprintAfter: Int, peak: Int
        var line: String {
            String(format: "ears heard seconds=%.2f windows=%d audio_tokens=%d ids=%d total_ms=%.1f mel_ms=%.1f tower_load_ms=%.1f tower_ms=%.1f tower_gpu_ms=%.1f llm_load_ms=%.1f prefill_ms=%.1f decode_ms=%.1f detok_ms=%.1f llm=%@ footprint_mb=%d->%d peak_mb=%d",
                   seconds, windows, audioTokens, ids.count, totalMs, melMs, towerLoadMs, towerMs, towerGpuMs, llmLoadMs, prefillMs, decodeMs,
                   detokMs, EarsASR.llmOnGPU ? "gpu" : "cpu", footprintBefore, footprintAfter, peak)
        }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private(set) static var last: Heard?
    nonisolated(unsafe) private static var dumped = 0

    /// Qwen3-ASR answers "language English<asr_text>The words." — the words.
    static func words(_ raw: String) -> String {
        guard let r = raw.range(of: "<asr_text>") else { return raw.trimmingCharacters(in: .whitespacesAndNewlines) }
        return String(raw[r.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The lifetime high-water mark of phys_footprint (the Jetsam ledger), MB.
    static func peakFootprint() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return kr == KERN_SUCCESS ? Int(info.ledger_phys_footprint_peak / 1_048_576) : -1
    }

    /// One wav (16 kHz mono, s16 or f32) → the text. Blocking; serialized.
    static func transcribe(wav: URL, seconds: Double) throws -> Heard {
        lock.lock(); defer { lock.unlock() }
        let llm = dir.appendingPathComponent(llmFile).path, tower = dir.appendingPathComponent(towerFile).path
        let cap = 512
        var ids = [Int32](repeating: 0, count: cap), top1 = [Float](repeating: 0, count: cap), top2 = [Float](repeating: 0, count: cap)
        var st = ears_stats()
        var err = [CChar](repeating: 0, count: 512)
        let before = VisionCheck.footprint()
        var embdOut: String?
        if let dump = value("-earsDump") {
            try? FileManager.default.createDirectory(atPath: dump, withIntermediateDirectories: true)
            dumped += 1
            let stem = URL(fileURLWithPath: dump).appendingPathComponent(String(format: "turn%02d", dumped))
            try? FileManager.default.copyItem(at: wav, to: stem.appendingPathExtension("wav"))
            embdOut = stem.appendingPathExtension("embd.f32").path
        }
        let n = ears_transcribe(llm, tower, wav.path, Int32(cap), llmOnGPU ? 1 : 0, embdOut, &ids, &top1, &top2, Int32(cap), &st, &err)
        guard n > 0 else { throw VoiceError("ears: \(Self.string(err))") }
        let t0 = Voice.now()
        var out = [CChar](repeating: 0, count: 8192)
        let k = ears_detokenize(llm, ids, n, &out, Int32(out.count), &err)
        guard k >= 0 else { throw VoiceError("ears: \(Self.string(err))") }
        let raw = string(out)
        if let m = (0..<Int(n)).min(by: { top1[$0] - top2[$0] < top1[$1] - top2[$1] }) {
            Voice.log(String(format: "ears closest step=%d id=%d top1=%a top2=%a margin=%.6f", m, ids[m], top1[m], top2[m], top1[m] - top2[m]))
        }
        let heard = Heard(raw: raw, text: words(raw), ids: Array(ids.prefix(Int(n))), windows: Int(st.n_windows), audioTokens: Int(st.n_audio_tokens),
                          seconds: seconds, melMs: st.mel_ms, towerLoadMs: st.tower_load_ms, towerMs: st.tower_ms, towerGpuMs: st.tower_gpu_ms,
                          llmLoadMs: st.llm_load_ms, prefillMs: st.prefill_ms, decodeMs: st.decode_ms, totalMs: st.total_ms,
                          detokMs: Voice.now() - t0, footprintBefore: before, footprintAfter: VisionCheck.footprint(), peak: peakFootprint())
        last = heard
        Voice.log(heard.line)
        return heard
    }

    private static func string(_ c: [CChar]) -> String {
        String(decoding: c.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// 16 kHz mono IEEE float wav (what ears' read_wav takes as is: no requantization).
    static func writeWav(_ samples: [Float], to url: URL) throws {
        var d = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        let bytes = UInt32(samples.count * 4)
        d.append(contentsOf: Array("RIFF".utf8)); u32(36 + bytes); d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(3); u16(1); u32(16000); u32(64000); u16(4); u16(32)
        d.append(contentsOf: Array("data".utf8)); u32(bytes)
        samples.withUnsafeBytes { d.append(contentsOf: $0) }
        try d.write(to: url)
    }
}

/// One turn heard by EARS: audio in (any PCM format, converted to 16 kHz mono float as it comes), the text at the end.
/// Under 0.3 s of audio (prepare's warm-up silence, a tap) is nothing said: "" without loading anything.
final class EarsListener: Hearing, @unchecked Sendable {
    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    private let lock = NSLock()
    private var samples: [Float] = []
    private var converter: AVAudioConverter?
    private var cancelled = false
    private let onPartial: @Sendable (String) -> Void

    init(onPartial: @escaping @Sendable (String) -> Void) { self.onPartial = onPartial }

    private func append(_ b: AVAudioPCMBuffer) {
        guard let ch = b.floatChannelData else { return }
        lock.withLock { samples.append(contentsOf: UnsafeBufferPointer(start: ch[0], count: Int(b.frameLength))) }
    }

    /// 16 kHz mono float in any layout (a fixture's AVAudioFile format does not compare == to ours) is taken as is: the
    /// samples ears hears are the clip's, bit for bit. Anything else goes through the converter (the mic: 48 kHz).
    func feed(_ buffer: AVAudioPCMBuffer) {
        let f = buffer.format
        if f.commonFormat == .pcmFormatFloat32, f.sampleRate == 16000, f.channelCount == 1 { append(buffer); return }
        let converter = lock.withLock { () -> AVAudioConverter? in
            if self.converter?.inputFormat != buffer.format { self.converter = AVAudioConverter(from: buffer.format, to: format) }
            return self.converter
        }
        guard let converter else { return }
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * format.sampleRate / buffer.format.sampleRate + 64)
        guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else { return }
        var given = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if given { status.pointee = .noDataNow; return nil }
            given = true
            status.pointee = .haveData
            return buffer
        }
        if error == nil, out.frameLength > 0 { append(out) }
    }

    /// The converter keeps its last partial chunk until told the stream ended: drain it, or a turn loses its final ~35 ms
    /// (found by G38's ragged20 check: a strict prefix, 567 samples short, flipped "Rocky" to "rocky").
    private func drain() {
        guard let converter = lock.withLock({ converter }) else { return }
        while let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4096) {
            var error: NSError?
            converter.convert(to: out, error: &error) { _, status in status.pointee = .endOfStream; return nil }
            if error != nil || out.frameLength == 0 { break }
            append(out)
        }
    }

    func finish() async throws -> String {
        drain()
        let pcm = lock.withLock { samples }
        guard pcm.count >= 4800, !lock.withLock({ cancelled }) else { return "" }
        let wav = FileManager.default.temporaryDirectory.appendingPathComponent("ears-turn-\(UUID().uuidString).wav")
        try EarsASR.writeWav(pcm, to: wav)
        defer { try? FileManager.default.removeItem(at: wav) }
        let seconds = Double(pcm.count) / 16000
        let heard = try await Task.detached(priority: .userInitiated) { try EarsASR.transcribe(wav: wav, seconds: seconds) }.value
        onPartial(heard.text)
        return heard.text
    }

    func cancel() async { lock.withLock { cancelled = true; samples = [] } }
}
#endif
