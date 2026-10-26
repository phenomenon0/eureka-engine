import AVFoundation
import CoreML
import Foundation

/// VOICESHOP: the app's voice layer (VoiceLibrary + adapters + KokoroVoice + NeuralVoice + VoiceShop) on the Mac, for
/// receipts and the shop's preview clips (scripts/voice-mac.sh). The same Swift the app compiles.
///
///   voice-mac list                                  the packs the library sees (BEND_KOKORO_DIR, BEND_VOICES_DIR)
///   voice-mac phonemes "<text>" [gb]                the G2P's output + the words it did not know
///   voice-mac render <pack-id> "<text>" <out.wav>   one render, timed
///   voice-mac swap <A> <B> <outdir>                 A speaks the demo line, B takes over (timed), B speaks it
///   voice-mac shop <catalog-url> <id>               the shop's catalog → download → sha256 → install → load → speak
///   voice-mac firstsound <dir> [gap] [prep] [cfgs]  G28: first-chunk render per Core ML config (warm / idle / prewarmed)
let args = Array(CommandLine.arguments.dropFirst())
func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }
func out(_ s: String) { print(s); fflush(stdout) }

func writeWav(_ samples: [Float], rate: Int, to path: String) throws {
    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(rate), channels: 1, interleaved: false)!
    let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
    buf.frameLength = buf.frameCapacity
    samples.withUnsafeBufferPointer { buf.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
    let settings: [String: Any] = [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: rate, AVNumberOfChannelsKey: 1,
                                   AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false]
    try? FileManager.default.removeItem(atPath: path)
    let file = try AVAudioFile(forWriting: URL(fileURLWithPath: path), settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
    try file.write(from: buf)
}

let library = VoiceLibrary.shared
func load(_ id: String) throws -> (VoicePack, any VoiceEngine, Double) {
    guard let pack = library.pack(id) else { throw NeuralVoiceError("no pack \(id) (voice-mac list)") }
    let t0 = now()
    let v = try library.adapter(pack.engine)!.load(pack)
    return (pack, v, now() - t0)
}
func speak(_ v: any VoiceEngine, _ text: String, _ path: String?) throws -> String {
    let t0 = now()
    let s = try v.synthesize(text)
    let ms = now() - t0, audio = Double(s.count) / Double(v.sampleRate) * 1000
    if let path { try writeWav(s, rate: v.sampleRate, to: path) }
    return String(format: "render_ms=%.0f audio_ms=%.0f rtf=%.3f", ms, audio, ms / max(audio, 1))
}

do {
    switch args.first {
    case "list":
        for p in library.packs(refresh: true) {
            out("\(p.id)\t\(p.name)\t\(p.summary)\t\(p.origin.rawValue)\t\(p.bytes) B\t\(p.license)\t\((p.path as NSString).lastPathComponent)")
        }
    case "phonemes":
        let rt = try KokoroRuntime(dir: KokoroRuntime.directory()!)
        let g = try rt.phonemizer(british: args.count > 2 && args[2] == "gb")
        let ps = g.phonemize(args[1])
        out(ps)
        out("ids=\(rt.ids(ps).count) unknown=\(g.unknown)")
    case "render":
        let (pack, v, loadMs) = try load(args[1])
        out(String(format: "pack=%@ load_ms=%.0f", pack.id, loadMs))
        _ = try v.synthesize("Ready.")  // as the app's warm-up
        out("render \(try speak(v, args[2], args[3])) → \(args[3])")
    case "swap":
        let dir = args[3]
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let (a, va, loadA) = try load(args[1])
        _ = try va.synthesize("Ready.")
        out(String(format: "A=%@ load_ms=%.0f", a.id, loadA))
        out("A \(try speak(va, VoiceLibrary.demoLine, "\(dir)/A-\(a.id.replacingOccurrences(of: ":", with: "_")).wav"))")
        let (b, vb, swapMs) = try load(args[2])
        out(String(format: "swap A→B=%@ swap_ms=%.1f (same engine: the style table only)", b.id, swapMs))
        out("B \(try speak(vb, VoiceLibrary.demoLine, "\(dir)/B-\(b.id.replacingOccurrences(of: ":", with: "_")).wav"))")
    case "shop":
        // The app's VoiceShop, end to end: catalog → download → sha256 → install → the library lists it → load → speak.
        setenv("BEND_VOICES_CATALOG_URL", args[1], 1)
        let shop = VoiceShop.shared
        await shop.load()
        out("catalog \(shop.listing) url=\(VoiceCatalog.url.absoluteString) entries=\(shop.entries.map(\.id))")
        guard args.count > 2 else { break }
        guard let e = shop.entries.first(where: { $0.id == args[2] }) else { throw NeuralVoiceError("no \(args[2]) in the catalog") }
        let t0 = now()
        shop.download(e)
        while true {
            try await Task.sleep(for: .milliseconds(50))
            if case .installed = shop.rows[e.id] { break }
            if case .failed(let m) = shop.rows[e.id] { throw NeuralVoiceError(m) }
        }
        shop.receipts.forEach { out("shop \($0)") }
        out(String(format: "installed %@ in %.0f ms → %@", e.packID, now() - t0, shop.dest(e).path))
        let (pack, v, loadMs) = try load(e.packID)
        out(String(format: "library lists %@ (origin=%@) load_ms=%.1f", pack.id, pack.origin.rawValue, loadMs))
        if args.count > 3 { out("speak \(try speak(v, VoiceLibrary.demoLine, args[3])) → \(args[3])") }
    case "firstsound":
        // G28: the first chunk's render per Kokoro Core ML config, three ways: warm (back to back, median of 5), after an
        // idle gap (the person talking: `gap` ms, default 3000), and after the same gap with a speech-end prewarm render
        // `prep` ms (default 600: the phone's fastest engine first token) before it. Median of 3 for the last two. WAVs of
        // the warm run per (config, text) → <outdir>; each bucket config is compared with cpu (the shipped chain) on the
        // same ids: corr / SNR over the shorter of the two, and the length difference.
        //   voice-mac firstsound <outdir> [gap_ms] [prep_ms] [configs: cpu,cpu-f120,ane,ane-f120]
        let dir = args[1], gap = args.count > 2 ? Double(args[2])! : 3000, prep = args.count > 3 ? Double(args[3])! : 600
        let want = (args.count > 4 ? args[4] : "cpu,cpu-f120,ane,ane-f120").split(separator: ",").map(String.init)
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let texts: [(String, String)] = [("c4a", "I recommend Leaders Eat"), ("c4b", "I would start with"), ("c4c", "Sure, here is one."),
                                         ("c4d", "Your next meeting is"), ("c4e", "That sounds like a"), ("c8", "I recommend Leaders Eat Last by Simon Sinek"),
                                         ("c12", "because it focuses on how creating a safe environment for your team")]
        let base = try KokoroRuntime(dir: KokoroRuntime.directory()!, threads: 1, coreml: false)
        let g2p = try base.phonemizer(british: false)
        let ids = try texts.map { base.ids(try g2p.phonemize($0.1)) }
        let warmIds = base.ids(try g2p.phonemize("Sure, here is one."))
        guard let bella = library.pack(VoiceLibrary.defaultID) else { throw NeuralVoiceError("no default pack") }
        let style = try KokoroVoice.style(at: bella.url)
        let configs: [(String, MLComputeUnits, Bool)] = [("cpu", .cpuOnly, false), ("cpu-f120", .cpuOnly, true),
                                                         ("ane", .cpuAndNeuralEngine, false), ("ane-f120", .cpuAndNeuralEngine, true)]
        var ref: [String: [Float]] = [:]
        func median(_ x: [Double]) -> Double { x.sorted()[x.count / 2] }
        func pause(_ ms: Double) { usleep(useconds_t(ms * 1000)) }
        for (label, units, bucket) in configs where want.contains(label) {
            let t0 = now()
            let rt = try KokoroANE(dir: KokoroANE.directory()!, units: units, bucket: bucket)
            let loadMs = now() - t0
            let t1 = now()
            _ = try rt.run(base.ids(try g2p.phonemize("Ready.")), style: style, speed: 1)  // as the app's warm-up
            out(String(format: "fs config=%@ load_ms=%.0f first_run_ms=%.0f", label, loadMs, now() - t1))
            for (i, (id, text)) in texts.enumerated() {
                var warm: [Double] = [], audio: [Float] = []
                for _ in 0 ..< 5 { let t = now(); audio = try rt.run(ids[i], style: style, speed: 1); warm.append(now() - t) }
                let stages = KokoroANE.Stage.allCases.map { String(format: "%@=%.1f", String($0.rawValue.dropFirst(6)), rt.lastStageMs[$0] ?? -1) }.joined(separator: " ")
                let bucketed = rt.lastBucketed, frames = rt.lastFrames
                var idle: [Double] = [], pre: [Double] = []
                for _ in 0 ..< 3 {
                    pause(gap); let t = now(); _ = try rt.run(ids[i], style: style, speed: 1); idle.append(now() - t)
                    pause(gap - prep); _ = try rt.run(warmIds, style: style, speed: 1); pause(prep)
                    let u = now(); _ = try rt.run(ids[i], style: style, speed: 1); pre.append(now() - u)
                }
                try writeWav(audio.map { $0 * 1.45 }, rate: KokoroANE.sampleRate, to: "\(dir)/\(label)-\(id).wav")
                var cmp = ""
                if label == "cpu" { ref[id] = audio } else if let r = ref[id] {
                    let n = min(r.count, audio.count)
                    var dot = 0.0, ra = 0.0, aa = 0.0, err = 0.0
                    for k in 0 ..< n { let x = Double(r[k]), y = Double(audio[k]); dot += x * y; ra += x * x; aa += y * y; err += (x - y) * (x - y) }
                    cmp = String(format: " vs_cpu corr=%.4f snr_db=%.1f len_diff_ms=%.0f", dot / max(sqrt(ra * aa), 1e-12), 10 * log10(ra / max(err, 1e-12)),
                                 Double(audio.count - r.count) / 24.0)
                }
                out(String(format: "fs config=%@ text=%@ words=%d ids=%d frames=%d bucket=%@ warm_ms=%.1f idle_ms=%.1f prewarmed_ms=%.1f audio_ms=%.0f%@ | %@",
                           label, id, text.split(separator: " ").count, ids[i].count, frames, bucketed ? "f120" : "none",
                           median(warm), median(idle), median(pre), Double(audio.count) / 24.0, cmp, stages))
            }
        }
    default:
        out("usage: voice-mac list | phonemes <text> [gb] | render <id> <text> <out.wav> | swap <A> <B> <dir> | shop <catalog> <id>")
    }
} catch {
    out("ERROR \(error.localizedDescription)")
    exit(1)
}
