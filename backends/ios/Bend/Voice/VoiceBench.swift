import CoreML
import Foundation

/// VOICESHOP2: `-voiceBench`: what the first chunk costs to render on this device, per Kokoro runtime, before any turn.
/// The early-speak pieces (4, 8 and 14 words: Speaker.earlySplit's caps) in Bella, each runtime loaded alone and then
/// released: onnxruntime on 2 / 4 CPU threads, and the Neural Engine chain (KokoroANE) when its models are bundled.
/// One line per (runtime, text): load, cold first call, warm median of 3; then `bench best …` and `bench done`.
/// Nothing else runs meanwhile (no chat turn), so these are the floor; a turn's `timer` line is the number under load.
enum VoiceBench {
    static let texts = [
        "Sure, here's one.",
        "I would start with Leaders Eat Last by",
        "Simon Sinek, because it is short and every chapter ends with something you can try.",
    ]

    static func run() async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            DispatchQueue.global(qos: .userInitiated).async { measure(); c.resume() }
        }
    }

    private static func measure() {
        guard let dir = KokoroRuntime.directory(),
              let bella = VoiceLibrary.shared.packs(refresh: true).first(where: { $0.id == VoiceLibrary.defaultID }),
              let style = try? KokoroVoice.style(at: bella.url),
              let base = try? KokoroRuntime(dir: dir, threads: 1, coreml: false), let g2p = try? base.phonemizer(british: false),
              let ids = try? texts.map({ base.ids(try g2p.phonemize($0)) })
        else { Voice.log("bench done: no Kokoro files"); return }
        var best: (String, Double)?
        func one(_ label: String, load: () throws -> ([Int64]) throws -> [Float]) {
            let t0 = Voice.now()
            let synth: ([Int64]) throws -> [Float]
            do { synth = try load() } catch { Voice.log("bench runtime=\(label) load failed: \(error.localizedDescription)"); return }
            let loadMs = Voice.now() - t0
            for (i, text) in texts.enumerated() {
                var times: [Double] = [], samples = 0
                for _ in 0 ..< 4 {
                    let t = Voice.now()
                    guard let out = try? synth(ids[i]) else { Voice.log("bench runtime=\(label) words=\(Speaker.words(text)) failed"); return }
                    times.append(Voice.now() - t)
                    samples = out.count
                }
                let warm = times.dropFirst().sorted()[1]
                let audio = Double(samples) / Double(KokoroRuntime.sampleRate) * 1000
                Voice.log(String(format: "bench runtime=%@ words=%d load_ms=%.0f cold_ms=%.0f warm_ms=%.0f audio_ms=%.0f rtf=%.2f", label, Speaker.words(text), loadMs, times[0], warm, audio, warm / max(audio, 1)))
                if i == 1, best == nil || warm < best!.1 { best = (label, warm) }
            }
        }
        for threads: Int32 in [2, 4] {
            one("onnx-cpu-\(threads)t") {
                let rt = try KokoroRuntime(dir: dir, threads: threads, coreml: false)
                return { try rt.run($0, style: style, speed: 1) }
            }
        }
        if let ane = KokoroANE.directory() {
            // G28: + the 120-frame decoder bucket (the 4-word piece fits it; the 8- and 14-word ones show the fallback).
            for (label, units, bucket) in [("ane", MLComputeUnits.cpuAndNeuralEngine, false), ("coreml-cpu", MLComputeUnits.cpuOnly, false),
                                           ("ane-f120", MLComputeUnits.cpuAndNeuralEngine, true), ("coreml-cpu-f120", MLComputeUnits.cpuOnly, true)] {
                one(label) {
                    let rt = try KokoroANE(dir: ane, units: units, bucket: bucket)
                    return { try rt.run($0, style: style, speed: 1) }
                }
            }
        } else {
            Voice.log("bench runtime=ane not bundled (ios/Vendor/voices/kokoro-ane)")
        }
        if let best { Voice.log(String(format: "bench best runtime=%@ first_chunk_8w_ms=%.0f", best.0, best.1)) }
        Voice.log("bench done")
    }
}
