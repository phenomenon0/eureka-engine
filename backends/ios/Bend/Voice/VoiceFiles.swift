import AVFoundation
import Foundation
import Speech

// MACVOICE: the file modes of the voice contract, the headless testable path (no mic, no speaker, no grant):
//   POST /voice/synthesize {text, file?}   AVSpeechSynthesizer renders `text` into a 16-bit mono WAV, nothing is played
//   POST /voice/transcribe {file}          the recognizer (SpeechAnalyzer; SFSpeechRecognizer on-device as the fallback)
//                                          hears the audio file, fed faster than real time → {text, heardBy, …}
//   POST /voice/permission                 asks for the microphone (the system prompt, once) → {mic}
// The same Hearing / voice the live turn uses; only the source (a file) and the sink (a file) differ.

extension Voice {
    #if !os(iOS)
    /// The first real press asks for the microphone (the system's prompt, once); a refusal says where to fix it.
    static func askMic() async throws {
        if micState == "undetermined" { _ = await AVCaptureDevice.requestAccess(for: .audio) }
        guard micState == "granted" else {
            throw VoiceError("Eureka can’t hear you: allow the microphone in System Settings › Privacy & Security › Microphone.")
        }
    }
    #endif

    static func askMicState() async -> [String: Any] {
        #if !os(iOS)
        try? await askMic()
        #endif
        return ["mic": micState]
    }

    static var outputDir: URL { Speaker.dir.appendingPathComponent("files") }

    /// `text` → a 16-bit mono WAV at Apple's voice rate (22.05 kHz). Never plays.
    static func synthesize(_ text: String, file: String?, voice: AVSpeechSynthesisVoice) async throws -> [String: Any] {
        let words = text.split(whereSeparator: { $0 == " " || $0 == "\n" }).count
        guard words > 0 else { throw VoiceError("nothing to say") }
        let url: URL
        if let file, !file.isEmpty { url = URL(fileURLWithPath: (file as NSString).expandingTildeInPath) } else {
            try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
            url = outputDir.appendingPathComponent("say-\(Int(Date().timeIntervalSince1970)).wav")
        }
        try? FileManager.default.removeItem(at: url)
        let t0 = now()
        let synth = AVSpeechSynthesizer()
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = voice
        utterance.prefersAssistiveTechnologySettings = false
        let (frames, rate): (Int, Double) = try await withCheckedThrowingContinuation { (c: CheckedContinuation<(Int, Double), Error>) in
            let state = NSLock()
            nonisolated(unsafe) var out: AVAudioFile?
            nonisolated(unsafe) var total = 0, rate = 0.0, done = false, failure: Error?
            synth.write(utterance) { buffer in
                guard let pcm = buffer as? AVAudioPCMBuffer else { return }
                state.lock(); defer { state.unlock() }
                if done { return }
                if pcm.frameLength == 0 {
                    done = true
                    if let failure { c.resume(throwing: failure) } else { c.resume(returning: (total, rate)) }
                    return
                }
                do {
                    if out == nil {
                        rate = pcm.format.sampleRate
                        out = try AVAudioFile(forWriting: url, settings: [
                            AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: rate, AVNumberOfChannelsKey: 1,
                            AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false,
                        ], commonFormat: pcm.format.commonFormat, interleaved: pcm.format.isInterleaved)
                    }
                    try out?.write(from: pcm)
                    total += Int(pcm.frameLength)
                } catch { failure = failure ?? error }
            }
        }
        guard frames > 0 else { throw VoiceError("the synthesizer produced no audio for “\(text)”") }
        let bytes = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? Int ?? 0
        let result: [String: Any] = [
            "file": url.path, "bytes": bytes, "audioMs": (Double(frames) / rate * 1000).rounded(), "sampleRate": Int(rate),
            "renderMs": (now() - t0).rounded(), "voice": voice.name, "voiceQuality": voice.quality.rawValue, "text": text, "engine": "AVSpeechSynthesizer",
        ]
        log("synthesize text=\"\(text)\" → \(url.path) bytes=\(bytes) audio_ms=\(result["audioMs"]!) render_ms=\(result["renderMs"]!) voice=\(voice.name)")
        return result
    }

    /// An audio file (any format AVAudioFile reads) → its words, through the recognizer a live turn uses.
    static func transcribe(file path: String) async throws -> [String: Any] {
        let url = try fixtureURL((path as NSString).expandingTildeInPath)
        let audio = try AVAudioFile(forReading: url)
        let t0 = now()
        let hearing = try await Self.hearing(ask: true, onPartial: { _ in })
        let step = AVAudioFrameCount(audio.processingFormat.sampleRate)  // one-second buffers, as fast as the analyzer takes them
        while audio.framePosition < audio.length {
            guard let buffer = AVAudioPCMBuffer(pcmFormat: audio.processingFormat, frameCapacity: step) else { break }
            try audio.read(into: buffer, frameCount: step)
            if buffer.frameLength == 0 { break }
            hearing.feed(buffer)
        }
        // Half a second of quiet after the last word, as a person's pause would be: the recognizer finalizes cleanly.
        if let tail = AVAudioPCMBuffer(pcmFormat: audio.processingFormat, frameCapacity: AVAudioFrameCount(audio.processingFormat.sampleRate / 2)) {
            tail.frameLength = tail.frameCapacity
            hearing.feed(tail)
        }
        let text = try await hearing.finish()
        let ms = (now() - t0).rounded()
        let audioMs = (Double(audio.length) / audio.processingFormat.sampleRate * 1000).rounded()
        log("transcribe file=\(url.path) audio_ms=\(audioMs) took_ms=\(ms) heardBy=\(recognizer) text=\"\(text)\"")
        return ["text": text, "heardBy": recognizer, "file": url.path, "audioMs": audioMs, "ms": ms]
    }
}
