import Foundation

/// IOS-FOLD F1–F3: the provider's five verb handlers.
///
/// These are route *semantics*, not a vendored `serve.c`: the app already owns the engine (`LLBEngine`, static, in this
/// process), the lifecycle (`admit`/`ended`), the vision path (`VisionLook`), the speech pipeline (`Voice`) and the
/// voice library (`VoiceLibrary`/Kokoro). Each handler parses the contract body, refuses invalid input with the
/// canonical envelope, then runs the real in-process path on the engine thread (or a detached task for voice work).
extension EngineManager {
    // MARK: decide — POST /v1/decide (F2)

    /// `/v1/decide` is the five-verb name for `/v1/systemone`; `/v1/systemone` stays for the web app.
    /// Body is forwarded with `model` filled when absent; `int` is refused with `param: questions.<name>.type`.
    nonisolated func decide(_ x: HTTPExchange) {
        let request: [String: Any]
        do {
            request = try ProviderContract.object(x.body)
            try Self.validateDecide(request)
        } catch let e as ProviderContract.APIError {
            return x.fail(e)
        } catch {
            return x.fail(.invalid(error.localizedDescription))
        }
        let model = ProviderContract.model(request, EngineID.decision.model)
        // The body is forwarded byte for byte (llb_decide owns its parser); `model` is filled in the reply instead.
        let body = String(decoding: x.body, as: UTF8.self)
        guard present(.decision, x) else { return }
        Task { @MainActor in
            guard await self.admit(.decision, x) else { return }
            self.runDecide(x, body: body, model: model)
        }
    }

    /// SCHEMAS §2 plus the pocket head's declared subset. Throws `ProviderContract.APIError` (400, with `param`).
    nonisolated static func validateDecide(_ request: [String: Any]) throws {
        guard let state = request["state"] as? String, !state.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ProviderContract.invalid("state: expected a non-empty string", param: "state")
        }
        guard let questions = request["questions"] as? [String: Any], !questions.isEmpty else {
            throw ProviderContract.invalid("questions: expected a non-empty object of question specs", param: "questions")
        }
        for (name, value) in questions {
            guard let q = value as? [String: Any] else {
                throw ProviderContract.invalid("questions.\(name): expected an object", param: "questions.\(name)")
            }
            guard let type = q["type"] as? String else {
                throw ProviderContract.invalid("questions.\(name).type: expected a string", param: "questions.\(name).type")
            }
            switch type {
            case "noul", "score", "choice":
                break
            case "int":
                throw ProviderContract.invalid(
                    "questions.\(name).type: \"int\" is not supported by this device's decide head (noul|score|choice); the clef head that serves int does not fit the 6 GB law",
                    param: "questions.\(name).type")
            default:
                throw ProviderContract.invalid("questions.\(name).type: unknown type \"\(type)\" (noul|score|choice)",
                                               param: "questions.\(name).type")
            }
            if let instructions = q["instructions"], !(instructions is String) {
                throw ProviderContract.invalid("questions.\(name).instructions: expected a string", param: "questions.\(name).instructions")
            }
        }
    }

    // MARK: see — POST /v1/see (F3)

    /// One image in, text out (`VisionLook` → `llb_generate_rgb`), or `task: "detect"` for boxes through the pocket
    /// detector when it is staged. Unsupported detect is an honest 501 with its reasons in the body; missing vision is
    /// 503 `no-vision: …` (the same bytes in simulator and device).
    nonisolated func see(_ x: HTTPExchange) {
        let request: [String: Any]
        do {
            request = try ProviderContract.object(x.body)
        } catch let e as ProviderContract.APIError {
            return x.fail(e)
        } catch {
            return x.fail(.invalid(error.localizedDescription))
        }
        let model = ProviderContract.model(request, EngineID.chat.model)
        guard let source = (request["image"] as? String) ?? (request["image_b64"] as? String) else {
            return x.fail(.invalid("image: expected a data: URL (or image_b64)", param: "image"))
        }
        let task = (request["task"] as? String) ?? "text"
        if task == "detect" {
            return detect(x, source: source, model: model)
        }
        guard ["text", "describe", "look"].contains(task) else {
            return x.fail(.invalid("task: expected \"text\" or \"detect\"", param: "task"))
        }
        do {
            _ = try ProviderContract.base64(source, param: "image")  // 400 before admitting anything
            _ = try VisionLook.decode(source)
        } catch let e as ProviderContract.APIError {
            return x.fail(e)
        } catch {
            return x.fail(.invalid("image: \(error.localizedDescription)", param: "image"))
        }
        let prompt = ProviderContract.optionalString(request, "prompt") ?? VisionLook.defaultQuestion
        let asked = (request["max_tokens"] as? Int) ?? VisionLook.defaultMaxNew
        let maxNew = min(max(asked, 1), VisionLook.maxNew)
        #if BEND_CAMERA
        if let why = VisionLook.unavailableReason {
            return x.fail(.unavailable("no-vision: \(why)"))
        }
        guard present(.chat, x) else { return }
        Task { @MainActor in
            guard await self.admit(.chat, x) else { return }
            Self.onEngineThread("bend.engine.see") {
                let t0 = Date()
                do {
                    let image = try VisionLook.decode(source)
                    let r = try VisionLook.shared.look(image, question: prompt, maxNew: maxNew)
                    x.json(200, ["model": model, "text": r.text,
                                 "usage": ["prompt_tokens": r.promptIds, "completion_tokens": r.ids.count,
                                           "total_tokens": r.promptIds + r.ids.count],
                                 "llb": ["images": 1, "encode_ms": r.towerMsSum, "tower_calls": r.towerCalls,
                                         "image_tokens": r.imageTokens, "rows_kept": r.rowsKept,
                                         "first_ms": r.firstMs, "total_ms": r.totalMs,
                                         "wall_ms": Date().timeIntervalSince(t0) * 1000,
                                         "program": LLBEngine.programTag]])
                    Self.log(String(format: "see model=%@ text_chars=%d image_tokens=%d first_ms=%.0f total_ms=%.0f footprint_mb=%d",
                                    model, r.text.count, r.imageTokens, r.firstMs, r.totalMs, Self.footprint()))
                    Task { @MainActor in self.ended(.chat, loadMs: 0, died: nil) }
                } catch {
                    Self.log("see error: \(error.localizedDescription)")
                    x.fail(.internalError("see: \(error.localizedDescription)"))
                    Task { @MainActor in self.ended(.chat, loadMs: 0, died: nil) }
                }
            }
        }
        #else
        x.fail(.unavailable("no-vision: this build has no vision path (the camera engine is compiled into the Bend target only)"))
        #endif
    }

    /// `task: "detect"`. The detector is the flag-only pocket stack (RF-DETR-S int8 / D-FINE-N + segmenter); when it is
    /// not enabled or not staged this is a 501/503 with reasons, never a guessed box.
    nonisolated private func detect(_ x: HTTPExchange, source: String, model: String) {
        do {
            _ = try ProviderContract.base64(source, param: "image")
        } catch let e as ProviderContract.APIError {
            return x.fail(e)
        } catch {
            return x.fail(.invalid("image: \(error.localizedDescription)", param: "image"))
        }
        guard VisionPocket.enabled else {
            return x.fail(.unsupported("see task \"detect\": no detection pipeline in this build (the vision pocket is off)",
                                       param: "task"),
                          extra: ["supported": ["text"],
                                  "reasons": ["the detector stack (RF-DETR-S int8 / D-FINE-N ONNX + segmenter) is flag-only on iOS (-visionPocket) and its packages are not in ModelManifest",
                                              "the native camera path returns a yes/no verdict from llb_classify_rgb, not boxes",
                                              "see text is served by VisionLook + the 4B vision tower"]])
        }
        Task.detached(priority: .userInitiated) {
            guard let pocket = await PocketHost.shared.get(log: { Self.log($0) }) else {
                return x.fail(.unavailable("no-vision: the pocket packages are not staged (Documents/ml)"),
                              extra: ["supported": ["text"], "reasons": ["-visionPocket is on but the detector/segmenter packages were not found under Documents/ml"]])
            }
            do {
                let image = try VisionLook.decode(source)
                let a = try pocket.run(image)
                var body = VisionPocket.json(a, segName: pocket.segName, detName: pocket.detName, withMask: false)
                body["model"] = model
                body["task"] = "detect"
                x.json(200, body)
                Self.log("see detect boxes=\(a.boxes.count) det=\(pocket.detName) seg=\(pocket.segName) total_ms=\(a.total)")
            } catch {
                Self.log("see detect error: \(error.localizedDescription)")
                x.fail(.internalError("see detect: \(error.localizedDescription)"))
            }
        }
    }

    // MARK: hear — POST /v1/hear (F4 bridge; device WER gate is F4)

    /// One WAV data URL → transcript through the same speech pipeline the voice surface uses (`Voice.transcribe`:
    /// SpeechAnalyzer, SFSpeechRecognizer fallback, or Ears when that flag is on). `timestamps: true` is refused: segment
    /// timing is not wired on iOS yet.
    nonisolated func hear(_ x: HTTPExchange) {
        let request: [String: Any]
        do {
            request = try ProviderContract.object(x.body)
        } catch let e as ProviderContract.APIError {
            return x.fail(e)
        } catch {
            return x.fail(.invalid(error.localizedDescription))
        }
        if (request["timestamps"] as? Bool) == true {
            return x.fail(.invalid("timestamps: segment timing is not wired on iOS yet; send timestamps:false",
                                   param: "timestamps"))
        }
        guard let source = (request["audio"] as? String) ?? (request["audio_b64"] as? String) else {
            return x.fail(.invalid("audio: expected a WAV data: URL (or audio_b64)", param: "audio"))
        }
        let data: Data
        do {
            data = try ProviderContract.base64(source, param: "audio")
        } catch let e as ProviderContract.APIError {
            return x.fail(e)
        } catch {
            return x.fail(.invalid("audio: \(error.localizedDescription)", param: "audio"))
        }
        guard ProviderContract.isWAV(data) else {
            return x.fail(.invalid("audio: expected a RIFF/WAVE PCM16 data: URL", param: "audio"))
        }
        #if targetEnvironment(simulator)
        // The simulator has no SpeechTranscriber asset (IOS-PREP §3.5); Voice.transcribe waits for one that never
        // arrives. Say so instead of hanging. The real F4 gate is a device run.
        return x.fail(.unavailable("hear: the simulator has no SpeechTranscriber asset (no on-device ASR); F4's transcript gate runs on device"))
        #else
        let model = ProviderContract.model(request, "apple-speechanalyzer")
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("bend-hear-\(UUID().uuidString).wav")
        do {
            try data.write(to: url, options: .atomic)
        } catch {
            return x.fail(.internalError("hear: couldn't stage the clip: \(error.localizedDescription)"))
        }
        Task.detached(priority: .userInitiated) {
            defer { try? FileManager.default.removeItem(at: url) }
            do {
                let r = try await Self.transcribeBounded(url.path, seconds: 30)
                let text = r["text"] as? String ?? ""
                let audioMs = (r["audioMs"] as? NSNumber)?.doubleValue ?? 0
                let took = (r["ms"] as? NSNumber)?.doubleValue ?? 0
                x.json(200, ["model": model, "text": text,
                             "duration_s": (audioMs / 100).rounded() / 10,
                             "usage": ["audio_tokens": 0],
                             "llb": ["decode_ms": took, "engine": r["heardBy"] as? String ?? "unknown"]])
                Self.log("hear text_chars=\(text.count) audio_ms=\(Int(audioMs)) took_ms=\(Int(took))")
            } catch {
                Self.log("hear error: \(error.localizedDescription)")
                x.fail(.unavailable("hear: \(error.localizedDescription)"))
            }
        }
        #endif
    }

    /// `Voice.transcribe` bounded: an absent speech asset must not hold an HTTP request open forever.
    nonisolated static func transcribeBounded(_ path: String, seconds: Double) async throws -> [String: Any] {
        let once = OnceFlag()
        return try await withCheckedThrowingContinuation { (c: CheckedContinuation<[String: Any], Error>) in
            Task.detached(priority: .userInitiated) {
                do {
                    let r = try await Voice.transcribe(file: path)
                    if once.claim() { c.resume(returning: r) }
                } catch {
                    if once.claim() { c.resume(throwing: error) }
                }
            }
            Task.detached {
                try? await Task.sleep(for: .seconds(seconds))
                if once.claim() {
                    c.resume(throwing: ProviderContract.unavailable("transcription timed out after \(Int(seconds)) s"))
                }
            }
        }
    }

    // MARK: speak — POST /v1/speak (F5 bridge; RIFF gate is F5)

    /// Text → 24 kHz mono PCM16 WAV through the bundled Kokoro voice (`VoiceLibrary` → `KokoroVoice`), the same engine
    /// the speaker uses. The WAV header is validated before the 200; an empty render is 503, never a truncated body.
    nonisolated func speak(_ x: HTTPExchange) {
        let request: [String: Any]
        do {
            request = try ProviderContract.object(x.body)
            let text = try ProviderContract.string(request, "text", param: "text")
            let scalars = text.unicodeScalars
            guard !text.isEmpty, scalars.count <= 2000 else {
                return x.fail(.invalid("text: expected 1–2000 characters", param: "text"))
            }
            guard scalars.allSatisfy({ (32...126).contains($0.value) || $0 == "\n" || $0 == "\t" }) else {
                return x.fail(.invalid("text: expected printable ASCII (English)", param: "text"))
            }
            let style = ProviderContract.optionalString(request, "style") ?? "neutral"
            let speed: Float
            switch style {
            case "neutral": speed = 1.0
            case "slow": speed = 0.8
            case "fast": speed = 1.25
            case "pitch_up", "pitch_down":
                return x.fail(.invalid("style: \"\(style)\" is deferred on iOS (Kokoro has no f0-offset drive here); use neutral|slow|fast",
                                       param: "style"))
            default:
                return x.fail(.invalid("style: expected neutral|slow|fast", param: "style"))
            }
            let asked = ProviderContract.optionalString(request, "voice") ?? "af_heart"
            let id = asked.contains(":") ? asked : "kokoro:\(asked)"
            let library = VoiceLibrary.shared
            _ = library.packs(refresh: true)
            guard let pack = library.pack(id) ?? library.pack(asked) else {
                return x.fail(.invalid("voice: \"\(asked)\" is not installed on this phone (voice_not_installed)",
                                       param: "voice"))
            }
            guard let adapter = library.adapter(pack.engine), adapter.available else {
                return x.fail(.unavailable("speak: the \(pack.engine) engine is not available in this build"))
            }
            let bare = pack.id.split(separator: ":", maxSplits: 1).last.map(String.init) ?? pack.id
            let t0 = Date()
            Task.detached(priority: .userInitiated) {
                do {
                    let engine = try adapter.load(pack)
                    if let kokoro = engine as? KokoroVoice { kokoro.speed = speed }
                    let samples = try engine.synthesize(text)
                    let wav = ProviderContract.wav(samples, rate: engine.sampleRate)
                    guard ProviderContract.isWAV(wav) else {
                        return x.fail(.unavailable("speak: the synthesizer produced an empty or truncated WAV"))
                    }
                    let ms = Date().timeIntervalSince(t0) * 1000
                    x.reply(200, wav, type: "audio/wav", headers: [
                        "X-LLB-Sample-Rate": "\(engine.sampleRate)",
                        "X-LLB-Voice": bare,
                        "X-LLB-Style": style,
                        "X-LLB-G2P": engine is KokoroVoice ? "misaki" : "native",
                        "X-LLB-Speak-Ms": String(format: "%.1f", ms),
                    ])
                    Self.log(String(format: "speak voice=%@ style=%@ samples=%d rate=%d bytes=%d ms=%.1f",
                                    bare, style, samples.count, engine.sampleRate, wav.count, ms))
                } catch {
                    Self.log("speak error: \(error.localizedDescription)")
                    x.fail(.unavailable("speak: synthesis failed: \(error.localizedDescription)"))
                }
            }
        } catch let e as ProviderContract.APIError {
            return x.fail(e)
        } catch {
            return x.fail(.invalid(error.localizedDescription))
        }
    }
}

/// First caller wins; the loser leaves the continuation alone.
private final class OnceFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false

    func claim() -> Bool {
        lock.withLock {
            if done { return false }
            done = true
            return true
        }
    }
}
